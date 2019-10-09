import os
import sys
sys.path.insert(0, '../')
sys.path.insert(0, '../../')
import random
import numpy as np
import torch
import torch.nn.functional as F

from collections import defaultdict
from rotm2euler import *
from test_utils import *
from experiment_utils import *
from layers import *

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import collections as mc
import pdb

def read_pair_list(pair_path):
    f = open(pair_path)
    lines = f.readlines()

    frames = []
    for ln in lines:
        split = ln.strip().split()
        scn = split[0]
        tgt_id, src_id = int(split[1]), int(split[2])
        tup = (scn, tgt_id, src_id)
        frames.append(tup)

    # print the first 3 results
    for f in frames[:3]:
        print(f)
    
    return frames

def load_data(record, root, img_width=320, img_height=240):
    """
    -------Input-------
    record: string including scene, tgt_frame, src_frame. ex. "scene0020_00 123 222"
    root: dictionary containing different dataset directories
    -------Output------
    data: dictionary with different data
     * 'tgt': target rgb image [H, W, 3]
     * 'src': source(1) rgb image [H, W, 3]
     * 'pose': pose from target to source [4, 4]
     * 'K': intrinsic matrix [3, 3]
     * 'norm': normal map of tgt [H, W, 3]
     * 'depth': depth map of tgt [H, W]
     * 'mask': mask that masks out empty areas on the depth and normal map [H, W]
     
    """
    
    def load_image(data_root, scn, fid, w=320):
        img_path = os.path.join(data_root, scn, str(fid) + ".jpg")
        im = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) / 255 # convert to [0, 1]
        im = im.astype(np.float32)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        
        tgt, src1 = im[:, w:w*2, :], im[:, w*2:, :]
        return tgt, src1
    
    def load_intrinsics(data_root, scn, fid):
        intr_path = img_path = os.path.join(data_root, scn, str(fid) + "_cam.txt")
        mat = np.loadtxt(intr_path, delimiter=',').reshape(3, 3)
        return mat
    
    def load_pose(pose_root, scn, tgt_id, src_id):
        pose_path = os.path.join(pose_root, scn+'.txt')
        with open(pose_path, 'r') as f:
            pose_all = f.readlines()

        for p in pose_all:
            char = p.split(" ")
            if char[0] == str(tgt_id) and char[1] == str(src_id):
                pose = untransform(np.array([float(c.replace(",", "")) for c in char[2:]]))
                break
            elif char[0] == str(src_id) and char[1] == str(tgt_id):
                tmp = np.array([float(c.replace(",", "")) for c in char[2:]])
                tmp = untransform(tmp)
                pose_inv = inversePose(tmp)
                pose = pose_inv
                break
        return pose

    def load_depth(depth_root, scn, fid, h=240, w=320):
        img_path = os.path.join(depth_root, scn, 'depth', str(fid)+".png")
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) / 1000
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_NEAREST)
        mask = (image != 0).astype(int)
        return image, mask

    def load_normal(normal_root, scn, fid, h=240, w=320):
        img_path = os.path.join(normal_root, scn, 'normal', str(fid)+".png")
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) / 255 # convert to [0, 1]
        image = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_NEAREST)
        
        black = np.zeros_like(image)
        mask = (image != black).astype(int)[:, :, 0]
        
        image = unplot_normal(image)
        return image, mask
    
    scn, tgt_id, src_id = record
    
    data = dict()
    data['tgt'], data['src'] = load_image(root['data'], scn, tgt_id)
    data['pose'] = load_pose(root['pose'], scn, tgt_id, src_id)
    data['K'] = load_intrinsics(root['data'], scn, tgt_id)
    data['depth'], depth_mask = load_depth(root['depth'], scn, tgt_id)
    data['norm'], norm_mask = load_normal(root['gt'], scn, tgt_id)
    data['mask'] = depth_mask & norm_mask

    return data

def data_to_device_tensor(data):
    """
    -------Input-------
    data: dictionary with different data
     * 'tgt': target rgb image [H, W, 3]
     * 'src': source(1) rgb image [H, W, 3]
     * 'pose': pose from target to source [4, 4]
     * 'K': intrinsic matrix [3, 3]
     * 'norm': normal map of tgt [H, W, 3]
     * 'depth': depth map of tgt [H, W]
     * 'mask': mask that masks out empty areas on the depth and normal map [H, W]
    -------Output------
    device_dict: data -> tensor dictionary
     * 'tgt': target rgb image [1, 3, H, W]
     * 'src': source(1) rgb image [1, 3, H, W]
     * 'pose': pose from target to source [1, 4, 4]
     * 'K': intrinsic matrix [1, 3, 3]
     * 'norm': normal map of tgt [1, 3, H, W]
     * 'depth': depth map of tgt [1, 1, H, W]
     * 'mask': mask that masks out empty areas on the depth and normal map [1, 1, H, W]
    """
    device_dict = dict()
    imgs = ['tgt', 'src', 'depth', 'norm', 'mask']
    for k, v in data.items():
        value = np.expand_dims(v, 0).astype(np.float32)
        if k == 'depth' or k == 'mask':
            value = np.expand_dims(value, -1)
        if k in imgs:
            value = value.transpose((0, 3, 1, 2))
        value = torch.from_numpy(value).cuda()
        device_dict[k] = value

    return device_dict

def enlarge_K(K):
    nK = torch.zeros([4, 4]).cuda()
    nK[3, 3] = 1
    nK[:3, :3] = K
    return nK

def get_ofs(patch_size, dilation=1):
    psize_eff = 1 + dilation * (patch_size - 1) # effective patch size
    ofs = (psize_eff - 1) // 2                  # offset
    return ofs

def test_square_patch(samples, frame, layers, psize_list, gt_deviation, step):
    batch, _, height, width = frame['tgt'].shape 
    eps_lst = np.arange(-gt_deviation, gt_deviation, step)
    gt_depth = frame['depth']
    inv_K = torch.inverse(frame['K'])
    K_4x4 = enlarge_K(frame['K'])
    samples_xy = torch.ones_like(samples)
    samples_xy[:, 0], samples_xy[:, 1] = samples[:, 1], samples[:, 0]
    _, _, height, width = frame['src'].shape
    
    ret_min = defaultdict(list)
    ret_curve = dict()
    for patch_size in psize_list:
        print('Square Patch of Size {}'.format(patch_size))
        ofs = get_ofs(patch_size, dilation=1)
        loss_maps = []
        for i, eps in enumerate(eps_lst):
            depth = gt_depth + eps

            # warp
            cam_points = layers['backproject_depth'](depth, inv_K)
            pix_coords = layers['project_3d'](cam_points, K_4x4, frame['pose'])
            ### TODO: need to fix pix_coords / width, / height
            warped = F.grid_sample(frame['src'], pix_coords, padding_mode="border")

            # unfold [B, 3, n_samples, patch_size, patch_size]
            src_patch = sampled_intensity(samples_xy, warped, patch_size) 
            tgt_patch = sampled_intensity(samples_xy, tgt_patch, patch_size)

            # calculate loss 
            abs_diff = torch.abs(frame['tgt'] - warped)
            l1_loss = abs_diff.mean(1, True)
            if patch_size == 1:
                loss = l1_loss
            else:
                l1 = np.abs(src_patch - tgt_patch)
                center = int((patch_size - 1) / 2)
                l1_loss_point = l1[:, :, :, center, center].mean(1, True) #[B, 1, N]
                #l1_loss_patch = l1.mean(4).mean(3).mean(1, True)
                loss = 0.15 * l1_loss_point + 0.85 * ssim_patch_sampled_pts(tgt_patch, warped_patch) # [B, 1, N]
            if ofs == 0:
                loss = loss * frame['mask']
            else:
                ## sample from mask
                grid = sample_xy.unsqueeze(0).unsqueeze(2)
                grid[..., 0] /= width - 1 
                grid[..., 1] /= height - 1 
                mask = F.grid_sample(frame['mask'], grid, mode='nearest', padding_mode='border') # [1, 3, n_samples, patch_size*patch_size]
                mask = mask.squeeze()
                loss = loss * mask
            # stack loss 
            loss_maps.append(loss)

        # take minimums and corresponding eps
        loss_stack = torch.cat(loss_maps, dim=1) # [1, len(eps_lst), H, W]
        min_loss, min_idx = torch.min(loss_stack, dim=1) # both [1, H, W]
        min_loss = min_loss.squeeze(dim=0) # [H, W]
        min_idx = min_idx.squeeze(dim=0)   # [H, W]

        # select sampled locations
        curve = np.zeros([len(eps_lst)])
        for _, _, y, x in samples:
            min_eps = eps_lst[min_idx[y, x]]
            min_loss_value = min_loss[y, x]
            result_frame = ((x, y), min_eps, min_loss_value)
            ret_min[patch_size].append(result_frame)
            curve += loss_stack[0, :, y, x].cpu().numpy()
        ret_curve[patch_size] = curve/samples.shape[0]
        
        # clear unused variables
        del loss_maps

    return ret_min, ret_curve
        
def test_deform_patch(samples, frame, layers, psize_list, gt_deviation, step):
    eps_lst = np.arange(-gt_deviation, gt_deviation, step)
    gt_depth = frame['depth']
    inv_K = torch.inverse(frame['K'])

    ret_min = defaultdict(list)
    ret_curve = dict()
    for patch_size in psize_list[1:]: # skip patch size of 1
        print('Deformed Patch of Size {}'.format(patch_size))
        ofs = get_ofs(patch_size, dilation=1)
        loss_maps = []
        for eps in eps_lst:
            depth = gt_depth + eps
            cam_points = layers['backproject_depth'](depth, inv_K)
            
            loss, _ = patch_center_loss(
                            frame['tgt'],
                            frame['src'],
                            frame['K'],
                            cam_points,
                            frame['pose'],
                            depth,
                            frame['norm'],
                            patch_size=patch_size,
                            dilation=1)
            if ofs == 0:
                loss = loss * frame['mask']
            else:
                loss = loss * frame['mask'][:, :, ofs:-ofs, ofs:-ofs]
            loss_maps.append(loss)

        # take minimums and corresponding eps
        loss_stack = torch.cat(loss_maps, dim=1) # [1, len(eps_lst), H, W]
        min_loss, min_idx = torch.min(loss_stack, dim=1) # both [1, H, W]
        min_loss = min_loss.squeeze(dim=0) # [H, W]
        min_idx = min_idx.squeeze(dim=0)   # [H, W]

        # select sampled locations
        curve = np.zeros([len(eps_lst)])
        for _, _, y, x in samples:
            min_eps = eps_lst[min_idx[y, x]]
            min_loss_value = min_loss[y, x]
            result_frame = ((x, y), frame['norm'][0, :, y, x], min_eps, min_loss_value)
            ret_min[patch_size].append(result_frame)
            curve += loss_stack[0, :, y, x].cpu().numpy()
        ret_curve[patch_size] = curve/samples.shape[0]
        
        # clear unused variables
        del loss_maps

    return ret_min, ret_curve

def run_test(frame_t,
             frame,
             agg_results,
             layers,
             psize_list,
             gt_deviation, step, n_samples=100):
    # randomly sample pixels in the valid region
    psize_max = psize_list[-1]
    ofs_max = get_ofs(psize_max, dilation=1)
    frame['mask'][:, :, 0:ofs_max, :] = 0 # up
    frame['mask'][:, :, -ofs_max:, :] = 0 # down
    frame['mask'][:, :, :, 0:ofs_max] = 0 # left
    frame['mask'][:, :, :, -ofs_max:] = 0 # right
    
    valid_idxs = frame['mask'].nonzero()
    perm = torch.randperm(valid_idxs.size(0))
    selected_idxs = perm[:n_samples]
    samples = valid_idxs[selected_idxs]

    # generate loss curve for all samples with square/deformed patch
    with torch.no_grad():
        r_square_min, r_square_curve = test_square_patch(samples, frame, layers, psize_list, gt_deviation, step)
        r_deform_min, r_deform_curve = test_deform_patch(samples, frame, layers, psize_list, gt_deviation, step)

    # dump r_x_min containing (pixel location, absolute depth error) 
    dump_result(frame_t, r_square_min, r_deform_min, r_square_curve, r_deform_curve)

    # aggregate loss curve
    for i, patch_size in enumerate(psize_list):
        agg_results['square'][i] += r_square_curve
        agg_results['deform'][i] += r_deform_curve
    return agg_results # return updated aggregated results

def dump_result(frame_t, r_square_min, r_deform_min, r_square_curve, r_deform_curve):
    scene, tgt, src = frame_t
    file_path = os.path.join(dump_root, scene, '{}_{}.txt'.format(tgt, src))
    
    record_dict = dict()
    record_dict['square'] = r_square_min
    record_dict['deform'] = r_deform_min
    record_dict['square_curve'] = r_square_curve
    record_dict['deform_curve'] = r_deform_curve

    np.savez(file_path, record_dict)


if __name__ == '__main__':
    list_path = "/viscompfs/users/sawang/ScanNet/analyze_list_new.txt"
    root = dict()
    root['data'] = "/viscompfs/users/sawang/ScanNet/data_motion_range"
    root['gt'] =   "/viscompfs/users/sawang/ScanNet/data_gt"
    root['pose'] = "/viscompfs/users/sawang/ScanNet/data_pose"
    root['depth'] = "/viscompfs/users/sawang/ScanNet/data/scans"
    img_w, img_h = 320, 240
    deviation = 0.2
    step = 1e-4
    psize_list = [1] + [2**i+1 for i in range(1, 7)] # [1, 3, 5, 9, 17, 33, 65]
    eps_lst = np.arange(-deviation, deviation, step)

    # set random seed
    seed = 732
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # define layers
    layers = dict()
    layers['backproject_depth'] = BackprojectDepth(1, img_h, img_w).cuda()
    layers['project_3d'] = Project3D(1, img_h, img_w).cuda()
    for patch_size in psize_list:
        layers['unfold_{}'.format(patch_size)] = \
            torch.nn.Unfold(kernel_size=(patch_size, patch_size), dilation=1).cuda()

    agg_results = dict()
    agg_results['square'] = np.zeros([len(psize_list), len(eps_lst)])
    agg_results['deform'] = np.zeros([len(psize_list), len(eps_lst)])

    frame_list = read_pair_list(list_path)
    for frame_t in frame_list:
        frame = load_data(frame_t, root, img_w, img_h)
        frame = data_to_device_tensor(frame)
        agg_results = run_test(frame_t,
                               frame,
                               agg_results,
                               layers,
                               psize_list,
                               deviation, step, n_samples=100)
        