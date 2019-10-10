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
import time

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
    
    ## contruct sample_xy, [n_samples, 2], each entry is (col, row) = [x, y]
    samples_xy = torch.ones_like(samples[:, :2])
    samples_xy[:, 0], samples_xy[:, 1] = samples[:, 3], samples[:, 2]
    
    ret_min_pts, ret_min_patch = defaultdict(list), defaultdict(list)
    ret_curve_pts, ret_curve_patch = dict(), dict()
    for patch_size in psize_list:
        print('Square Patch of Size {}'.format(patch_size))
        ofs = get_ofs(patch_size, dilation=1)
        loss_maps = []
        for i, eps in enumerate(eps_lst):
            depth = gt_depth + eps

            # warp
            cam_points = layers['backproject_depth'](depth, inv_K)
            pix_coords = layers['project_3d'](cam_points, K_4x4, frame['pose'])

            pix_coords[..., 0] /= width - 1 
            pix_coords[..., 1] /= height - 1 
            warped = F.grid_sample(frame['src'], pix_coords, padding_mode="border")

            # unfold [B, 3, n_samples, patch_size, patch_size]
            if patch_size != 1:
                src_patch = sampled_intensity(samples_xy, warped, patch_size) 
                tgt_patch = sampled_intensity(samples_xy, frame['tgt'], patch_size)

            # calculate loss 
            abs_diff = torch.abs(frame['tgt'] - warped)
            l1_loss = abs_diff.mean(1, True)
            if patch_size == 1:
                grid = torch.zeros_like(samples_xy)
                grid[..., 0] = samples_xy[..., 0] / (width - 1)
                grid[..., 1] = samples_xy[..., 1] / (height - 1)
                grid = grid.unsqueeze(0).unsqueeze(2)
                loss = F.grid_sample(l1_loss, grid.to(torch.float), padding_mode="border").squeeze(-1)
                loss = torch.stack([loss, loss], dim=-1)
            else:
                l1 = torch.abs(src_patch - tgt_patch)
                center = int((patch_size - 1) / 2)
                l1_loss_point = l1[:, :, :, center, center].mean(1, True) #[B, 1, N]
                l1_loss_patch = l1.mean(4).mean(3).mean(1, True)
                loss_point = 0.15 * l1_loss_point + 0.85 * ssim_patch_sampled_pts(tgt_patch, src_patch) # [B, 1, N]
                loss_patch = 0.15 * l1_loss_patch + 0.85 * ssim_patch_sampled_pts(tgt_patch, src_patch) # [B, 1, N]
                loss = torch.stack([loss_point, loss_patch], dim=-1)
                
            # stack loss 
            loss_maps.append(loss)
        # take minimums and corresponding eps
        loss_stack = torch.cat(loss_maps, dim=1) # [1, len(eps_lst), N, 2]
        min_loss, min_idx = torch.min(loss_stack, dim=1) # both [1, N, 2]
        min_loss = min_loss.squeeze(dim=0).cpu().numpy() # [N, 2]
        min_idx = min_idx.squeeze(dim=0).cpu().numpy()   # [N, 2]
        
        # select sampled locations
        pts_curve, patch_curve = np.zeros([len(eps_lst)]), np.zeros([len(eps_lst)])
        for idx, (_, _, y, x) in enumerate(samples):
            ret_min_pts[patch_size].append(make_result_frame(eps_lst, min_idx[:, 0], min_loss[:, 0], idx, x, y))
            ret_min_patch[patch_size].append(make_result_frame(eps_lst, min_idx[:, 1], min_loss[:, 1], idx, x, y)) 
            pts_curve += loss_stack[0, :, idx, 0].cpu().numpy()
            patch_curve += loss_stack[0, :, idx, 1].cpu().numpy()
        ret_curve_pts[patch_size] = pts_curve/samples.shape[0]
        ret_curve_patch[patch_size] = patch_curve/samples.shape[0]
        
        # clear unused variables
        del loss_maps

    return (ret_min_pts, ret_min_patch), (ret_curve_pts, ret_curve_patch)

def make_result_frame(eps_lst, min_idx, min_loss, idx, x, y):
    min_eps = eps_lst[min_idx[idx]]
    min_loss_value = min_loss[idx]
    result_frame = ((x, y), min_eps, min_loss_value)
    return result_frame
        
def test_deform_patch(samples, frame, layers, psize_list, gt_deviation, step):
    batch = frame['tgt'].shape[0]
    eps_lst = np.arange(-gt_deviation, gt_deviation, step)
    gt_depth = frame['depth']
    inv_K = torch.inverse(frame['K'])
    n_samples = samples.shape[0]

    ## contruct sample_xy, [n_samples, 2], each entry is (col, row) = [x, y]
    samples_xy = torch.ones_like(samples[:, :2])
    samples_xy[:, 0], samples_xy[:, 1] = samples[:, 3], samples[:, 2]

    loss_tensor_point = torch.zeros([batch, len(eps_lst), len(psize_list), n_samples])
    loss_tensor_patch = torch.zeros([batch, len(eps_lst), len(psize_list), n_samples])
    # fill patch size=1 locations with constants
    loss_tensor_point[:, :, 0, :] = 1e6 
    loss_tensor_patch[:, :, 0, :] = 1e6

    for eps_idx, eps in enumerate(eps_lst):
        print('eps: {}'.format(eps))
        depth = gt_depth + eps
        cam_points = layers['backproject_depth'](depth, inv_K)

        # generate target patches, shape [1, 3, n_samples, patch_size, patch_size]
        tgt_patches = sampled_intensity(samples_xy, frame['tgt'], patch_size) 

        # generate source patches by warping with the largest patch size
        # shape [1, 3, n_samples, patch_size, patch_size]
        src_patches, src_coords = sampled_patch_center_loss(
                                    samples[:, 2:],
                                    frame['tgt'],
                                    frame['src'],
                                    frame['K'],
                                    cam_points,
                                    frame['pose'],
                                    depth,
                                    frame['norm'],
                                    patch_size=psize_list[-1], # largest patch size
                                    dilation=1)
        
        for patch_size in psize_list[::-1][:-1] # skip patch size=1
            # crop patches
            prev_patch_size = tgt_patches.shape[-1]
            ctr = get_ofs(prev_patch_size, dilation=1) # old center index
            ofs = get_ofs(patch_size, dilation=1)      # offset for new patch size
            tgt_patches = tgt_patches[:, :, :, ctr-ofs:ctr+ofs+1, ctr-ofs:ctr+ofs+1]
            src_patches = src_patches[:, :, :, ctr-ofs:ctr+ofs+1, ctr-ofs:ctr+ofs+1]

            # compute loss
            l1_loss_patch = torch.abs(tgt_patches - src_patches).mean(1, True) # [B, 1, n_samples, patch_size, patch_size]
            l1_loss_point= l1_loss[:, :, :, ofs, ofs] # [B, 1, n_samples]
            dssim = ssim_patch_sampled_pts(tgt_patch, src_patch)
            loss_point = 0.15 * l1_loss_point + 0.85 * dssim # [B, 1, N]
            loss_patch = 0.15 * l1_loss_patch + 0.85 * dssim # [B, 1, N]
            
            # log loss
            # stack ordered according to patch size from small to large
            patch_size_idx = psize_list.index(patch_size) 
            loss_tensor_point[:, eps_idx, patch_size_idx, :] = loss_point
            loss_tensor_patch[:, eps_idx, patch_size_idx, :] = loss_patch

        # take minimums and corresponding eps
        min_point_loss, min_point_idx = torch.min(loss_tensor_point, dim=1) # [B, n_size, n_samples]
        min_patch_loss, min_patch_idx = torch.min(loss_tensor_patch, dim=1) # [B, n_size, n_samples]

    mean_point_loss = torch.mean(loss_tensor_point, dim=3) #[batch, len(eps_lst), len(psize_list)]
    mean_patch_loss = torch.mean(loss_tensor_patch, dim=3) #[batch, len(eps_lst), len(psize_list)]

    ret_min_pts, ret_min_patch = defaultdict(list), defaultdict(list)
    ret_curve_pts, ret_curve_patch = dict(), dict()
    for patch_size in psize_list[1:]: # skip patch size=1
        patch_size_idx = psize_list.index(patch_size)

        for sample_idx, (_, _, y, x) in enumerate(samples):
            min_pt = min_point_idx[0, patch_size_idx, sample_idx].cpu().numpy()
            min_ph = min_patch_idx[0, patch_size_idx, sample_idx].cpu().numpy()
            ret_min_pts[patch_size].append(((x, y), eps_lst[min_pt], min_point_loss[0, patch_size_idx, sample_idx].cpu().numpy()))
            ret_min_patch[patch_size].append(((x, y), eps_lst[min_ph], min_patch_loss[0, patch_size_idx, sample_idx].cpu().numpy())) 

        # create curve
        ret_curve_pts[patch_size] = mean_point_loss[0, :, patch_size_idx].cpu().numpy()
        ret_curve_patch[patch_size] = mean_patch_loss[0, :, patch_size_idx].cpu().numpy()

    return (ret_min_pts, ret_min_patch), (ret_curve_pts, ret_curve_patch)

def run_test(frame_t,
             frame,
             agg_results,
             layers,
             psize_list,
             gt_deviation, step, root, n_samples=100):
    #start = time.time()
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
        #r_square_min, r_square_curve = test_square_patch(samples, frame, layers, psize_list, gt_deviation, step)
        r_deform_min, r_deform_curve = test_deform_patch(samples, frame, layers, psize_list, gt_deviation, step)

    # dump r_x_min containing (pixel location, absolute depth error) 
    #dump_result(frame_t, r_square_min, r_deform_min, r_square_curve, r_deform_curve)
    dump_result(frame_t, r_square_min, None, r_square_curve, None, root['dump'])

    # aggregate loss curve
    for i, patch_size in enumerate(psize_list):
        agg_results['square_pts'][i] += r_square_curve[0][patch_size]
        agg_results['square_patch'][i] += r_square_curve[1][patch_size]
        #agg_results['deform'][i] += r_deform_curve
    
    #print(time.time() - start)

    return agg_results # return updated aggregated results

def dump_result(frame_t, r_square_min, r_deform_min, r_square_curve, r_deform_curve, dump_root):
    scene, tgt, src = frame_t
    dir_path = os.path.join(dump_root, scene)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        
    file_path = os.path.join(dir_path, '{}_{}'.format(tgt, src))
    
    record_dict = dict()
    record_dict['square_pt'] = r_square_min[0]
    #record_dict['deform_pt'] = r_deform_min[0]
    record_dict['square_curve_pt'] = r_square_curve[0]
    #record_dict['deform_curve_pt'] = r_deform_curve[0]

    record_dict['square_patch'] = r_square_min[1]
    #record_dict['deform_patch'] = r_deform_min[1]
    record_dict['square_curve_patch'] = r_square_curve[1]
    #record_dict['deform_curve_patch'] = r_deform_curve[1]
    

    np.savez(file_path, record_dict)


if __name__ == '__main__':
    list_path = "/viscompfs/users/sawang/ScanNet/analyze_list_new.txt"
    root = dict()
    root['data'] = "/viscompfs/users/sawang/ScanNet/data_motion_range"
    root['gt'] = "/viscompfs/users/sawang/ScanNet/data_gt"
    root['pose'] = "/viscompfs/users/sawang/ScanNet/data_pose"
    root['depth'] = "/viscompfs/users/sawang/ScanNet/data/scans"
    root['dump'] = "/viscompfs/users/sawang/ScanNet/analysis"
    img_w, img_h = 320, 240
    deviation = 0.2
    step = 5e-3
    n_samples = 50
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
    agg_results['square_pts'] = np.zeros([len(psize_list), len(eps_lst)])
    agg_results['square_patch'] = np.zeros([len(psize_list), len(eps_lst)])
    agg_results['deform_pts'] = np.zeros([len(psize_list), len(eps_lst)])
    agg_results['deform_patch'] = np.zeros([len(psize_list), len(eps_lst)])

    frame_list = read_pair_list(list_path)
    for frame_t in frame_list:
        frame = load_data(frame_t, root, img_w, img_h)
        frame = data_to_device_tensor(frame)
        agg_results = run_test(frame_t,
                               frame,
                               agg_results,
                               layers,
                               psize_list,
                               deviation, step, root, n_samples=n_samples)
    
