import os
import sys
sys.path.insert(0, '../')
sys.path.insert(0, '../../')
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.multiprocessing import Pool, Process, set_start_method

from collections import defaultdict
from rotm2euler import *
from test_utils import *
from experiment_utils import *
from layers import *

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import collections as mc

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

def get_meshgrid(w, h, x_ctr, y_ctr):
    x_upper_left = x_ctr - w//2
    y_upper_left = y_ctr - h//2
    
    x = np.arange(0, w)
    y = np.arange(0 ,h)
    xv, yv = np.meshgrid(x, y)
    xv = xv.astype(np.float)
    yv = yv.astype(np.float)
    xv += x_upper_left
    yv += y_upper_left
    return xv, yv

def get_tgt_patch(pt, tgt, patch_size):
    x, y = pt[0, 0], pt[1, 0]
    ofs = (patch_size - 1) // 2
    return tgt[y-ofs:y+ofs+1, x-ofs:x+ofs+1]

def warp_patch(pt, depth, K, rel_pose, src, patch_size):
    ps_ctr = warp_location_torch(pt, K, rel_pose, depth)
    xv, yv = get_meshgrid(patch_size, patch_size, ps_ctr[0, 0], ps_ctr[1, 0]) # x-coord, y-coord
    xv = xv.flatten()
    yv = yv.flatten()
    ps = np.vstack([xv, yv]).cuda() # [2, patch_size**2]
    
    patch_img = torch.cuda.FloatTensor([patch_size, patch_size, 3]).fill_(0)
    for x in range(patch_size):
        for y in range(patch_size):
            i = y * patch_size + x
            patch_img[y, x] = bilinear_pt_torch(ps[:, i].unsqueeze(-1), src)
    return ps, patch_img

def warp_patch_homography_torch(pt_ctr, K, pose, depth, normal, src, patch_size=7):
    xv, yv = get_meshgrid(patch_size, patch_size, pt_ctr[0, 0], pt_ctr[1, 0]) # x-coord, y-coord
    xv = xv.flatten()
    yv = yv.flatten()
    mesh_flatten = np.vstack([xv, yv, np.ones_like(yv)]).cuda() # [3, patch_size**2]

    H = compute_homography_torch(pt_ctr, K, pose, depth, normal)
    ps = torch.matmul(H, mesh_flatten)
    ps = ps[:2] / (ps[2] + 1e-8)
    
    patch_img = np.zeros([patch_size, patch_size, 3])
    for x in range(patch_size):
        for y in range(patch_size):
            i = y * patch_size + x
            patch_img[y, x] = bilinear_pt_torch(ps[:, i].unsqueeze(-1), src)
    return ps, patch_img 

def test_single_pixel_square(pt, frame, layers, patch_size, gt_deviation, step):
    """For fixed pixel location and patch size, test all inverse depth values with square patch"""
    if print_msg:
        print('{}, {} worker started'.format(pt, patch_size), flush=True)
    y, x = pt
    n_mid = (patch_size**2 - 1)//2
    ofs = get_ofs(patch_size, dilation=1)
    eps_lst = np.arange(-gt_deviation, gt_deviation, step)
    gt_inv_depth = 1 / (frame['depth'][0, 0, y, x] + 1e-6)
    gt_inv_depth = torch.clamp(gt_inv_depth, 0, 10) 
    tgt = frame['tgt'].permute([0, 2, 3, 1]).squeeze(dim=0)
    tgt_patch = get_tgt_patch(pt, tgt, patch_size).squeeze() # single-pixel or patch

    loss_values = np.zeros_like(eps_lst)
    for i, eps in enumerate(eps_lst):
        inv_depth = gt_inv_depth + eps
        depth = 1 / inv_depth
        
        if patch_size == 1:
            ps = warp_location_torch(pt.float(), frame['K'], frame['pose'], depth)
            src_patch = bilinear_pt_torch(ps[:, 0].unsqueeze(-1), frame['src'].squeeze(0)) # single-pixel
            diff = torch.mean(torch.abs(tgt_patch - src_patch))
        else:
            src_coords, src_patch = warp_patch_torch(pt.float(), depth, frame['K'], frame['pose'], frame['src'], patch_size)
            ps = src_coords[:, n_mid].unsqueeze(-1)
            diff = experiment_loss_torch(tgt_patch, src_patch)

        loss_values[i] = diff.cpu().numpy()

    min_idx = np.argmin(loss_values)
    min_eps, min_loss = eps_lst[min_idx], loss_values[min_idx]
    result_t = ((x, y), patch_size, gt_inv_depth, min_eps, min_loss)
    if print_msg:
        print('{}, {} worker completed'.format(pt, patch_size), flush=True)
    return (result_t, loss_values)

def test_single_pixel_deform(pt, frame, layers, patch_size, gt_deviation, step):
    """For fixed pixel location and patch size, test all inverse depth values with deformed patch"""
    y, x = pt
    n_mid = (patch_size**2 - 1)//2
    eps_lst = np.arange(-gt_deviation, gt_deviation, step)
    gt_inv_depth = 1 / (frame['depth'][0, 0, y, x] + 1e-6)
    gt_inv_depth = torch.clamp(gt_inv_depth, 0, 10) 
    gt_normal = frame['norm'][y, x]
    tgt = frame['tgt'].permute([0, 2, 3, 1]).squeeze(dim=0)
    tgt_patch = get_tgt_patch(pt, tgt, patch_size).squeeze() # single-pixel or patch

    loss_values = np.zeros_like(eps_lst)
    for i, eps in enumerate(eps_lst):
        inv_depth = gt_inv_depth + eps
        depth = 1 / inv_depth
        
        if patch_size == 1:
            ps = warp_location_torch(pt.float(), frame['K'], frame['pose'], depth)
            src_patch = bilinear_pt_torch(ps[:, 0].unsqueeze(-1), frame['src'].squeeze(0)) # single-pixel
            diff = torch.mean(torch.abs(tgt_patch - src_patch))
        else:
            src_coords, src_patch = warp_patch_homography_torch(
                                        pt.float(),
                                        frame['K'],
                                        frame['pose'],
                                        depth,
                                        gt_normal,
                                        frame['src'],
                                        patch_size)
            ps = src_coords[:, n_mid].squeeze(-1)
            diff = experiment_loss_torch(tgt_patch, src_patch)

        loss_values[i] = diff.cpu().numpy()
    
    min_idx = np.argmin(loss_values)
    min_eps, min_loss = eps_lst[min_idx], loss_values[min_idx]
    result_t = ((x, y), patch_size, gt_inv_depth, gt_normal, min_eps, min_loss)
    return (result_t, loss_values)

def test_square_patch(samples, frame, layers, psize_list, gt_deviation, step):    
    ret_min = defaultdict(list)
    ret_curve = dict()

    with Pool(32) as p:
        for patch_size in psize_list:
            if print_msg:
                print('\t Square patch of size {}'.format(patch_size))
            # prepare subprocess args
            #args = []
            results = []
            for pt in samples: # pt: [0, 0, y, x]
                pt = pt[2:].unsqueeze(-1)
                tup = (pt, frame, layers, patch_size, gt_deviation, step)
                r = test_single_pixel_square(*tup)
                #args.append(tup)
                results.append(r)
                

            # get results
#             results = p.starmap(test_single_pixel_square, args)
#             if print_msg:
#                 print('\t Subprocess completed')
            
            # aggregate results
            curve = np.zeros([len(eps_lst)])
            for (result_t, loss_values) in results:
                ret_min[patch_size].append(result_t)
                curve += loss_values
            ret_curve[patch_size] = curve/samples.shape[0]

    return ret_min, ret_curve
        
def test_deform_patch(samples, frame, layers, psize_list, gt_deviation, step):
    ret_min = defaultdict(list)
    ret_curve = dict()

    with Pool(32) as p:
        for patch_size in psize_list:
            if print_msg:
                print('\t Deformed patch of size {}'.format(patch_size))
            # prepare subprocess args
            args = []
            for pt in samples: # pt: [0, 0, y, x]
                pt = pt[2:].unsqueeze(-1)
                tup = (pt, frame, layers, patch_size, gt_deviation, step)
                args.append(tup)

            # get results
            results = p.starmap(test_single_pixel_deform, args)
            if print_msg:
                print('\t Subprocess completed')
            
            # aggregate results
            curve = np.zeros([len(eps_lst)])
            for (result_t, loss_values) in results:
                ret_min[patch_size].append(result_t)
                curve += loss_values
            ret_curve[patch_size] = curve/samples.shape[0]
            
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
    global print_msg 
    print_msg = True

    # set random seed
    seed = 732
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # torch multiprocessing
    try:
         set_start_method('forkserver')
    except RuntimeError:
        pass
    
    # define layers
    layers = dict()
    layers['backproject_depth'] = BackprojectDepth(1, img_h, img_w).cuda()
    layers['project_3d'] = Project3D(1, img_h, img_w).cuda()
    for patch_size in psize_list:
        layers['unfold_{}'.format(patch_size)] = \
            torch.nn.Unfold(kernel_size=(patch_size, patch_size), dilation=1).cuda()
    
    # initialize aggregated results
    agg_results = dict()
    agg_results['square'] = np.zeros([len(psize_list), len(eps_lst)])
    agg_results['deform'] = np.zeros([len(psize_list), len(eps_lst)])

    frame_list = read_pair_list(list_path)
    for i, frame_t in enumerate(frame_list):
        tic = time.time()
        print('Running frame {}/{}'.format(i+1, len(frame_list)))
        frame = load_data(frame_t, root, img_w, img_h)
        frame = data_to_device_tensor(frame)
        agg_results = run_test(frame_t,
                               frame,
                               agg_results,
                               layers,
                               psize_list,
                               deviation, step, n_samples=100)
        toc = time.time()
        print('{}s/it'.format(toc-tic))
        