import os
import sys
sys.path.insert(0, '/viscompfs/users/tedyu/monodepth2/tests/')
sys.path.insert(0, '/viscompfs/users/tedyu/monodepth2/')
import random
import argparse
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
from matplotlib.pyplot import cm
from matplotlib import collections as mc
#import pdb
import time
#import seaborn as sns

def set_parse():
    """Set arguments for parser"""

    parser = argparse.ArgumentParser(
                description=\
                    'Compute loss values on ScanNet dataset with different patch sizes and orientation')
    parser.add_argument('--parts', default=4, help='the number of partitions the dataset is divided into')
    parser.add_argument('--split', default=-1, help='Which partition will this application be responsible for' + \
                                                      'should be an integer raning from 1~parts, -1 denotes the entire dataset')
    parser.add_argument('--type_name', default='both', help="square/deform/both")
    return parser

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

def clip_percentage(loss, c_min, c_max):
    """set losses below `c_min` percentage and `c_max` percentage to zero
    Args:
        loss: [B, 1, N]
        c_min: min percentage to clip (out of 100%)
        c_max: max percentage to clip (out of 100%)
    Returns:
        clipped_min_max: clipped loss
        n_remain: number of nonzero loss elements
    """
    batch, _, n_samples = loss.shape
    min_k = int(n_samples * (c_min / 100))
    max_k = int(n_samples * ((100-c_max) / 100))
    _, min_idxs = torch.topk(loss, min_k, dim=-1, largest=False)
    _, max_idxs = torch.topk(loss, max_k, dim=-1, largest=True)
    clipped_min = loss.scatter(-1, min_idxs, 0)
    clipped_min_max = clipped_min.scatter(-1, max_idxs, 0)
    n_remain = n_samples - min_k - max_k

    return clipped_min_max, n_remain

def generate_plot(samples_xy, loss):
    """Given torch.cuda.tensor, return a [H, W, 1] loss map of type numpy array """
    ret = np.zeros([img_h, img_w])
    loss_np = loss.cpu().numpy()
    for i, (x, y) in enumerate(samples_xy.cpu().numpy()):
        ret[y, x] = loss_np[0, 0, i]
    
    return ret
    
def show_loss_map(samples_xy, loss_point, loss_patch, plot_title):
    """loss_point, loss_patch both [B, 1, N]"""
    fig = plt.figure(figsize=(16, 24))
    ax1 = fig.add_subplot(211)
    #ax1.imshow(generate_plot(samples_xy, loss_point))
    sns.heatmap(generate_plot(samples_xy, loss_point), vmin=0, vmax=1, ax=ax1)
    ax1.set_title(plot_title + '(Point)')
    ax1.axis('off')
    ax2 = fig.add_subplot(212)
    #ax2.imshow(generate_plot(samples_xy, loss_patch))
    sns.heatmap(generate_plot(samples_xy, loss_patch), vmin=0, vmax=1, ax=ax2)
    ax2.set_title(plot_title + '(Patch)')
    ax2.axis('off')
    plt.show()

def test_square_patch(samples, frame, layers, patch_config, thresh_list, gt_deviation, step):
    batch, _, height, width = frame['tgt'].shape 
    patch_size, dilation = patch_config
    eps_lst = np.arange(-gt_deviation, gt_deviation, step)
    gt_depth = frame['depth']
    inv_K = torch.inverse(frame['K'])
    K_4x4 = enlarge_K(frame['K'])
    n_samples = samples.shape[0]
    
    ## contruct sample_xy, [n_samples, 2], each entry is (col, row) = [x, y]
    samples_xy = torch.ones_like(samples[:, :2])
    samples_xy[:, 0], samples_xy[:, 1] = samples[:, 3], samples[:, 2]

    tgt_patches = sampled_intensity(samples_xy, frame['tgt'],
                                    patch_size=patch_size,
                                    dilation=dilation)

    loss_tensor_point = torch.zeros([batch, len(eps_lst), len(thresh_list)])
    loss_tensor_patch = torch.zeros([batch, len(eps_lst), len(thresh_list)])
    for eps_idx, eps in enumerate(eps_lst):
        depth = gt_depth + eps
        cam_points = layers['backproject_depth'](depth, inv_K)

        # warp 
        cam_points = layers['backproject_depth'](depth, inv_K)
        pix_coords = layers['project_3d'](cam_points, K_4x4, frame['pose'])
        warped = F.grid_sample(frame['src'], pix_coords, padding_mode="border")

        # unfold image to generate patches
        # shape [1, 3, n_samples, patch_size, patch_size]
        src_patches = sampled_intensity(samples_xy, warped, patch_size, dilation)

        # calculate loss 
        abs_diff = torch.abs(frame['tgt'] - warped)
        l1_loss = abs_diff.mean(1, True)
        
        # calculate l1 point loss for samples
        norm_x = samples_xy[..., 0].to(torch.float) / float(width - 1)
        norm_y = samples_xy[..., 1].to(torch.float) / float(height - 1)
        grid = (torch.stack([norm_x, norm_y], 1) - 0.5) * 2
        grid = grid.unsqueeze(0).unsqueeze(2)
        l1_loss_point = F.grid_sample(l1_loss, grid, padding_mode="border").squeeze(-1)


        l1 = torch.abs(src_patches - tgt_patches) # patch l1 difference 
        l1_loss_patch = l1.mean(4).mean(3).mean(1, True) 
        dssim = ssim_patch_sampled_pts(tgt_patches, src_patches) #[B, 1, N]
        loss_point = 0.15 * l1_loss_point + 0.85 * dssim # [B, 1, N]
        loss_patch = 0.15 * l1_loss_patch + 0.85 * dssim # [B, 1, N]

        if run_single_image and np.abs(eps) < 1e-3:
            show_loss_map(samples_xy, loss_point, loss_patch, "Square Loss")
        for tid, (clip_min, clip_max) in enumerate(thresh_list):
            # [B, 1, N], scalar
            clipped_loss_point, n_point = clip_percentage(loss_point, clip_min, clip_max)
            clipped_loss_patch, n_patch = clip_percentage(loss_patch, clip_min, clip_max)

            # log loss
            loss_tensor_point[:, eps_idx, tid] = clipped_loss_point.sum(dim=-1) / n_point
            loss_tensor_patch[:, eps_idx, tid] = clipped_loss_patch.sum(dim=-1) / n_patch

            # show figure
            #if run_single_image and np.abs(eps) < 1e-3:
                #show_loss_map(samples_xy, clipped_loss_point, clipped_loss_patch, "Clipped Square Loss ({}%, {}%)".format(clip_min, clip_max))

    # create curve
    ret_curve_pts, ret_curve_patch = dict(), dict()
    for tid, (c_min, c_max) in enumerate(thresh_list):
        ret_curve_pts[(patch_config, (c_min, c_max))] = loss_tensor_point[0, :, tid].cpu().numpy()
        ret_curve_patch[(patch_config, (c_min, c_max))] = loss_tensor_patch[0, :, tid].cpu().numpy()
    
    if run_single_image:
        fig = plt.figure(figsize=(20, 10))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        
        color=cm.rainbow(np.linspace(0, 1, len(thresh_list)))
        for tid, (c_min, c_max) in enumerate(thresh_list):
            ax1.plot(eps_lst, ret_curve_pts[(patch_config, (c_min, c_max))], label='{}, {}'.format(c_min, c_max), c=color[tid])
            ax2.plot(eps_lst, ret_curve_patch[(patch_config, (c_min, c_max))], label='{}, {}'.format(c_min, c_max), c=color[tid])
        
        ax1.legend()
        ax2.legend()
        ax1.set_title('Square Patch (Points)')
        ax2.set_title('Square Patch (Patch)')
        plt.show()
            
    return (ret_curve_pts, ret_curve_patch)
        
def test_deform_patch(samples, frame, layers, patch_config, thresh_list, gt_deviation, step):
    batch = frame['tgt'].shape[0]
    patch_size, dilation = patch_config
    eps_lst = np.arange(-gt_deviation, gt_deviation, step)
    gt_depth = frame['depth']
    inv_K = torch.inverse(frame['K'])
    n_samples = samples.shape[0]

    ## contruct sample_xy, [n_samples, 2], each entry is (col, row) = [x, y]
    samples_xy = torch.ones_like(samples[:, :2])
    samples_xy[:, 0], samples_xy[:, 1] = samples[:, 3], samples[:, 2]

    loss_tensor_point = torch.zeros([batch, len(eps_lst), len(thresh_list)])
    loss_tensor_patch = torch.zeros([batch, len(eps_lst), len(thresh_list)])

    # generate target patches, shape [1, 3, n_samples, patch_size, patch_size]
    tgt_patches = sampled_intensity(samples_xy, frame['tgt'], 
                                    patch_size=patch_size,
                                    dilation=dilation)

    for eps_idx, eps in enumerate(eps_lst):
        depth = gt_depth + eps
        cam_points = layers['backproject_depth'](depth, inv_K)
        
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
                                    patch_size=patch_size, 
                                    dilation=dilation)
        
        # compute loss
        ofs = get_ofs(patch_size, dilation=1)
        l1_loss_patch = torch.abs(tgt_patches - src_patches).mean(1, True) # [B, 1, n_samples, patch_size, patch_size]
        l1_loss_point= l1_loss_patch[:, :, :, ofs, ofs] # [B, 1, n_samples]
        dssim = ssim_patch_sampled_pts(tgt_patches, src_patches)
        loss_point = 0.15 * l1_loss_point + 0.85 * dssim # [B, 1, N]
        loss_patch = 0.15 * l1_loss_patch.mean(dim=(3, 4)) + 0.85 * dssim # [B, 1, N]
        
        if run_single_image and np.abs(eps) < 1e-3:
            show_loss_map(samples_xy, loss_point, loss_patch, "Deform Loss")
        for tid, (clip_min, clip_max) in enumerate(thresh_list):
            clipped_loss_point, n_point = clip_percentage(loss_point, clip_min, clip_max)
            clipped_loss_patch, n_patch = clip_percentage(loss_patch, clip_min, clip_max)

            # log loss
            loss_tensor_point[:, eps_idx, tid] = clipped_loss_point.sum(dim=-1) / n_point
            loss_tensor_patch[:, eps_idx, tid] = clipped_loss_patch.sum(dim=-1) / n_patch

            # show figure
            #if run_single_image and np.abs(eps) < 1e-3:
                #show_loss_map(samples_xy, clipped_loss_point, clipped_loss_patch, "Clipped Deformed Loss ({}%, {}%)".format(clip_min, clip_max))

    # create curve
    ret_curve_pts, ret_curve_patch = dict(), dict()
    for tid, (c_min, c_max) in enumerate(thresh_list):
        ret_curve_pts[(patch_config, (c_min, c_max))] = loss_tensor_point[0, :, tid].cpu().numpy()
        ret_curve_patch[(patch_config, (c_min, c_max))] = loss_tensor_patch[0, :, tid].cpu().numpy()

    if run_single_image:
        fig = plt.figure(figsize=(20, 10))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        
        color=cm.rainbow(np.linspace(0, 1, len(thresh_list)))
        for tid, (c_min, c_max) in enumerate(thresh_list):
            ax1.plot(eps_lst, ret_curve_pts[(patch_config, (c_min, c_max))], label='{}, {}'.format(c_min, c_max), c=color[tid])
            ax2.plot(eps_lst, ret_curve_patch[(patch_config, (c_min, c_max))], label='{}, {}'.format(c_min, c_max), c=color[tid])
        
        ax1.legend()
        ax2.legend()
        ax1.set_title('Deform Patch (Points)')
        ax2.set_title('Deform Patch (Patch)')
        plt.show()

    return (ret_curve_pts, ret_curve_patch)

def run_test(frame_t,
             frame,
             layers,
             patch_config,
             thresh_list,
             gt_deviation, step, root, type_name):
    # randomly sample pixels in the valid region
    patch_size, dilation = patch_config
    ofs_max = get_ofs(patch_size, dilation=dilation)
    frame['mask'][:, :, 0:ofs_max, :] = 0 # up
    frame['mask'][:, :, -ofs_max:, :] = 0 # down
    frame['mask'][:, :, :, 0:ofs_max] = 0 # left
    frame['mask'][:, :, :, -ofs_max:] = 0 # right
    
    valid_idxs = frame['mask'].nonzero()
    samples = valid_idxs
    
    # generate loss curve for all samples with square/deformed patch
    with torch.no_grad():
        if type_name == 'square' or type_name == 'both':
            r_square_curve = test_square_patch(samples, frame, layers, patch_config, thresh_list, gt_deviation, step)
            if not run_single_image:
                dump_result(frame_t, r_square_curve, root['dump'], 'square')

        if type_name == 'deform' or type_name == 'both':
            r_deform_min, r_deform_curve = test_deform_patch(samples, frame, layers, patch_config, thresh_list, gt_deviation, step)
            if not run_single_image:
                dump_result(frame_t, r_deform_curve, root['dump'], 'deform')
    return 

def dump_result(frame_t, r_curve, dump_root, type_name):
    scene, tgt, src = frame_t
    dir_path = os.path.join(dump_root, scene)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        
    file_path = os.path.join(dir_path, '{}_{}_{}'.format(tgt, src, type_name))
    
    record_dict = dict()
    record_dict['{}_curve_pt'.format(type_name)] = r_curve[0]
    record_dict['{}_curve_patch'.format(type_name)] = r_curve[1]
    
    np.save(file_path, record_dict)   

if __name__ == '__main__':
    global run_single_image
    run_single_image = False
    list_path = "/viscompfs/users/sawang/ScanNet/analyze_list_new.txt"
    root = dict()
    root['data'] = "/viscompfs/users/sawang/ScanNet/data_motion_range"
    root['gt'] = "/viscompfs/users/sawang/ScanNet/data_gt"
    root['pose'] = "/viscompfs/users/sawang/ScanNet/data_pose"
    root['depth'] = "/viscompfs/users/sawang/ScanNet/data/scans"
    root['dump'] = "/viscompfs/users/sawang/ScanNet/analysis_clip_params"
    img_w, img_h = 320, 240
    deviation = 0.2
    step = 5e-3
    patch_config = (5, 4) # (patch_size, dilation)
    eps_lst = np.arange(-deviation, deviation, step)
    clip_min_list = [0, 10, 20, 30, 40, 50]
    clip_max_list = [60, 70, 80, 90, 100]
    thresh_list = np.array(np.meshgrid(clip_min_list, clip_max_list)).T.reshape(-1, 2)
    

    # parse arguments
    parser = set_parse()
    args = parser.parse_args()
    type_name = args.type_name
    parts = int(args.parts)
    split = int(args.split) # [1, parts], for parts=4, can be 1, 2, 3, 4
    if split > parts:
        raise ValueError("`split` should be less or equal to `parts`")
    print('Experiment type: {}'.format(args.type_name))
    

    # set random seed
    seed = 732
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # define layers
    layers = dict()
    layers['backproject_depth'] = BackprojectDepth(1, img_h, img_w).cuda()
    layers['project_3d'] = Project3D(1, img_h, img_w).cuda()

    log_iter = 1
    tic = time.time()
    frame_list = read_pair_list(list_path)

    # select partition
    partition_size = len(frame_list) // parts
    start = (split - 1) * partition_size
    end = start + partition_size if split < parts else len(frame_list)
    partition = frame_list[start:end]

    for i, frame_t in enumerate(partition, start):
        frame = load_data(frame_t, root, img_w, img_h)
        frame = data_to_device_tensor(frame)
        try:
            agg_results = run_test(frame_t,
                                frame,
                                layers,
                                patch_config,
                                thresh_list,
                                deviation, step, root, type_name)
            if run_single_image:
                input("Press Enter to continue...")
        except:
            print('Experiment at', frame_t, 'failed')

        if i % log_iter == 0:
            toc = time.time()
            print('Frame {}/{}, {}s/it'.format(i+1, len(frame_list), (toc-tic)/log_iter))
            tic = toc