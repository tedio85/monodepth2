# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import random

import argparse
import numpy as np
import PIL.Image as pil

from utils import readlines
from kitti_utils import generate_depth_map

def scene_key(x):
    # example: scene0002_00
    scn_name = x.replace('scene', '')
    sid, num = scn_name.split('_')
    ret = int(sid + num) # 0002_00 => 200
    return ret

def set_parse():
    parser = argparse.ArgumentParser(description='export_gt_depth')

    parser.add_argument('--data_path',
                        type=str,
                        help='path to the root of the test dataset',
                        required=True)
    parser.add_argument('--split',
                        type=str,
                        help='which split to export gt from',
                        required=True,
                        choices=["scannet", "eigen", "eigen_benchmark"])
    parser.add_argument('--samples_per_scene', 
                        type=int,
                        help='number of ground truth depth maps selected for testing in each scene,\
                                this is only used when `split` is set to `scannet`',
                        default=10)
    parser.add_argument('--output_dir',
                        type=str,
                        help='where to export the test files list and ground truth depths,\
                                this is only used when `split` is set to `scannet`',
                        required=False,
                        default='/viscompfs/users/sawang/ScanNet/data/scans_test/')

    return parser.parse_args()

def export_gt_depths_kitti(opt):
    split_folder = os.path.join(os.path.dirname(__file__), "splits", opt.split)
    lines = readlines(os.path.join(split_folder, "test_files.txt"))

    print("Exporting ground truth depths for {}".format(opt.split))

    gt_depths = []
    for line in lines:

        folder, frame_id, _ = line.split()
        frame_id = int(frame_id)

        if opt.split == "eigen":
            calib_dir = os.path.join(opt.data_path, folder.split("/")[0])
            velo_filename = os.path.join(opt.data_path, folder,
                                         "velodyne_points/data", "{:010d}.bin".format(frame_id))
            gt_depth = generate_depth_map(calib_dir, velo_filename, 2, True)
        elif opt.split == "eigen_benchmark":
            gt_depth_path = os.path.join(opt.data_path, folder, "proj_depth",
                                         "groundtruth", "image_02", "{:010d}.png".format(frame_id))
            gt_depth = np.array(pil.open(gt_depth_path)).astype(np.float32) / 256

        gt_depths.append(gt_depth.astype(np.float32))

    output_path = os.path.join(split_folder, "gt_depths.npz")

    print("Saving to {}".format(opt.split))

    np.savez_compressed(output_path, data=np.array(gt_depths))

def generate_test_list_scannet(data_path, samples_per_scene=2):
    """Randomly select 2000 images from the test set"""
    scenes = [x for x in os.listdir(data_path) if x.startswith('scene')]
    scenes = sorted(scenes, key=scene_key)

    all_files = [] # [(scene_id_str, frame_id_str), ...], ordered by scene_id then by frame_id
    for scn in scenes:
        scn_path = os.path.join(data_path, scn, 'depth')
        scn_files = [int(x.split('.')[0]) for x in os.listdir(scn_path)] # list of file ids
        rand_files = random.sample(scn_files, samples_per_scene)    # randomly sample images
        rand_files.sort() # sort according to file id
        rand_files = [(scn, fid) for fid in rand_files]
        all_files.extend(rand_files)
    
    # write to file
    lst_path = os.path.join(data_path, 'test_list.txt')
    template = "{} {}\n"
    with open(lst_path, 'w') as f:
        for (scn, fid) in all_files:
            f.write(template.format(scn, fid))

def export_gt_depths_scannet(data_path, output_dir):
    """Export ground truth depths and corresponding scene and frame ids"""
    # read test list
    lst_path = os.path.join(data_path, 'test_list.txt')
    all_files = []
    with open(lst_path, 'r') as f:
        for line in f:
            scn, fid = line.split()
            all_files.append((scn, fid))
    
    # load depth map
    gt_depths = []
    for (scn, fid) in all_files:
        gt_depth_path = os.path.join(data_path, scn, 'depth', fid+'.png')
        gt_depth = np.array(pil.open(gt_depth_path)).astype(np.float32)
        gt_depths.append(gt_depth)

    # save to .npz file
    output_path = os.path.join(output_dir, "gt_depths.npz")
    print("Saving {} files to {}".format(len(gt_depths), output_path))
    np.savez_compressed(output_path, data=np.array(gt_depths))


if __name__ == "__main__":
    seed = 732
    random.seed(seed)
    np.random.seed(seed)
    
    opt = set_parse()
    if opt.split == "scannet":
        generate_test_list_scannet(opt.data_path, opt.samples_per_scene)
        export_gt_depths_scannet(opt.data_path, opt.output_dir)
    else:
        export_gt_depths_kitti(opt)
