# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
import os
import numpy as np
import torch
import hashlib
import zipfile
import torch

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib import collections as mc
plt.switch_backend('agg')

from six.moves import urllib

def freeze_weights(model, models_to_freeze, model_name):
    if model_name in models_to_freeze:
        for param in model.parameters():
            param.requires_grad = False


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines

def plot_normal(normal):
    x = (normal[0, :, :] + 1) / 2
    y = (normal[1, :, :] * -1 + 1) / 2
    z = (normal[2, :, :] / 2) * -1 + 0.5

    normal = torch.stack([x, y, z], dim=0)
    return normal

def normalize_image(x):
    """Rescale image pixels to span range [0, 1]
    """
    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d

def normalize_loss_map(x):
    """Rescale loss map to span range [0, 1], removing extreme values"""
    mean = float(x.mean().cpu().data)
    std = float(x.std().cpu().data)
    max_thresh = mean + 2*std
    min_thresh = mean - 2*std

    x = torch.clamp(x, min=min_thresh, max=max_thresh)
    x = normalize_image(x)
    return x

def get_angle(axisangle):
    vec = axisangle[:, 0]
    angle = torch.norm(vec, p=2, dim=2)
    return angle

def sec_to_hm(t):
    """Convert time in seconds to time in hours, minutes and seconds
    e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s


def sec_to_hm_str(t):
    """Convert time in seconds to a nice string
    e.g. 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)


def download_model_if_doesnt_exist(model_name):
    """If pretrained kitti model doesn't exist, download and unzip it
    """
    # values are tuples of (<google cloud URL>, <md5 checksum>)
    download_paths = {
        "mono_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_640x192.zip",
             "a964b8356e08a02d009609d9e3928f7c"),
        "stereo_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_640x192.zip",
             "3dfb76bcff0786e4ec07ac00f658dd07"),
        "mono+stereo_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_640x192.zip",
             "c024d69012485ed05d7eaa9617a96b81"),
        "mono_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_no_pt_640x192.zip",
             "9c2f071e35027c895a4728358ffc913a"),
        "stereo_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_no_pt_640x192.zip",
             "41ec2de112905f85541ac33a854742d1"),
        "mono+stereo_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_no_pt_640x192.zip",
             "46c3b824f541d143a45c37df65fbab0a"),
        "mono_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_1024x320.zip",
             "0ab0766efdfeea89a0d9ea8ba90e1e63"),
        "stereo_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_1024x320.zip",
             "afc2f2126d70cf3fdf26b550898b501a"),
        "mono+stereo_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_1024x320.zip",
             "cdc5fc9b23513c07d5b19235d9ef08f7"),
        }

    if not os.path.exists("models"):
        os.makedirs("models")

    model_path = os.path.join("models", model_name)

    def check_file_matches_md5(checksum, fpath):
        if not os.path.exists(fpath):
            return False
        with open(fpath, 'rb') as f:
            current_md5checksum = hashlib.md5(f.read()).hexdigest()
        return current_md5checksum == checksum

    # see if we have the model already downloaded...
    if not os.path.exists(os.path.join(model_path, "encoder.pth")):

        model_url, required_md5checksum = download_paths[model_name]

        if not check_file_matches_md5(required_md5checksum, model_path + ".zip"):
            print("-> Downloading pretrained model to {}".format(model_path + ".zip"))
            urllib.request.urlretrieve(model_url, model_path + ".zip")

        if not check_file_matches_md5(required_md5checksum, model_path + ".zip"):
            print("   Failed to download a file which matches the checksum - quitting")
            quit()

        print("   Unzipping model...")
        with zipfile.ZipFile(model_path + ".zip", 'r') as f:
            f.extractall(model_path)

        print("   Model unzipped to {}".format(model_path))

def denormalize_coords(coords, width, height):
    """Denormalize the sample coordinates ranging ing [-1, 1] 
    
    Args:
        coords: coordinates of shape [B, 2, H, W], 
        width: image width
        height: image height
    Returns:
        coords: [B, 2, H, W] inhomogeneous coordinates
    """
    coords = coords / 2 + 0.5
    coords[..., 0] *= (width - 1)
    coords[..., 1] *= (height - 1)
    return coords

def make_plot(src_img, tgt_img, patch_coords, pixel_coords, patch_centers, writer, patch_size, dilation, step):
    """Draw the patch onto the image according to the patch coordinates
    Args:
        src_img: source image, tensor of shape [B, 3, H, W]
        tgt_img: target image, tensor of shape [B, 3, H, W]
        patch_coords: patch coordinates normalized to [-1, 1], 
                        tensor of shape [B, (H - 2*offset)*(W - 2*offset), patch_size * patch_size, 2]
                        offset = (patch_size - 1)//2
        pixel_coords: pixel coordinates normalized to [-1, 1], tensor of shape [B, H, W, 2]
        patch_centers: patch centers in target image coordinates to plot for, numpy array of shape [2, n_points]
        writer: tensorboardX summary writer
    Returns:
        fig: matplotlib figure
    """
    batch_size, _, height, width = src_img.shape
    _, n_pts = patch_centers.shape
    sz = patch_size
    ofs = (dilation * (sz-1)) // 2
    h_ofs = height - 2*ofs
    w_ofs = width - 2*ofs 

    src = src_img[0, :, :, :].cpu().numpy()
    src = np.transpose(src, [1, 2, 0]) # [H, W, 3]
    tgt = tgt_img[0, :, :, :].cpu().numpy()
    tgt = np.transpose(tgt, [1, 2, 0]) # [H, W, 3]

    patch_coords = denormalize_coords(patch_coords, width, height)
    patch_coords = patch_coords.permute(0, 3, 1, 2) # [B, 2, H*W, p_size * p_size]
    patch_coords = patch_coords.contiguous().view(batch_size, 2, h_ofs, w_ofs, sz, sz)
    patch_coords = patch_coords.cpu().detach().numpy()

    pixel_coords = denormalize_coords(pixel_coords, width, height) 
    pixel_coords = pixel_coords.permute(0, 3, 1, 2) # [B, 2, H, W]
    pixel_coords = pixel_coords.cpu().detach().numpy()

    color=cm.rainbow(np.linspace(0, 1, n_pts))
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.imshow(tgt)
    ax2.imshow(src)
    for i, c in zip(range(n_pts), color):
        ctr_x, ctr_y = patch_centers[:, i].astype(int)

        # plot patch centers on target image
        ax1.scatter(ctr_x, ctr_y, color=c, s=4)

        # plot patch boundaries on target image
        upper_left  = (ctr_x-ofs, ctr_y-ofs)
        upper_right = (ctr_x+ofs, ctr_y-ofs)
        lower_left =  (ctr_x-ofs, ctr_y+ofs)
        lower_right = (ctr_x+ofs, ctr_y+ofs)
        lines = [[upper_left, upper_right], 
                [upper_right, lower_right],
                [lower_right, lower_left],
                [lower_left, upper_left]]
        lc = mc.LineCollection(lines, colors=c, linewidths=2)
        ax1.add_collection(lc)

        # plot patch centers on source image
        ctr_x_s, ctr_s_y = pixel_coords[0, :, ctr_y, ctr_x]
        ax2.scatter(ctr_x_s, ctr_s_y, color=c, s=4)

        # plot patch boundaries on source image
        ctr_x -= ofs
        ctr_y -= ofs
        coord = patch_coords[0, :, ctr_y, ctr_x, :, :] # [2, patch_size, patch_size]

        # for plotting lines, refer to 
        # https://stackoverflow.com/questions/21352580/matplotlib-plotting-numerous-disconnected-line-segments-with-different-colors
        upper_left  = coord[:,  0,  0]
        upper_right = coord[:,  0, -1]
        lower_left  = coord[:, -1,  0]
        lower_right = coord[:, -1, -1]
        lines = [[upper_left, upper_right], 
                [upper_right, lower_right],
                [lower_right, lower_left],
                [lower_left, upper_left]]
        lc = mc.LineCollection(lines, colors=c, linewidths=2)
        ax2.add_collection(lc)

    # set `close=True` to automatically close the figure
    writer.add_figure('patch_reprojection', fig, global_step=step, close=True)
    return writer
