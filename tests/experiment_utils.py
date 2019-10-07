import os
import numpy as np
import imageio
import cv2
import copy
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import defaultdict, namedtuple
import sys
sys.path.insert(0, '/viscompfs/users/sawang/PatchSfM/src/')

from utils import meshgrid, pixel2cam, cam2pixel, bilinear_sampler
from rotm2euler import isRotationMatrix, rotationMatrixToEulerAngles, eulerAnglesToRotationMatrix


######################### Basic functions ###########################

def to_homog(x):
    return np.vstack([x, [1]])

def to_homog_torch(x):
    return torch.stack([x, [1]], dim=0)

def de_homog(x):
    return x[:2, 0, np.newaxis] / x[2, 0]

def de_homog_torch(x):
    return x[:2, 0].unsqueeze(dim=-1) / x[2, 0]

######################### Pose-related functions ####################

def inversePose(T):
    '''Inverse a 4x4 pose matrix (ex.src -> tgt to tgt -> src)'''
    R = T[:3, :3]
    t = T[:3, 3]
    
    R_inv = np.linalg.inv(R)
    t_inv = np.dot(R_inv, t.reshape((3, -1))) * -1
    
    T_inv = np.zeros_like(T)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv.reshape(-1)
    T_inv[3, 3] = 1
    return T_inv
    

def transform(T):
    """Transform a 4x4 matrix to a 6-vector [tx, ty, tz, rx, ry, rz]"""
    R = T[:3, :3]
    t = T[:3, 3]
    r = rotationMatrixToEulerAngles(R)
    return np.hstack([t, r])

def untransform(t):
    """Transform a 6-vector [tx, ty, tz, rx, ry, rz] to  a 4x4 matrix"""
    T = np.zeros([4, 4])
    R = eulerAnglesToRotationMatrix(t[3:])
    T[:3, :3] = R
    T[:3, 3] = t[:3]
    T[3, 3] = 1
    return T

def get_rel_pose(pose_src, pose_tgt):
    """Generate relative transformation from target to source frame.

        Inputs:
            pose_src: 4x4 camera pose (camera to world)
            pose_tgt: 4x4 camera pose (camera to world)
        Returns:
            T0: 4x4 relative camera pose (from target to source)
            t0: 6-vector [tx, ty, tz, rx, ry, rz]
            containsNaN: whether the input pose is valid, (True if pose is invalid)
    """
    R_src0, t_src0 = pose_src[:3, :3], pose_src[:3, 3]
    R_tgt, t_tgt = pose_tgt[:3, :3], pose_tgt[:3, 3]

    # calculate relative pose according to
    # https://math.stackexchange.com/questions/709622/relative-camera-matrix-pose-from-global-camera-matrixes
    # q_src = T @ q_tgt
    T0 = np.zeros([4, 4])
    T0[:3, :3] = R_src0.T @ R_tgt
    T0[:3, 3] = R_src0.T @ (t_tgt - t_src0)
    T0[3, 3] = 1

    # transform to tx, ty, tz, rx, ry, rz
    t0 = transform(T0)
    containsNaN = np.isnan(t0).any()

    return T0, t0, containsNaN

def matchORB(src, tgt, show_matches=False):
    """find matching point locations with ORB descriptors"""
    # Initiate detector
    orb = cv2.ORB_create()

    # find the keypoints and descriptors with ORB
    kp_src, des_src = orb.detectAndCompute(src,None)
    kp_tgt, des_tgt = orb.detectAndCompute(tgt,None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des_tgt, des_src) # a list of DMatch objects
    
    # https://stackoverflow.com/questions/13318853/opencv-drawmatches-queryidx-and-trainidx
    # for each DMatch object, `queryIdx` refers to tgt index
    # `trainIdx` refers to src index
    tgt_idx = [m.queryIdx for m in matches]
    src_idx = [m.trainIdx for m in matches]
    
    tgt_pts = np.array([kp_tgt[idx].pt for idx in tgt_idx])
    src_pts = np.array([kp_src[idx].pt for idx in src_idx])

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    
    # Draw first 10 matches.
    if show_matches:
        match_img = cv2.drawMatches(tgt, kp_tgt, src, kp_src, matches[:10], None, flags=2)
        plt.imshow(match_img)
        plt.show()
        
    return matches, src_pts, tgt_pts

def estimate_pose(intrinsics, src_pts, tgt_pts):
    assert intrinsics[0, 0] == intrinsics[1, 1]
    f = intrinsics[0, 0]
    C = intrinsics[:2, 2]

    # Alternative:
    # https://stackoverflow.com/questions/33906111/how-do-i-estimate-positions-of-two-cameras-in-opencv
    E, mask = cv2.findEssentialMat(tgt_pts,
                                   src_pts,
                                   method=cv2.RANSAC,
                                   focal=f, pp=tuple(C), threshold=0.001)

    points, R, t, mask = cv2.recoverPose(E, tgt_pts, src_pts)

    pose = np.zeros([4, 4])
    pose[:3, :3] = R
    pose[:3,  3] = t.flatten()
    pose[ 3,  3] = 1
    return pose

######################### Numpy Warping ############################

def bilinear_pt(coord, src):
    """Given a single point coordinate of shape [3, 1], bilinear
        sample image intensity on the source image"""
    
    H, W, _ = src.shape
    x, y = coord[0, 0], coord[1, 0]
    x0 = np.floor(x)
    x1 = x0 + 1
    y0 = np.floor(y)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, W)
    x1 = np.clip(x1, 0, W)
    y0 = np.clip(y0, 0, H)
    y1 = np.clip(y1, 0, H)

    wt_x0 = x1 - x
    wt_x1 = x - x0
    wt_y0 = y1 - y
    wt_y1 = y - y0
    #print(wt_x0, wt_x1, wt_y0, wt_y1)

    w00 = wt_x0 * wt_y0
    w01 = wt_x0 * wt_y1
    w10 = wt_x1 * wt_y0
    w11 = wt_x1 * wt_y1
    # print(w00, w01, w10, w11)

    # check if coordinates are in range
    ret = 0
    if 0<=x0 and x1<W and 0<=y0 and y1<H:

        i00 = src[int(y0), int(x0)]
        i01 = src[int(y1), int(x0)]
        i10 = src[int(y0), int(x1)]
        i11 = src[int(y1), int(x1)]
        ret = w00 * i00 + w01 * i01 + w10 * i10 + w11 * i11

    return ret

def bilinear_pt_torch(coord, src):
    """Given a single point coordinate of shape [3, 1], bilinear
        sample image intensity on the source image"""
    
    H, W, _ = src.shape
    x, y = coord[0, 0], coord[1, 0]
    x0 = torch.floor(x)
    x1 = x0 + 1
    y0 = torch.floor(y)
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, W)
    x1 = torch.clamp(x1, 0, W)
    y0 = torch.clamp(y0, 0, H)
    y1 = torch.clamp(y1, 0, H)

    wt_x0 = x1 - x
    wt_x1 = x - x0
    wt_y0 = y1 - y
    wt_y1 = y - y0
    #print(wt_x0, wt_x1, wt_y0, wt_y1)

    w00 = wt_x0 * wt_y0
    w01 = wt_x0 * wt_y1
    w10 = wt_x1 * wt_y0
    w11 = wt_x1 * wt_y1
    # print(w00, w01, w10, w11)

    # check if coordinates are in range
    ret = 0
    if 0<=x0 and x1<W and 0<=y0 and y1<H:

        i00 = src[int(y0), int(x0)]
        i01 = src[int(y1), int(x0)]
        i10 = src[int(y0), int(x1)]
        i11 = src[int(y1), int(x1)]
        ret = w00 * i00 + w01 * i01 + w10 * i10 + w11 * i11

    return ret

def warp_location(pt, intrinsics, rel_pose, depth):
    """Obtain the corresponding source coordinate for a given target coordinate"""
    # create 4x4 intrinsics
    K = np.zeros([4, 4])              # 4x4
    K[:3, :3] = intrinsics
    K[3, 3] = 1
    inv_K = np.linalg.inv(intrinsics) # 3x3

    cam_coord = depth * (inv_K @ to_homog(pt)) # [3, 1]
    cam_coord = to_homog(cam_coord)  # [4, 1]
    ps =(K @ rel_pose @ cam_coord)  # [4, 1]
    ps = de_homog(ps)               # [2, 1]
    return ps

def warp_location_torch(pt, intrinsics, rel_pose, depth):
    # create 4x4 intrinsics
    K = torch.zeros([4, 4]).cuda()              # 4x4
    K[:3, :3] = intrinsics
    K[3, 3] = 1
    inv_K = torch.inverse(intrinsics) # 3x3

    cam_coord = depth * torch.matmul(inv_K, to_homog_torch(pt)) # [3, 1]
    cam_coord = to_homog(cam_coord)  # [4, 1]
    cam_coord = torch.matmul(rel_pose, cam_coord)
    ps = torch.matmul(K, cam_coord)  # [4, 1]
    ps = de_homog_torch(ps)          # [2, 1]
    return ps

#################### File reading utilities #######################

def filename_to_sort_key(fname):
    """Sort the list of scenes according to id number
        usage: sorted(list_to_sort, key=filename_to_sort_key)
        example file name: scene0016_02
    """
    splits = fname.split('_')
    prefix = splits[0][-4:]
    postfix = splits[1]
    return int(prefix + postfix)

def read_intrinsics(path):
    with open(path, 'r') as f:
        line = f.read()
        n_str = line.split(',')
        val = np.array([float(x) for x in n_str]).reshape(3, 3)

    intr = val
    return intr

def scale_intrinsics(intrinsics, oldH, oldW, newH, newW):
    x_scaling = newW / oldW
    y_scaling = newH / oldH
    intr = copy.deepcopy(intrinsics)
    
    intr[0, 0] *= x_scaling # fx
    intr[1, 1] *= y_scaling # fy
    intr[0, 2] *= x_scaling # cx
    intr[1, 2] *= y_scaling # cy
    return intr

#################### Tensorflow ##################################

def a2t(x):
    """convert numpy arry to tensor, add extra `batch`"""
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    x = tf.expand_dims(x, axis=0)
    return x

def warp(img, depth, pose, intrinsics):
    """Inverse warp a source image to the target image plane based on projection.

    Args:
    img: the source image [batch, height_s, width_s, 3]
    depth: depth map of the target image [batch, height_t, width_t]
    pose: [batch, 4, 4]
    intrinsics: camera intrinsics [batch, 3, 3]
    Returns:
    Source image inverse warped to the target image plane [batch, height_t,
    width_t, 3]
    """
    batch, height, width, _ = img.get_shape().as_list()
    # Construct pixel grid coordinates
    pixel_coords = meshgrid(batch, height, width)
    # Convert pixel coordinates to the camera frame
    cam_coords = pixel2cam(depth, pixel_coords, intrinsics)
    # Construct a 4x4 intrinsic matrix (TODO: can it be 3x4?)
    filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
    filler = tf.tile(filler, [batch, 1, 1])
    intrinsics = tf.concat([intrinsics, tf.zeros([batch, 3, 1])], axis=2)
    intrinsics = tf.concat([intrinsics, filler], axis=1)
    # Get a 4x4 transformation matrix from 'target' camera frame to 'source'
    # pixel frame.
    proj_tgt_cam_to_src_pixel = tf.matmul(intrinsics, pose)
    src_pixel_coords = cam2pixel(cam_coords, proj_tgt_cam_to_src_pixel)
    output_img = bilinear_sampler(img, src_pixel_coords)
    return output_img


def warp_image(src, rel_pose, intrinsics, dmap):
    """Wrapper for warp(), this function consumes numpy arguments"""
    batch = 1
    depths = np.arange(0.01, 100, 0.01)
    # warp images with tensorflow
    src = a2t(src / 255.0)
    rel_pose = a2t(rel_pose)
    intrinsics = a2t(intrinsics)
    dmap = a2t(dmap)
    proj_img = warp(src, dmap, rel_pose, intrinsics)
    proj_img = tf.image.convert_image_dtype(proj_img, tf.float32)
    return proj_img

###################### Experiment plots #########################

def plot_normal(normal):
    ones = np.ones_like(normal[:, :, :1])
    x = (normal[:, :, :1] + ones) / 2.0
    y = (normal[:, :, 1:2] * -1 + ones) / 2.0
    z = (normal[:, :, 2:] / 2.0 ) * -1 + 0.5 * ones

    plot = np.dstack((x, y, z))
    return plot

def unplot_normal(normal):
    ones = np.ones_like(normal[:, :, :1])
    x = (normal[:, :, :1] * 2 - ones) #* -1
    y = (normal[:, :, 1:2] * 2 - ones) * -1
    z = (normal[:, :, 2:] - 0.5 * ones) * -1 * 2.0

    plot = np.dstack((x, y, z))
    return plot

def plot_inv_curve(loc, colors,
                   tgt, src, rel_pose, intrinsics, dmap):
    """Plot intensity difference versus inverse depth"""
    s = 3 # rectangle size
    # get warped image
    warped_src = warp_image(src, rel_pose, intrinsics, dmap)
    with tf.Session() as sess:
        proj_img = sess.run(warped_src)
    plt.imshow(np.squeeze(proj_img))
    plt.title('Warped image')
    plt.show()
    
    # prepare figure
    fig = plt.figure(figsize=(20, 10))
    ax_tgt = fig.add_subplot(121)
    ax_src = fig.add_subplot(122)
    ax_tgt.imshow(tgt)
    ax_src.imshow(src)
    ax_tgt.set_title('Target')
    ax_src.set_title('Source')
    ax_src.set_ylim(bottom=H, top=0)

    # warp provided coordinates
    inv_depth_values = np.arange(0.0001, 1.01, 0.0001)
    locations = np.split(loc, 3, axis=1)

    # will contain tuple of ('pt', 'ps', 'It', 'Is', 'inv_depth', 'diff')
    result = defaultdict(list)
    r = namedtuple('r', 'pt ps It Is inv_depth diff')
    for i, (pt, c) in enumerate(zip(locations, colors)):
        It = tgt[tuple(pt)[::-1]] # target intensity
        for inv_d in inv_depth_values:
            # warp
            d = 1/inv_d
            ps = warp_location(pt, intrinsics, d)
            Is = bilinear_pt(ps, src) # source intensity
            diff = np.mean(np.abs(It-Is))
            tmp = r(pt=pt, ps=ps, It=It, Is=Is, inv_depth=inv_d, diff=diff)
            result[i].append(tmp)

    # plot RGB images for comparison
    print('Generating plots')
    for i, (pt, c) in enumerate(zip(locations, colors)):
        # plot rectangles on target image
        rect = patches.Rectangle((pt[0, 0]-s//2, pt[1, 0]-s//2), s, s,
                                 linewidth=1,edgecolor=c,facecolor='none')
        ax_tgt.add_patch(rect)

        # plot ground-truth correspondence on source image
        gt_corr = warp_location(pt, intrinsics, dmap[tuple(pt)[::-1]])
        rect2 = patches.Rectangle((gt_corr[0, 0]-s//2, gt_corr[1, 0]-s//2), s, s,
                                   linewidth=1,edgecolor=c,facecolor='none')
        ax_src.add_patch(rect2)

        # plot lowest photometric loss on source image
        diffs = [x.diff for x in result[i]]
        lowest_idx = np.argmin(diffs)
        lowest_ps = result[i][lowest_idx].ps
        circle = patches.Circle((lowest_ps[0, 0], lowest_ps[1, 0]), radius=s//2,
                                 linewidth=1,edgecolor=c,facecolor='none')
        ax_src.add_patch(circle)
    plt.show()

    # plot error curve
    for i, c in enumerate(colors):
        diffs = [x.diff for x in result[i]]
        lowest_idx = np.argmin(diffs)
        lowest_inv_depth = result[i][lowest_idx].inv_depth
        gt_depth = dmap[tuple(locations[i])[::-1]]
        print(c+' line ground truth depth', ':', gt_depth[0])
        print(c+' line depth w/ lowest error:', 1/lowest_inv_depth)

        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111)
        ax.plot(inv_depth_values, diffs, c=c, label=locations[i].flatten())
        ax.axvline(x=1/gt_depth, c=c)
        ax.axvline(lowest_inv_depth, c='k')
        ax.set_xlabel('Inverse Depth')
        ax.set_ylabel('Photometric Error')
        ax.legend()
        plt.show()

    plt.close('all')
    return result
