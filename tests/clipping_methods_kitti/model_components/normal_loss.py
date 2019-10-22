import torch
import numpy as np
import torch.nn.functional as F

def ssim_patch(x, y):
    """Computes the patch SSIM loss
        Input:
        x: [B, 3, H, W, patch_size, patch_size]
        y: [B, 3, H, W, patch_size, patch_size]
        Output:
        SSIM: SSIM of all patches, [B, 1, H, W]
    """
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    B, _, H, W, size, _ = x.shape
    x = x.contiguous().view(B, 3, H, W, size * size)
    y = y.contiguous().view(B, 3, H, W, size * size)

    mu_x = torch.mean(x, dim=4)
    mu_y = torch.mean(y, dim=4)

    sig_x = torch.mean(x ** 2, dim=4) - mu_x ** 2
    sig_y = torch.mean(y ** 2, dim=4) - mu_y ** 2
    sig_xy = torch.mean(x * y, dim=4) - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sig_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sig_x + sig_y + C2) + 1e-10
    SSIM = torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

    SSIM = torch.mean(SSIM, dim=1, keepdim=True)
    return SSIM

def compute_cos_dist(pred_normal, cam_coords, alpha=0.1, n_shift=3):
    """Computes the depth and normal orthogonality loss
        Input:
        pred_normal: predicted normal of the network, [B, 3, H, W]
        cam_coords: Homogeneous 3D points of each pixel, [B, 4, H * W]
    """
    ## Remove Homogenous coord
    B, _, H, W = pred_normal.shape
    pts_3d_map = cam_coords[:, :-1, :].view(B, 3, H, W)

    nei = n_shift
    # shift the 3d pts map by nei along 8 directions
    pts_3d_map_ctr = pts_3d_map[:, :, nei:-nei, nei:-nei]
    pts_3d_map_x0 = pts_3d_map[:, :, nei:-nei, 0:-(2*nei)]
    pts_3d_map_y0 = pts_3d_map[:, :, 0:-(2*nei), nei:-nei]
    pts_3d_map_x1 = pts_3d_map[:, :, nei:-nei, 2*nei:]
    pts_3d_map_y1 = pts_3d_map[:, :, 2*nei:, nei:-nei]
    #pts_3d_map_x0y0 = pts_3d_map[:, :, 0:-(2*nei), 0:-(2*nei)]
    #pts_3d_map_x0y1 = pts_3d_map[:, :, 2*nei:, 0:-(2*nei)]
    #pts_3d_map_x1y0 = pts_3d_map[:, :, 0:-(2*nei), 2*nei:]
    #pts_3d_map_x1y1 = pts_3d_map[:, :, 2*nei:, 2*nei:]

    # generate difference between the central pixel and one of 8 neighboring pixels
    # each `diff` has shape [batch, 3, H-2*nei, W-2*nei]
    diff_x0 = pts_3d_map_ctr - pts_3d_map_x0
    diff_x1 = pts_3d_map_ctr - pts_3d_map_x1
    diff_y0 = pts_3d_map_y0 - pts_3d_map_ctr
    diff_y1 = pts_3d_map_y1 - pts_3d_map_ctr
    #diff_x0y0 = pts_3d_map_x0y0 - pts_3d_map_ctr
    #diff_x0y1 = pts_3d_map_ctr - pts_3d_map_x0y1
    #diff_x1y0 = pts_3d_map_x1y0 - pts_3d_map_ctr
    #diff_x1y1 = pts_3d_map_ctr - pts_3d_map_x1y1
    diff_stk = [diff_x0, diff_y0, diff_x1, diff_y1]

    
    # crop pred_norm
    cropped_norm = pred_normal[:, :, nei:-nei, nei:-nei]
    n_norm = torch.norm(cropped_norm, dim=1, keepdim=True)

    # calculate loss by dot product
    cos_dist_all = torch.zeros_like(n_norm)
    for diff in diff_stk:
        v_norm = torch.norm(diff, dim=1)
        cos_dist = torch.sum(diff * cropped_norm, 1, keepdim=True) / (v_norm * n_norm + 1e-10)
        cos_dist_abs = torch.abs(cos_dist)
        cos_dist_all = cos_dist_all + cos_dist_abs
    
    cos_dist_all = cos_dist_all / 4
    return cos_dist_all

def compute_normal_regularizar(pred_norm, pts_3d_map):
    """Computes the normal regularizar between tgt and src
    Input:
        pts_3d_map: backprojected 3D points (K_inv @ x * d), [B, 4, H * W]
        pred_norm: predicted surface normal, [B, 3, H, W]
    Output:
        reg: normal regulizar loss at each pixel. [B, 1, H, W]
    """
    ## multiply direction with -1 making it pointing towards camera center and normalize viewing direction vector to unit vector
    B, _, H, W = pred_norm.shape
    pts_3d_map = pts_3d_map[:, :-1, :].view(B, 3, H, W)
    view_direction = pts_3d_map * -1
    norm = torch.norm(view_direction, dim=1, keepdim=True)
    view_direction = view_direction / (norm + 1e-10)
    reg = torch.sum(view_direction * pred_norm, 1, keepdim=True) * -1
    
    return reg

def compute_identity_patch_difference(tgt_img, src_img, patch_size=7, dilation=1, type="median"):
    """Computes the patch intensity difference between target and source
    Input:
        tgt_img: target image, [B, 3, H, W]
        src_img: source image, [B, 3, H, W]
    Output:
        patch_diff: patch intensity difference, value in each pixel indicates max/average intensity 
        difference for that patch. [B, 1, H-2*offset, W-2*offset] 
    """

    batch, _, height, width = tgt_img.shape
    unfold = torch.nn.Unfold(kernel_size=(patch_size, patch_size), dilation=dilation)

    ## sample patch intensities, 
    # final resulting `tgt_intensities` has shape [B, 3, H-2*offset, W-2*offset, patch_size, patch_size]
    psize_eff = 1 + dilation * (patch_size - 1) # effective patch size
    ofs = (psize_eff - 1) // 2                  # offset
    tgt_intensities = unfold(tgt_img)       # [B, 3*psize*psize, (H-2*offset)*(W-2*offset)]
    tgt_intensities = tgt_intensities.view(batch, 3, patch_size, patch_size, height-2*ofs, width-2*ofs)
    tgt_intensities = tgt_intensities.permute(0, 1, 4, 5, 2, 3) 

    src_intensities = unfold(src_img)       # [B, 3*psize*psize, (H-2*offset)*(W-2*offset)]
    src_intensities = src_intensities.view(batch, 3, patch_size, patch_size, height-2*ofs, width-2*ofs)
    src_intensities = src_intensities.permute(0, 1, 4, 5, 2, 3) 

    # take max/avg/median over patch 
    if type == "mean":
        patch_abs = torch.abs(tgt_intensities - src_intensities).mean(dim=1, keepdim=True).mean(dim=(4, 5)) # [B, 1, H - 2*offset, W - 2*offset]
    else:
        patch_abs = torch.abs(tgt_intensities - src_intensities).mean(dim=1, keepdim=True)
        patch_abs = patch_abs.view(batch, 1, height-2*ofs, width-2*ofs, -1) # [B, 1, H-2*offset, W-2*offset, patch_size * patch_size]
        if type == "max":
            patch_abs, _ = patch_abs.max(dim=-1) # [B, 1, H-2*offset, W-2*offset]
        elif type == "median":
            patch_abs, _ = patch_abs.median(dim=-1)
        else:
            raise ValueError("Operation type not recognized.")

    patch_diff = patch_abs * 0.15 + ssim_patch(tgt_intensities, src_intensities) * 0.85

    return patch_diff


def compute_patch_difference(tgt_img, src_img, K, cam_coords,
                             pred_pose, pred_depth, pred_norm,
                             patch_size=7, dilation=1, type="median"):
    """Computes the patch intensity difference between tgt and src
    Input:
        tgt_img: target image, [B, 3, H, W]
        src_img: source image, [B, 3, H, W]
        K: intrinsics, [B, 4, 4]
        cam_coords: backprojected 3D points (K_inv @ x * d), [B, 4, H * W]
        pred_pose: predicted poses, [B, 4, 4]
        pred_depth: predicted depth, [B, 1, H, W]
        pred_norm: predicted surface normal, [B, 3, H, W]
        patch_size: patch size, integer (default = 7)
        dilation: dilation factor for patch (default = 1)
    Output:
        patch_diff: patch intensity difference, value in each pixel indicates max/average intensity 
        difference for that patch. [B, 1, H, W]
    """
    batch, _, height, width = pred_norm.shape
    cam_coords = cam_coords[:, :-1, :].view(batch, 3, height, width)
    K = K[:, :3, :3]

    unfold = torch.nn.Unfold(kernel_size=(patch_size, patch_size), dilation=dilation)

    ## sample tgt patch intensities, 
    # final resulting `tgt_intensities` has shape [B, 3, H-2*offset, W-2*offset, patch_size, patch_size]
    psize_eff = 1 + dilation * (patch_size - 1) # effective patch size
    ofs = (psize_eff - 1) // 2                  # offset
    tgt_intensities = unfold(tgt_img)       # [B, 3*psize*psize, (H-2*offset)*(W-2*offset)]
    tgt_intensities = tgt_intensities.view(batch, 3, patch_size, patch_size, height-2*ofs, width-2*ofs)
    tgt_intensities = tgt_intensities.permute(0, 1, 4, 5, 2, 3) 

    ## sample src patch intensities, [B, 3, H - 2 * offset, W - 2 * offset, patch_size, patch_size]
    src_intensities, src_coords = sample_src_intensity(src_img, K, cam_coords, pred_pose, pred_depth, pred_norm, patch_size, dilation)

    ## patch_difference
    if type == "mean":
        patch_abs = torch.abs(tgt_intensities - src_intensities).mean(dim=1, keepdim=True).mean(dim=(4, 5)) # [B, 1, H - 2*offset, W - 2*offset]
    else:
        patch_abs = torch.abs(tgt_intensities - src_intensities).mean(dim=1, keepdim=True)
        patch_abs = patch_abs.view(batch, 1, height-2*ofs, width-2*ofs, -1) # [B, 1, H-2*offset, W-2*offset, patch_size * patch_size]
        if type == "max":
            patch_abs, _ = patch_abs.max(dim=-1) # [B, 1, H-2*offset, W-2*offset]
        elif type == "median":
            patch_abs, _ = patch_abs.median(dim=-1)
        else:
            raise ValueError("Operation type not recognized.")

    patch_diff = patch_abs * 0.15 + ssim_patch(tgt_intensities, src_intensities) * 0.85

    return patch_diff, src_coords

def sample_src_intensity(src_img, K, cam_coords, pred_pose, pred_depth, pred_norm, patch_size, dilation):
    """Computes src patch intensity
    Input:
        src_img: source image, [B, 3, H, W]
        K: intrinsics, [B, 3, 3]
        cam_coords: backprojected 3D points (K_inv @ x * d), [B, 3, H, W]
        pred_pose: predicted poses, [B, 4, 4]
        pred_depth: predicted depth, [B, 1, H, W]
        pred_norm: predicted surface normal, [B, 3, H, W]
        patch_size: patch size, integer (default = 7)
        dilation: dilation factor for patch (default = 1)
    Output:
        patch_intensities: src patch intensity, [B, 3, H - 2 * offset, W - 2 * offset, patch_size, patch_size]
        (offset = (patch size - 1) / 2)
        patch_coords: patch coordinates [B, 2, (H - 2*offset)*(W - 2*offset), patch_size * patch_size]
    """
    batch, _, height, width = pred_norm.shape
    psize_eff = 1 + dilation * (patch_size - 1) # effective patch size
    offset = (psize_eff - 1) // 2       
    
    ## sample src patch intensities
    H = calculate_homography(pred_pose, K, pred_norm, cam_coords) #[B, 3, 3, H, W]

    ## cam_coords = (K_inv @ pts * d) / d 
    cam_coords = cam_coords / pred_depth

    ## get sample grid for each patch
    patch_coords = get_patch_coords(H, cam_coords, patch_size, dilation) #[B, 3, H - 2*offset, W - 2*offset, p_size, p_size]
    patch_coords = patch_coords.view(batch, 2, -1, patch_size * patch_size)
    patch_coords = patch_coords.permute(0, 2, 3, 1)
    
    ## normalize coordinates from sampling
    patch_coords[..., 0] /= width - 1 
    patch_coords[..., 1] /= height - 1 
    patch_coords = (patch_coords - 0.5) * 2

    ## sample intensity from src img
    patch_intensities = F.grid_sample(src_img, patch_coords, padding_mode='border')
    patch_intensities = patch_intensities.view(batch, 3, height - 2* offset, width - 2* offset, patch_size, patch_size)
    
    return patch_intensities, patch_coords

def get_patch_coords(H, cam_coords, patch_size, dilation):
    """ Get the src patch coordinates 
    Input:
        H:  K(R - tn^T/d), [B, 3, 3, H, W]
        cam_coords: backprojected 3D points / d (K_inv @ x), [B, 3, H, W]
        patch_size: patch size, integer (default = 7)
        dilation: dilation factor for patch
    Output:
        patch_coords: sample idx for each patch, [B, 2, H - 2 * offset, W - 2 * offset, patch_size, patch_size] 
        (offset = (patch size - 1) / 2)
    """
    batch, _, height, width = cam_coords.shape
    psize_eff = 1 + dilation * (patch_size - 1) # effective patch size
    offset = (psize_eff - 1) // 2               # offset

    unfold = torch.nn.Unfold(kernel_size=(patch_size, patch_size), dilation=dilation)
    patch_coords = unfold(cam_coords) # # [B, 3*psize*psize, (H-2*offset)*(W-2*offset)]
    patch_coords = patch_coords.view(batch, 3, patch_size, patch_size, height-2*offset, width-2*offset)
    patch_coords = patch_coords.permute(0, 1, 4, 5, 2, 3) #[B, 3, H-2*offset, W-2*offset, patch_size, patch_size]
    H = H[:, :, :, offset:-offset, offset:-offset]
    
    patch_coords = torch.einsum("mnpqr, mpqrst -> mnqrst", H, patch_coords) #[B, 3, H, W, patch_size, patch_size]
    patch_coords = patch_coords[:, :2, :, :, :, :] / (patch_coords[:, 2:, :, :, :, :] + 1e-10) # dehomogenize
    return patch_coords

def calculate_homography(pose, K, pred_norm, cam_coords):
    """Computes the homography of each pixel 
    Input:
        K: intrinsics, [B, 3, 3]
        cam_coords: backprojected 3D points (K_inv @ x * d), [B, 3, H, W]
        pred_pose: predicted poses, [B, 4, 4]
        pred_norm: predicted surface normal, [B, 3, H, W]
    Output:
        H: sample idx for each patch, [B, 3, 3, H, W] 
    """
    B, _, H, W = pred_norm.shape

    # compute d = - (nx*x + ny*y + nz*z), [B, 1, H, W]
    d = torch.sum(pred_norm * cam_coords, dim=1, keepdim=True) * -1

    # construct homography matrix H = (R - tn^T/d), [B, 3, H, W]
    scaled_norm = pred_norm / (d + 1e-10)

    R = pose[:, :3, :3].view(B, 3, 3)
    t = pose[:, :3, 3].view(B, 3)
    R = torch.unsqueeze(torch.unsqueeze(R, -1), -1) # [B, 3, 3, 1, 1]
    R = R.repeat(1, 1, 1, H, W)
    
    # outer product: outer[m, q, n, p, r] = t[m, q] * scaled_norm[m, n, p, r]
    outer = torch.einsum('mq, mnpr -> mqnpr', t, scaled_norm)
    H = R - outer # [B, 3, 3, H, W]

    # pre-multiply with K, H = K(R - tn^T/d)
    H = torch.einsum('bcd, bdehw -> bcehw', K, H)

    return H
