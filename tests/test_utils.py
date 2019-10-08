import numpy as np
import torch 
from normal_loss import *

def barron_loss(x, alpha=1, c=0.01):
    t1 = np.abs(alpha-2) / alpha # term 1
    base = ((x/c)**2) / np.abs(alpha-2) + 1
    loss = t1 * (np.power(base, alpha/2) - 1)
    return loss

def DSSIM(I_x, I_y,
          C1=0.01 ** 2,
          C2=0.03 ** 2):
    """Given two windows of the same shape, calculate DSSIM"""
    mu_x = np.mean(I_x)
    mu_y = np.mean(I_y)
    sigma_x  = np.mean(I_x ** 2) - mu_x ** 2
    sigma_y  = np.mean(I_y ** 2) - mu_y ** 2
    sigma_xy = np.mean(I_x*I_y) - mu_x * mu_y

    numer = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denom = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return np.clip((1 - numer/denom)/2, a_min=0, a_max=1)

def loss_func():
    """return the loss as a function"""
    def loss(tgt_patch, src_patch, alpha=0.85):
        assert tgt_patch.shape == src_patch.shape  
        patch_size, _, _ = tgt_patch.shape
        ofs = (patch_size-1) // 2
        L1 = np.mean(np.abs(tgt_patch[ofs, ofs] - src_patch[ofs, ofs]))
        dssim = DSSIM(tgt_patch, src_patch)
        return (1-alpha) * L1 + alpha * dssim
    return loss

def loss_func_torch():
    def loss(tgt_patch, src_patch, alpha=0.85):
        assert tgt_patch.shape == src_patch.shape  
        patch_size, _, _ = tgt_patch.shape
        ofs = (patch_size-1) // 2
        L1 = torch.mean(torch.abs(tgt_patch[ofs, ofs] - src_patch[ofs, ofs]))
        dssim = ssim_patch(tgt_patch, src_patch)
        return (1-alpha) * L1 + alpha * dssim
    return loss

def experiment_loss_torch(tgt_patch, src_patch, alpha=0.85):
    assert tgt_patch.shape == src_patch.shape  
    patch_size, _, _ = tgt_patch.shape
    ofs = (patch_size-1) // 2
    L1 = torch.mean(torch.abs(tgt_patch[ofs, ofs] - src_patch[ofs, ofs]))
    dssim = ssim_patch(tgt_patch, src_patch)
    return (1-alpha) * L1 + alpha * dssim

def compute_homography_torch(loc, K, pose, depth, normal):
    x, y = loc
    z, n = depth, normal
    K = K[:3, :3]
    inv_K = torch.inverse(K)
    
    R, t = pose[:3, :3], pose[:3, 3].reshape((3, 1))
    loc = to_homog(loc)
    pts_3d = torch.matmul(inv_K, loc) * z
    
    d = -1 * torch.matmul(n.transpose(0, 1), pts_3d)
    H = R - torch.ger(t ,(n.flatten() / d))
    H = torch.matmul(K, H)
    H = torch.matmul(H, inv_K)
    return H
    
def patch_center_loss(tgt_img, src_img, K, cam_coords,
                      pred_pose, pred_depth, pred_norm,
                      patch_size=7, dilation=1):
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
    mid_idx = (patch_size-1) // 2
    src_center_intensities = src_intensities[:, :, :, :, mid_idx, mid_idx] # [B, 3, H-2*offset, W-2*offset]
    pixel_abs = torch.abs(tgt_img[:, :, ofs:-ofs, ofs:-ofs] - src_center_intensities)
    pixel_abs = pixel_abs.mean(1, True)

    # pixel-wise L1 + patch-wise DSSIM
    patch_diff = pixel_abs * 0.15 + ssim_patch(tgt_intensities, src_intensities) * 0.85

    return patch_diff, src_coords