import numpy as np
import torch 
import torch.nn.functional as F
from normal_loss import *
import pdb

import pdb

def barron_loss(x, alpha=1, c=0.01):
    t1 = np.abs(alpha-2) / alpha # term 1
    base = ((x/c)**2) / np.abs(alpha-2) + 1
    loss = t1 * (np.power(base, alpha/2) - 1)
    return loss

def ssim_patch_sampled_pts(x, y):
    """Computes the patch SSIM loss
        Input:
        x: [B, 3, N, patch_size, patch_size]
        y: [B, 3, N, patch_size, patch_size]
        Output:
        SSIM: SSIM of all patches, [B, 1, N]
    """
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    B, _, N, size, _ = x.shape
    x = x.contiguous().view(B, 3, N, size * size)
    y = y.contiguous().view(B, 3, N, size * size)

    mu_x = torch.mean(x, dim=3)
    mu_y = torch.mean(y, dim=3)

    sig_x = torch.mean(x ** 2, dim=3) - mu_x ** 2
    sig_y = torch.mean(y ** 2, dim=3) - mu_y ** 2
    sig_xy = torch.mean(x * y, dim=3) - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sig_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sig_x + sig_y + C2) + 1e-10
    SSIM = torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

    SSIM = torch.mean(SSIM, dim=1, keepdim=True)
    return SSIM

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
        L1 = np.mean(np.abs(tgt_patch - src_patch))
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

def form_meshgrid(samples, patch_size, dilation=1):
    """
    Args:
        samples: [n_samples, 2], each entry is (row, col) = [x, y]
        patch_size: patch size, integer
    Returns:
        mesh: a meshgrid of shape [1, n_samples, patch_size*patch_size, 2] 
                where each element is centered at (x, y)
    """
    N, _ = samples.shape
    # tile center coordinates
    samples = samples.unsqueeze(0).unsqueeze(2) # [1, n_samples, 1, 2]
    samples = samples.repeat([1, 1, patch_size*patch_size, 1]) # [1, n_samples, patch_size*patch_size, 2] 
    # create meshgrid
    effective_patch_size = 1 + (patch_size - 1) * dilation
    ofs = (effective_patch_size - 1) // 2
    x = torch.arange(start=-ofs, end=ofs+1, step=dilation).cuda()
    grid_y, grid_x = torch.meshgrid(x, x) # both [patch_size, patch_size]
    grid_x = grid_x.contiguous().view(-1) # [patch_size*patch_size, ]
    grid_y = grid_y.contiguous().view(-1) # [patch_size*patch_size, ]
    mesh = torch.stack([grid_x, grid_y], dim=1) # [patch_size*patch_size, 2]
    mesh = mesh.unsqueeze(0).unsqueeze(0) # [1, 1, patch_size*patch_size, 2]
    # use broadcast to generate [1, n_samples, patch_size*patch_size, 2]
    mesh = mesh.cuda() + samples 

    return mesh

def sampled_intensity(samples, img, patch_size):
    """Use to sample intensities at target and source images at corresponding patch
        locations in the square patch experiment
    Args:
        samples: [n_samples, 2], each entry is (row, col) = [x, y]
        img: image of shape [1, 3, H, W] 
        patch_size: patch size, integer
    Returns:
        sampled intensities for each patch [1, 3, n_samples, patch_size, patch_size]
    """
    _, _, height, width = img.shape
    n_samples = samples.shape[0]
    mesh = form_meshgrid(samples, patch_size, dilation=1).to(torch.float) # [1, n_samples, patch_size*patch_size, 2] 
    
    mesh[..., 0] /= (width - 1)
    mesh[..., 1] /= (height - 1)
    mesh = (mesh - 0.5) * 2

    intensities = F.grid_sample(img, mesh, mode='nearest', padding_mode='border') # [1, 3, n_samples, patch_size*patch_size]
    intensities = intensities.view(1, 3, n_samples, patch_size, patch_size)
    return intensities

def compute_sampled_homography(samples, K, cam_coords,
                               pred_pose, pred_norm,
                               patch_size=7, dilation=1):
    """Compute homography for sampled locations
    Input:
        samples: [n_samples, 2], each entry is (row, col) = [y, x]
        K: intrinsics, [B, 4, 4]
        cam_coords: backprojected 3D points (K_inv @ x * d), [B, 4, H * W]
        pred_pose: predicted poses, [B, 4, 4]
        pred_norm: predicted surface normal, [B, 3, H, W]
        patch_size: patch size, integer (default = 7)
        dilation: dilation factor for patch (default = 1)

    """
    batch, _, height, width = pred_norm.shape
    n_samples = samples.shape[0]
    cam_coords = cam_coords[:, :-1, :].view(batch, 3, height, width) # [B, 3, H, W]
    K = K[:, :3, :3] 
    psize_eff = 1 + dilation * (patch_size - 1) # effective patch size
    ofs = (psize_eff - 1) // 2       
    
    # compute homography for every pixel location
    H_all = calculate_homography(pred_pose, K, pred_norm, cam_coords) #[B, 3, 3, H, W]
    
    # gather H with python-style indexing
    H = H_all[:, :, :, samples[:, 0], samples[:, 1]]   # [B, 3, 3, n_samples]
    H = H.permute(0, 3, 1, 2) # [B, n_samples, 3, 3]
    H = H.unsqueeze(2) # [B, n_samples, 1, 3, 3]
    return H

def sampled_patch_center_loss(samples, tgt_img, src_img,
                              K,
                              cam_coords,                      
                              pred_pose, pred_depth, pred_norm,
                              patch_size=7, dilation=1):
    """Computes the patch intensity difference between tgt and src
    Input:
        samples: [n_samples, 2], each entry is (row, col) = [y, x]
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
        patch_intensities: sampled intensities for each patch [1, 3, n_samples, patch_size, patch_size]
        patch_coords: sampled coordinates [B, n_samples, size*patch_size, 2]
    """
    batch, _, height, width = pred_norm.shape
    n_samples = samples.shape[0]
    cam_coords = cam_coords[:, :-1, :].view(batch, 3, height, width) # [B, 3, H, W]
    K = K[:, :3, :3] 
    psize_eff = 1 + dilation * (patch_size - 1) # effective patch size
    ofs = (psize_eff - 1) // 2       
    
    # compute homography for every pixel location
    H_all = calculate_homography(pred_pose, K, pred_norm, cam_coords) #[B, 3, 3, H, W]
    
    # gather H with python-style indexing
    H = H_all[:, :, :, samples[:, 0], samples[:, 1]]   # [B, 3, 3, n_samples]
    H = H.permute(0, 3, 1, 2) # [B, n_samples, 3, 3]
    H = H.unsqueeze(2) # [B, n_samples, 1, 3, 3]
    
    # cam_coords = (K_inv @ pts * d) / d 
    cam_coords = cam_coords / pred_depth # [B, 3, H, W]

    # unfold cam_coords at sampled locations
    patch_coords = []
    for y, x in samples:
        if ofs>=0:
            coords = cam_coords[:, :, y-ofs:y+ofs+1, x-ofs:x+ofs+1] # [B, 3, patch_size, patch_size]
            patch_coords.append(coords)
    patch_coords = torch.stack(patch_coords, dim=-1) # [B, 3, patch_size, patch_size, n_samples]
    patch_coords = patch_coords.permute(0, 4, 2, 3, 1) # [B, n_samples, patch_size, patch_size, 3]
    patch_coords = patch_coords.view(batch, n_samples, -1, 3) # [B, n_samples, patch_size*patch_size, 3]
    patch_coords = patch_coords.unsqueeze(-1) # [B, n_samples, patch_size*patch_size, 3, 1]
        

    # matmul with the last two dimensions aligned
    warped_coords = torch.matmul(H, patch_coords) # [B, n_samples, size*patch_size, 3, 1]
    warped_coords = warped_coords.squeeze(-1) # [B, n_samples, size*patch_size, 3]


    # dehomogenize
    warped_coords = warped_coords[:, :, :, :2] / (warped_coords[:, :, :, 2:] + 1e-10) # [B, n_samples, size*patch_size, 2]
    warped_coords[..., 0] /= width - 1
    warped_coords[..., 1] /= height - 1

    # grid sample
    patch_intensities = F.grid_sample(src_img, warped_coords, padding_mode='border') # [B, 3, n_samples, patch_size*patch_size]
    patch_intensities = patch_intensities.view(batch, 3, n_samples, patch_size, patch_size)

    return patch_intensities, patch_coords




    
