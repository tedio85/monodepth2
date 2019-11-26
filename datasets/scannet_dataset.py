from __future__ import absolute_import, division, print_function

import os
import random
import skimage.transform
import numpy as np
import PIL.Image as pil

import torch
import torch.utils.data as data
from torchvision import transforms

from .mono_dataset import MonoDataset


class ScanNetDataset(MonoDataset):
    """Superclass for different types of ScanNet dataset loaders
    """
    def __init__(self, *args, gt_path=None, **kwargs):
        self.gt_path = gt_path
        super(ScanNetDataset, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index'.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        do_color_aug = self.data_aug and self.is_train and random.random() > 0.5
        do_flip = self.data_aug and self.is_train and random.random() > 0.5

        line = self.filenames[index].split()
        folder, frame_index = line[0], int(line[1])

        # load sequence
        inputs = self.get_sequence(folder, frame_index, do_flip)
        ori_w, ori_h = inputs[("color", 0, -1)].size # pil function

        # load 4x4 intrinsics
        intrinsics = self.get_scaled_intrinsics(folder, frame_index, ori_h, ori_w)
        
        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = intrinsics.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)
            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        # generate multi-scale images with data augmentation
        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        # load ground truth depth map
        # self.load_depth = self.check_depth() defined in class MonoDataset
        if self.load_depth: 
            depth_gt = self.get_depth(self.gt_path, folder, frame_index, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))
            return inputs
            
        return inputs

    def get_sequence(self, frame_index, do_flip):
        """ Get a sequence of images in the form of a dictionary
            inputs[("color", frame_offset, -1)]
            where frame_offset = 0 is target image,
                 +1 for the next frame
                 -1 for the previous frame
        """
        raise NotImplementedError

    def get_scaled_intrinsics(folder, frame_index, ori_h, ori_w):
        """Get camera intrinsics matrix with 1st and 2nd row divided by  
            the image width and height, respectively
        """
        raise NotImplementedError
    
    def get_color(self, folder, frame_index, side, do_flip):
        assert side is None
        color = self.loader(self.get_image_path(folder, frame_index))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def check_depth(self):
        """Determine whether depth maps are loaded
             Returns true if ground truth path is provided and depth map is present in file
             Returns false otherwise
        """
        if self.gt_path is None:
            return False
        else:
            line = self.filenames[0].split()
            folder, frame_idx = line[0], line[1]
            f_str = "{}.png".format(frame_idx)
            depth_filename = os.path.join(self.gt_path, folder, 'depth', f_str)
            return os.path.isfile(depth_filename)

    def get_depth(self, gt_path, folder, frame_index, do_flip):
        f_str = "{}.png".format(frame_index)
        depth_filename = os.path.join(gt_path, folder, 'depth', f_str)

        with open(depth_filename, 'rb') as f:
            depth_pil = pil.open(f)
            depth_gt = np.asarray(depth_pil) / 1000.0 # millimeters -> meters
        
        # resize to match shape of model prediction 
        model_output_size = (self.width, self.height)
        depth_gt = skimage.transform.resize(
            depth_gt, model_output_size[::-1], order=0, preserve_range=True, mode='constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt

class ScanNetRAWDataset(ScanNetDataset):
    """ScanNet dataset which loads the raw images and depth maps."""
    def __init__(self, *args, gt_path=None, **kwargs):
        super(ScanNetRAWDataset, self).__init__(*args, gt_path=gt_path, **kwargs)
        
    def get_image_path(self, folder, frame_index):
        f_str = "{}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(self.data_path, folder, "color", f_str)
        return image_path

    def get_sequence(self, folder, frame_index, do_flip):
        inputs = {}
        for i in self.frame_idxs:
            inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side=None, do_flip=do_flip)
        return inputs

    def get_scaled_intrinsics(self, folder, frame_index, ori_h, ori_w):
        # NOTE: frame_index is not used in this function, a scene shares a common intrinsics matrix
        intr_path = os.path.join(self.data_path, folder, 'intrinsic', 'intrinsic_color.txt')
        intrinsics = np.loadtxt(intr_path) # 4x4 matrix

        # divide fx, cx by image width
        # and fy, cy by image height
        intrinsics[0, :] /= ori_w
        intrinsics[1, :] /= ori_h
        return intrinsics

    
class ScanNetProcDataset(ScanNetDataset):
    """ScanNet dataset which loads the pre-processed sequences consisting of 3 images concatenated together.
        If the argument `gt_path` is given, include the ground truth depth map.
    """
    def __init__(self, *args, gt_path=None, **kwargs):
        super(ScanNetProcDataset, self).__init__(*args, gt_path=gt_path, **kwargs)
        
    def get_image_path(self, folder, frame_index):
        f_str = "{}{}".format(frame_index, self.img_ext)
        path = os.path.join(self.data_path, folder, f_str)
        return path

    def get_sequence(self, folder, frame_index, do_flip):
        inputs = {}

        # 3 images concatenated together
        seq = self.get_color(folder, frame_index, side=None, do_flip=False)
        seq_w, seq_h = seq.size
        seq_len = len(self.frame_idxs)

        im_w = seq_w//seq_len
        im_h = seq_h
        left = 0
        for i in sorted(self.frame_idxs): # for 3 frames, [-1, 0, 1]
            box = (left, 0, left+im_w, im_h) # [left, upper, right, lower]
            img = seq.crop(box)
            if do_flip:
                img = img.transpose(pil.FLIP_LEFT_RIGHT)
            inputs[("color", i, -1)] = img
            left += im_w
        
        return inputs

    def get_scaled_intrinsics(self, folder, frame_index, ori_h, ori_w):
        f_str = "{}_cam.txt".format(frame_index)
        path = os.path.join(self.data_path, folder, f_str)
        with open(path, 'r') as f:
            num_lst = f.read().split(',')

        intrinsics = np.array([float(num) for num in num_lst], dtype=np.float32)
        intrinsics = intrinsics.reshape(3, 3)

        # divide fx, cx by image width
        # and fy, cy by image height
        intrinsics[0, :] /= ori_w
        intrinsics[1, :] /= ori_h
        
        # transform to a 4x4 matrix
        ret = np.zeros([4, 4], dtype=np.float32)
        ret[:3, :3] = intrinsics
        ret[ 3,  3] = 1
        return ret
            