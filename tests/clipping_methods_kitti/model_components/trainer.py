# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

from utils import *
from kitti_utils import *
from layers import *
from normal_loss import *

import datasets
import networks



class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])
        self.use_normal_net = (self.opt.orth_weight > 0 or self.opt.patch_weight > 0)

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")
        
        #### DEPTH network
        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["encoder"].to(self.device)
        freeze_weights(self.models["encoder"], self.opt.models_to_freeze, "encoder")
        self.parameters_to_train += [{'params': self.models["encoder"].parameters(),
                                      'lr': self.opt.learning_rate[0]}]

        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"].to(self.device)
        freeze_weights(self.models["depth"], self.opt.models_to_freeze, "depth")
        self.parameters_to_train += [{'params': self.models["depth"].parameters(),
                                       'lr': self.opt.learning_rate[0]}]
        
        #### POSE network
        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                self.models["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)

                self.models["pose_encoder"].to(self.device)
                freeze_weights(self.models["pose_encoder"], self.opt.models_to_freeze, "pose_encoder")
                self.parameters_to_train += [{'params': self.models["pose_encoder"].parameters(),
                                              'lr': self.opt.learning_rate[1]}]

                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

            elif self.opt.pose_model_type == "shared":
                self.models["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames)

            elif self.opt.pose_model_type == "posecnn":
                self.models["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)

            self.models["pose"].to(self.device)
            freeze_weights(self.models["pose"], self.opt.models_to_freeze, "pose")
            self.parameters_to_train += [{'params': self.models["pose"].parameters(),
                                          'lr': self.opt.learning_rate[1]}]
        
        self.parameters_to_train_no_pretrain = []
        #### NORMAL network
        if self.use_normal_net:
            if self.opt.normal_model_type == "separate_resnet":
                self.models["normal_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained")

                self.models["normal_encoder"].to(self.device)
                freeze_weights(self.models["normal_encoder"], self.opt.models_to_freeze, "normal_encoder")
                if self.opt.load_weights_folder != None and "normal_encoder" not in self.opt.models_to_load:
                    self.parameters_to_train_no_pretrain += [{'params': self.models["normal_encoder"].parameters(),
                                                              'lr':self.opt.learning_rate[2]}]
                else:
                    self.parameters_to_train += [{'params': self.models["normal_encoder"].parameters(),
                                                  'lr':self.opt.learning_rate[2]}]

                self.models["normal"] = networks.NormalDecoder(
                    self.models["normal_encoder"].num_ch_enc, self.opt.scales)

            elif self.opt.normal_model_type == "shared":
                self.models["normal"] = networks.NormalDecoder(
                    self.models["encoder"].num_ch_enc, self.opt.scales)

            self.models["normal"].to(self.device)
            freeze_weights(self.models["normal"], self.opt.models_to_freeze, "normal")
            if self.opt.load_weights_folder != None and 'normal' not in self.opt.models_to_load:
                self.parameters_to_train_no_pretrain += [{'params': self.models["normal"].parameters(),
                                                          'lr':self.opt.learning_rate[2]}]
            else:
                self.parameters_to_train += [{'params': self.models["normal"].parameters(),
                                              'lr':self.opt.learning_rate[2]}]
        #### Predictive mask
        if self.opt.predictive_mask:
            # Our implementation of the predictive masking baseline has the the same architecture
            # as our depth decoder. We predict a separate mask for each source frame.
            self.models["predictive_mask"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1))
            self.models["predictive_mask"].to(self.device)
            self.parameters_to_train += list(self.models["predictive_mask"].parameters())

        #### Optimizer
        self.model_optimizer = optim.Adam(self.parameters_to_train)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()
        
        for param_dict in self.parameters_to_train_no_pretrain:
            self.model_optimizer.add_param_group(param_dict)

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        #### Data Loader
        datasets_dict = {"scannet_proc": datasets.ScanNetProcDataset,
                         "kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        if self.opt.dataset.startswith("scannet"):
            fpath = os.path.join(self.opt.data_path, '{}.txt')
        else:
            fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext, data_aug=not self.opt.disable_data_aug)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)

        if self.opt.dataset.startswith("scannet"):
            val_dataset = datasets.ScanNetProcDataset(
                self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
                self.opt.frame_ids, 4, is_train=False, # set is_train to false 
                gt_path=self.opt.dmap_path,            # provide gt_path
                img_ext=img_ext)
        else:
            val_dataset = self.dataset(
                self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
                self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        #### Logging
        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        #### Inverse warp calculations
        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        #### Validation
        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

        #### Logging
        w, h = self.opt.width, self.opt.height
        self.patch_centers = np.array([[w//2, 0.2*w, 0.8*w],
                                      [h//2, 0.8*h, 0.2*h]])

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        self.model_lr_scheduler.step()

        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()
            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        if self.opt.pose_model_type == "shared":
            # If we are using a shared encoder for both depth and pose (as advocated
            # in monodepthv1), then all images are fed separately through the depth encoder.
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]

            outputs = self.models["depth"](features[0])
        else:
            # Otherwise, we only feed the image with frame_id 0 through the depth encoder
            features = self.models["encoder"](inputs["color_aug", 0, 0])
            outputs = self.models["depth"](features)

        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features))
        
        if self.use_normal_net:
            if self.opt.normal_model_type == "shared":
                # Feed depth encoder features into normal decoder
                outputs.update(self.models["normal"](features))
            else:
                # Feed the image with frame_id 0 / finest scale through the normal encoder
                features = self.models["normal_encoder"](inputs["color_aug", 0, 0])
                outputs.update(self.models["normal"](features))

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.use_normal_net:
                normal = outputs[("norm", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                if self.use_normal_net:
                    normal = F.interpolate(
                        normal, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)

                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth
            cam_points = self.backproject_depth[source_scale](
                depth, inputs[("inv_K", source_scale)])
            
            ## NORMAL REGULARIZAR calculations
            if self.use_normal_net:
                outputs[("norm_reg", scale)] = compute_normal_regularizar(normal, cam_points)
            ## ORTH loss calculations
            if self.opt.orth_weight > 0 and scale in self.opt.orth_loss_layers:
                outputs[("cos_dis", scale)] = compute_cos_dist(normal, cam_points)

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":

                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T) # [B, H, W, 2]

                if self.opt.patch_weight > 0 and scale == 0 and frame_id == -1:
                    # only record pixel coordinates when plotting patch reprojection
                    outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    pix_coords,
                    padding_mode="border")
                                
                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]
                ## PATCH difference 
                if self.opt.patch_weight > 0 and scale in self.opt.patch_loss_layers:
                    outputs[("patch_id_diff", frame_id, scale)] = compute_identity_patch_difference(
                                                                    inputs[("color", 0, source_scale)], #target
                                                                    inputs[("color", frame_id, source_scale)], #src
                                                                    patch_size=self.opt.patch_size,
                                                                    dilation=self.opt.dilation_factor,
                                                                    type=self.opt.patch_mask_type)

                    patch_diff, src_coords = compute_patch_difference(
                                                inputs[("color", 0, source_scale)], #target
                                                inputs[("color", frame_id, source_scale)], #src
                                                inputs[("K", source_scale)], #intrinsic
                                                cam_points,
                                                T,
                                                depth,
                                                normal,
                                                patch_size=self.opt.patch_size,
                                                dilation=self.opt.dilation_factor,
                                                type=self.opt.patch_mask_type)
                    outputs[("patch_warp_diff", frame_id, scale)] = patch_diff
                    if scale == 0 and frame_id == -1:
                        outputs[("src_patch_coords", frame_id, scale)] = src_coords

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)
        
        if self.opt.no_ssim:
            reprojection_loss = l1_loss
            ssim_loss = torch.zeros_like(l1_loss).to(self.device)
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss, ssim_loss, l1_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reproj_loss, ssim_loss, l1_loss = self.compute_reprojection_loss(pred, target)
                reprojection_losses.append(reproj_loss)
                
                # log losses
                losses[("reproj_error", frame_id, scale)] = reproj_loss
                losses[("ssim", frame_id, scale)] = ssim_loss
                losses[("l1", frame_id, scale)] = l1_loss    

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    reproj_loss, _, _ = self.compute_reprojection_loss(pred, target)
                    identity_reprojection_losses.append(reproj_loss)

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.opt.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape).cuda() * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (idxs > 1).float()

            loss += self.opt.pixel_weight * to_optimise.mean() 

            ### SMOOTHNESS loss (depth)
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)
            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)

            ### ORTH loss
            if self.opt.orth_weight > 0 and scale in self.opt.orth_loss_layers:
                orth_loss = outputs[('cos_dis', scale)]
                # not self.opt.disable_automasking:
                #    orth_loss = orth_loss * outputs["identity_selection/0"]
                orth_loss = orth_loss.mean()
                loss += self.opt.orth_weight * orth_loss
                losses['orth_loss/s{}'.format(scale)] = orth_loss
            
            ### NORMAL REG loss & SMOOTHNESS loss (normal)
            if self.use_normal_net:
                norm_reg_loss = outputs[("norm_reg", scale)].mean()
                loss += self.opt.normal_regularizar * norm_reg_loss
                losses["norm_reg_loss/s{}".format(scale)] = norm_reg_loss

                norm_smooth_loss = get_smooth_loss(outputs[("norm", scale)], color)
                loss += self.opt.normal_smoothness * norm_smooth_loss
                losses["norm_smooth_loss/s{}".format(scale)] = norm_smooth_loss
            
            ### PATCH loss
            if self.opt.patch_weight > 0 and scale in self.opt.patch_loss_layers:
                patch_diff_stack = []
                for frame_id in self.opt.frame_ids[1:]:
                    diff = outputs[("patch_warp_diff", frame_id, scale)]
                    patch_diff_stack.append(diff)
                patch_diff_stack = torch.cat(patch_diff_stack, dim=1)
                
                if self.opt.disable_automasking:
                    ## directily compute the mean of the patch difference
                    patch_loss = patch_diff_stack.mean(dim=1)
                    outputs[("patch_diff", scale)] = patch_loss
                    patch_loss = patch_loss.mean()
                else:
                    patch_id_diff_stack = []
                    for frame_id in self.opt.frame_ids[1:]:
                        id_diff = outputs[("patch_id_diff", frame_id, scale)]
                        patch_id_diff_stack.append(id_diff)
                    patch_id_diff_stack = torch.cat(patch_id_diff_stack, dim=1)

                    # add random numbers to break ties
                    patch_id_diff_stack += torch.randn(patch_id_diff_stack.shape).cuda() * 0.00001
                    patch_combined = torch.cat((patch_id_diff_stack, patch_diff_stack), dim=1)
                    if combined.shape[1] == 1:
                        patch_diff = patch_combined
                    else:
                        patch_diff, idxs = torch.min(patch_combined, dim=1, keepdim=True)
                    outputs["patch_identity_selection/{}".format(scale)] = (idxs > 1).float()
                    outputs[("patch_diff", scale)] = patch_diff
                    patch_loss = patch_diff.mean()

                loss += self.opt.patch_weight * patch_loss
                losses["patch_loss/s{}".format(scale)] = patch_loss

            total_loss += loss
            losses["loss/scale{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(depth_pred, self.opt.min_depth, self.opt.max_depth)
        if self.opt.dataset != "scannet_proc":
            depth_pred = torch.clamp(F.interpolate(
                depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # crop n pixels near borders
        n = 10
        crop_mask = torch.ones_like(mask)
        crop_mask[:, :,  :n,    :] = 0
        crop_mask[:, :, -n:,    :] = 0
        crop_mask[:, :,    :,  :n] = 0
        crop_mask[:, :,    :, -n:] = 0
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=self.opt.min_depth, max=self.opt.max_depth)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]

        # scalars
        disp_median = torch.median(outputs[("disp", 0)]) # median of all disp in a batch
        for k, v in self.get_lr().items():
            writer.add_scalar("learning_rate/{}".format(k), v, self.step)
        writer.add_scalar("disp_median", disp_median, self.step)
        for l, v in losses.items():
            w_str = "{}_{}_s{}".format(*l) if isinstance(l, tuple) else "{}".format(l)
            val = v.mean() if len(v.shape) > 1 else v
            writer.add_scalar(w_str, val, self.step)

        # patch reprojection
        if self.opt.patch_weight > 0:
            writer = make_plot(inputs[("color", -1, 0)], # previous frame
                               inputs[("color",  0, 0)], 
                               outputs[("src_patch_coords", -1, 0)],
                               outputs[("sample", -1, 0)], # coordinates in previous frame
                               self.patch_centers,
                               writer,
                               self.opt.patch_size,
                               self.opt.dilation_factor,
                               self.step) 

        # images
        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_s{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        # warped image
                        writer.add_image(
                            "color_pred_{}_s{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)
                        # loss maps
                        writer.add_image(
                            "reproj_error_{}_s{}/{}".format(frame_id, s, j),
                            normalize_loss_map(losses[("reproj_error", frame_id, s)][j]),
                            self.step)
                        writer.add_image(
                            "ssim_{}_s{}/{}".format(frame_id, s, j),
                            normalize_loss_map(losses[("ssim", frame_id, s)][j]),
                            self.step)
                        writer.add_image(
                            "l1_{}_s{}/{}".format(frame_id, s, j),
                            normalize_loss_map(losses[("l1", frame_id, s)][j]),
                            self.step)
                        # pose   
                        trans = outputs[("translation", 0, frame_id)] # [B, 2, 1, 3]
                        tx = trans[:, :, 0, 0]
                        ty = trans[:, :, 0, 1]
                        tz = trans[:, :, 0, 2]
                        writer.add_histogram("tx", tx, self.step)
                        writer.add_histogram("ty", ty, self.step)
                        writer.add_histogram("tz", tz, self.step)
                        writer.add_histogram("angle", get_angle(outputs[("axisangle", 0, frame_id)]), self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)
                if self.use_normal_net:
                    writer.add_image(
                        "norm_{}/{}".format(s,j),
                        plot_normal(outputs[("norm", s)][j]).data, self.step)

                if self.opt.patch_weight > 0 and s in self.opt.patch_loss_layers:
                    writer.add_image(
                        "patch_automask_{}/{}".format(s, j),
                        outputs["patch_identity_selection/{}".format(s)][j].data, self.step)
                    writer.add_image(
                        "patch_diff_{}/{}".format(s,j),
                        normalize_image(outputs[("patch_diff", s)][j]).data, self.step)

                writer.add_histogram("disp_{}/{}".format(s, j), outputs[("disp", s)][j].data, self.step)

                if self.opt.predictive_mask:
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        writer.add_image(
                            "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                            outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                            self.step)

                elif not self.opt.disable_automasking:
                    writer.add_image(
                        "automask_{}/{}".format(s, j),
                        outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

    def get_lr(self):
        """Get current learning rate
        """
        lr_dict = {}
        for i, param_group in enumerate(self.model_optimizer.param_groups):
            lr_dict[str(i)] = param_group['lr']
        return lr_dict

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
