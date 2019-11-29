# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from networks.depth_decoder import DepthDecoder
from layers import *


class RGBDecoder(DepthDecoder):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=3, use_skips=True):
        super(RGBDecoder, self).__init__(num_ch_enc, scales, num_output_channels, use_skips)

    def forward(self, input_features, feat_mask):
        self.outputs = {}
        self.decoder_features = [0] * len(self.scales)

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)

            # mask regions with binary mask
            if i in self.scales:
                x = x * feat_mask[i] 
            
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.decoder_features[i] = x 
                self.outputs[("rgb_recon", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return self.outputs, self.decoder_features
