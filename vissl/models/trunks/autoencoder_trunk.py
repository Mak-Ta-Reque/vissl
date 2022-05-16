# Copyright MD ABDUL KADIR and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
from vissl.config import AttrDict
from vissl.models.model_helpers import Flatten, get_trunk_forward_outputs_module_list
from vissl.models.trunks import register_model_trunk
from typing import List

@register_model_trunk("autoencoder_trunk")
class Encoder(nn.Module):
    def __init__(self, model_config: AttrDict, model_name: str):
        super().__init__()
        self.model_config = model_config
        # get the params trunk takes from the config
        trunk_config = self.model_config.TRUNK.ENCODER
        num_input_channels = trunk_config.num_input_channels
        c_hid = trunk_config.c_hid
        latent_dim = trunk_config.latent_dim


        # implement the model trunk and construct all the layers that the trunk uses
        conv1_relu = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, stride=2, padding=1), # 224 x224 => 112 x112
            nn.ReLU(inplace=True)
        )
        conv2_relu = nn.Sequential(
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1), #
            nn.ReLU(inplace=True)
        )

        conv3_relu = nn.Sequential(
            nn.Conv2d(c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 112 x 112 => 56 x 56
            nn.ReLU(inplace=True), # 16x16 => 8x8
        ) 

        conv4_relu = nn.Sequential(
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), # 16x16 => 8x8
        ) 
        conv5_relu = nn.Sequential(
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 56 x 56 => 28 x 28
            nn.ReLU(inplace=True), # 16x16 => 8x8
        ) 
        conv6_relu = nn.Sequential(
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 28 x 28 => 14 x 14
            nn.ReLU(inplace=True), # 16x16 => 8x8
        )
        conv7_relu = nn.Sequential(
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 14 x 14 => 7 x 7
            nn.ReLU(inplace=True), # 16x16 => 8x8
        )

        flatten = nn.Sequential(
            nn.Flatten(), # Image grid to single feature vector
        ) 
        encoding = nn. Sequential(
                nn.Linear(2*7*7*c_hid, latent_dim)
        )

        self._feature_blocks = nn.ModuleList(
            [
                conv1_relu,
                conv2_relu,
                conv3_relu,
                conv4_relu,
                conv5_relu,
                conv6_relu,
                conv7_relu,
                flatten,
                encoding
            ]
        )
        self.all_feat_names = [
            "conv1",
            "conv2",
            "conv3",
            "conv4",
            "conv5",
            "conv6",
            "conv7",
            "flatten",
            "encoding"

        ]
        assert len(self.all_feat_names) == len(self._feature_blocks)


        # give a name to the layers of your trunk so that these features
        # can be used for other purposes: like feature extraction etc.
        # the name is fully upto user descretion. User may chose to
        # only name one layer which is the last layer of the model.
    def forward(self, x: torch.Tensor, out_feat_keys: List[str] = None) -> List[torch.Tensor]:
    # implement the forward pass of the model. See the forward pass of resnext.py
    # for reference.
    # The output would be a list. The list can have one tensor (the trunk output)
    # or mutliple tensors (corresponding to several features of the trunk)
        feat = x
        out_feats = get_trunk_forward_outputs_module_list(
            feat,
            out_feat_keys,
            self._feature_blocks,
            self.all_feat_names,
        )
        return out_feats