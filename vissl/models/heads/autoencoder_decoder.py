import torch
import torch.nn as nn
from vissl.models.heads import register_model_head
from vissl.models.model_helpers import Flatten, get_trunk_forward_outputs_module_list
from vissl.config import AttrDict
from typing import List

@register_model_head("autoencoder_head")
class Decoder(nn.Module):
    """
    Add documentation on what this head does and also link any papers where the head is used
    """

    def __init__(self, model_config: AttrDict, num_input_channels= 3, c_hid= 64, latent_dim= 1792):

        """
        Args:
            add documentation on what are the parameters to the head
        """
        super().__init__()
        #self.model_config = model_config
        # get the params trunk takes from the config
        #head_config = self.model_config.HEAD.DECODER
        #num_input_channels = head_config.num_input_channels
        #c_hid = trunk_config.c_hid
        #latent_dim = trunk_config.latent_dim
        self.decoding = nn.Sequential(
            nn.Linear(latent_dim, 2*7*7*c_hid),
            nn.ReLU(inplace=True),
        )

        convTrans1_layer = nn.Sequential(
            nn.ConvTranspose2d(2*c_hid, 2*c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 7x7 => 14x14
             nn.GELU()      
        )
        convTrans2_layer = nn.Sequential(
            nn.ConvTranspose2d(2*c_hid, 2*c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 14x14 => 28x28
             nn.GELU()      
        )
        convTrans3_layer = nn.Sequential(
            nn.ConvTranspose2d(2*c_hid, 2*c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 28x28 => 56x56
             nn.GELU()      
        )
        convTrans4_layer = nn.Sequential(
            nn.ConvTranspose2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
             nn.GELU()      
        )
        convTrans5_layer = nn.Sequential(
            nn.ConvTranspose2d(2*c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 56x56 => 112x112
             nn.GELU()      
        )
        convTrans6_layer = nn.Sequential(
            nn.ConvTranspose2d(c_hid, c_hid, kernel_size=3, padding=1),
            nn.GELU()      
        )
        convTrans7_layer = nn.Sequential(
            nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2), # 112x112 => 224x224
             nn.GELU()      
        )

        self._decoder_feature_blocks = nn.ModuleList(
            [
                convTrans1_layer,
                convTrans2_layer,
                convTrans3_layer,
                convTrans4_layer,
                convTrans5_layer,
                convTrans6_layer,
                convTrans7_layer,
            ]
        )
        self.all_feat_names = [
            "convTrans1",
            "convTrans2",
            "convTrans3",
            "convTrans4",
            "convTrans5",
            "convTrans6",
            "convTrans7",

        ]
        assert len(self.all_feat_names) == len(self._decoder_feature_blocks)





        
        # implement what the init of head should do. Example, it can construct the layers in the head
        # like FC etc., initialize the parameters or anything else

    # the input to the model should be a torch Tensor or list of torch tensors.
    def forward(self, feat: torch.Tensor or List[torch.Tensor],  out_feat_keys: List[str] = None) :
        """
        add documentation on what the head input structure should be, shapes expected
        and what the output should be
        """
        # implement the forward pass of the head
        feat = self.decoding(feat)
        print(feat.shape[0])
        feat = feat.reshape(feat.shape[0], -1, 7, 7)
        out_feats = get_trunk_forward_outputs_module_list(
            feat,
            out_feat_keys,
            self._decoder_feature_blocks,
            self.all_feat_names,
        )
        return out_feats