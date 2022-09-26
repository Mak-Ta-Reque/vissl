# -*- coding: utf-8 -*-
# @Time    : 2020-02-26 17:58
# @Author  : Zonas
# @Email   : zonas.wang@gmail.com
# @File    : unet_base.py
"""

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from vissl.models.heads import register_model_head
from vissl.config import AttrDict
from typing import List


@register_model_head("unet_head")
class UNet(nn.Module):
    def __init__(self, model_config: AttrDict, n_classes:int ):
        super().__init__()

    def forward(self, batch: torch.Tensor or List[torch.Tensor]):
        return batch

