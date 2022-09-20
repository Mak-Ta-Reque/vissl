# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List
from torchvision import transforms
import torch
from vissl.data.collators import register_collator
from PIL import Image

def load_mask(file_list, size: tuple):
    convert_tensor = transforms.ToTensor()
    masks = [Image.open(file).resize(size) for file in file_list]
    masks = [convert_tensor(msk) for msk in masks]
    return masks


@register_collator("mask_collator")
def mask_collator(batch: List[Dict[str, Any]], mask_size: tuple, transform: Any, ) -> Dict[str, List[torch.Tensor]]:
    mask_size = eval(mask_size)
    """
    This collator is specific to load mask image from file

    """
    assert "data" in batch[0], "data not found in sample"
    assert "label" in batch[0], "label not found in sample"

    #data = [torch.stack(x["data"]) for x in batch]

    # laad the image and normalize to mask

    for each in batch:
        each["label"] = load_mask(each["label"], mask_size)
    data = torch.stack([x["data"][0] for x in batch])
    data_valid = torch.stack([torch.tensor(x["data_valid"][0]) for x in batch])
    data_idx = torch.stack([torch.tensor(x["data_idx"][0]) for x in batch])
    labels = torch.stack([x["label"][0] for x in batch])
    output_batch = {
        "data": [data],
        "label": [labels],
        "data_valid": [data_valid],
        "data_idx": [data_idx],
    }
    return output_batch
