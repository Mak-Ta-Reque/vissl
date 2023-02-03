"""
    Author: Md Abdul Kadir, DFKI
"""
from typing import List
import torch
from torch import nn
from classy_vision.losses import ClassyLoss, register_loss
from vissl.config import AttrDict


class DiceLossCriterion(nn.Module):
    def __init__(self, class_codes: List, softmax: bool, weight: List, smooth: float):
        super(DiceLossCriterion, self).__init__()
        self.classes = class_codes
        self.weight = weight
        self.smooth = smooth
        self.softmax = softmax
        if self.weight and len(self.weight) != len(self.classes): raise Exception(f"Dimention miss match between class_codes and weight,\
             e. g: {len(self.weight)} and {len(self.classes)}")

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in self.classes:
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + self.smooth) / (z_sum + y_sum + self.smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target):
        if self.softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if self.weight is None:
            self.weight = [1] * len(self.classes)
        
        assert inputs.size()[2:4] == target.size()[2:4], 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, len(self.classes)):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * self.weight[i]
        return loss / len(self.classes)


@register_loss("dice_loss")
class DiceLoss(ClassyLoss):
    """
    Initializer for the sum of the Dice loss. For multiclass segmantion

    Config params:
        weight: weight of sample, optional
        ignore_index: sample should be ignored for loss, optional
        reduction: specifies reduction to apply to the output, optional
        label_smoothing: specific a label smoothing factor between 0.0 and 1.0 (default is 0.0)
    """

    def __init__(self, loss_config: AttrDict):
        super(DiceLoss, self).__init__()
        self.softmax = loss_config.get("softmax", False)
        self._weight = loss_config.get("weight", None)
        self.class_codes = loss_config.get("class_codes") # how mask is encoded in a single image
        self.smooth = loss_config.get("smooth", 1e-5)
        self.criterion = DiceLossCriterion(
            class_codes=self.class_codes, 
            softmax = self.softmax,
            weight=self._weight,
            smooth=self.smooth
           
        )

    @classmethod
    def from_config(cls, loss_config: AttrDict):
        """
        Instantiates Dice Loss from configuration.

        Args:
            loss_config: configuration for the loss

        Returns:
           DiceLoss instance.
        """
        return cls(loss_config)

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        """
        For each btch output and single encoded target, loss is calculated.
        The returned loss value is the sum loss across all outputs.
        """
        return self.criterion(output, target)