"""
    Author: Md Abdul Kadir, DFKI
"""
from typing import List
import torch
from torch import nn
from torch.nn.modules.loss import CrossEntropyLoss
from classy_vision.losses import ClassyLoss, register_loss
from vissl.config import AttrDict
from .dice_loss import DiceLossCriterion


@register_loss("dice_ce_loss")
class DiceCELoss(ClassyLoss):
    """
    Initializer for the sum of the Dice loss and Cross entropy loss. For multiclass segmantion

    Config params:
        softmax: Apply softmax on the output before dice operation.\
            For CELoss it is not necessary becase CELoss always use softmax operation.
        lambda: weight of to the loss. lambda * loss_ce + (1- lambda) * loss_dice

        ignore_index: sample should be ignored for loss, optional
        reduction: specifies reduction to apply to the output, optional
        label_smoothing: specific a label smoothing factor between 0.0 and 1.0 (default is 0.0)
    """

    def __init__(self, loss_config: AttrDict):
        super(DiceCELoss, self).__init__()
        self.softmax = loss_config.get("softmax", True)
        self.lamda = loss_config.get("lambda", 0.0) 
        self._weight = loss_config.get("weight", None)
        self.class_codes = loss_config.get("class_codes") # how mask is encoded in a single image
        self.smooth = loss_config.get("smooth", 1e-5)
        self.criterion_dice = DiceLossCriterion(
            class_codes=self.class_codes, 
            softmax = self.softmax,
            weight=self._weight,
            smooth=self.smooth       
        )
        self.criterion_ce = CrossEntropyLoss()

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
        return self.lamda * self.criterion_ce(output, target[:].long()) + (1.0 - self.lamda) * self.criterion_dice(output, target)