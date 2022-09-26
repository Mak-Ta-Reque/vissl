import torch.nn.functional as F
import torch
from torch.autograd import Function
import numpy as np
print("The loss calculation is varified for n_channel = 1 output, for multiple chanel ooutput result is not varified")
class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)


def batch_evaluation(model_output, target, deepsupervision= False, threshold=0.5, n_classes=1):
    """
    Evaluation without the densecrf with the dice coefficient
    """
    masks_preds = model_output
    true_masks = target
    tot = 0
    # compute loss
    if deepsupervision:
       
        loss = 0

        for masks_pred in masks_preds:
            tot_cross_entropy = 0
            for true_mask, pred in zip(true_masks, masks_pred):
                pred = (pred > threshold).float()
                if n_classes > 1:
                    sub_cross_entropy = F.cross_entropy(pred.unsqueeze(dim=0), true_mask.unsqueeze(dim=0).squeeze(1)).item()
                else:
                    sub_cross_entropy = dice_coeff(pred, true_mask.squeeze(dim=1)).item()
                tot_cross_entropy += sub_cross_entropy
            tot_cross_entropy = tot_cross_entropy / len(masks_preds)
            tot += tot_cross_entropy
    else:
        for true_mask, pred in zip(true_masks, masks_pred):
            pred = (pred > threshold).float()
            if n_classes > 1:
                tot += F.cross_entropy(pred.unsqueeze(dim=0), true_mask.unsqueeze(dim=0).squeeze(1)).item()
            else:
                tot += dice_coeff(pred, true_mask.squeeze(dim=1)).item()


    return tot/ model_output.shape[0]

def dice_coefficient(pred, true, smooth=1e-15):
    tot = 0.0
    for p, t in zip(pred, true):
        intersection = 2 * (torch.sum((torch.logical_and(t, p)))).item()
        union = torch.sum(t).item() + torch.sum(p).item()
        tot += (intersection + smooth) / (union + smooth)
    return tot/ pred.shape[0]


# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import torch
from classy_vision.generic.distributed_util import all_reduce_sum, gather_from_all
from classy_vision.meters import ClassyMeter, register_meter
from vissl.config import AttrDict
from vissl.utils.env import get_machine_local_and_dist_rank
from vissl.utils.svm_utils.evaluate import get_precision_recall


@register_meter("dice_score")
class DiceScore(ClassyMeter):
    """
    Meter to calculate mean AP metric for multi-label image classification task.

    Args:
        meters_config (AttrDict): config containing the meter settings

    meters_config should specify the num_classes
    """

    def __init__(self, meters_config: AttrDict):
        self.num_classes = meters_config.get("n_classes")
        self.threshold = meters_config.get("threshold") # Threshold for converting unet output to mask
        self._total_sample_count = None
        self._curr_sample_count = None
        self.deepsupervision = meters_config.get("deepsupervision")
        self.reset()

    @classmethod
    def from_config(cls, meters_config: AttrDict):
        """
        Get the AccuracyListMeter instance from the user defined config
        """
        return cls(meters_config)

    @property
    def name(self):
        """
        Name of the meter
        """
        return "mean_dice_index_meter"

    @property
    def value(self):
        """
        Value of the meter globally synced. mean AP and AP for each class is returned
        """
        _, distributed_rank = get_machine_local_and_dist_rank()
        logging.info(
            f"Rank: {distributed_rank} Mean dice score meter: "
            f"scores: {self._scores.shape}"
        )
        ap_matrix = {k:0 for k in range(self.num_classes)}#torch.ones(self.num_classes, dtype=torch.float32) * -1
        # targets matrix = 0, 1, -1
        # unknown matrix = 0, 1 where 1 means that it's an unknown
        #unknown_matrix = torch.eq(self._targets, -1.0).float().detach().numpy()
        for cls_num in range(self.num_classes):
            avg_score = self._scores[:,cls_num].mean().item()
            assert avg_score <= 1.0, " Calculation error in Dice score, dice score cant be largeer than 1.0" 
            ap_matrix[cls_num] = avg_score
        return {"dice score": ap_matrix }

    def gather_scores(self, scores: torch.Tensor):
        """
        Do a gather over all embeddings, so we can compute the loss.
        Final shape is like: (batch_size * num_gpus) x embedding_dim
        """
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # gather all embeddings.
            scores_gathered = gather_from_all(scores)
        else:
            scores_gathered = scores
        return scores_gathered

    def gather_targets(self, targets: torch.Tensor):
        """
        Do a gather over all embeddings, so we can compute the loss.
        Final shape is like: (batch_size * num_gpus) x embedding_dim
        """
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # gather all embeddings.
            targets_gathered = gather_from_all(targets)
        else:
            targets_gathered = targets
        return targets_gathered

    def sync_state(self):
        """
        Globally syncing the state of each meter across all the trainers.
        We gather scores, targets, total sampled
        """
        # Communications
        self._curr_sample_count = all_reduce_sum(self._curr_sample_count)
        self._scores = self.gather_scores(self._scores)
        self._targets = self.gather_targets(self._targets)

        # Store results
        self._total_sample_count += self._curr_sample_count

        # Reset values until next sync
        self._curr_sample_count.zero_()

    def reset(self):
        """
        Reset the meter
        """
        self._scores = torch.zeros(0, self.num_classes, dtype=torch.float32)
        self._targets = torch.zeros(0, self.num_classes, dtype=torch.int8)
        self._total_sample_count = torch.zeros(1)
        self._curr_sample_count = torch.zeros(1)

    def __repr__(self):
        return repr({"name": self.name, "value": self.value})

    def set_classy_state(self, state):
        """
        Set the state of meter
        """
        assert (
            self.name == state["name"]
        ), f"State name {state['name']} does not match meter name {self.name}"
        assert self.num_classes == state["num_classes"], (
            f"num_classes of state {state['num_classes']} "
            f"does not match object's num_classes {self.num_classes}"
        )

        # Restore the state -- correct_predictions and sample_count.
        self.reset()
        self._total_sample_count = state["total_sample_count"].clone()
        self._curr_sample_count = state["curr_sample_count"].clone()
        self._scores = state["scores"]
        self._targets = state["targets"]

    def get_classy_state(self):
        """
        Returns the states of meter
        """
        return {
            "name": self.name,
            "num_classes": self.num_classes,
            "scores": self._scores,
            "targets": self._targets,
            "total_sample_count": self._total_sample_count,
            "curr_sample_count": self._curr_sample_count,
        }

    def verify_target(self, target):
        """
        Verify that the target contains {-1, 0, 1} values only
        """
        max_ = torch.max(target)
        min_ = torch.min(target)
        assert max_ <= 1.0, "Target max values should be <= 1.0"
        assert min_ >= 0.0, "Target min values should be >= 0.0"

    def update(self, model_output, target):
        """
        Update the scores and targets
        """
        self.validate(model_output, target)
        self.verify_target(target)

        self._curr_sample_count += model_output.shape[0]
        curr_dice_scores = self._scores
        sample_count_so_far = curr_dice_scores.shape[0]
        self._scores = torch.zeros(
            int(self._curr_sample_count[0]), self.num_classes, dtype=torch.float32
        )
        
        dice_scores = torch.zeros(self.num_classes)

        #for cls in range(self.num_classes):
            #dice_scores[:,cls] = batch_evaluation(model_output[:,cls,:,:], target[:,cls,:,:], deepsupervision=self.deepsupervision, threshold=self.threshold, n_classes=self.num_classes)
        model_output = model_output.permute(1,0,2,3)
        for label, pred_mask in enumerate(model_output):
            decoded_mask = torch.squeeze(target == label)
            pred_mask = (pred_mask > self.threshold).float()

            dice_scores[label] = dice_coefficient(pred_mask, decoded_mask)


        if sample_count_so_far > 0:
            self._scores[:sample_count_so_far, :] = curr_dice_scores
        self._scores[sample_count_so_far:, :] = dice_scores

        del curr_dice_scores

    def validate(self, model_output, target):
        """
        Validate that the input to meter is valid
        """
        assert len(model_output.shape) == 4, "model_output should be a 4D tensor"
        assert len(target.shape) == 4, "target should be a 4D tensor"
        assert (
            model_output.shape[0] == target.shape[0]
        ), "Expect same shape in model output and target.  Please check the ground truth dimention"
        assert (
            model_output.shape[1] == self.num_classes
        ), "Expect same shape in model output and mask in target"
        assert (
            model_output.shape[2] == target.shape[2]
        ), "Expect same shape in model output and target"
        assert (
            model_output.shape[3] == target.shape[3]
        ), "Expect same shape in model output and target"
        num_classes = target.shape[1]
        source = "'https://github.com/zonasw/unet-nested-multiple-classification/blob/master/losses.py'"
        logging.info(f"Number of classes in the model output is {model_output.shape[1]} but the dataloader supply ony {num_classes} as as ground trth. The meter is followed from {source}")
       

def main():
    input_ = torch.zeros(2, 1, 224, 224)
    output =  torch.zeros(2, 1, 224, 224)
    result = batch_evaluation(input_, output, deepsupervision= False, threshold=0.5, n_classes=1)
    assert result == 1.0, "For two black image dice coefficient is not 1.0"

    input_ = torch.ones(1, 2, 224, 224)
    output =  torch.ones(1, 2, 224, 224)
    result = batch_evaluation(input_, output, deepsupervision= False, threshold=0.5, n_classes=1)
    assert result == 1.0, "For two white image dice coefficient is not 1.0"

    input_ = torch.zeros(1, 2, 224, 224)
    output =  torch.ones(1, 2, 224, 224)
    result = batch_evaluation(input_, output, deepsupervision= False, threshold=0.5, n_classes=1)
    print((result))
    assert np.isclose(result, 0.0, rtol=1e-09, atol=1e-08, equal_nan=False), "For black and white image dice coefficient is not close to zero"

    input_ = torch.zeros(1, 2, 224, 224)
    output =  torch.zeros(1, 2, 224, 224)
    result = batch_evaluation(input_, output, deepsupervision= False, threshold=0.5, n_classes=2)
    assert result == 0.0, "For two black image  coefficient is not 0.0"


    input_ = torch.ones(1, 2, 224, 224)
    output =  torch.ones(1, 2, 224, 224)
    result = batch_evaluation(input_, output, deepsupervision= False, threshold=0.5, n_classes=5)
    print(result)
    #assert result == 0.0, "For two white image dice coefficient is not 0.0"

    input_ = torch.zeros(1,2)
    output =  torch.ones(1,2)
    print(output)
    result = batch_evaluation(input_, output, deepsupervision= False, threshold=0.5, n_classes=2)
    print((result))
    assert np.isclose(result, 0.0, rtol=1e-09, atol=1e-08, equal_nan=False), "For black and white image dice coefficient is not close to zero"


if __name__ == "__main__":
    main()

