# Author: Md ABdul Kadir
import torch.nn.functional as F
import torch
from medpy import metric
from torch.autograd import Function
import logging
import numpy as np
import torch
from classy_vision.generic.distributed_util import all_reduce_sum, gather_from_all
from classy_vision.meters import ClassyMeter, register_meter
from vissl.config import AttrDict
from vissl.utils.env import get_machine_local_and_dist_rank
from vissl.utils.svm_utils.evaluate import get_precision_recall

@register_meter("hd95")
class HD95(ClassyMeter):
    """
    Meter to calculate HD95 metric for multi-label image segmention task.

    Args:
        meters_config (AttrDict): config containing the meter settings

    meters_config should specify the encoded_classes
    """

    def __init__(self, meters_config: AttrDict):
        self.encoded_classes = meters_config.get("encoded_classes")
        self.num_classes = len(self.encoded_classes)
        self._softmax = meters_config.get("softmax", True)
        #self.threshold = meters_config.get("threshold") # Threshold for converting unet output to mask
        self._total_sample_count = None
        self._curr_sample_count = None
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
        return "hd95"

    @property
    def value(self):
        """
        Value of the meter globally synced.
        """
        _, distributed_rank = get_machine_local_and_dist_rank()
        logging.info(
            f"Rank: {distributed_rank} hd95 meter: "
            f"scores: {self._scores.shape}"
        )
        ap_matrix = {k:0 for k in range(len(self.encoded_classes))}#torch.ones(self.num_classes, dtype=torch.float32) * -1
        # targets matrix = 0, 1, -1
        # unknown matrix = 0, 1 where 1 means that it's an unknown
        #unknown_matrix = torch.eq(self._targets, -1.0).float().detach().numpy()
        for cls_num in range(len(self.encoded_classes)):
            avg_score = self._scores[:,cls_num].mean().item()
            #assert avg_score <= 1.0, " Calculation error in Dice score, dice score cant be largeer than 1.0" 
            ap_matrix[cls_num] = avg_score
        return {"hd95 score": ap_matrix }

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
        #max_ = torch.max(target)
        #min_ = torch.min(target)
        #assert max_ >= 1.0, "Target max values should be > 1.0"
        #assert min_  == 0, "Target min values should be == 0.0"
        return True
    
    def calculate_metric_percase(self, pred, gt):
        pred[pred > 0] = 1
        gt[gt > 0] = 1
        if pred.sum() > 0 and gt.sum()>0:
            #dice = metric.binary.dc(pred, gt)
            hd95 = metric.binary.hd95(pred, gt)
            return hd95
        elif pred.sum() > 0 and gt.sum()==0:
            return 0
        else:
            return 0

    def update(self, model_output, target):
        """
        Update the scores and targets
        """
        
        if self._softmax:
            model_output = torch.softmax(model_output, dim=1)

        
        model_output = torch.argmax(model_output, dim=1)
        self.validate(model_output.size(), target.size())
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
        #model_output = model_output.permute(1,0,2,3)
        #mask_type = torch.float32 if self.num_classes == 1 else torch.long
        
        #for label, pred_mask in enumerate(model_output):
        #    pred_mask = pred_mask.to(mask_type)
        #    decoded_mask = torch.squeeze(target == label)
        #    pred_mask = torch.squeeze(pred_mask > self.threshold).float()

        #    dice_scores[label] = dice_coefficient(pred_mask, decoded_mask)

        for i in range(1, self.num_classes):
            score = self.calculate_metric_percase(model_output == i, target == i)
            #pred_mask = pred_mask.to(mask_type)
            #decoded_mask = torch.squeeze(target == label)
            #pred_mask = torch.squeeze(pred_mask > self.threshold).float()

            dice_scores[i -1] = score

        if sample_count_so_far > 0:
            self._scores[:sample_count_so_far, :] = curr_dice_scores
        self._scores[sample_count_so_far:, :] = dice_scores

        del curr_dice_scores

    def validate(self, model_output: tuple , target: tuple):
        """
        Validate that the input to meter is valid
        """
        assert model_output[1] == self.num_classes, "model_output should have equal number of class as encoded number of class"
        assert model_output[0] == target[0], "model_output should have has same batch as target encoded tensor"
        assert model_output[-2:] == target[-2:], "image size should be same"
        source = "'https://github.com/tifat58/TransUNet/blob/d68a53a2da73ecb496bb7585340eb660ecda1d59/utils.py'"
        logging.info(f"Number of classes in the model output is {target} but the dataloader supply ony {self.num_classes} as as ground trth. The meter is followed from {source}")
       

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

