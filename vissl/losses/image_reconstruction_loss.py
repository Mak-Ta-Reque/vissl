import pprint
from classy_vision.losses import ClassyLoss, register_loss
import torch.nn.functional as F
from vissl.config import AttrDict
@register_loss("image_reconstruction_loss")
class AutoencoderLoss(ClassyLoss):
    """
    Add documentation for what the loss does

    Config params:
        document what parameters should be expected for the loss in the defaults.yaml
        and how to set those params
    """

    def __init__(self, loss_config: AttrDict, device: str = "gpu"):
        super(AutoencoderLoss, self).__init__()

        self.loss_config = loss_config
        # implement what the init function should do
        ...

    @classmethod
    def from_config(cls, loss_config: AttrDict):
        """
        Instantiates MyNewLoss from configuration.

        Args:
            loss_config: configuration for the loss

        Returns:
            MyNewLoss instance.
        """
        return cls(loss_config)


    def forward(self, output, target):
        # implement how the loss should be calculated. The output should be
        # torch.Tensor or List[torch.Tensor] and target should be torch.Tensor
        loss =  F.mse_loss(target, output[0], reduction="none")
        loss = loss.sum(dim=[1,2,3]).mean(dim=[0])
        return loss