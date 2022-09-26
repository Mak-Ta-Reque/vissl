import pprint
from classy_vision.losses import ClassyLoss, register_loss
from vissl.config import AttrDict
@register_loss("lovasz_loss")
class LovaszLoss(ClassyLoss):
    """
    

    Config params:
        document what parameters should be expected for the loss in the defaults.yaml
        and how to set those params
    """

    def __init__(self, loss_config: AttrDict, device: str = "gpu"):
        super(LovaszLoss, self).__init__()

        self.loss_config = loss_config
        self.loss_type = self.loss_config.loss_type
        self.update_memory_emb_index = self.loss_config.update_mem_with_emb_index
        self.lovasz_loss_average = 0 # Call the class that calculate the loss

        # implement what the init function should do
        

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

    def __repr__(self):
        # implement what information about loss params should be
        # printed by print(loss). This is helpful for debugging
        repr_dict = {
            "name": self._get_name(),
            "lovasz_loss_average": self.lovasz_loss_average,
            "loss_type": self.loss_type,
            "update_emb_index": self.update_memory_emb_index,
        }
        return pprint.pformat(repr_dict, indent=2)

    def forward(self, output, target):
        # implement how the loss should be calculated. The output should be
        # torch.Tensor or List[torch.Tensor] and target should be torch.Tensor
        ...
        ...

        return None