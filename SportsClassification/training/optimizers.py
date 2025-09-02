# utils/optimizers.py
from torch import nn
import torch.optim as optim
from utils.helpers import filter_kwargs


OPTIMIZERS_DICT = {
    "adam": optim.Adam,
    "sgd": optim.SGD,
}


def setup_optimizer(model: nn.Module, optimizer_config: dict):
    optimizer_type = optimizer_config["type"]

    optimizer_cls = OPTIMIZERS_DICT[optimizer_type]
    
    kwargs = filter_kwargs(optimizer_cls, optimizer_config)
    return optimizer_cls(model.parameters(), **kwargs)
