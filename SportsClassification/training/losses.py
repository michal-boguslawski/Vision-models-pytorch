from torch import nn
from utils.helpers import filter_kwargs
from typing import Any


LOSSES_DICT: dict[str, type[nn.Module]] = {
    "cross_entropy": nn.CrossEntropyLoss
}


def setup_loss(loss_config: dict[str, Any]):
    loss_type = loss_config["type"]

    loss_cls = LOSSES_DICT[loss_type]
    
    kwargs = filter_kwargs(loss_cls, loss_config)
    return loss_cls(**kwargs)
