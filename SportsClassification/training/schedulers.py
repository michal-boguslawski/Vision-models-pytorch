from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, OneCycleLR, CosineAnnealingWarmRestarts
from torch.optim import Optimizer
from utils.helpers import filter_kwargs


SCHEDULERS_DICT = {
    "step_lr": StepLR,
    "reduce_lr_on_plateau": ReduceLROnPlateau,
    "one_cycle_lr": OneCycleLR,
    "cosine_annealing_warm_restarts": CosineAnnealingWarmRestarts
}


def setup_scheduler(optimizer: Optimizer,scheduler_config: dict | None = None):

    if scheduler_config:
        scheduler_type = scheduler_config["type"]

        scheduler_cls = SCHEDULERS_DICT[scheduler_type]
        
        kwargs = filter_kwargs(scheduler_cls, scheduler_config)
        return scheduler_cls(optimizer, **kwargs)

    return None
