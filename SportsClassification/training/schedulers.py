from typing import Any
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, OneCycleLR, CosineAnnealingWarmRestarts, ExponentialLR
from torch.optim import Optimizer
from utils.helpers import filter_kwargs
import math
from torch.optim.lr_scheduler import LRScheduler


class CosineAnnealingWarmRestartsDecay(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        T_0: int,
        T_mult: int=1,
        eta_min: float=0.,
        gamma: float=0.5,
        last_epoch: int=-1
    ):
        """
        CosineAnnealingWarmRestarts with decaying max LR after each restart.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            T_0 (int): Number of iterations for the first restart.
            T_mult (int): Factor to increase cycle length after each restart.
            eta_min (float): Minimum learning rate.
            last_epoch (int): The index of the last epoch.
        """
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_epoch
        self.cycle_length = T_0
        self.restart_count = 0
        self.gamma = gamma
        self.base_max_lrs = [group['lr'] for group in optimizer.param_groups]  # starting max LRs
        super().__init__(optimizer, last_epoch)

    def get_last_lr(self) -> list[float]:
        return self.get_lr()

    def get_lr(self) -> list[float]:
        # scale factor = 1 / (2^restart_count)
        scale = self.gamma ** self.restart_count
        lrs: list[float] = []
        for base_lr in self.base_max_lrs:
            max_lr = base_lr * scale
            lr = self.eta_min + (max_lr - self.eta_min) * (
                1 + math.cos(math.pi * self.T_cur / self.cycle_length)) / 2
            lrs.append(lr)
        return lrs

    def step(self, epoch: int | None = None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.T_cur += 1

        if self.T_cur >= self.cycle_length:
            # Restart
            self.restart_count += 1
            self.T_cur = 0
            self.cycle_length = self.cycle_length * self.T_mult

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


SCHEDULERS_DICT: dict[str, type[LRScheduler]] = {
    "step_lr": StepLR,
    "reduce_lr_on_plateau": ReduceLROnPlateau,
    "one_cycle_lr": OneCycleLR,
    "cosine_annealing_warm_restarts": CosineAnnealingWarmRestarts,
    "exponential_lr": ExponentialLR,
    "cosine_annealing_warm_restarts_with_decay": CosineAnnealingWarmRestartsDecay,
}


def setup_scheduler(optimizer: Optimizer, scheduler_config: dict[str, Any] | None = None):

    if scheduler_config:
        scheduler_type = scheduler_config["type"]

        scheduler_cls = SCHEDULERS_DICT[scheduler_type]
        
        kwargs = filter_kwargs(scheduler_cls, scheduler_config)
        return scheduler_cls(optimizer, **kwargs)

    return None
