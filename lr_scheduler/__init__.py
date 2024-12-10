from typing import Literal

from .base import BaseLrScheduler
from .const import ConstLrScheduler
from .cos import CosineScheduler
from .inv_sqrt import InvSqrtLrScheduler
from .warmup import LRWarmup, WarmupModes

type LrSchedulerKind = Literal["const", "cos", "inv-sqrt"]

LR_SCHEDULERS = {
    scheduler.kind: scheduler
    for scheduler in [ConstLrScheduler, CosineScheduler, InvSqrtLrScheduler]
}


def create_lr_scheduler(kind: LrSchedulerKind, *args, **kwargs) -> BaseLrScheduler:
    Scheduler = LR_SCHEDULERS.get(kind)
    if Scheduler is None:
        options = " | ".join([repr(m) for m in LR_SCHEDULERS])
        raise ValueError(
            f"No LR scheduler for`kind={kind!r}`, must be one of: {options}"
        )
    return Scheduler(*args, **kwargs)


__all__ = [
    "BaseLrScheduler",
    "ConstLrScheduler",
    "CosineScheduler",
    "InvSqrtLrScheduler",
    "LRWarmup",
    "WarmupModes",
    "LrSchedulerKind",
    "LR_SCHEDULERS",
    "create_lr_scheduler",
]
