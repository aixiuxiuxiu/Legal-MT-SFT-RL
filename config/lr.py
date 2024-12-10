import typing
from dataclasses import dataclass

import torch.optim as optim
from simple_parsing import choice, field
from torch.utils.data import DataLoader

from lr_scheduler import (
    BaseLrScheduler,
    LrSchedulerKind,
    WarmupModes,
    create_lr_scheduler,
)


@dataclass
class LrConfig:
    """
    Learning rate configuration
    """

    # Peak learning rate to use
    peak_lr: float = field(default=2e-5, alias=["-l", "--learning-rate"])
    # Learning rate scheduler kind to use
    scheduler: LrSchedulerKind = choice(
        *typing.get_args(LrSchedulerKind.__value__), default="cos"
    )
    # Number of linear warmup steps for the learning rate. Can also be given as
    # a percentage of the whole run, must be in range [0, 1].
    # If not specified, will default to 20%% of the first epoch.
    warmup_steps: int | float | None = None
    # Learning rate to start the warmup from
    warmup_start_lr: float = 0.0
    # How the warmup is performed
    warmup_mode: str = choice(*typing.get_args(WarmupModes.__value__), default="linear")

    def calculate_warmup_steps(
        self, data_loader: DataLoader, num_epochs: int = 1
    ) -> int:
        if self.warmup_steps is None:
            # Warmup for 20% of the first epoch.
            # TODO: Should probably be adapted to the size of the dataset
            return int(len(data_loader) * 0.2)
        elif self.warmup_steps <= 1:
            return int(len(data_loader) * num_epochs * self.warmup_steps)
        else:
            return int(self.warmup_steps)

    def create_scheduler(
        self, optimiser: optim.Optimizer, data_loader: DataLoader, num_epochs: int = 1
    ) -> BaseLrScheduler:
        return create_lr_scheduler(
            self.scheduler,
            optimiser,
            peak_lr=self.peak_lr,
            warmup_steps=self.calculate_warmup_steps(
                data_loader, num_epochs=num_epochs
            ),
            total_steps=len(data_loader) * num_epochs,
            end_lr=1e-8,
            # To not crash when choosing schedulers that don't support all arguments.
            allow_extra_args=True,
        )
