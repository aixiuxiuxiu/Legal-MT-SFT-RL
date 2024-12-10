import os
import typing
from dataclasses import dataclass

import torch.multiprocessing as mp
from simple_parsing import choice, field

from utils.hardware import HardwareManager, MixedPrecisionChoice


@dataclass
class HardwareConfig:
    """
    Hardware related configuration
    """

    # Random seed for reproducibility
    seed: int = field(default=1234, alias="-s")
    # Batch size per GPU
    batch_size: int = field(default=8, alias="-b")
    # Number of workers to use for data loading. If not specified, it will use the
    # number of available CPUs equally distributed across the GPUs.
    # Note: Specifying this value signifies the number of workers per GPU not the total.
    num_workers: int | None = field(default=None, alias="-w")
    # Do not persist workers after the epoch ends but reinitialise them at the start of
    # every epoch. (Slower but uses much less RAM)
    no_persistent_workers: bool = field(action="store_true")
    # Type for mixed-precision. By default it will use auto, which will use bf16 if the
    # GPU supports it, and otherwise fallback to fp16. To disable it, use "none" as the
    # argument."
    mixed_precision: MixedPrecisionChoice = choice(
        *typing.get_args(MixedPrecisionChoice.__value__), default="auto"
    )
    # Set the device to run the model on. Can be set to "auto" (default), in which case
    # the best device for the current hardware is used, such as CUDA or MPS if they are
    # available.
    device: str = "auto"
    # Split the model across available GPUs. Not allowed when launching the training
    # script distributed, as that would be conflicting.
    split_model: bool = field(action="store_true")

    def calculate_num_workers(self) -> int:
        num_workers = self.num_workers
        if num_workers is None:
            num_processes = int(os.environ.get("WORLD_SIZE", "1"))
            # When the number of workers was not specified they will be partitioned such
            # that all GPUs get an equal number of workers.
            # This is not done when the option is specified manually, since that is on
            # a per-worker basis rather than the total (similar to batch size)
            num_workers = mp.cpu_count() // num_processes
        return num_workers

    def has_persistent_workers(self) -> bool:
        num_workers = self.calculate_num_workers()
        return num_workers > 0 and not self.no_persistent_workers

    def create_manager(self) -> HardwareManager:
        return HardwareManager(
            device=self.device,
            mixed_precision=self.mixed_precision,
        )
