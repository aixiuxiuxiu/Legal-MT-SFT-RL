import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from simple_parsing import field

from .entry import ConfigEntry
from .hardware import HardwareConfig
from .image import ImageConfig
from .lora import LoraConfig
from .lr import LrConfig
from .optim import OptimConfig


@dataclass
class TrainConfig(ConfigEntry):
    """
    Training configuration
    """

    # Path to train dataset, either a directory with JSON files or a TSV file listing
    # the JSON files to use.
    train_data: Path
    # Path to validation dataset, either a directory with JSON files or a TSV file
    # listing the JSON files to use.
    validation_data: Path
    # Name or path of the pre-trained LLM to fine-tune
    model: str = field(alias="-m")
    # Path to a JSON file with prompts that should be used to get variations.
    # Mostly system prompts, but also questions for datasets where the same question
    # would be used for all tasks, e.g. for classification.
    # For validation, the first one in the list will always be used.
    # Structure: {"system": [], "question": []}
    prompts: Path | None = field(default=None, alias="-p")
    # Set a different padding token, as sometimes the one defined in the model
    # config may not work correctly, e.g. when it's the <eos> token, the model would
    # just never learn when to stop.
    pad_token: str | None = None
    # Number of epochs to train
    num_epochs: int = field(default=10, alias="-n")
    # Name of the experiment for the logging and saved model. If not specified, the
    # current date and time will be used.
    name: str | None = None
    # Name of the project for the logging
    project: str = "vision-finetune"
    # Additional tags to add to the run in wandb
    tags: list[str] = field(default_factory=lambda: [])

    image: ImageConfig = field(default_factory=ImageConfig)
    lr: LrConfig = field(default_factory=LrConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)

    def get_name(self) -> str:
        name = self.name
        if name is None:
            timestamp = datetime.now()
            name = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        return name

    def get_log_dir(self, base_dir: str | os.PathLike = "./log") -> Path:
        return Path(base_dir) / self.get_name()
