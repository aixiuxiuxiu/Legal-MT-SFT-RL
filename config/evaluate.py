from dataclasses import dataclass
from pathlib import Path

from simple_parsing import field

from .entry import ConfigEntry
from .hardware import HardwareConfig
from .image import ImageConfig


@dataclass
class EvaluateConfig(ConfigEntry):
    """
    Evaluation configuration
    """

    # Path to dataset to be evalulated, either a directory with JSON files or a TSV file
    # listing the JSON files to use.
    data: Path = field(alias="-d")
    # Name or path of the trained LLM
    model: str = field(alias="-m")
    # Output directory to save the results
    out_dir: Path = field(default=Path("results"), alias="-o")
    # Path to a JSON file with prompts that should be used that are shared across the
    # dataset.
    # Mostly system prompts, but also questions for datasets where the same question
    # would be used for all tasks, e.g. for classification.
    # Currently only the first one in the list will be used.
    # Structure: {"system": [], "question": []}
    prompts: Path | None = field(default=None, alias="-p")
    # Set a different padding token, as sometimes the one defined in the model
    # config may not work correctly. Should only ever be necessary for models that were
    # not fine-tuned.
    pad_token: str | None = None
    # Maximum number of new tokens that are generated before stopping it manually if it
    # fails to produce and end of sequence token.
    max_new_tokens: int | None = None

    image: ImageConfig = field(default_factory=ImageConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
