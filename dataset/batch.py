from dataclasses import dataclass, field
from typing import Any

from transformers import BatchEncoding


@dataclass
class Batch:
    data: BatchEncoding
    answers: list[str]
    info: dict[str, list[Any]] = field(default_factory=lambda: {})
