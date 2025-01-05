from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from utils.nested_dict import NestedDict


@dataclass
class TrainResult:
    lr: float
    metrics: NestedDict[float]
    time_elapsed: float

    def to_dict(self) -> dict:
        return asdict(self)

    def to_log_dict(self) -> NestedDict[float]:
        return dict(lr=self.lr, **self.metrics)


@dataclass
class Example:
    path: str
    pred: str
    target: str


@dataclass
class ValidationResult:
    metrics: NestedDict[float]
    examples: list[Example]
    time_elapsed: float

    def to_dict(self) -> dict:
        return asdict(self)

    def to_log_dict(self) -> NestedDict[float]:
        return self.metrics


@dataclass
class TrainOutput:
    loss: torch.Tensor
    metrics: NestedDict[float]
    info: dict[str, list[Any]] = field(default_factory=lambda: {})


@dataclass
class ValidationOutput:
    preds: list[str]
    target: list[str]
    metrics: NestedDict[float]
    info: dict[str, list[Any]] = field(default_factory=lambda: {})
