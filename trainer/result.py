from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from metric import BaseMetric


@dataclass
class TrainResult:
    loss: float
    lr: float
    metrics: dict[str, BaseMetric]

    def to_dict(self) -> dict:
        return dict(
            loss=self.loss,
            lr=self.lr,
            metrics={name: metric.to_dict() for name, metric in self.metrics.items()},
        )

    def to_log_dict(self) -> dict[str, float]:
        metrics = {name: metric.get_value() for name, metric in self.metrics.items()}
        return dict(loss=self.loss, lr=self.lr, **metrics)


@dataclass
class Example:
    path: str
    pred: str
    target: str


@dataclass
class ValidationResult:
    metrics: dict[str, BaseMetric]
    examples: list[Example]

    def to_dict(self) -> dict:
        return dict(
            metrics={name: metric.to_dict() for name, metric in self.metrics.items()},
            example=[asdict(ex) for ex in self.examples],
        )

    def to_log_dict(self) -> dict[str, float]:
        return {name: metric.get_value() for name, metric in self.metrics.items()}


@dataclass
class TrainOutput:
    loss: torch.Tensor
    metrics: dict[str, BaseMetric]
    info: dict[str, list[Any]] = field(default_factory=lambda: {})


@dataclass
class ValidationOutput:
    preds: list[str]
    target: list[str]
    metrics: dict[str, BaseMetric]
    info: dict[str, list[Any]] = field(default_factory=lambda: {})
