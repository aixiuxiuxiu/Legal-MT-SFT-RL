from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Self

import torch


@dataclass
class BaseMetric(ABC):
    @classmethod
    def mean_of_list(cls, values: list[Self]) -> Self:
        values_dict = {}
        for val in values:
            for k, v in val.to_dict().items():
                if k not in values_dict:
                    values_dict[k] = []
                values_dict[k].append(v)
        mean_dict = {
            k: torch.mean(torch.tensor(v, dtype=torch.float)).item()
            for k, v in values_dict.items()
        }
        return cls(**mean_dict)

    # This seems really unnecessary, since it just calls `asdict` from dataclasses, but
    # it's just simpler, as the user does not need to import asdict just for this.
    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @abstractmethod
    def get_value(self) -> float:
        raise NotImplementedError("get_value method is not implemented")

    @abstractmethod
    def is_better_than(self, other: Self | None) -> bool:
        raise NotImplementedError("is_better_than method is not implemented")


@dataclass
class TokenAccuracy(BaseMetric):
    value: float

    @classmethod
    def compute(
        cls,
        preds: torch.Tensor,
        target: torch.Tensor,
        ignore_index: int = -100,
    ) -> Self:
        keep = target != ignore_index
        return cls(
            value=torch.mean(preds[keep] == target[keep], dtype=torch.float).item(),
        )

    def get_value(self) -> float:
        return self.value

    def is_better_than(self, other: Self | None) -> bool:
        if other is None:
            return True
        return self.value >= other.value


@dataclass
class ClassificationAccuracy(BaseMetric):
    value: float

    @classmethod
    def compute(
        cls,
        preds: list[str],
        target: list[str],
        ignore_case: bool = False,
    ) -> Self:
        result = []
        for pred, tar in zip(preds, target):
            if ignore_case:
                result.append(pred.lower() == tar.lower())
            else:
                result.append(pred == tar)

        return cls(
            value=torch.mean(torch.tensor(result), dtype=torch.float).item(),
        )

    def get_value(self) -> float:
        return self.value

    def is_better_than(self, other: Self | None) -> bool:
        if other is None:
            return True
        return self.value >= other.value


def summarise_metrics(metrics: dict[str, list[BaseMetric]]) -> dict[str, BaseMetric]:
    summary = {}
    for name, values in metrics.items():
        if len(values) == 0:
            # Skip empty metrics
            continue
        cls = values[0].__class__
        summary[name] = cls.mean_of_list(values)
    return summary


def summarise_list_of_metrics(
    metrics: list[dict[str, BaseMetric]],
) -> dict[str, BaseMetric]:
    metrics_by_key = {}
    for metric in metrics:
        for name, value in metric.items():
            if name not in metrics_by_key:
                metrics_by_key[name] = []
            metrics_by_key[name].append(value)
    return summarise_metrics(metrics_by_key)


# This is used to restore the metrics after synchronising them, which requries them to
# be converted to a dict.
def restore_dict_of_metrics(
    metrics_dict: dict[str, dict[str, float]], metrics_orig: dict[str, BaseMetric]
) -> dict[str, BaseMetric]:
    restored = {}
    for name, d in metrics_dict.items():
        cls = metrics_orig[name].__class__
        restored[name] = cls(**d)
    return restored
