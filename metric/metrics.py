from dataclasses import dataclass
from typing import Literal

type MetricOrder = Literal["min", "max"]
type MetricWhen = Literal["train", "validation"]


@dataclass
class Metric:
    name: str
    key: str
    order: MetricOrder
    when: MetricWhen | None = None
    short_name: str | None = None

    def get_short_name(self) -> str:
        return self.short_name or self.name

    def is_new_best(self, old: float | None, new: float | None) -> bool:
        match (old, new):
            case (_, None):
                return False
            case (None, _):
                return True
            case (old, new):
                match self.order:
                    case "min":
                        return new <= old
                    case "max":
                        return new >= old


# Only works for training, since there is no loss during validation (it uses generate)
TRAIN_LOSS = Metric(name="Loss", key="loss", order="min", when="train")

# Only available during validation for now, since the unsloth is broken for retuning
# anything but just the loss from the model in the forward pass (during training).
CLASS_ACCURACY = Metric(
    name="Class Accuracy",
    key="accuracy.class",
    order="max",
    when="validation",
    short_name="Class",
)
CLASS_ACCURACY_UNCASED = Metric(
    name="Class Accuracy (ignore case)",
    key="accuracy.class_uncased",
    order="max",
    when="validation",
    short_name="Class (uncased)",
)
