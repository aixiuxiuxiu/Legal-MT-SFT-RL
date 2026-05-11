import re
import typing
from typing import Literal, override

from torchmetrics.text import CHRFScore

from .reward import RewardFn

REGEX_THINK = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.MULTILINE)

# For now, the metrics are the same as the translation quality.
ThinkingMetric = Literal["chrf", "chrf++"]


def extract_think(completion: str) -> str | None:
    matches = REGEX_THINK.search(completion)
    return matches.group(1) if matches else None


class ThinkingReward(RewardFn):
    key = "thinking-reward"

    def __init__(
        self,
        metric: ThinkingMetric = "chrf",
        name: str | None = None,
        weight: float = 1.0,
    ):
        super().__init__(name=name, weight=weight)
        self.metric_name = metric
        match self.metric_name:
            case "chrf":
                self.metric = CHRFScore(n_word_order=0)
            case "chrf++":
                self.metric = CHRFScore(n_word_order=2)
            case other:  # pyright: ignore[reportUnnecessaryComparison]
                options = " | ".join(
                    [repr(m) for m in typing.get_args(ThinkingMetric.__value__)]
                )
                raise ValueError(
                    f"metric={other!r} not supported, choose one of {options}"
                )

    @override
    def calculate(
        self, completion: str, answer: str, thinking: str | None = None
    ) -> float:
        extracted = extract_think(completion)
        if extracted is None:
            return 0.0
        score = self.metric([extracted], [thinking])
        return float(score)
