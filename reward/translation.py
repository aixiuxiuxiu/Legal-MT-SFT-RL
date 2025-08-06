import re
import typing
from typing import Literal, override

from torchmetrics.text import CHRFScore

from .reward import RewardFn

REGEX_TRANSLATION = re.compile(
    r"<translation>\s*(.*?)\s*</translation>", re.DOTALL | re.MULTILINE
)

TranslationMetric = Literal["chrf", "chrf++"]


def extract_translation(completion: str) -> str | None:
    matches = REGEX_TRANSLATION.search(completion)
    return matches.group(1) if matches else None


class TranslationReward(RewardFn):
    key = "translation-reward"

    def __init__(
        self,
        metric: TranslationMetric = "chrf",
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
                    [repr(m) for m in typing.get_args(TranslationMetric.__value__)]
                )
                raise ValueError(
                    f"metric={other!r} not supported, choose one of {options}"
                )

    @override
    def calculate(self, completion: str, answer: str) -> float:
        extracted = extract_translation(completion) or completion
        score = self.metric([extracted], [answer])
        return float(score)
