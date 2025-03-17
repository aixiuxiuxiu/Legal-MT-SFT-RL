import re
from typing import override

from .reward import RewardFn

REGEX_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.MULTILINE)


def extract_answer(completion: str) -> str | None:
    matches = REGEX_ANSWER.search(completion)
    return matches.group(1) if matches else None


class ClassificationReward(RewardFn):
    key = "classification-reward"

    def __init__(
        self,
        value: float = 1.0,
        name: str | None = None,
        weight: float = 1.0,
    ):
        super().__init__(name=name, weight=weight)
        self.value = value

    @override
    def calculate(self, completion: str, answer: str) -> float:
        if completion == answer:
            return self.value
        extracted = extract_answer(completion)
        if extracted == answer:
            return self.value
        return 0.0
