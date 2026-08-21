import re
from typing import override

from .reward import RewardFn

REGEX_ANSWER = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.MULTILINE)


# Extracts the final answer, i.e. the last <answer></answer> tag.
def extract_answer(completion: str) -> str | None:
    matches = REGEX_ANSWER.findall(completion)
    return None if len(matches) == 0 else matches[-1]


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
    def calculate(
        self, completion: str, answer: str, thinking: str | None = None
    ) -> float:
        if completion == answer:
            return self.value
        extracted = extract_answer(completion)
        if extracted == answer:
            return self.value
        return 0.0
