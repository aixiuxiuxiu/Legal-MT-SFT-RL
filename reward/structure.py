import re
from typing import override

from .reward import RewardFn


class StructureReward(RewardFn):
    key = "structure-reward"

    def __init__(
        self,
        pattern: str,
        value: float = 1.0,
        strict: bool = False,
        name: str | None = None,
        weight: float = 1.0,
    ):
        super().__init__(name=name, weight=weight)
        if strict:
            # Strict means that the whole string must match the pattern, not just parts
            # of it
            pattern = rf"\A{pattern}\Z"
        self.regex = re.compile(pattern, re.DOTALL | re.MULTILINE)
        self.value = value

    @override
    def calculate(self, completion: str, answer: str) -> float:
        return self.value if self.regex.search(completion) else 0.0
