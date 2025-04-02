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
        max_count: int | None = None,
        count_penalty: float = 0.5,
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
        self.max_count = max_count
        self.count_penalty = count_penalty

    @override
    def calculate(self, completion: str, answer: str) -> float:
        if self.max_count:
            match len(self.regex.findall(completion)):
                case 0:
                    return 0.0
                case count if count <= self.max_count:
                    return self.value
                case count:
                    return self.value - self.count_penalty * (count - self.max_count)
        else:
            return self.value if self.regex.search(completion) else 0.0
