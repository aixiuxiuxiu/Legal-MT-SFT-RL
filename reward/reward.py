from abc import ABC, abstractmethod
from typing import ClassVar


class RewardFn(ABC):
    key: ClassVar[str]

    def __init__(self, name: str | None = None, weight: float = 1.0):
        self.name = name if name else self.key
        self.weight = weight

    @abstractmethod
    def calculate(self, completion: str, answer: str) -> float:
        raise NotImplementedError("calculate method is not implemented")

    def __call__(self, completion: str, answer: str) -> float:
        return self.calculate(completion, answer) * self.weight
