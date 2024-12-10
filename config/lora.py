from dataclasses import dataclass

from simple_parsing import field


@dataclass
class LoraConfig:
    """
    Low-Rank Adaptation (LoRA) configuration
    """

    # Rank for LoRA
    rank: int = field(default=16, alias="-r")
    # LoRA alpha scaling factor, recommended  to use α = 2·r, which will automatically
    # be used if it was not manually specified.
    alpha: int | None = field(default=None, alias="-a")

    def calculate_alpha(self) -> int:
        return self.alpha if self.alpha else 2 * self.rank
