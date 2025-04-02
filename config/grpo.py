from dataclasses import dataclass


@dataclass
class GrpoConfig:
    """
    Configuration related to the Group Relative Preference Optimisation (GRPO)
    """

    # Number of generations per sample for GRPO.
    num_generations: int = 8
    # Clip range of the advantage term. A value of 0.2 means that it will clipped in the
    # range [0.8, 1.2].
    clip_advantage: float = 0.2
    # Weight for the KL-Divergence loss term. DeepSeek-R1 used 0.04, but that seems to
    # be too high, hence the default is 0.01.
    kl_weight: float = 0.01
    # Temperature to use for the generation of the group samples for GRPO.
    temperature: float = 0.6
    # Top-P to use for the generation of the group samples for GRPO.
    top_p: float = 0.92
