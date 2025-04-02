from dataclasses import dataclass, field
from typing import Any, Generator, Self

import torch
from transformers import BatchEncoding

from reward import RewardFn


@dataclass
class Batch:
    data: BatchEncoding
    answers: list[str]
    info: dict[str, list[Any]] = field(default_factory=lambda: {})


@dataclass
class GroupedBatch:
    """
    Batches for grouped generations.
    """

    data: list[BatchEncoding]
    completions: list[list[str]]
    answers: list[str]
    prompt_len: int
    info: dict[str, list[Any]] = field(default_factory=lambda: {})

    @classmethod
    def from_generations(
        cls,
        batches: list[BatchEncoding],
        completions: list[list[str]],
        reference: Batch,
    ) -> Self:
        """
        Create a grouped batch from the generated completions.

        Important: This assumes that every batch in the list of batches, has the same
        order of samples, meaning that if there are 4 different prompts, each of the
        batch needs to have all 4 of these prompts (in the same order).
        Otherwise the grouping would be incorrect.

        Also the answers and info are shared across the completions, so only one of them
        needs to be preserved, hence the same order is required as well.
        """
        return cls(
            data=batches,
            completions=completions,
            # Use the same from the reference batch, as they are shared across batches.
            answers=reference.answers,
            prompt_len=reference.data.input_ids.size(1),
            info=reference.info,
        )

    def completions_by_group(self) -> list[list[str]]:
        """
        Get the completions by group rather than by batch.

        The completions are stored by batch [num_completions x batch_size], but in order
        to calculate the group averages, it needs to be transposed to
        [batch_size x num_completions].
        """
        if len(self.completions) == 0:
            return []
        batch_size = len(self.completions[0])
        return [[batch[i] for batch in self.completions] for i in range(batch_size)]

    def compute_rewards(
        self,
        reward_fns: list[RewardFn],
        eps: float = 1e-5,
        scale: bool | float | None = True,
    ) -> torch.Tensor:
        """
        Compute the rewards for all the given reward functions within the group.

        Args:
            reward_fns (list[RewardFn]): List of reward functions.
            eps (float): Epsilon to avoid division by zero. [Default: 1e-5]
            scale (bool | float, optional): How to scale the rewards, if set to True the
                rewards will be scaled by the standard deviation of the group. A float
                value will scale them by that value (e.g. max_length).
                [Default: True]

        Returns:
            advantages (torch.Tensor): Advantages for all completions and batches.
                [Dimension: num_completions x batch_size]
        """
        advantages = []
        # This iterates over the samples per batch. Meaning that the first value
        # contains all generated num_completions of that sample.
        for group_completions, answer in zip(self.completions_by_group(), self.answers):
            # Rewards for each completion in the group, with each reward function.
            # Dimension: num_completions x num_fns
            rewards_matrix = torch.tensor(
                [
                    [fn(completion, answer) for fn in reward_fns]
                    for completion in group_completions
                ]
            )
            # Sum all the various rewards for each completion to get a single number.
            rewards_per_completion = rewards_matrix.sum(dim=1)
            # Group stats
            group_mean = torch.mean(rewards_per_completion)
            # The (dis-)advtange is how much better/worse a completion is compared to
            # the whole group.
            advantage = rewards_per_completion - group_mean
            match scale:
                case True:
                    group_std = torch.std(rewards_per_completion)
                    scale_factor = group_std + eps
                case False | None:
                    scale_factor = 1.0
                case val:
                    scale_factor = val
            advantage /= scale_factor
            advantages.append(advantage)
        return torch.stack(advantages, dim=1)

    def iter_batches(
        self,
        reward_fns: list[RewardFn],
        scale: bool | float | None = True,
    ) -> Generator[Batch, None, None]:
        """
        Create an iterator over the batches.

        This returns a batch at each iteration, where the advantages within the group
        are added to the info dict. That is a bit clunky, but for the moment it will do
        the job to keep it inline with the trainer API.

        Args:
            reward_fns (list[RewardFn]): List of reward functions.
            scale (bool | float, optional): How to scale the rewards, if set to True the
                rewards will be scaled by the standard deviation of the group. A float
                value will scale them by that value (e.g. max_length).
                [Default: True]

        Returns:
            iter (Generator[Batch]): Iterator of batches.
        """
        batch_size = len(self.answers)
        advatanges = self.compute_rewards(reward_fns, scale=scale)
        for data, adv in zip(self.data, advatanges.tolist()):
            batch = Batch(
                data=data,
                answers=self.answers,
                info=dict(
                    **self.info,
                    prompt_len=[self.prompt_len] * batch_size,
                    advantages=adv,
                ),
            )
            yield batch
