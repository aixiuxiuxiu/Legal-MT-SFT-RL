from dataclasses import dataclass

import torch

from dataset.chat.processing import MessageBoundaries

from .boundary import TokenBoundary

MULTI_TOKEN = dict(start=[0, 1, 2], end=[3, 4])

MULTI_TOKEN_BOUNDARIES = MessageBoundaries(
    start=torch.tensor(MULTI_TOKEN["start"]), end=torch.tensor(MULTI_TOKEN["end"])
)


@dataclass
class MultiTokenBoundary(TokenBoundary):
    BOUNDARIES = MessageBoundaries(
        start=torch.tensor(MULTI_TOKEN["start"]), end=torch.tensor(MULTI_TOKEN["end"])
    )


def test_mask_empty_message():
    MultiTokenBoundary.validate_messages_matrix(max_len=0)


def test_mask_no_message():
    MultiTokenBoundary.validate_messages_matrix(num_messages=0)


def test_mask_no_boundaries():
    boundary = MultiTokenBoundary.single_message(boundaries=False)
    boundary.validate(include_start=True, include_end=True)
    assert not torch.any(boundary.expected)


def test_mask_messages():
    for i in range(5):
        MultiTokenBoundary.validate_messages_matrix(num_messages=i)


def test_mask_messages_interleaved():
    for i in range(5):
        for nth in range(1, 6):
            MultiTokenBoundary.validate_messages_matrix(
                num_messages=i, boundaries_every_nth=nth
            )
