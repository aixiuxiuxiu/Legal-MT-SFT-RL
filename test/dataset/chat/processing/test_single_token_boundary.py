from dataclasses import dataclass

import torch

from dataset.chat.processing import MessageBoundaries

from .boundary import TokenBoundary

SINGLE_TOKEN = dict(start=0, end=1)


@dataclass
class SingleTokenBoundary(TokenBoundary):
    BOUNDARIES = MessageBoundaries(
        start=torch.tensor([SINGLE_TOKEN["start"]]),
        end=torch.tensor([SINGLE_TOKEN["end"]]),
    )


def test_mask_empty_message():
    SingleTokenBoundary.validate_messages_matrix(max_len=0)


def test_mask_no_message():
    SingleTokenBoundary.validate_messages_matrix(num_messages=0)


def test_mask_no_boundaries():
    boundary = SingleTokenBoundary.single_message(boundaries=False)
    boundary.validate(include_start=True, include_end=True)
    assert not torch.any(boundary.expected)


def test_mask_messages():
    for i in range(5):
        SingleTokenBoundary.validate_messages_matrix(num_messages=i)


def test_mask_messages_interleaved():
    for i in range(5):
        for nth in range(1, 6):
            SingleTokenBoundary.validate_messages_matrix(
                num_messages=i, boundaries_every_nth=nth
            )
