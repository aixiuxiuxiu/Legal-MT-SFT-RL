from dataclasses import dataclass
from typing import Self

import torch

from dataset.chat.processing import MessageBoundaries

SINGLE_TOKEN = dict(start=0, end=1)

SINGLE_TOKEN_BOUNDARIES = MessageBoundaries(
    start=torch.tensor([SINGLE_TOKEN["start"]]), end=torch.tensor([SINGLE_TOKEN["end"]])
)


@dataclass
class BoundaryCase:
    start: bool
    end: bool


BOUNDARY_MATRIX = [
    BoundaryCase(start=True, end=True),
    BoundaryCase(start=True, end=False),
    BoundaryCase(start=False, end=True),
    BoundaryCase(start=False, end=False),
]


@dataclass
class SingleTokenBoundary:
    input: torch.Tensor
    expected: torch.Tensor

    @classmethod
    def single_message(
        cls,
        boundaries: bool = True,
        start: bool = False,
        end: bool = False,
        max_len: int = 20,
        vocab_size: int = 100,
    ) -> Self:
        if max_len == 0:
            msg = torch.tensor([])
        else:
            msg = torch.randint(
                SINGLE_TOKEN["end"] + 1,
                vocab_size,
                size=(int(torch.randint(max_len, size=(1,))),),
            )
        if boundaries:
            input = torch.cat(
                [SINGLE_TOKEN_BOUNDARIES.start, msg, SINGLE_TOKEN_BOUNDARIES.end]
            )
            expected = torch.full_like(input, True, dtype=torch.bool)
            expected[0] = start
            expected[-1] = end
        else:
            input = msg
            expected = torch.full_like(input, False, dtype=torch.bool)
        return cls(
            input=input,
            expected=expected,
        )

    @classmethod
    def multiple_messages(
        cls,
        num_messages: int,
        boundaries_every_nth: int = 1,
        start: bool = False,
        end: bool = False,
        max_len: int = 20,
        vocab_size: int = 100,
    ) -> Self:
        inputs = cls(
            input=torch.tensor([]), expected=torch.tensor([], dtype=torch.bool)
        )
        for i in range(num_messages):
            next_input = cls.single_message(
                boundaries=(i + 1) % boundaries_every_nth == 0,
                start=start,
                end=end,
                max_len=max_len,
                vocab_size=vocab_size,
            )
            inputs = inputs.append(next_input)
        return inputs

    @classmethod
    def validate_messages_matrix(
        cls,
        num_messages: int = 1,
        boundaries_every_nth: int = 1,
        max_len: int = 20,
        vocab_size: int = 100,
    ):
        for case in BOUNDARY_MATRIX:
            self = cls.multiple_messages(
                num_messages=num_messages,
                boundaries_every_nth=boundaries_every_nth,
                start=case.start,
                end=case.end,
                max_len=max_len,
                vocab_size=vocab_size,
            )
            self.validate(include_start=case.start, include_end=case.end)

    def validate(
        self,
        include_start: bool = False,
        include_end: bool = False,
    ):
        mask = SINGLE_TOKEN_BOUNDARIES.mask(
            self.input, include_start=include_start, include_end=include_end
        )
        assert torch.equal(mask, self.expected)

    def append(self, other: Self) -> Self:
        return self.__class__(
            input=torch.cat([self.input, other.input]),
            expected=torch.cat([self.expected, other.expected]),
        )


def test_mask_single_token_empty_message():
    SingleTokenBoundary.validate_messages_matrix(max_len=0)


def test_mask_single_token_no_message():
    SingleTokenBoundary.validate_messages_matrix(num_messages=0)


def test_mask_single_token_no_boundaries():
    boundary = SingleTokenBoundary.single_message(boundaries=False)
    boundary.validate(include_start=True, include_end=True)
    assert not torch.any(boundary.expected)


def test_mask_single_token_messages():
    for i in range(5):
        SingleTokenBoundary.validate_messages_matrix(num_messages=i)


def test_mask_single_token_messages_interleaved():
    for i in range(5):
        for nth in range(1, 6):
            SingleTokenBoundary.validate_messages_matrix(
                num_messages=i, boundaries_every_nth=nth
            )
