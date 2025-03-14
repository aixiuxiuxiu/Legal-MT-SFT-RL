from dataclasses import dataclass
from typing import Self

import torch
from transformers import PreTrainedTokenizerBase

from .messages import ChatMessage


# Just a small wrapper around apply_chat_template, mostly to also get the correct types
# from them.
def tokenise_chat(
    processor: PreTrainedTokenizerBase,
    messages: list[ChatMessage],
    add_generation_prompt: bool = False,
) -> list[int]:
    tokens = processor.apply_chat_template(
        # HuggingFace got the type annotations wrong of the chat messages.
        [message.as_chat() for message in messages],  # pyright: ignore[reportArgumentType]
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    assert isinstance(tokens, list) and isinstance(tokens[0], list), (
        "Tokenised messages are not a batch of tokens (list[list[int]])"
    )
    return tokens[0]


@dataclass
class MessageBoundaries:
    """
    Chat message boundaries defined by an opening and closing sequence of tokens.

    For example:
        - `<|im_start|>assistant\\n` as the opening sequence
        - `<|im_end|>` as the closing sequence

    These are then tokenised, such that the sequences are a list of tokens, which allows
    to determine the boundaries in any already tokenised input. It is primarily used to
    determine the parts of the labels that should be used for training, which generally
    are only the assistant messages, whereas all others are just context.
    """

    start: torch.Tensor
    end: torch.Tensor

    @classmethod
    def identify_assistant(cls, processor: PreTrainedTokenizerBase) -> Self:
        """
        Identify the message boundaries of the assistant, which can be used to determine
        the tokens that should contribute to the loss.

        This is much more complicated than it ought to be, since different tokenisers
        define the start/end tokens differently. To make it more robust, the chat
        template is exploited, such that it starts empty (with the system message), then
        the generation prompt is added, which is the start of the turn for the
        assistant, and after that the assistant message is completed.
        With that approach, each new addition, the tokens for the start and end can be
        identified.

        Note: Pixtral does not work with this approach.

        Args:
            processor (PreTrainedTokenizerBase): Processor / tokeniser of the model
        """
        # The user message must be defined, because some chat templates require
        # alternating user/assistant, which must start with user. Some will add
        # a default system message if none is provided, so the user message ensures that
        # it is consistent.
        # Also since apply_chat_templates requires at least one message, this can be
        # used as the single one.
        user_message = ChatMessage.from_inputs([""], role="user")
        assistant_message = ChatMessage.from_inputs([""], role="assistant")

        system_only = tokenise_chat(
            processor,
            [user_message],
            add_generation_prompt=False,
        )
        with_assistant_start = tokenise_chat(
            processor,
            [user_message],
            add_generation_prompt=True,
        )
        with_assistant = tokenise_chat(
            processor,
            [user_message, assistant_message],
            add_generation_prompt=False,
        )

        # HuggingFace is really annoying with types that are just unions of all
        # possibilities. There exist ways to create an overload for cases where one
        # argument is set to True.
        start = with_assistant_start[len(system_only) :]
        end = with_assistant[len(with_assistant_start) :]

        # FIXME: Fix this type annotation for Image/Text processors.
        # But this is at least safe.
        tokeniser = (
            processor.tokenizer
            if hasattr(processor, "tokenizer")
            and isinstance(processor.tokenizer, PreTrainedTokenizerBase)
            else processor
        )

        # Remove the trailing whitespace of the end, as there is often a new line at the
        # end, that would only be there if there were another message, as the generation
        # stops before that.
        end = tokeniser.encode(tokeniser.decode(end).rstrip(), add_special_tokens=False)

        instance = cls(
            start=torch.tensor(start),
            end=torch.tensor(end),
        )
        # Make sure that the boundaries work correctly with the tokeniser.
        instance._sanity_check(processor)
        return instance

    def _sanity_check(self, processor: PreTrainedTokenizerBase):
        assert len(self.start) > 0, "Start boundary was not found (empty)"
        assert len(self.end) > 0, "End boundary was not found (empty)"
        msg = "Hello"
        # User message is necessary for chat templates that require alternative between
        # user/assistant (need to start with user)
        user_message = ChatMessage.from_inputs([""], role="user")
        assistant_message = ChatMessage.from_inputs([msg], role="assistant")
        tokens = torch.tensor(
            tokenise_chat(
                processor,
                [user_message, assistant_message],
                add_generation_prompt=False,
            )
        )
        # Extract just the text itself.
        mask = self.mask(tokens, include_start=False, include_end=False)
        content_only = tokens[mask]
        decoded_msg = processor.decode(content_only)
        assert decoded_msg == msg, (
            f"Boundary check failed, expected extracted message to be {msg!r} "
            f"but got {decoded_msg!r}"
        )

    def _find_sequence_matches(
        self, input: torch.Tensor, sequence: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): Tokenised input in which to search the sequence.
                Can have any additional leading dimensions, but the last dimension needs
                to be the sequence dimension.
                [Dimension: * x text_seq_len]
            sequence (torch.Tensor): Sequence of tokens to search for.
                [Dimension: len]

        Returns:
            matches (torch.Tensor): Matches with the corresponding start and end
                indices. Each match contains two values (start, end) which is given as
                the indices of all dimensions to index (similar to torch.nonzero).
                [Dimension: num_matches x 2 x num_dimensions]
        """
        assert len(sequence) > 0, "Sequence to find must not be empty"
        matches = []
        # Find the first token of the sequence, then include the following tokens that
        # should be part of the sequence.
        initial_match = input == sequence[0]
        for pos in initial_match.nonzero():
            *leading, start = pos.unbind()
            end = start + sequence.size(-1)
            # Not enough tokens in the input to be a match.
            if end > input.size(-1):
                continue
            # Get the sequence of start tokens
            start_tokens = input[*leading, start:end]
            # Verify that the whole sequence is correct, not only the first.
            # Otherwise it's just a partial match and it should be ignored.
            if not torch.all(start_tokens == sequence):
                continue
            matches.append([[*leading, start], [*leading, end]])
        return torch.tensor(matches)

    def mask(
        self, input: torch.Tensor, include_start: bool = False, include_end: bool = True
    ) -> torch.Tensor:
        """
        Get the mask for the messages inside the boundaries, i.e. the assistant messages.

        For the generation it is often expected to also include the end boundary of the
        message, but not the beginning. This is the default but can be customised to
        include either of the parts if necessary.

        Args:
            input (torch.Tensor): Tokenised input [Dimension: * x text_seq_len]
            include_start (bool): Whether to include the start boundary of the message
                e.g. "<|im_start|>assistant\\n"
                [Default: False]
            include_end (bool): Whether to include the end boundary of the message
                e.g. "<|im_end|>"
                [Default: True]
        """
        mask = torch.full_like(input, False, dtype=torch.bool)
        # Dimension: num_matches x 2 x num_dims
        starts = self._find_sequence_matches(input, self.start)
        ends = self._find_sequence_matches(input, self.end)

        # No match possible as either side of the boundary was missing.
        if starts.numel() == 0 or ends.numel() == 0:
            return mask

        # The end of start need to be matched with the beginning of the end, as they
        # are given as a range of [begin, end] (2nd dimension).
        ends_first = ends[:, 0]
        for start in starts:
            start_last = start[1]
            # Make sure the indices of the leading dimensions (non-sequence dimensions)
            # are the same as the start candidate, otherwise it spills into the other
            # dimensions, such as the batch dimension, hence the torch.all(..., dim=-1)
            # And the last index needs of the end needs to come after the start.
            end_candidates = torch.all(
                ends_first[:, :-1] == start_last[:-1], dim=-1
            ) & (ends_first[:, -1] >= start_last[-1])
            if torch.sum(end_candidates) == 0:
                # No end match that could close the start, therefore it should be
                # everything until the end.
                # It is safe to be outside of the indices, as it will be used as an end
                # index, but since it's exclusive, it should also include all the tokens
                # even if the `include_end=False`, since it doesn't exist.
                end_selected = start.clone()
                end_selected[:, -1] = mask.size(-1)
            else:
                end_selected = ends[end_candidates]
                # Use the first selection, which is automatically the closest,
                # torch.nonzero returns an ordered list.
                # This is guaranteed to have at least one.
                end_selected = end_selected[0]

            match_start = start[0] if include_start else start[1]
            *leading_dims, start_index = match_start.unbind()
            end_index = end_selected[1, -1] if include_end else end_selected[0, -1]
            mask[*leading_dims, start_index:end_index] = True

        return mask
