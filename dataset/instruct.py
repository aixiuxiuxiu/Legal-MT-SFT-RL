import csv
import glob
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from dataset.resize import ImageResizer

from .chat.messages import ChatMessage


@dataclass
class PromptSelection:
    system: list[str] | None = None
    question: list[str] | None = None
    first_only: bool = False

    @classmethod
    def from_json(cls, path: str | os.PathLike, first_only: bool = False) -> Self:
        with open(path, "r", encoding="utf-8") as fd:
            data = json.load(fd)
        return cls(
            system=data.get("system"),
            question=data.get("question"),
            first_only=first_only,
        )

    def _random_selection(self, prompts: list[str] | None) -> str | None:
        if prompts is None or len(prompts) == 0:
            return None
        if self.first_only:
            return prompts[0]
        index = int(torch.randint(len(prompts), size=(1,)))
        return prompts[index]

    def random_system_prompt(self) -> str | None:
        return self._random_selection(self.system)

    def random_question_prompt(self) -> str | None:
        return self._random_selection(self.question)


@dataclass
class InstructSample:
    """
    A single sample for the instruction tuning.

    Expects the message to be given in a dictionary of
    """

    messages: list[ChatMessage]
    answer: str
    info: dict = field(default_factory=lambda: {})

    @classmethod
    def from_json(
        cls,
        path: str | os.PathLike,
        prompts: PromptSelection = PromptSelection(),
        resizer: ImageResizer = ImageResizer(),
    ) -> Self:
        path = Path(path)
        with open(path, "r", encoding="utf-8") as fd:
            data = json.load(fd)
        messages = []
        system = prompts.random_system_prompt() or data.get("system")
        if system:
            messages.append(ChatMessage.from_inputs([system], role="system"))
        question = prompts.random_question_prompt() or data.get("question")
        assert question is not None, (
            f"Sample `{path}` does not contain `question`, nor were any "
            "additonal question prompts given to be randomly selected"
        )
        image_path = data.get("image")
        if image_path is None:
            messages.append(ChatMessage.from_inputs([question], role="user"))
        else:
            image = Image.open(path.parent / image_path).convert("RGB")
            image = resizer(image)
            messages.append(ChatMessage.from_inputs([image, question], role="user"))
        answer = data.get("answer")
        assert answer is not None, f"Sample `{path}` does not contain `answer`"
        return cls(messages=messages, answer=answer, info=dict(path=path))

    def as_chat(
        self,
        include_answer: bool = True,
        prefill: str | None = None,
    ) -> list[dict[str, str | list[dict[str, str]]]]:
        if include_answer and prefill is not None:
            raise ValueError("Cannot use `prefill` together with `include_answer=True`")
        messages = [msg.as_chat() for msg in self.messages]
        if include_answer:
            answer = ChatMessage.from_inputs([self.answer], role="assistant")
            messages.append(answer.as_chat())
        if prefill:
            start_of_answer = ChatMessage.from_inputs([prefill], role="assistant")
            messages.append(start_of_answer.as_chat())
        return messages

    def get_images(self) -> list[Image.Image]:
        images = []
        for msg in self.messages:
            images.extend(msg.get_images())
        return images


class InstructDataset(Dataset):
    def __init__(
        self,
        path: str | os.PathLike,
        processor: PreTrainedTokenizerBase,
        prompts: str | os.PathLike | None = None,
        first_prompt_only: bool = False,
        ignore_index: int = -100,
        image_resizer: ImageResizer = ImageResizer(),
    ):
        """
        Args:
            path (str | os.PathLike): Path to either a directory of JSON files, where
                each JSON file corresponds to one sample, or path to a TSV file that
                lists all the JSON files that should be used (must be relative to the
                directory of the TSV file).
            processor (PreTrainedTokenizerBase): Processor of the model.
            prompts (str | os.PathLike, optional): Path to a JSON file that contains
                prompts for system/question, which will be randomly sampled to get some
                variation in the data.
                Structure of the JSON: {{"system": [], "question": []}}
            first_prompt_only (bool): Whether to only use the first prompt from the
                prompt selection. Helpful for the validation, so that the prompt remains
                consistent. [Default: False]
            ignore_index (int): Label value that is ignored in the loss. [Default: -100]
            image_resizer (ImageResizer): Image resizer to apply to each image. The
                default is a no-op.
        """
        self.path = Path(path)
        self.processor = processor
        self.first_prompt_only = first_prompt_only
        self.ignore_index = ignore_index
        if self.path.is_dir():
            self.dir = self.path
            # When a directory is given, consider every JSON files in that directory as
            # a sample.
            self.files = [Path(path) for path in glob.glob(str(self.path / "*.json"))]
        else:
            self.dir = self.path.parent
            # A TSV file lists all samples to be used with the path to their JSON file
            # in the first column.
            with open(self.path, "r", encoding="utf-8") as fd:
                reader = csv.reader(fd, delimiter="\t")
                self.files = [self.dir / line[0] for line in reader]
        self.prompt_selection = (
            PromptSelection.from_json(prompts, first_only=self.first_prompt_only)
            if prompts
            else PromptSelection(first_only=self.first_prompt_only)
        )
        self.image_resizer = image_resizer

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int) -> InstructSample:
        return InstructSample.from_json(
            self.files[i], prompts=self.prompt_selection, resizer=self.image_resizer
        )
