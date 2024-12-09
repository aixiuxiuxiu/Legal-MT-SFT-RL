import csv
import glob
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from .chat.messages import ChatMessage


@dataclass
class InstructSample:
    """
    A single sample for the instruction tuning.

    Expects the message to be given in a dictionary of
    """

    messages: list[ChatMessage]
    info: dict = field(default_factory=lambda: {})

    @classmethod
    def from_json(cls, path: str | os.PathLike) -> Self:
        path = Path(path)
        with open(path, "r", encoding="utf-8") as fd:
            data = json.load(fd)
        messages = []
        if system_message := data.get("system"):
            messages.append(ChatMessage.from_inputs([system_message], role="system"))
        messages.append(
            ChatMessage.from_inputs([data["image"], data["question"]], role="user")
        )
        messages.append(ChatMessage.from_inputs([data["answer"]], role="assistant"))
        return cls(messages=messages, info=dict(path=path))

    def as_chat(self) -> list[dict[str, str | list[dict[str, str]]]]:
        return [msg.as_chat() for msg in self.messages]

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
        ignore_index: int = -100,
    ):
        """
        Args:
            path (str | os.PathLike): Path to either a directory of JSON files, where
                each JSON file corresponds to one sample, or path to a TSV file that
                lists all the JSON files that should be used (must be relative to the
                directory of the TSV file).
            processor (PreTrainedTokenizerBase): Processor of the model.
            ignore_index (int): Label value that is ignored in the loss. [Default: -100]
        """
        self.path = Path(path)
        self.processor = processor
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

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int) -> InstructSample:
        return InstructSample.from_json(self.files[i])
