# All this is just to have a type checked version of the chat messages.
from dataclasses import dataclass
from typing import Literal, Self

from PIL import Image

type InputType = str | Image.Image


@dataclass
class ChatMessageText:
    text: str
    type: Literal["text"] = "text"

    def as_chat(self) -> dict[str, str]:
        # NOTE(model-specific): Most models accept the content as a dict int the form
        # of: {"type": "text", "text": "..."}, whereas some others want just the text
        # directly. Usually, only models that are text only have this problem, even
        # though some newer ones also switched to dicts even for text only.
        # For now, if you want to use one of those models, you have to manually swap
        # the following two lines (and in ChatMessage below too).
        return dict(type=self.type, text=self.text)
        # return self.text


@dataclass
class ChatMessageImage:
    image: Image.Image
    type: Literal["image"] = "image"

    def as_chat(self) -> dict[str, str]:
        # The image is not included in the chat.
        return dict(type=self.type)


def create_content(input: InputType) -> ChatMessageText | ChatMessageImage:
    if isinstance(input, str):
        return ChatMessageText(input)
    elif isinstance(input, Image.Image):
        return ChatMessageImage(input)


@dataclass
class ChatMessage:
    content: list[ChatMessageText | ChatMessageImage]
    role: str

    @classmethod
    def from_inputs(cls, inputs: InputType | list[InputType], role: str) -> Self:
        if not isinstance(inputs, list):
            inputs = [inputs]
        return cls(
            content=[create_content(inp) for inp in inputs],
            role=role,
        )

    def as_chat(self) -> dict[str, str | list[dict[str, str]]]:
        content = [content.as_chat() for content in self.content]
        return dict(
            role=self.role,
            # NOTE(model-specific): Most models accept a list of inputs, mostly when
            # having more than just text. Other however, only accept a string directly.
            # Usually, only models that are text only have this problem.
            # For now, if you want to use one of those models, you have to manually swap
            # the following two lines (and in ChatMessageText above too).
            content=content,
            # content=content[0] if len(content) == 1 else content,
        )

    def get_images(self) -> list[Image.Image]:
        return [msg.image for msg in self.content if isinstance(msg, ChatMessageImage)]
