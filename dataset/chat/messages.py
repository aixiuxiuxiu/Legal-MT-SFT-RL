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
        return dict(type=self.type, text=self.text)


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
        return dict(
            role=self.role,
            content=[content.as_chat() for content in self.content],
        )

    def get_images(self) -> list[Image.Image]:
        return [msg.image for msg in self.content if isinstance(msg, ChatMessageImage)]
