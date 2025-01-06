from dataclasses import dataclass
from typing import Literal

from PIL import Image

type ResizeWhen = Literal["always", "never", "when-smaller", "when-larger"]


def resize_larger_side(
    img: Image.Image, size: int, when: ResizeWhen = "always"
) -> Image.Image:
    """
    Resize images where the large side is resized to the specified size while keeping
    the aspect ratio.

    Args:
        img (Image.Image): Image to resize
        size (int): Size to which the larger side is resized
        resize (Resize): When to resize the image:
            - "when-larger" means that it is only resized if the larger side is larger
                than the size it would be resized to (i.e. smaller ones are untouched
                instead of being up-scaled)
            - "when-smaller" means that it is only resized if the larger side is smaller
                than the size it would be resized to (i.e. larger ones are untouched
                instead of being down-scaled).
            [Default: "always"]


    Returns:
        out (Image.Image): Resized image
    """
    width, height = img.size
    if height >= width:
        new_height = size
        new_width = width * size // height
    else:
        new_width = size
        new_height = height * size // width
    larger_side = max(height, width)
    match when:
        case "never":
            return img
        case "always":
            resize_larger = True
            resize_smaller = True
        case "when-smaller":
            resize_larger = False
            resize_smaller = larger_side < size
        case "when-larger":
            resize_larger = larger_side > size
            resize_smaller = False
    if (new_width > width and resize_smaller) or (new_width < width and resize_larger):
        return img.resize(size=(new_width, new_height))
    return img


@dataclass
class ImageResizer:
    """
    A small helper class to have a default that does not resize the image at all, but
    only when a size is specified. This is different from setting when="never" as the
    default should be "always" but only when a value has been given.
    """

    larger_side: int | None = None
    when: ResizeWhen = "always"

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.larger_side:
            return resize_larger_side(img, size=self.larger_side, when=self.when)
        return img
