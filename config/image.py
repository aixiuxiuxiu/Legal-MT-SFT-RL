from dataclasses import dataclass

from dataset.resize import ResizeWhen

# HACK: simple-parsing does not support types defined with the `type` keyword, which are
# lazily evaluated, so in order to use it as the type of the args, the right-hand side
# needs to be used (i.e. the actual value should needs to be extracted)
ResizeWhenTy = ResizeWhen.__value__


@dataclass
class ImageConfig:
    """
    Configuration related to the images
    """

    # Size to which the longer side of the images are resized. The other side is resized
    # accordingly such that the aspect ratio remains unchanged.
    size: int | None = None
    # When to resize the images. This only has an effect if `--size` is specified as
    # well. Use "when-smaller" or "when-larger" if you want to ensure a minimum or
    # maximum size, respectively.
    resize: ResizeWhenTy = "always"
