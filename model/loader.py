import os

from peft.peft_model import PeftModel
from transformers import PreTrainedTokenizerBase
from unsloth import FastModel

from model.utils import unwrap_tokeniser


# TODO: Figure out type for Processor, as there does not seem to be any base class that
# would be correct since it uses a Mixin and there are some definitions missing.
def create_lora_model(
    name_or_path: str | os.PathLike,
    rank: int = 16,
    alpha: int = 32,
    freeze_vit: bool = True,
    freeze_llm: bool = False,
    pad_token: str | None = None,
    device_map: dict | str | None = None,
) -> tuple[PeftModel, PreTrainedTokenizerBase]:
    """
    Creates the (Vision) Languge Model with LoRA adapters and the corresponding
    processor (image and text tokeniser).

    Args:
        name_or_path (str): Name or path to the model
        rank (int): LoRA rank [Default: 16]
        alpha (int): LoRA scaling factor, recommended α = 2·r [Default: 32]
        freeze_vit (bool): Whether to freeze the Vision Transformer (ViT)
            Only applies to Vision Language Models (VLM).
            [Default: True]
        freeze_llm (bool): Whether to freeze the Language Model (LLM).
            Only applies to Vision Language Models (VLM).
            [Default: False]
        pad_token (str, optional): Set the pad token to the given token. Might be
            necessary for some models, as they sometimes don't define a padding token at
            all, or use <eos> as the padding token, which prevents the model from
            learning when to stop.
        device_map (dict | str, optional): Mapping of the parts of the models to the
            respective devices, as defined by accelerate. May also be a string, such as
            "auto" being very common.

    Returns:
        model (PeftModel): Pre-trained (V)LM with LORA adapters.
        processor (PreTrainedTokenizerBase): Processor of the model (image / tokeniser).
    """
    assert freeze_vit, (
        "Unsloth is currently broken for vision fine-tuning. "
        "Hence freeze_vit=True is required"
    )
    model, processor = FastModel.from_pretrained(
        str(name_or_path),
        load_in_4bit=True,
        device_map=device_map,  # pyright: ignore[reportArgumentType]
    )
    # Unsloth raises an error when it already has LoRA adapters, but if you want to
    # continue from a checkpoint with LoRA adapters, you would still load it first.
    # So if that is the case, don't add new adapters.
    if not isinstance(model, PeftModel):
        model = FastModel.get_peft_model(
            model,
            r=rank,
            lora_alpha=alpha,
            finetune_vision_layers=not freeze_vit,
            finetune_language_layers=not freeze_llm,
        )
    if pad_token:
        tokeniser = unwrap_tokeniser(processor)
        tokeniser.pad_token = pad_token
    return model, processor


def load_model_for_inference(
    name_or_path: str | os.PathLike,
    pad_token: str | None = None,
    device_map: dict | str | None = None,
) -> tuple[PeftModel, PreTrainedTokenizerBase]:
    """
    Loads the trained (Vision) Languge Model including LoRA adapters if it was
    trained with LoRA and the corresponding processor (image and text tokeniser).
    This prepares the model and processor for inference rather than training,
    e.g. enabling left side padding for the tokeniser.

    Args:
        name_or_path (str): Name or path to the model
        pad_token (str, optional): Set the pad token to the given token. Might be
            necessary for some models, as they sometimes don't define a padding token at
            all, or use <eos> as the padding token, which prevents the model from
            learning when to stop.
            If it was fine-tuned, this should already be fixed and therefore no longer
            necessary.
        device_map (dict | str, optional): Mapping of the parts of the models to the
            respective devices, as defined by accelerate. May also be a string, such as
            "auto" being very common.

    Returns:
        model (PeftModel): Trained (V)LM including possible LORA adapters.
        processor (PreTrainedTokenizerBase): Processor of the model (image / tokeniser).
    """
    model, processor = FastModel.from_pretrained(
        str(name_or_path),
        load_in_4bit=True,
        device_map=device_map,  # pyright: ignore[reportArgumentType]
    )
    tokeniser = unwrap_tokeniser(processor)
    if pad_token:
        tokeniser.pad_token = pad_token
    model = model.eval()
    FastModel.for_inference(model)
    # Left side padding for inference
    tokeniser.padding_side = "left"
    return model, processor
