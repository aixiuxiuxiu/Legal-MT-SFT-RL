import os

from peft.peft_model import PeftModel
from transformers import PreTrainedTokenizerBase
from unsloth import FastVisionModel


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
    Creates the Vision Languge Model (VLM) with LoRA adapters and the corresponding
    processor (image and text tokeniser).

    Args:
        name_or_path (str): Name or path to the model
        rank (int): LoRA rank [Default: 16]
        alpha (int): LoRA scaling factor, recommended α = 2·r [Default: 32]
        freeze_vit (bool): Whether to freeze the Vision Transformer (ViT)
            [Default: True]
        freeze_llm (bool): Whether to freeze the Language Model (LLM).
            [Default: False]
        pad_token (str, optional): Set the pad token to the given token. Might be
            necessary for some models, as they sometimes don't define a padding token at
            all, or use <eos> as the padding token, which prevents the model from
            learning when to stop.
        device_map (dict | str, optional): Mapping of the parts of the models to the
            respective devices, as defined by accelerate. May also be a string, such as
            "auto" being very common.

    Returns:
        model (PeftModel): Pre-trained VLM with LORA adapters.
        processor (PreTrainedTokenizerBase): Processor of the model (image / tokeniser).
    """
    assert freeze_vit, (
        "Unsloth is currently broken for vision fine-tuning. "
        "Hence freeze_vit=True is required"
    )
    model, processor = FastVisionModel.from_pretrained(
        str(name_or_path),
        load_in_4bit=True,
        device_map=device_map,  # pyright: ignore[reportArgumentType]
    )
    model = FastVisionModel.get_peft_model(
        model,
        r=rank,
        lora_alpha=alpha,
        finetune_vision_layers=not freeze_vit,
        finetune_language_layers=not freeze_llm,
    )
    if pad_token:
        processor.tokenizer.pad_token = pad_token
        model.config.pad_token_id = processor.tokenizer.pad_token_id
    return model, processor


def load_model_for_inference(
    name_or_path: str | os.PathLike,
    pad_token: str | None = None,
    device_map: dict | str | None = None,
) -> tuple[PeftModel, PreTrainedTokenizerBase]:
    """
    Loads the trained Vision Languge Model (VLM) including LoRA adapters if it was
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
        model (PeftModel): Trained VLM including possible LORA adapters.
        processor (PreTrainedTokenizerBase): Processor of the model (image / tokeniser).
    """
    model, processor = FastVisionModel.from_pretrained(
        str(name_or_path),
        load_in_4bit=True,
        device_map=device_map,  # pyright: ignore[reportArgumentType]
    )
    if pad_token:
        processor.tokenizer.pad_token = pad_token
        model.config.pad_token_id = processor.tokenizer.pad_token_id
    model = model.eval()
    FastVisionModel.for_inference(model)
    # Left side padding for inference
    processor.tokenizer.padding_side = "left"
    return model, processor
