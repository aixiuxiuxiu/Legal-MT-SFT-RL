import torch.nn as nn
from transformers import PreTrainedTokenizerBase


def unwrap_tokeniser(processor: PreTrainedTokenizerBase) -> PreTrainedTokenizerBase:
    tokeniser = (
        # FIXME: Fix this type annotation for Image/Text processors.
        # But this is at least safe.
        processor.tokenizer
        if hasattr(processor, "tokenizer")
        and isinstance(processor.tokenizer, PreTrainedTokenizerBase)
        else processor
    )
    return tokeniser


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Unwraps a model to the core model, which can be across multiple layers with
    wrappers such as DistributedDataParallel.
    """
    while hasattr(model, "module") and isinstance(model.module, nn.Module):
        model = model.module
    return model
