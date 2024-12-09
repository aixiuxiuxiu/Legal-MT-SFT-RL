from typing import Literal, Self

import torch
from torch._prims_common import DeviceLikeType


def has_tensor_cores() -> bool:
    """
    Check whether the GPU has Tensor cores and therefore supports FP16.
    This is the case for CUDA compute capability >= 7.0
    """
    if not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability()
    return major >= 7


MixedPrecisionChoice = Literal["fp16", "bf16", "auto", "none"]


def get_mixed_precision_dtype(
    choice: MixedPrecisionChoice | None = "auto",
) -> torch.dtype | None:
    """
    Get the mixed precision type based on the selected choice.

    If None or "none" is given, this returns None.
    """
    if choice is None or choice == "none":
        return None
    elif choice == "bf16":
        assert torch.cuda.is_bf16_supported(), "bf16 is not supported by the hardware"
        return torch.bfloat16
    elif choice == "fp16":
        assert has_tensor_cores(), "fp16 is not supported by the hardware"
        return torch.float16
    else:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        elif has_tensor_cores():
            return torch.float16
        else:
            raise RuntimeError("Neither bf16 nor fp16 are supported by the hardware")


def get_best_device() -> torch.device:
    """
    Get the best possible device for the training.
    This is usually CUDA, if it's supported, or MPS for Macs.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


DeviceChoice = Literal["auto"] | torch.device


class HardwareConfig:
    def __init__(
        self,
        device: DeviceChoice = "auto",
        mixed_precision: MixedPrecisionChoice | None = None,
    ):
        self.device = get_best_device() if device == "auto" else torch.device(device)
        self.mixed_precision = get_mixed_precision_dtype(mixed_precision)
        self.grad_scaler = self.mixed_precision and torch.amp.GradScaler(  # pyright: ignore[reportAttributeAccessIssue]
            self.device.type
        )

    def to(self, device: DeviceLikeType) -> Self:
        self.device = torch.device(device)
        return self

    def autocast(self) -> torch.autocast:
        """
        Run it in mixed precision with the configured dtype.

        If the hardware does not support it or it mixed precision is disable, the
        autocast will be disabled.
        """
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.mixed_precision,
            enabled=self.mixed_precision is not None,
        )

    def is_cuda(self) -> bool:
        return self.device.type == "cuda"
