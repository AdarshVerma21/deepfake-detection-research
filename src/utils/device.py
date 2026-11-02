"""Runtime device and mixed-precision utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.utils.errors import DeviceConfigurationError


@dataclass(frozen=True)
class RuntimeDeviceConfig:
    """Container for selected device and precision support."""

    device: torch.device
    use_mixed_precision: bool


def get_runtime_device(requested: str = "auto", mixed_precision: bool = True) -> RuntimeDeviceConfig:
    """Return resolved device and effective mixed-precision capability."""
    request = requested.lower().strip()
    cuda_available = torch.cuda.is_available()

    if request == "auto":
        device = torch.device("cuda" if cuda_available else "cpu")
    elif request == "cuda":
        if not cuda_available:
            raise DeviceConfigurationError("CUDA requested but not available on this machine.")
        device = torch.device("cuda")
    elif request == "cpu":
        device = torch.device("cpu")
    else:
        raise DeviceConfigurationError(f"Unsupported device option: {requested}")

    use_amp = bool(mixed_precision and device.type == "cuda")
    return RuntimeDeviceConfig(device=device, use_mixed_precision=use_amp)
