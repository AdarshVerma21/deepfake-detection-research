import pytest

from src.utils.device import get_runtime_device
from src.utils.errors import DeviceConfigurationError


def test_auto_device_returns_valid_type() -> None:
    runtime = get_runtime_device("auto", mixed_precision=True)
    assert runtime.device.type in {"cpu", "cuda"}


def test_cpu_device_disables_mixed_precision() -> None:
    runtime = get_runtime_device("cpu", mixed_precision=True)
    assert runtime.device.type == "cpu"
    assert runtime.use_mixed_precision is False


def test_invalid_device_raises() -> None:
    with pytest.raises(DeviceConfigurationError):
        get_runtime_device("invalid_device", mixed_precision=True)
