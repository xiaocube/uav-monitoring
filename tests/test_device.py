from __future__ import annotations

import pytest

from skyguard.core.exceptions import DeviceError
from skyguard.utils.device import detect_device, resolve_torch_device


def test_detect_auto_returns_valid_device():
    info = detect_device("auto")
    assert info.selected in {"cpu", "cuda", "mps"}
    assert info.platform
    assert info.python_version
    assert isinstance(info.cuda_available, bool)
    assert isinstance(info.mps_available, bool)


def test_detect_rejects_unknown():
    with pytest.raises(DeviceError):
        detect_device("tpu")


def test_detect_cpu_always_works():
    info = detect_device("cpu")
    assert info.selected == "cpu"


@pytest.mark.parametrize("req", ["cpu", "auto"])
def test_resolve_returns_torch_device(req):
    dev = resolve_torch_device(req)
    import torch

    assert isinstance(dev, torch.device)


def test_requested_cuda_without_backend_errors(monkeypatch):
    import skyguard.utils.device as d

    monkeypatch.setattr(d, "_detect_torch_backends", lambda: (False, False, "fake"))
    with pytest.raises(DeviceError):
        detect_device("cuda")


def test_requested_mps_without_backend_errors(monkeypatch):
    import skyguard.utils.device as d

    monkeypatch.setattr(d, "_detect_torch_backends", lambda: (False, False, "fake"))
    with pytest.raises(DeviceError):
        detect_device("mps")
