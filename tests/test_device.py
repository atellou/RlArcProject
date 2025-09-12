import pytest

from rlarcworld.utils.device import enable_cuda


class DummyCuda:
    def __init__(self, available: bool, major_cap: int | None = None):
        self._available = available
        self._major = major_cap

    def is_available(self):
        return self._available

    def get_device_count(self, index):
        return self._major

    def get_device_name(self, index):
        return "dummy"

    def get_device_capability(self, index):
        # Return (major, minor) when available; mimic torch API
        if not self._available:
            raise RuntimeError("CUDA not available")
        major = 0 if self._major is None else self._major
        return (major, 0)


@pytest.mark.parametrize(
    "available,major,amp_in,use_ckpt_in,exp",
    [
        # No CUDA => CPU, no AMP, no checkpointing
        (
            False,
            None,
            True,
            True,
            {"amp_enabled": False, "use_checkpointing": False, "device": "cpu"},
        ),
        # CUDA with old capability (<7) => treated as not AMP-available
        (
            True,
            6,
            True,
            True,
            {"amp_enabled": False, "use_checkpointing": False, "device": "cuda"},
        ),
        # CUDA with capability >=7, AMP enabled, checkpointing on
        (
            True,
            7,
            True,
            True,
            {"amp_enabled": True, "use_checkpointing": True, "device": "cuda"},
        ),
        # CUDA with capability >=7 but amp_enabled=False => device stays cpu; ckpt still allowed by function logic when requested
        (
            True,
            8,
            False,
            True,
            {"amp_enabled": False, "use_checkpointing": True, "device": "cuda"},
        ),
    ],
)
def test_enable_cuda(monkeypatch, available, major, amp_in, use_ckpt_in, exp):
    # Monkeypatch torch.cuda methods
    import torch

    dummy = DummyCuda(available=available, major_cap=major)
    monkeypatch.setattr(torch, "cuda", dummy, raising=True)

    cfg = enable_cuda(amp_enabled=amp_in, use_checkpointing=use_ckpt_in)

    assert cfg["amp_enabled"] is exp["amp_enabled"]
    assert cfg["use_checkpointing"] is exp["use_checkpointing"]
    assert cfg["device"] == exp["device"]
