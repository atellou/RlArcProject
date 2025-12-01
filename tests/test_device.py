import types

import pytest

from rlarcworld.utils.device import AmpScaler, enable_cuda


class DummyCuda:
    def __init__(self, available: bool, major_cap: int | None = None):
        """Initialize a minimal CUDA stub for testing."""
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
        # CUDA with capability >=7 but amp_enabled=False => device stays cpu;
        # checkpointing still allowed by function logic when requested
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


def test_amp_scaler_cpu_noop():
    import torch

    scaler = AmpScaler(amp_enabled=True, device="cpu")

    # ctx manager should be a no-op on CPU
    with scaler as ctx:
        assert ctx is None

    # scale should return the same tensor when disabled
    t = torch.tensor(1.0)
    assert scaler.scale(t) is t

    # Dummy optimizer to verify step() path without scaler
    class DummyOpt:
        def __init__(self):
            self.stepped = False

        def step(self):
            self.stepped = True

    opt = DummyOpt()
    scaler.step(opt)
    assert opt.stepped is True

    # unscale should be a no-op
    scaler.unscale(opt)  # should not raise


def test_amp_scaler_cuda_enabled(monkeypatch):
    """Validate AmpScaler behavior when CUDA."""
    import torch

    # Create dummy torch.amp namespace
    dummy_amp = types.SimpleNamespace()

    class DummyContext:
        def __init__(self):
            self.entered = False

        def __enter__(self):
            self.entered = True
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.entered = False
            return False

    class DummyGradScaler:
        def __init__(self, device):
            self.device = device
            self.updated = False
            self.unscaled = False

        def scale(self, x):
            # Return a new tensor to signal scaling happened
            return x + 0.0

        def step(self, opt):
            opt.stepped = True

        def update(self):
            self.updated = True

        def unscale_(self, opt):
            self.unscaled = True

    dummy_amp.autocast = lambda device_type: DummyContext()
    dummy_amp.GradScaler = lambda device: DummyGradScaler(device=device)

    # Patch torch.amp and pretend CUDA device is in use
    monkeypatch.setattr(torch, "amp", dummy_amp, raising=True)

    scaler = AmpScaler(amp_enabled=True, device="cuda")

    # ctx manager should enter/exit
    with scaler as ctx:
        assert isinstance(ctx, DummyContext)
        assert ctx.entered is True
    assert ctx.entered is False

    # scale should return a (numerically equal) new tensor
    t = torch.tensor(2.0)
    t2 = scaler.scale(t)
    assert torch.allclose(t, t2)
    assert t2 is not t  # indicates scaling wrapper was used

    # Dummy optimizer to verify step/update and unscale paths
    class DummyOpt:
        def __init__(self):
            self.stepped = False

        def step(self):
            # Should not be called directly when scaler is present
            raise AssertionError("Optimizer.step() should be called via GradScaler")

    opt = DummyOpt()
    scaler.step(opt)
    # Access underlying dummy scaler via the object we created through monkeypatch
    # We cannot access it directly; instead, ensure the effects we expect occurred.
    assert opt.stepped is True

    # unscale should route to GradScaler.unscale_
    scaler.unscale(opt)  # should not raise
