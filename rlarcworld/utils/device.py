"""CUDA/AMP device configuration helpers."""

import importlib.util
import logging
from typing import TypedDict

import torch

logger = logging.getLogger(__name__)

TORCH_XLA_AVAILABLE = importlib.util.find_spec("torch_xla") is not None

if TORCH_XLA_AVAILABLE:
    import torch_xla
else:
    xm = None


class CudaConfig(TypedDict):
    """Typed configuration returned by ``enable_cuda``.

    Keys:
    - ``amp_enabled``: whether Automatic Mixed Precision should be enabled
    - ``use_checkpointing``: whether gradient checkpointing should be enabled
    - ``device``: selected device string ("cuda" or "cpu")
    """

    amp_enabled: bool
    use_checkpointing: bool
    device: str


def enable_cuda(
    amp_enabled: bool = True,
    use_checkpointing: bool = False,
    device_index: int | None = None,
) -> CudaConfig:
    """Configures CUDA settings for AMP and gradient checkpointing.

    Args:
        amp_enabled (bool, optional):
            Flag to indicate whether to use Automatic Mixed Precision.
            Defaults to True.
        use_checkpointing (bool, optional):
            Flag to indicate whether to use gradient checkpointing.
            Defaults to False.
        device_index (int | None, optional):
            If provided, select this CUDA device index (e.g. LOCAL_RANK under
            torchrun). Defaults to None.

    Returns:
        CudaConfig: Configuration with keys 'amp_enabled', 'use_checkpointing',
            and 'device'.
    """
    cuda_available = torch.cuda.is_available()
    if cuda_available and device_index is not None:
        # Ensure we query and use the intended device (e.g. LOCAL_RANK under torchrun)
        try:
            torch.cuda.set_device(device_index)
        except Exception:  # pragma: no cover
            pass

    if TORCH_XLA_AVAILABLE:
        device_name = str(torch_xla.device())
        capability_major = 7
        device_type = "XLA"
    # Only query device properties if CUDA is available
    elif cuda_available:
        try:
            idx = int(device_index) if device_index is not None else 0
            capability_major = torch.cuda.get_device_capability(idx)[0]
            device_name = torch.cuda.get_device_name(idx)
        except Exception:  # pragma: no cover - defensive: some backends may differ
            capability_major = 0
            device_name = "cuda"
        device_type = "CUDA"
    else:
        capability_major = 0
        device_name = "cpu"
        device_type = "CPU"

    amp_available = capability_major >= 7
    logger.info(
        "Device type: %s, AMP available: %s, Device: %s, Capability: %s",
        device_type,
        amp_available,
        device_name,
        capability_major,
    )
    logger.info(
        "Gradient checkpointing enabled: %s, use_checkpointing: %s",
        amp_available,
        use_checkpointing,
    )

    return {
        "amp_enabled": amp_enabled and amp_available,
        "use_checkpointing": use_checkpointing and amp_available,
        "device": (
            f"cuda:{int(device_index)}"
            if cuda_available and device_index is not None
            else "cuda"
        )
        if cuda_available
        else "cpu",
    }


class AmpScaler:
    """Small wrapper that makes AMP usage optional and ergonomic.

    Can be used as a ctx manager to enter autocast, and exposes scaler helpers
    for training when AMP is enabled. If AMP is disabled, all methods are no-ops.
    """

    def __init__(self, amp_enabled: bool, device: str | torch.device) -> None:
        """Initialize AMP scaler and autocast ctx if enabled.

        Args:
            amp_enabled: Whether AMP should be enabled.
            device: Target device ("cuda"/"cpu" or torch.device) used to select
                the autocast/GradScaler device.
        """
        # Normalize device to string device type
        device_type = device.type if isinstance(device, torch.device) else str(device)
        if amp_enabled and device_type == "cuda":
            # Using torch.amp API (PyTorch 2.x) on CUDA only.
            self.scaler = torch.amp.GradScaler(device=device_type)
            self._cm = torch.amp.autocast(device_type="cuda")
        else:
            self.scaler = None
            self._cm = None

    def __enter__(self):
        """Enter autocast ctx if AMP is enabled."""
        if self._cm is not None:
            return self._cm.__enter__()
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit autocast ctx if AMP is enabled."""
        if self._cm is not None:
            return self._cm.__exit__(exc_type, exc_val, exc_tb)
        return False

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        """Conditionally scale a loss tensor when AMP is enabled."""
        if self.scaler is not None:
            return self.scaler.scale(loss)
        return loss

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """Conditionally step optimizer and update scaler."""
        if self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()

    def unscale(self, optimizer: torch.optim.Optimizer) -> None:
        """Unscale gradients for a given optimizer when AMP is enabled.

        Call this before gradient clipping to ensure clipping happens on unscaled grads.
        """
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)
