"""CUDA/AMP device configuration helpers."""

import logging
from typing import TypedDict

import torch

logger = logging.getLogger(__name__)


class CudaConfig(TypedDict):
    """Typed configuration returned by ``enable_cuda``.

    Keys:
    - ``amp_enabled``: whether AMP should be enabled
    - ``use_checkpointing``: whether gradient checkpointing should be enabled
    - ``device``: selected device string ("cuda" or "cpu")
    """

    amp_enabled: bool
    use_checkpointing: bool
    device: str


def enable_cuda(
    amp_enabled: bool = True, use_checkpointing: bool = False
) -> CudaConfig:
    """Configures CUDA settings for AMP and gradient checkpointing.

    Args:
        amp_enabled (bool, optional): Flag to indicate whether to use AMP.
            Defaults to True.
        use_checkpointing (bool, optional): Flag to indicate whether to use
            gradient checkpointing. Defaults to False.

    Returns:
        CudaConfig: Configuration with keys 'amp_enabled', 'use_checkpointing',
            and 'device'.
    """
    cuda_available = torch.cuda.is_available()
    # Only query device properties if CUDA is available
    if cuda_available:
        try:
            capability_major = torch.cuda.get_device_capability(0)[0]
            device_name = torch.cuda.get_device_name(0)
        except Exception:  # pragma: no cover - defensive: some backends may differ
            capability_major = 0
            device_name = "cuda"
    else:
        capability_major = 0
        device_name = "cpu"

    amp_available = cuda_available and capability_major >= 7
    logger.info(
        "CUDA available: %s, AMP available: %s, Device: %s, Capability: %s",
        cuda_available,
        amp_available,
        device_name,
        capability_major,
    )
    gradient_checkpointing_enabled = amp_available
    logger.info(
        "Gradient checkpointing enabled: %s, use_checkpointing: %s",
        gradient_checkpointing_enabled,
        use_checkpointing,
    )

    return {
        "amp_enabled": amp_enabled and amp_available,
        "use_checkpointing": use_checkpointing and gradient_checkpointing_enabled,
        "device": "cuda" if cuda_available else "cpu",
    }
