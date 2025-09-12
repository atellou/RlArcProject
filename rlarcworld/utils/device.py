import torch


def enable_cuda(amp_enabled: bool = True, use_checkpointing: bool = False):
    """
    Configures CUDA settings for AMP and gradient checkpointing.

    Args:
        amp_enabled (bool, optional): Flag to indicate whether to use AMP. Defaults to True.
        use_checkpointing (bool, optional): Flag to indicate whether to use gradient checkpointing. Defaults to False.

    Returns:
        dict: Configuration dictionary with keys 'amp_enabled', 'use_checkpointing', and 'device'.
    """
    amp_available = (
        torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 7
    )
    gradient_checkpointing_enabled = amp_available

    return {
        "amp_enabled": amp_enabled and amp_available,
        "use_checkpointing": use_checkpointing and gradient_checkpointing_enabled,
        "device": "cuda" if torch.cuda.is_available() and amp_enabled else "cpu",
    }
