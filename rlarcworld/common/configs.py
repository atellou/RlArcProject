"""Common configurations."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class NStepMinVMaxConfig:
    """Neural network configuration.

    Attributes:
        n_steps (int): Number of steps for n-step returns.
        pixel_wise_min_v (Optional[float]): Size of the embedding.
        pixel_wise_max_v (Optional[float]): Size of the embedding.
        binary_min_v (Optional[float]): Size of the embedding.
        binary_max_v (Optional[float]): Size of the embedding.
    """

    n_steps: int = 1
    pixel_wise_min_v: Optional[float] = None
    pixel_wise_max_v: Optional[float] = None
    binary_min_v: Optional[float] = None
    binary_max_v: Optional[float] = None

    def get_pixel_wise_minmax(self) -> dict[str, float] | None:
        """Get the pixel-wise minimum value and maximum value."""
        if self.pixel_wise_min_v is None or self.pixel_wise_max_v is None:
            return None
        return {"min": self.pixel_wise_min_v, "max": self.pixel_wise_max_v}

    def get_binary_minmax(self) -> dict[str, float] | None:
        """Get the binary minimum value and maximum value."""
        if self.binary_min_v is None or self.binary_max_v is None:
            return None
        return {"min": self.binary_min_v, "max": self.binary_max_v}
