"""Minimal replay buffer implementations for testing."""

import torch
from tensordict import TensorDict


class MiniBuffer:
    """Minimal replay buffer that stores and returns the last batch.

    Useful for tests that need to bypass warmup and just need basic
    extend/sample functionality without full replay buffer overhead.
    """

    def __init__(self, max_size: int = 1):
        """Initialize minimal replay buffer.

        Args:
            max_size: Maximum number of items the buffer can hold
        """
        self._max_size = max_size
        self._len = 0
        self._last: TensorDict | None = None

    @property
    def storage(self):
        """Fake storage object with max_size and __len__."""
        buffer = self

        class _Storage:
            @property
            def max_size(self):
                return buffer._max_size

            def __len__(self):
                return buffer._len

        return _Storage()

    def __len__(self):
        """Get the current number of items in the buffer."""
        return self._len

    def extend(self, batch: torch.Tensor):
        """Store batch (squeezed) and return indices."""
        self._last = batch.squeeze(0)
        self._len = min(self._len + 1, self._max_size)
        return [0]

    def sample(self, batch_size: int, return_info: bool = False) -> torch.Tensor:
        """Return last stored batch."""
        if self._last is None:
            raise RuntimeError("MiniBuffer.sample() called before any data was added")
        if return_info:
            # Some trainer code expects PER-style info["index"] to have the same
            # leading dimension as the sampled batch.
            try:
                n = int(self._last.batch_size[0])
            except Exception:
                try:
                    n = int(self._last.shape[0])
                except Exception:
                    n = int(batch_size)

            device = getattr(self._last, "device", None)
            if device is None:
                try:
                    # tensordict.device can be None even if underlying tensors are
                    # on CUDA. Find the first tensor leaf and use its device.
                    stack = [self._last]
                    while stack:
                        x = stack.pop()
                        if isinstance(x, torch.Tensor):
                            device = x.device
                            break
                        if hasattr(x, "values"):
                            try:
                                stack.extend(list(x.values()))
                            except Exception:
                                pass
                except Exception:
                    device = None

            return self._last, {"index": torch.arange(n, device=device)}
        return self._last

    def update_priority(self, indices, priorities):
        """No-op for compatibility with PER interfaces."""
        pass


class TrackingMiniBuffer(MiniBuffer):
    """MiniBuffer that tracks update_priority calls for testing."""

    def __init__(self, max_size: int = 1):
        """Initialize tracking mini buffer.

        Args:
            max_size: Maximum size of the buffer
        """
        super().__init__(max_size)
        self.priority_calls: list[tuple] = []

    def update_priority(self, indices, priorities):
        """Record priority update calls."""
        self.priority_calls.append((indices, priorities.detach().cpu()))
