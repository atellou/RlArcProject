"""Typed tensor-backed FIFO queue utilities.

This module defines `TorchQueue`, a thin tensor subclass that maintains a
bounded queue along a specified dimension. It provides convenient `push` and
`pop` operations while preserving device/dtype.
"""

import logging

import torch

logger = logging.getLogger(__name__)


class TorchQueue(torch.Tensor):
    """Tensor subclass implementing a bounded FIFO queue along one dimension.

    The queue is represented as a tensor and stores a fixed number of items
    along the configured `queue_dim`. When pushing beyond capacity, the oldest
    entries are discarded to maintain the maximum `queue_size`.
    """

    _queue_size: int
    _queue_dim: int

    def __new__(
        cls,
        data: torch.Tensor,
        queue_size: int,
        queue_dim: int = 0,
        *args,
        **kwargs,
    ):
        """Creates a new TorchQueue instance.

        Args:
            data (torch.Tensor): The data to be stored in the queue.
            queue_size (int): The maximum size of the queue.
            queue_dim (int, optional): The dimension along which the queue is
                stored. Defaults to 0.
            *args: Additional arguments to pass to the parent class.
            **kwargs: Additional keyword arguments to pass to the parent class.

        Returns:
            TorchQueue: A new TorchQueue instance.
        """
        obj = super().__new__(cls, data, *args, **kwargs)
        obj._attrs = {"queue_size": queue_size, "queue_dim": queue_dim}
        obj._queue_size = queue_size
        obj._queue_dim = queue_dim
        return obj

    def __init__(self, data: torch.Tensor, queue_size: int, queue_dim: int = 0):
        """Initializes a new TorchQueue instance.

        Args:
            data (torch.Tensor): The data to be stored in the queue.
            queue_size (int): The maximum size of the queue.
            queue_dim (int, optional): The dimension along which the queue is
                stored. Defaults to 0.

        Raises:
            AssertionError: If queue_size is not a positive integer.
            AssertionError: If queue_dim is not an integer.
            AssertionError: If the size of the queue dimension is greater than
                the queue size.
        """
        assert (
            isinstance(queue_size, int) and queue_size > 0
        ), "queue_size must be a positive integer."
        assert isinstance(queue_dim, int), "queue_dim must be an integer."
        assert (
            data.shape[queue_dim] <= queue_size
        ), "Data shape is {}, queue_size is {}".format(
            data.shape[queue_dim], queue_size
        )
        self._queue_size = queue_size
        self._queue_dim = queue_dim

    def clone(self, *args, **kwargs) -> "TorchQueue":
        """Clones the TorchQueue instance."""
        return TorchQueue(
            super().clone(*args, **kwargs),
            queue_size=self._queue_size,
            queue_dim=self._queue_dim,
        )

    def to(self, *args, **kwargs) -> "TorchQueue":
        """Return a copy on the specified device/dtype preserving queue attrs.

        Mirrors `torch.Tensor.to` but ensures the returned value is a
        `TorchQueue` with the same `queue_size` and `queue_dim` metadata.
        """
        new_obj = super().to(*args, **kwargs)
        if new_obj is self:
            # No-op device move, keep self
            return self

        if isinstance(new_obj, TorchQueue):
            if hasattr(self, "_queue_size") and not hasattr(new_obj, "_queue_size"):
                new_obj._queue_size = self._queue_size
            if hasattr(self, "_queue_dim") and not hasattr(new_obj, "_queue_dim"):
                new_obj._queue_dim = self._queue_dim
            return new_obj

        if not hasattr(self, "_queue_size") or not hasattr(self, "_queue_dim"):
            logger.debug(
                "TorchQueue.to called on object without queue metadata; "
                "returning base tensor"
            )
            return new_obj

        return TorchQueue(
            new_obj, queue_size=self._queue_size, queue_dim=self._queue_dim
        )

    @property
    def queue_size(self) -> int:
        """Returns the maximum size of the queue."""
        return self._queue_size

    @property
    def queue_dim(self) -> int:
        """Returns the dimension along which the queue is stored."""
        return self._queue_dim

    @property
    def is_full(self) -> bool:
        """Returns True if the queue is full, False otherwise."""
        return self.shape[self._queue_dim] == self._queue_size

    def push(self, item: torch.Tensor) -> "TorchQueue":
        """Add an item; if full, drop the oldest element along `queue_dim`.

        Args:
            item (torch.Tensor): The item to be added to the queue.

        Returns:
            torch.Tensor: The item that was added to the queue.
        """
        item = torch.cat([self, item], dim=self._queue_dim)
        if item.shape[self._queue_dim] > self._queue_size:
            item = self.__correct_queue_size(item)
            assert item.shape[self._queue_dim] == self._queue_size
        assert (
            item.shape[self._queue_dim] <= self._queue_size
        ), "Item shape is {}, queue_size is {}".format(
            item.shape[self._queue_dim], self._queue_size
        )
        return TorchQueue(item, queue_size=self._queue_size, queue_dim=self._queue_dim)

    def __correct_queue_size(self, item: torch.Tensor = None) -> "TorchQueue":
        """Correct the size of the queue along the specified dimension when full."""
        if item is None:
            item = self
        # Lazily initialize reversed indices if missing (can happen if object was
        # created or moved by PyTorch internals without calling our __init__).
        if not hasattr(self, "_TorchQueue__reversed_indices"):
            self.__reversed_indices = torch.arange(-1 - self._queue_size, 0).to(
                self.device
            )
        return TorchQueue(
            torch.index_select(
                item,
                self._queue_dim,
                self.__reversed_indices[1:] + self._queue_size + 1,
            ),
            queue_size=self._queue_size,
            queue_dim=self._queue_dim,
        )

    def pop(self) -> tuple[torch.Tensor, "TorchQueue"]:
        """Pops the oldest elements along the specified dimension.

        Returns:
            torch.Tensor: The oldest item in the queue.
            TorchQueue: The updated queue without the oldest item.
        """
        # Build indices to slice along queue_dim
        indices_oldest: list[object] = [slice(None)] * self.ndim
        indices_oldest[self._queue_dim] = 0
        oldest = self[tuple(indices_oldest)]

        indices_rest: list[object] = [slice(None)] * self.ndim
        indices_rest[self._queue_dim] = slice(1, None)
        rest = self[tuple(indices_rest)]
        return oldest, TorchQueue(
            rest, queue_size=self._queue_size, queue_dim=self._queue_dim
        )

    def __getitem__(self, idx) -> torch.Tensor:
        """Return a base torch.Tensor, not TorchQueue.

        Args:
            idx: The index to be used for indexing.

        Returns:
            torch.Tensor: The indexed tensor.
        """
        return torch.Tensor(super().__getitem__(idx))


# Register for torch serialization safe globals
torch.serialization.safe_globals([TorchQueue])
