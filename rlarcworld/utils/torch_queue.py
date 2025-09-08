from typing import Tuple
import torch


class TorchQueue(torch.Tensor):
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
        """
        Creates a new TorchQueue instance.

        Args:
            data (torch.Tensor): The data to be stored in the queue.
            queue_size (int): The maximum size of the queue.
            queue_dim (int, optional): The dimension along which the queue is stored. Defaults to 0.
            *args: Additional arguments to pass to the parent class.
            **kwargs: Additional keyword arguments to pass to the parent class.

        Returns:
            TorchQueue: A new TorchQueue instance.
        """
        obj = super().__new__(cls, data, *args, **kwargs)
        obj._attrs = {"queue_size": queue_size, "queue_dim": queue_dim}
        return obj

    def __init__(self, data: torch.Tensor, queue_size: int, queue_dim: int = 0):
        """
        Initializes a new TorchQueue instance.

        Args:
            data (torch.Tensor): The data to be stored in the queue.
            queue_size (int): The maximum size of the queue.
            queue_dim (int, optional): The dimension along which the queue is stored. Defaults to 0.

        Raises:
            AssertionError: If queue_size is not a positive integer.
            AssertionError: If queue_dim is not an integer.
            AssertionError: If the size of the queue dimension is greater than the queue size.
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
        self.__reversed_indices = torch.arange(-1 - queue_size, 0).to(self.device)

    def clone(self, *args, **kwargs) -> "TorchQueue":
        """
        Clones the TorchQueue instance.
        """
        return TorchQueue(
            super().clone(*args, **kwargs),
            queue_size=self._queue_size,
            queue_dim=self._queue_dim,
        )

    def to(self, *args, **kwargs) -> "TorchQueue":
        new_obj = super().to(*args, **kwargs)
        if new_obj is self:
            return self
        return TorchQueue(new_obj, queue_size=self.queue_size, queue_dim=self.queue_dim)

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
        """
        Adds an item to the queue. If the queue is full, the oldest item is removed first.

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
        """
        Correct the size of the queue along the specified dimension when full.
        """
        if item is None:
            item = self
        return TorchQueue(
            torch.index_select(
                item,
                self._queue_dim,
                self.__reversed_indices[1:] + self._queue_size + 1,
            ),
            queue_size=self._queue_size,
            queue_dim=self._queue_dim,
        )

    def pop(self) -> Tuple[torch.Tensor, "TorchQueue"]:
        """
        Pops the oldest elements along the specified dimension.

        Returns:
            torch.Tensor: The oldest item in the queue.
            TorchQueue: The updated queue without the oldest item.
        """
        return self[0], TorchQueue(
            self[1:], queue_size=self._queue_size, queue_dim=self._queue_dim
        )


# Register for torch serialization safe globals
torch.serialization.safe_globals([TorchQueue])
