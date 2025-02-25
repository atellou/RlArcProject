import os
import torch
import logging
from pythonjsonlogger import jsonlogger


def enable_cuda(use_amp=True, use_checkpointing=False):
    """
    Configures CUDA settings for AMP and gradient checkpointing.

    This function checks the availability of CUDA and its device capability
    to determine if Automatic Mixed Precision (AMP) and gradient checkpointing
    can be enabled. It returns a configuration dictionary indicating whether
    AMP and checkpointing should be used based on the provided flags and
    hardware capabilities.

    Args:
        use_amp (bool, optional): Flag to indicate whether to use AMP. Defaults to True.
        use_checkpointing (bool, optional): Flag to indicate whether to use gradient checkpointing. Defaults to True.

    Returns:
        dict: Configuration dictionary with keys 'use_amp' and 'use_checkpointing'
        indicating the enabled settings.
    """

    amp_available = (
        torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 7
    )
    gradient_checkpointing_enabled = amp_available  # Use only if CUDA is available

    config = {
        "use_amp": use_amp and amp_available,
        "use_checkpointing": use_checkpointing and gradient_checkpointing_enabled,
        "device": "cuda" if torch.cuda.is_available() and use_amp else "cpu",
    }

    return config


def get_nested_ref(dictionary: dict, key: str, sep: str = "/", default=None):
    if sep in key:
        first_key, other_keys = key.split(sep, 1)
        sub_dict = dictionary.setdefault(first_key, {})
        return get_nested_ref(sub_dict, other_keys, sep, default=default)

    if key not in dictionary:
        dictionary[key] = default() if callable(default) else default
    return dictionary, key


def categorical_projection(
    next_q_dist,
    rewards,
    terminated,
    gamma,
    z_atoms,
    v_min,
    v_max,
    num_atoms,
    apply_softmax=True,
    n_steps=1,
):
    """
    Projects the target distribution using the Bellman update.

    Args:
        next_q_dist (torch.Tensor): Next state Q-distribution (batch_size, num_atoms).
        rewards (torch.Tensor): Rewards from the batch (batch_size, 1).
        terminated (torch.Tensor): Done flags from the batch (batch_size, 1).
        gamma (float): Discount factor.
        z_atoms (torch.Tensor): Z reference distribution Atom values (num_atoms, 1).
        v_min (float): Minimum value for value distribution.
        v_max (float): Maximum value for value distribution.
        num_atoms (int): Number of atoms in the distribution.
        apply_softmax (bool): Whether to apply softmax to the projected distribution.
        n_steps (int): Number of steps in the Bellman update.

    Returns:
        torch.Tensor: Projected target distribution (batch_size, num_atoms).
    """
    delta_z = (v_max - v_min) / (num_atoms - 1)

    # Compute the target distribution support
    tz = rewards + (gamma**n_steps) * (1 - terminated) * z_atoms
    tz = tz.clamp(v_min, v_max)

    # Map values to categorical bins
    b = (tz - v_min) / delta_z
    l, u = b.floor().long(), b.ceil().long()
    l = l.clamp(0, num_atoms - 1)
    u = u.clamp(0, num_atoms - 1)

    # Distribute probability mass
    projected_dist = torch.zeros_like(next_q_dist).to(next_q_dist.device)
    projected_dist.scatter_add_(dim=-1, index=l, src=next_q_dist * (u.float() - b))
    projected_dist.scatter_add_(dim=-1, index=u, src=next_q_dist * (b - l.float()))

    return torch.softmax(projected_dist, dim=-1) if apply_softmax else projected_dist


class TorchQueue(torch.Tensor):
    _queue_size: tuple
    _queue_dim: int

    def __new__(cls, data, queue_size: int, queue_dim: int = 0, *args, **kwargs):
        obj = super().__new__(cls, data, *args, **kwargs)
        obj._attrs = {"queue_size": queue_size, "queue_dim": queue_dim}
        return obj

    def __init__(self, data, queue_size: int, queue_dim: int = 0):
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

    def clone(self, *args, **kwargs):
        return TorchQueue(
            super().clone(*args, **kwargs),
            queue_size=self._queue_size,
            queue_dim=self._queue_dim,
        )

    def to(self, *args, **kwargs):
        new_obj = super().to(*args, **kwargs)
        if new_obj is self:
            return self
        return TorchQueue(new_obj, queue_size=self.queue_size, queue_dim=self.queue_dim)

    @property
    def queue_size(self):
        return self._queue_size

    @property
    def queue_dim(self):
        return self._queue_dim

    @property
    def is_full(self):
        return self.shape[self._queue_dim] == self._queue_size

    def push(self, item):
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

    def __correct_queue_size(self, item=None):
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

    def pop(self):
        """
        Pops the oldest elements along the specified dimension.

        Returns:
            torch.Tensor: The oldest item in the queue.
            TorchQueue: The updated queue without the oldest item.
        """
        return self[0], TorchQueue(
            self[1:], queue_size=self._queue_size, queue_dim=self._queue_dim
        )


torch.serialization.safe_globals([TorchQueue])


class BetaScheduler:
    def __init__(self, start, end, steps):
        self.start = start
        self.end = end
        self.steps = steps
        self.internal_step = 0

    def step(
        self,
    ):
        self.internal_step += 1
        return min(
            self.end,
            self.start + (self.end - self.start) * self.internal_step / self.end,
        )


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Formats log lines in JSON."""

    def process_log_record(self, log_record):
        """Modifies fields in the log_record to match Cloud Logging's expectations."""
        log_record["severity"] = log_record["levelname"]
        log_record["timestampSeconds"] = int(log_record["created"])
        log_record["timestampNanos"] = int(
            (log_record["created"] % 1) * 1000 * 1000 * 1000
        )

        return log_record


def configure_logger(level="WARNING"):
    """Configures python logger to format logs as JSON."""
    formatter = CustomJsonFormatter(
        "%(name)s|%(levelname)s|%(message)s|%(created)f" "|%(lineno)d|%(pathname)s",
        "%Y-%m-%dT%H:%M:%S",
    )
    root_logger = logging.getLogger()
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, level.upper()))
