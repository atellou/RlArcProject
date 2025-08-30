import torch
import logging
from typing import Tuple, Any
from pythonjsonlogger import jsonlogger


def enable_cuda(amp_enabled: bool = True, use_checkpointing: bool = False):
    """
    Configures CUDA settings for AMP and gradient checkpointing.

    This function checks the availability of CUDA and its device capability
    to determine if Automatic Mixed Precision (AMP) and gradient checkpointing
    can be enabled. It returns a configuration dictionary indicating whether
    AMP and checkpointing should be used based on the provided flags and
    hardware capabilities.

    Args:
        amp_enabled (bool, optional): Flag to indicate whether to use AMP. Defaults to True.
        use_checkpointing (bool, optional): Flag to indicate whether to use gradient checkpointing. Defaults to True.

    Returns:
        dict: Configuration dictionary with keys 'amp_enabled' and 'use_checkpointing'
        indicating the enabled settings.
    """

    amp_available = (
        torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 7
    )
    gradient_checkpointing_enabled = amp_available  # Use only if CUDA is available

    config = {
        "amp_enabled": amp_enabled and amp_available,
        "use_checkpointing": use_checkpointing and gradient_checkpointing_enabled,
        "device": "cuda" if torch.cuda.is_available() and amp_enabled else "cpu",
    }

    return config


def get_nested_ref(dictionary: dict, key: str, sep: str = "/", default: Any = None):
    """
    Retrieves a nested value from a dictionary using a key with a specified separator.

    Args:
        dictionary (dict): The dictionary to retrieve the value from.
        key (str): The key to retrieve the value from.
        sep (str, optional): The separator to use when splitting the key. Defaults to "/".
        default (any, optional): The default value to return if the key is not found. Defaults to None.

    Returns:
        any: The value retrieved from the dictionary.
    """
    if sep in key:
        first_key, other_keys = key.split(sep, 1)
        sub_dict = dictionary.setdefault(first_key, {})
        return get_nested_ref(sub_dict, other_keys, sep, default=default)

    if key not in dictionary:
        dictionary[key] = default() if callable(default) else default
    return dictionary, key


def categorical_projection(
    next_q_dist: torch.Tensor,
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    gamma: float,
    z_atoms: torch.Tensor,
    v_min: float,
    v_max: float,
    num_atoms: int,
    apply_softmax: bool = True,
    n_steps: int = 1,
) -> torch.Tensor:
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
    bins = (tz - v_min) / delta_z
    lower, upper = bins.floor().long(), bins.ceil().long()
    lower = lower.clamp(0, num_atoms - 1)
    upper = upper.clamp(0, num_atoms - 1)

    # Distribute probability mass
    projected_dist = torch.zeros_like(next_q_dist).to(next_q_dist.device)
    projected_dist.scatter_add_(
        dim=-1, index=lower, src=next_q_dist * (upper.float() - bins)
    )
    projected_dist.scatter_add_(
        dim=-1, index=upper, src=next_q_dist * (bins - lower.float())
    )

    return torch.softmax(projected_dist, dim=-1) if apply_softmax else projected_dist


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


torch.serialization.safe_globals([TorchQueue])


class BetaScheduler:
    def __init__(self, start: float, end: float, steps: int, internal_step: int = 0):
        """
        Initializes a new BetaScheduler instance.

        Args:
            start (float): The initial value of beta.
            end (float): The final value of beta.
            steps (int): The number of steps to reach the final value of beta.
            internal_step (int, optional): The internal step counter. Defaults to 0.
        """
        self.start = start
        self.end = end
        self.steps = steps
        self.internal_step = internal_step
        self.beta = self.start

    def step(self) -> float:
        """
        Steps the scheduler forward by one step.

        Returns:
            float: The updated value of beta.
        """
        self.internal_step += 1
        self.beta = min(
            self.end,
            self.start + (self.end - self.start) * self.internal_step / self.end,
        )
        return self.beta

    def state_dict(self) -> dict:
        """
        Returns the state dictionary of the scheduler.

        Returns:
            dict: The state dictionary of the scheduler.
        """
        return {
            "start": self.start,
            "end": self.end,
            "steps": self.steps,
            "internal_step": self.internal_step,
            "beta": self.beta,
        }

    def load(self, state_dict: dict):
        """
        Loads the state dictionary of the scheduler.

        Args:
            state_dict (dict): The state dictionary of the scheduler.
        """
        self.start = state_dict["start"]
        self.end = state_dict["end"]
        self.steps = state_dict["steps"]
        self.internal_step = state_dict["internal_step"]
        self.beta = state_dict["beta"]


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Formats log lines in JSON."""

    def process_log_record(self, log_record: dict) -> dict:
        """Modifies fields in the log_record to match Cloud Logging's expectations."""
        log_record["severity"] = log_record["levelname"]
        log_record["timestampSeconds"] = int(log_record["created"])
        log_record["timestampNanos"] = int(
            (log_record["created"] % 1) * 1000 * 1000 * 1000
        )

        return log_record


def configure_logger(level: str = "WARNING", json_format: bool = True):
    """Configures python logger to format logs as JSON."""
    if json_format:
        formatter = CustomJsonFormatter(
            "%(name)s|%(levelname)s|%(message)s|%(created)f" "|%(lineno)d|%(pathname)s",
            "%Y-%m-%dT%H:%M:%S",
        )
    else:
        formatter = logging.Formatter(
            "%(lineno)d|%(pathname)s|%(name)s|%(levelname)s:%(message)s"
        )
    root_logger = logging.getLogger()
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, level.upper()))
