"""Minimal protocol for core algorithms to implement."""

from typing import Any, Iterator, Protocol

import torch
from tensordict import TensorDict


class ReplayBufferProto(Protocol):
    """A minimal protocol for prioritized or uniform replay buffers.

    The implementing buffer may be prioritized or uniform and should expose
    extend/sample/update_priority and an optional ``storage`` attribute.
    """

    def extend(self, batch: torch.Tensor) -> Any:
        """Append a batch of transitions and return indices/handles."""
        ...

    def sample(self, batch_size: int, return_info: bool = False) -> TensorDict:
        """Sample a batch of transitions as a TensorDict."""
        ...

    def update_priority(self, indices: Any, priorities: torch.Tensor) -> None:
        """Update sampling priorities for the given indices."""
        ...

    @property
    def storage(self) -> Any:  # noqa: D401 - property described in class doc
        """Underlying storage implementation (buffer-specific)."""
        ...

    def __len__(self) -> int:
        """Return the number of transitions in the buffer."""
        ...


class ActorCriticCoreProto(Protocol):
    """Minimal protocol a learning core must implement to work with the trainer."""

    actor: torch.nn.Module
    critic: torch.nn.Module
    device: str

    def compute_loss(
        self,
        batch: TensorDict,
        *,
        training: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, TensorDict]:
        """Compute the loss for the actor and critic networks.

        Returns the actor loss, critic loss, and TD error.
        """
        ...

    def update_target_networks(self) -> None:
        """Update the target networks."""
        ...

    def hyperparameters(self) -> dict:
        """Returns the hyperparameters of the algorithm."""
        ...


class RewardWorldModelProto(Protocol):
    """Protocol for RewardWorldModel."""

    def update(self, state: TensorDict, true_reward: TensorDict) -> None:
        """Update the world model with true rewards."""
        ...

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        """Return the parameters of the world model."""
        ...

    def forward(self, state: TensorDict) -> TensorDict:
        """Call the world model."""
        ...

    def state_dict(self) -> dict:
        """Return the state dict."""
        ...

    def load_state_dict(self, state_dict: dict) -> None:
        """Load state dict."""
        ...

    def enable_example_cache(self) -> None:
        """Enable example cache."""
        ...

    def clear_example_cache(self) -> None:
        """Clear example cache."""
        ...
