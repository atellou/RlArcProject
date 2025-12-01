"""Common types for the agent."""

from __future__ import annotations

from typing import NotRequired, TypedDict

import torch

from rlarcworld.common.enums import ActionKeys, StateKeys


class StateDict(TypedDict):
    """Typed mapping for state TensorDicts.

    Optional keys should be marked as NotRequired to align with training vs. eval
    differences (e.g., ``terminated`` is usually training-only).
    """

    # Core state tensors
    # Shapes are ctx-dependent; enforced at runtime by validators.
    # Using torch.Tensor type for generality.
    last_grid: torch.Tensor  # StateKeys.LAST_GRID
    grid: torch.Tensor  # StateKeys.GRID
    examples: torch.Tensor  # StateKeys.EXAMPLES
    initial: torch.Tensor  # StateKeys.INITIAL
    index: torch.Tensor  # StateKeys.INDEX

    # Training-only flag
    terminated: NotRequired[torch.Tensor]  # StateKeys.TERMINATED


class ActionDict(TypedDict):
    """Typed mapping for action TensorDicts emitted by the actor."""

    x_location: torch.Tensor  # ActionKeys.X_LOCATION
    y_location: torch.Tensor  # ActionKeys.Y_LOCATION
    color_values: torch.Tensor  # ActionKeys.COLOR_VALUES
    submit: torch.Tensor  # ActionKeys.SUBMIT


class RewardDict(TypedDict, total=False):
    """Typed mapping for reward TensorDicts used throughout training.

    ``n_reward`` is optional and present only when n-step returns are enabled.
    """

    pixel_wise: torch.Tensor  # RewardKeys.PIXEL_WISE
    binary: torch.Tensor  # RewardKeys.BINARY

    # Optional reward type, present only when n-step returns are enabled
    pixel_wise_n_reward: NotRequired[torch.Tensor]  # RewardKeys.PIXEL_WISE_N_REWARD
    binary_n_reward: NotRequired[torch.Tensor]  # RewardKeys.BINARY_N_REWARD


# Module-level canonical key sets for runtime validation or configuration
STATE_REQUIRED_KEYS = frozenset(
    {
        StateKeys.LAST_GRID.value,
        StateKeys.GRID.value,
        StateKeys.EXAMPLES.value,
        StateKeys.INITIAL.value,
        StateKeys.INDEX.value,
    }
)

ACTION_KEYS_SET = frozenset(
    {
        ActionKeys.X_LOCATION.value,
        ActionKeys.Y_LOCATION.value,
        ActionKeys.COLOR_VALUES.value,
        ActionKeys.SUBMIT.value,
    }
)
