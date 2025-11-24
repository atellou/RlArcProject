"""Common enums for the agent."""

from __future__ import annotations

from enum import StrEnum


class StateKeys(StrEnum):
    """State keys for the environment."""

    LAST_GRID = "last_grid"
    GRID = "grid"
    EXAMPLES = "examples"
    EXAMPLE_MASK = "example_mask"
    INITIAL = "initial"
    INDEX = "index"
    TERMINATED = "terminated"
    TARGET = "target"


class ActionKeys(StrEnum):
    """Action keys for the environment."""

    COLOR_VALUES = "color_values"
    X_LOCATION = "x_location"
    Y_LOCATION = "y_location"
    SUBMIT = "submit"


class RewardKeys(StrEnum):
    """Reward keys for the environment."""

    PIXEL_WISE = "pixel_wise"
    BINARY = "binary"
    N_REWARD = "n_reward"
