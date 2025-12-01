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
    PIXEL_WISE_N_REWARD = "pixel_wise_n_reward"
    BINARY_N_REWARD = "binary_n_reward"

    @staticmethod
    def get_n_step_keys() -> list[RewardKeys]:
        """Get the reward keys for the given n-step."""
        return [RewardKeys.PIXEL_WISE_N_REWARD, RewardKeys.BINARY_N_REWARD]

    @staticmethod
    def get_base_keys() -> list[RewardKeys]:
        """Get the reward keys."""
        return [RewardKeys.PIXEL_WISE, RewardKeys.BINARY]


class ArcEnvRunningMode(StrEnum):
    """Running mode for the ARC environment."""

    TRAIN = "train"
    INFERENCE = "inference"

    def is_train(self) -> bool:
        """Check if the current running mode is training."""
        return self == self.TRAIN
