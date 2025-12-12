"""Common enums for the agent."""

from __future__ import annotations

from enum import IntEnum, StrEnum


class StateKeys(StrEnum):
    """State keys for the environment."""

    LAST_GRID = "last_grid"
    GRID = "grid"
    EXAMPLES = "examples"
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


class GridEncoderIdsType(IntEnum):
    """Grid encoder type."""

    # Token IDs
    PAD_ID = 0
    COLOR_START_ID = 1
    COLOR_END_ID = 11
    BOS_ID = 12
    EOS_ID = 13
    ENDXGRID_ID = 14
    ENDYGRID_ID = 15
    ENDGRID_ID = 16
    SEP_ID = 17
    MASK_ID = 18
    EMPTY_ID = 19
    VOCAB_SIZE = 20  # 0-10 colors + 10 special tokens

    def valid_non_encoded_ids(self) -> list[int]:
        """Get the valid non-encoded IDs."""
        return list(range(self.COLOR_START_ID, self.COLOR_END_ID + 1)) + [self.PAD_ID]
