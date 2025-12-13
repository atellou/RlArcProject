"""Reward world model for the ARC environment.

This module implements a reward model that predicts ARC rewards from
environment state. It uses the same ViTARC-based architecture as the
actor/critic networks and outputs a TensorDict keyed by RewardKeys,
so it can be used by RewardWorldModelWrapper.
"""

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from tensordict import TensorDict

from rlarcworld.common.enums import RewardKeys, StateKeys
from rlarcworld.nn_modules import (
    RewardHeadFromEncoder,
    RlArcWorldViTARC,
    RlArcWorldViTARCConfig,
)
from rlarcworld.utils.device import enable_cuda

if TYPE_CHECKING:
    from rlarcworld.utils.device import CudaConfig

logger = logging.getLogger(__name__)


class ViTARCRewardModel(nn.Module):
    """ViTARC-based reward world model for the ARC environment.

    The model takes an ARC state TensorDict (same structure as used by
    ViTARCActor/VitARCCritic) and predicts per-sample rewards
    for different reward types. It uses the shared RlArcWorldViTARC
    encoder and a RewardHeadFromEncoder for predictions.
    """

    def __init__(
        self,
        color_values: int,
        rl_vit_arc_config: RlArcWorldViTARCConfig,
        config: "CudaConfig | None" = None,
    ) -> None:
        """Initialize the ViTARC reward world model.

        Args:
            color_values: Number of color values in the grid.
            rl_vit_arc_config: Configuration for the RlArcWorldViTARC encoder.
            config: CUDA/device configuration. Defaults to None.
        """
        super().__init__()
        self.color_values = color_values
        self.grid_size = rl_vit_arc_config.grid_size

        # Shared ViTARC encoder
        self.rl_vit_arc = RlArcWorldViTARC(rl_vit_arc_config)

        # Reward prediction head
        self.reward_head = RewardHeadFromEncoder(
            hidden_size=rl_vit_arc_config.d_model,
            num_heads=rl_vit_arc_config.num_heads,
            dropout=rl_vit_arc_config.dropout,
        )

        self.config = enable_cuda() if config is None else config
        self.device = self.config["device"]

        self.to(self.device)

    @torch.jit.export
    def output_val(self, rewards: TensorDict) -> None:
        """Validate that the output contains the required reward keys."""
        assert isinstance(rewards, TensorDict), TypeError(
            "Output rewards must be a TensorDict",
        )
        required_keys = {RewardKeys.BINARY.value, RewardKeys.PIXEL_WISE.value}
        assert set(rewards.keys()) == required_keys, ValueError(
            f"Reward keys must be {required_keys}. Keys found {set(rewards.keys())}",
        )

    def forward(self, state: TensorDict) -> TensorDict:
        """Forward pass of the reward world model.

        Args:
            state: Environment state TensorDict with keys like
                ``grid``, ``examples``, ``initial``, etc.

        Returns:
            TensorDict with reward predictions:
                - ``binary``: Softmax probabilities [B, 2] for (incorrect, correct)
                - ``pixel_wise``: Sigmoid output [B] for percentage of wrong pixels
        """
        state = state.clone()
        current_device = next(self.parameters()).device
        state = state.to(current_device)

        # Remove terminated key if present (only used in training)
        state.pop(StateKeys.TERMINATED.value, None)

        # Encode state through ViTARC
        if self.config.get("use_checkpointing"):
            encoder_output = checkpoint.checkpoint(
                self.rl_vit_arc, state, use_reentrant=False
            )
        else:
            encoder_output = self.rl_vit_arc(state)

        # Get reward predictions from head
        rewards = self.reward_head(encoder_output)

        self.output_val(rewards)
        return rewards


# Alias for backward compatibility
RewardWorldModel = ViTARCRewardModel
