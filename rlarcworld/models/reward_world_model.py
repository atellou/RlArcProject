"""Reward world model for the ARC environment.

This module implements a reward model that predicts ARC rewards from
environment state. It mirrors the input processing of the actor/critic
networks and outputs a TensorDict keyed by RewardKeys, so it can be used
by RewardWorldModelWrapper.
"""

import logging

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from tensordict import TensorDict

from rlarcworld.common.enums import StateKeys
from rlarcworld.common.types import STATE_REQUIRED_KEYS, RewardKeys
from rlarcworld.nn_modules import (
    CnnAttention,
    CnnPreTrainedModule,
    CrossAttentionClassifier,
)
from rlarcworld.utils.device import enable_cuda

logger = logging.getLogger(__name__)


class RewardWorldModel(nn.Module):
    """Reward world model for the ARC environment.

    The model takes an ARC state TensorDict (same structure as used by
    ArcActorNetwork/ArcCriticNetwork) and predicts per-sample rewards
    for different reward types. It is intended to be used by
    RewardWorldModelWrapper, which consumes a TensorDict with keys
    defined in RewardKeys (e.g. "binary", "pixel_wise").
    """

    def __init__(
        self,
        grid_size: int,
        color_values: int,
        embedding_size: int = 128,
        config=None,
    ) -> None:
        """Initialize the reward world model.

        Args:
            grid_size (int): Size of the ARC grid.
            color_values (int): Number of color values in the grid.
            embedding_size (int, optional): Embedding size for CNN features.
                Defaults to 128.
            config (CudaConfig, optional): CUDA configuration. Defaults to None.
        """
        super().__init__()
        self.grid_size = grid_size
        self.color_values = color_values

        self.inputs_layers = nn.ModuleDict(
            {
                StateKeys.LAST_GRID.value: CnnPreTrainedModule(
                    embedding_size=embedding_size,
                ),
                StateKeys.GRID.value: CnnPreTrainedModule(
                    embedding_size=embedding_size,
                ),
                StateKeys.EXAMPLES.value: CnnAttention(
                    embedding_size=embedding_size, nheads=4, dropout=0.2, bias=True
                ),
                StateKeys.INITIAL.value: CnnPreTrainedModule(
                    embedding_size=embedding_size,
                ),
                StateKeys.INDEX.value: CnnPreTrainedModule(
                    embedding_size=embedding_size,
                ),
            }
        )

        self.cross_attention_classifier = nn.Sequential(
            nn.LayerNorm(embedding_size),
            CrossAttentionClassifier(
                output_classes={
                    RewardKeys.BINARY.value: 2,  # Softmax over [incorrect, correct]
                    RewardKeys.PIXEL_WISE.value: 1,  # Direct regression
                },
                embedding_size=embedding_size,
                nheads=4,
                dropout=0.2,
                bias=True,
            ),
        )

        self.config = enable_cuda() if config is None else config
        self.device = self.config["device"]

        self.to(self.device)

    @torch.jit.export
    def scale_arc_grids(self, x: torch.Tensor) -> torch.Tensor:
        """Scale input grids by the number of color values.

        Args:
            x (torch.Tensor): Input tensor representing ARC grids.

        Returns:
            torch.Tensor: Scaled tensor.
        """
        return x / self.color_values

    @torch.jit.export
    def input_val(self, state: TensorDict) -> None:
        """Validate that the state contains exactly the required keys."""
        assert isinstance(state, TensorDict), TypeError(
            "Input State must be a TensorDict",
        )
        in_keys = set(STATE_REQUIRED_KEYS)
        assert set(state.keys()) == in_keys, ValueError(
            f"State keys must be {in_keys}. Keys found {set(state.keys())}",
        )

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

    def forward(self, state: TensorDict) -> torch.Tensor:
        """Forward pass of the reward world model.

        Args:
            state (TensorDict): Environment state TensorDict.

        Returns:
            torch.Tensor: Correct/Incorrect predictions.
        """
        state = state.clone()

        current_device = next(self.parameters()).device
        example_mask = state.pop(StateKeys.EXAMPLE_MASK.value, None)
        if example_mask is not None:
            example_mask = example_mask.to(current_device)
        # Only present in training
        state.pop(StateKeys.TERMINATED.value, None)
        self.input_val(state)

        # Broadcast and encode the state
        for key, value in state.items():
            value = value.to(current_device)
            if key == StateKeys.INDEX.value:
                max_value = torch.max(value)
                value = value.float() if max_value == 0 else value / max_value
            else:
                value = self.scale_arc_grids(value)

            if self.config.get("use_checkpointing"):
                state[key] = checkpoint.checkpoint(
                    self.inputs_layers[key], value, use_reentrant=False
                )
            else:
                state[key] = self.inputs_layers[key](value)

            if key != StateKeys.EXAMPLES.value:
                state[key] = state[key].unsqueeze(1)
            if torch.isnan(state[key]).any():
                raise ValueError(f"NaN in {key} layer")

        # If there is an example mask, apply it to zero out padded examples
        if example_mask is not None:
            mask = example_mask.unsqueeze(-1)
            state[StateKeys.EXAMPLES.value] = state[StateKeys.EXAMPLES.value] * mask

        # Concatenate the state embeddings
        state = torch.cat(tuple(state.values()), dim=1)

        if self.config.get("use_checkpointing"):
            state = checkpoint.checkpoint(
                self.cross_attention_classifier, state, use_reentrant=False
            )
        else:
            state = self.cross_attention_classifier(state)

        # Process outputs into reward predictions
        # Binary: softmax classification (better calibrated probabilities)
        binary_prob = torch.softmax(state[RewardKeys.BINARY.value], dim=-1)

        # Pixel-wise: sigmoid to predict percentage of wrong pixels in [0, 1]
        # This represents grid_diff / grid_size², bounded and interpretable
        pixel_wise_percentage = torch.sigmoid(
            state[RewardKeys.PIXEL_WISE.value].squeeze(-1)
        )

        rewards = TensorDict(
            {
                RewardKeys.BINARY.value: binary_prob,
                RewardKeys.PIXEL_WISE.value: pixel_wise_percentage,
            }
        )

        self.output_val(rewards)
        return rewards.auto_batch_size_()
