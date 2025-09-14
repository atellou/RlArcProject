"""Actor network for the ARC environment.

This module contains the implementation of the actor network for the ARC
environment. The actor network is a neural network that takes as input the
state of the environment and outputs a distribution over the possible actions.

The actor network is composed of several convolutional layers, a GRU layer,
and a linear layer. The convolutional layers are used to process the input
state, the GRU layer is used to process the output of the convolutional layers,
and the linear layer is used to output a distribution over the possible actions.

The actor network is trained using the policy gradient algorithm.
"""

import logging

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from tensordict import TensorDict

from rlarcworld.agent.nn_modules import (
    CnnAttention,
    CnnPreTrainedModule,
    CrossAttentionClassifier,
)
from rlarcworld.common.enums import ActionKeys, StateKeys
from rlarcworld.common.types import (
    ACTION_KEYS_SET,
    STATE_REQUIRED_KEYS,
)
from rlarcworld.utils.device import enable_cuda

logger = logging.getLogger()


class ArcActorNetwork(nn.Module):
    """Actor network for the ARC environment."""

    def __init__(
        self, grid_size: int, color_values: int, embedding_size=128, config=None
    ):
        """Actor network for the ARC environment.

        Args:
            grid_size (int): The size of the grid.
            color_values (int): The number of colors.
            embedding_size (int, optional): The size of the embedding. Defaults to 128.
            config (CudaConfig, optional): CUDA configuration. Defaults to None.
        """
        super().__init__()
        self.grid_size = grid_size
        self.color_values = color_values
        self.action_head_keys = set(ACTION_KEYS_SET)
        # Build module dict using canonical string keys
        self.inputs_layers = torch.nn.ModuleDict(
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

        self.cross_attention_classifier = torch.nn.Sequential(
            nn.LayerNorm(embedding_size),
            CrossAttentionClassifier(
                output_classes={
                    ActionKeys.X_LOCATION.value: self.grid_size,
                    ActionKeys.Y_LOCATION.value: self.grid_size,
                    ActionKeys.COLOR_VALUES.value: self.color_values,
                    ActionKeys.SUBMIT.value: 2,
                },
                embedding_size=128,
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
        """Scales the input tensor by the number of color values.

        Args:
            x (torch.Tensor): The input tensor to scale.

        Returns:
            torch.Tensor: The scaled tensor.
        """
        return x / self.color_values

    @torch.jit.export
    def input_val(self, state: TensorDict) -> None:
        """Validate that the state contains exactly the required keys."""
        assert isinstance(state, TensorDict), TypeError(
            "Input State must be a TensorDict"
        )
        assert set(state.keys()) == STATE_REQUIRED_KEYS, ValueError(
            f"State keys must be {STATE_REQUIRED_KEYS}. Keys found {set(state.keys())}"
        )

    @torch.jit.export
    def output_val(self, action: TensorDict) -> None:
        """Validate that the action dict contains exactly the expected heads."""
        assert isinstance(action, TensorDict), TypeError("Action must be a TensorDict")
        assert set(action.keys()) == self.action_head_keys, ValueError(
            f"Action keys must be {self.action_head_keys}. "
            f"Keys found {set(action.keys())}"
        )

    def forward(self, state: TensorDict) -> TensorDict:
        """Forward pass of the actor network.

        Args:
            state (TensorDict): The state of the environment.

        Returns:
            ActionDict: The output distribution of the actor network.

        Raises:
            TypeError: If the input state is not a StateDict.
            ValueError: If the state keys do not match the required keys.
            AssertionError: If a NaN is detected in the output.
        """
        state = state.clone()
        # Only present in training
        state.pop(StateKeys.TERMINATED.value, None)
        self.input_val(state)

        # Determine current device from module parameters
        # (in case module was moved after init)
        current_device = next(self.parameters()).device
        # Broadcast the state
        for key, value in state.items():
            value = value.to(current_device)
            if key == StateKeys.INDEX.value:
                max_value = torch.max(value)
                value = value.float() if max_value == 0 else value / max_value
            elif key != StateKeys.TERMINATED.value:
                value = self.scale_arc_grids(value)

            if self.config.get("use_checkpointing"):
                state[key] = checkpoint.checkpoint(
                    self.inputs_layers[key], value, use_reentrant=False
                )
            else:
                state[key] = self.inputs_layers[key](value)

            if key != StateKeys.EXAMPLES.value:
                state[key] = state[key].unsqueeze(1)
            assert not torch.isnan(state[key]).any(), f"NaN in {key} layer"

        # Concatenate flattned states
        state = torch.cat(
            tuple(state.values()),
            dim=1,
        )

        if self.config.get("use_checkpointing"):
            state = checkpoint.checkpoint(
                self.cross_attention_classifier, state, use_reentrant=False
            )
        else:
            state = self.cross_attention_classifier(state)

        # Apply softmax
        state = TensorDict(
            {
                action_type: torch.softmax(logits, dim=-1)
                for action_type, logits in state.items()
            },
        )
        self.output_val(state)
        return state.auto_batch_size_()
