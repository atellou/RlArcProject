"""
Actor network for the ARC environment.

This module contains the implementation of the actor network for the ARC
environment. The actor network is a neural network that takes as input the
state of the environment and outputs a distribution over the possible actions.

The actor network is composed of several convolutional layers, a GRU layer,
and a linear layer. The convolutional layers are used to process the input
state, the GRU layer is used to process the output of the convolutional layers,
and the linear layer is used to output a distribution over the possible actions.

The actor network is trained using the policy gradient algorithm.
"""

import os
import torch
import torch.nn as nn
from tensordict import TensorDict

import logging

logger = logging.getLogger(__name__)


class ArcActorNetwork(nn.Module):
    """Actor network for the ARC environment."""

    def __init__(self, size: int, color_values, epsilon: float = 1e-6):
        """
        Args:
            size (int): The size of the grid.
            color_values (int): The number of colors.
            epsilon (float, optional): The epsilon value for the GRU. Defaults to 1e-6.
        """
        super(ArcActorNetwork, self).__init__()
        self.size = size
        self.color_values = color_values
        self.inputs_layers = torch.nn.ModuleDict(
            {
                "last_grid": torch.nn.Conv2d(
                    in_channels=1, out_channels=1, kernel_size=3
                ),
                "grid": torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3),
                "examples": torch.nn.Conv3d(
                    in_channels=10, out_channels=1, kernel_size=(1, 3, 3)
                ),
                "initial": torch.nn.Conv2d(
                    in_channels=1, out_channels=1, kernel_size=3
                ),
                "index": torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3),
                "terminated": torch.nn.Linear(1, 1),
            }
        )
        self.linear1 = torch.nn.Linear(16465, 128)
        self.gru = torch.nn.GRU(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.outputs_layers = torch.nn.ModuleDict(
            {
                "x_location": torch.nn.Linear(256, self.size),
                "y_location": torch.nn.Linear(256, self.size),
                "color_values": torch.nn.Linear(256, self.color_values),
                "submit": torch.nn.Linear(256, 2),
            }
        )

    def scale_arc_grids(self, x: torch.Tensor):
        """
        Scales the input tensor by the number of color values.

        Args:
            x (torch.Tensor): The input tensor to scale.

        Returns:
            torch.Tensor: The scaled tensor.
        """

        return x / self.color_values

    def input_val(self, state: TensorDict):
        """
        Validates the input state TensorDict.

        Ensures that the input state is of type TensorDict and contains the
        required keys: "last_grid", "grid", "examples", "initial", "index", and "terminated".

        Args:
            state (TensorDict): The input state to validate.

        Raises:
            TypeError: If the input state is not a TensorDict.
            ValueError: If the state keys do not match the required keys.
        """

        assert isinstance(state, TensorDict), TypeError(
            "Input State must be a TensorDict"
        )
        in_keys = {
            "last_grid",
            "grid",
            "examples",
            "initial",
            "index",
            "terminated",
        }
        assert set(state.keys()) == in_keys, ValueError(
            "State keys must be {}. Keys found {}".format(in_keys, set(state.keys()))
        )

    def output_val(self, action: TensorDict):
        """
        Validates the output action TensorDict.

        Ensures that the output action is of type TensorDict and contains the
        required keys: "color_values", "submit", "x_location", and "y_location".

        Args:
            action (TensorDict): The output action to validate.

        Raises:
            TypeError: If the output action is not a TensorDict.
            ValueError: If the action keys do not match the required keys.
        """
        assert isinstance(action, TensorDict), TypeError("Action must be a TensorDict")
        in_keys = {"color_values", "submit", "x_location", "y_location"}
        assert set(action.keys()) == in_keys, ValueError(
            "Action keys must be {}. Keys found {}".format(in_keys, set(action.keys()))
        )

    def forward(self, state: TensorDict):
        """
        Forward pass of the actor network.

        Args:
            state (TensorDict): The state of the environment.

        Returns:
            TensorDict: The output distribution of the actor network.

        Raises:
            TypeError: If the input state is not a TensorDict.
            ValueError: If the state keys do not match the required keys.
            AssertionError: If a NaN is detected in the output.
        """
        state = state.clone()
        # Validate input
        self.input_val(state)

        # Brodcast the state
        for key, value in state.items():
            if key == "index":
                max_value = torch.max(value)
                value = value.float() if max_value == 0 else value / max_value
            elif key != "terminated":
                value = self.scale_arc_grids(value)
            state[key] = self.inputs_layers[key](value)
            state[key] = torch.relu(state[key])
            state[key] = state[key].view(state[key].shape[0], -1)
            assert not torch.isnan(state[key]).any(), f"NaN in {key} layer"

        # Concatenate flattned states
        state = torch.cat(
            tuple(state.values()),
            dim=1,
        )

        # Feed the state to the network
        state = self.linear1(state)
        state, _ = self.gru(state)

        state = TensorDict(
            {
                reward_type: torch.softmax(layer(state), dim=-1)
                for reward_type, layer in self.outputs_layers.items()
            },
        )
        self.output_val(state)

        for key in state.keys():
            assert not torch.isnan(
                state[key]
            ).any(), f"NaN detected in actor output for key {key}!"

        return state.auto_batch_size_()
