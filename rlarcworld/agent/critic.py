"""
Critic network for the ARC environment.

This module contains the implementation of the critic network for the ARC
environment. The critic network is a neural network that takes as input the
state of the environment and the action taken in the environment and outputs a
distribution over the possible rewards.

The critic network is composed of several convolutional layers, a GRU layer,
and a linear layer. The convolutional layers are used to process the input
state, the GRU layer is used to process the output of the convolutional layers,
and the linear layer is used to output a distribution over the possible rewards.

The critic network is trained using the distributional reinforcement learning
algorithm.

"""

import os
from typing import Dict
import torch
from tensordict import TensorDict

import logging

logger = logging.getLogger(__name__)


class ArcCriticNetwork(torch.nn.Module):
    r"""
    Args:
        size (int): Size of the grid.
        color_values (int): The number of color values in the grid.
        num_atoms (Dict[str, int]): The number of atoms in the distribution for each reward type.
        v_min (Dict[str, int]): The minimum value for each reward type.
        v_max (Dict[str, int]): The maximum value for each reward type.
        test (bool, optional): Whether to set the network to test mode. Defaults to False.
    """

    def __init__(
        self,
        size: int,
        color_values: int,
        num_atoms: Dict[str, int],
        v_min: Dict[str, int],
        v_max: Dict[str, int],
        test: bool = False,
    ):
        super(ArcCriticNetwork, self).__init__()
        self.num_atoms = num_atoms
        self.size = size
        self.color_values = color_values
        self.v_min = v_min
        self.v_max = v_max
        for key, min_val in v_min.items():
            assert (
                min_val < self.v_max[key]
            ), f"v_min[{key}]={min_val} is not lower than v_max[{key}]={self.v_max[key]}"
        self.z_atoms = {
            key: torch.linspace(v_min[key], v_max[key], value)
            for key, value in num_atoms.items()
        }
        self.no_scale_keys = [
            "x_location",
            "y_location",
            "color_values",
            "submit",
            "terminated",
        ]
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
                "x_location": torch.nn.Linear(self.size, 1),
                "y_location": torch.nn.Linear(self.size, 1),
                "color_values": torch.nn.Linear(self.color_values, 1),
                "submit": torch.nn.Linear(2, 1),
                "terminated": torch.nn.Linear(1, 1),
            }
        )

        self.linear1 = torch.nn.Linear(16469, 128)
        self.gru = torch.nn.GRU(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.outputs_layers = torch.nn.ModuleDict(
            {
                reward_type: torch.nn.Linear(
                    256,
                    num_atoms,
                )
                for reward_type, num_atoms in self.num_atoms.items()
            }
        )

    @torch.jit.export
    def scale_arc_grids(self, x: torch.Tensor):
        r"""
        Scales the input tensor by the number of color values.

        Args:
            x (torch.Tensor): The input tensor to scale.

        Returns:
            torch.Tensor: The scaled tensor.
        """
        return x / self.color_values

    @torch.jit.export
    def input_val(self, state: TensorDict):
        r"""
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

    @torch.jit.export
    def output_val(self, distribution: TensorDict):
        r"""
        Validates the output distribution TensorDict.

        Ensures that the output distribution is of type TensorDict and contains the
        required keys: "pixel_wise" and "binary".

        Args:
            distribution (TensorDict): The output distribution to validate.

        Raises:
            TypeError: If the output distribution is not a TensorDict.
            ValueError: If the distribution keys do not match the required keys.
        """
        assert isinstance(distribution, TensorDict), TypeError(
            "Distribution must be a TensorDict"
        )
        in_keys = set(self.num_atoms.keys())
        assert set(distribution.keys()) == in_keys, ValueError(
            "Distribution keys must be {}. Keys found {}".format(
                in_keys, set(distribution.keys())
            )
        )

    def forward(self, state: TensorDict, action: TensorDict):
        r"""
        Forward pass of the critic network.

        Args:
            state (TensorDict): The state of the environment.
            action (TensorDict): The action taken in the environment.

        Returns:
            TensorDict: The output distribution of the critic network.

        Raises:
            TypeError: If the input state or action is not a TensorDict.
            ValueError: If the state keys do not match the required keys.
            AssertionError: If a NaN is detected in the output.
        """
        state = state.clone()
        # Validate input
        self.input_val(state)
        assert isinstance(action, TensorDict), TypeError("Action must be a TensorDict")
        state.update(action)
        # Brodcast the state
        for key, value in state.items():
            if key == "index":
                max_value = torch.max(value)
                value = value.float() if max_value == 0 else value / max_value
            elif key not in self.no_scale_keys:
                value = self.scale_arc_grids(value)
            state[key] = self.inputs_layers[key](value.float())
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
                key: torch.softmax(layer(state), dim=-1)
                for key, layer in self.outputs_layers.items()
            }
        )

        return state.auto_batch_size_()
