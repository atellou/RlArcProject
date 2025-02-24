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
import copy
import torch
import torch.utils.checkpoint as checkpoint
from tensordict import TensorDict
from rlarcworld.agent.nn_modules import (
    CnnPreTrainedModule,
    CnnAttention,
    CrossAttentionClassifier,
)
from rlarcworld.utils import enable_cuda

import logging

logger = logging.getLogger(__name__)


class ArcCriticNetwork(torch.nn.Module):
    r"""
    Args:
        grid_size (int): Size of the grid.
        color_values (int): The number of color values in the grid.
        num_atoms (Dict[str, int]): The number of atoms in the distribution for each reward type.
        v_min (Dict[str, int]): The minimum value for each reward type.
        v_max (Dict[str, int]): The maximum value for each reward type.
        test (bool, optional): Whether to set the network to test mode. Defaults to False.
    """

    def __init__(
        self,
        grid_size: int,
        color_values: int,
        num_atoms: Dict[str, int],
        v_min: Dict[str, int],
        v_max: Dict[str, int],
        embedding_size: int = 128,
    ):
        super(ArcCriticNetwork, self).__init__()
        self.num_atoms = num_atoms
        self.grid_size = grid_size
        self.color_values = color_values
        self.v_min = v_min
        self.v_max = v_max
        self.config = enable_cuda()
        self.device = self.config["device"]
        for key, min_val in v_min.items():
            assert (
                min_val < self.v_max[key]
            ), f"v_min[{key}]={min_val} is not lower than v_max[{key}]={self.v_max[key]}"
        self.z_atoms = TensorDict(
            {
                key: torch.linspace(v_min[key], v_max[key], value)
                for key, value in num_atoms.items()
            }
        ).to(self.device)
        self.no_scale_keys = [
            "x_location",
            "y_location",
            "color_values",
            "submit",
            "terminated",
        ]

        self.actions_mlp = torch.nn.Sequential(
            torch.nn.Linear(
                (self.grid_size * 2) + self.color_values + 3, embedding_size * 2
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(embedding_size * 2, embedding_size),
        )
        self.inputs_layers = torch.nn.ModuleDict(
            {
                "last_grid": CnnPreTrainedModule(
                    embedding_size=embedding_size,
                ),
                "grid": CnnPreTrainedModule(
                    embedding_size=embedding_size,
                ),
                "examples": CnnAttention(
                    embedding_size=embedding_size, nheads=4, dropout=0.2, bias=True
                ),
                "initial": CnnPreTrainedModule(
                    embedding_size=embedding_size,
                ),
                "index": CnnPreTrainedModule(
                    embedding_size=embedding_size,
                ),
            }
        )

        self.cross_attention_classifier = torch.nn.Sequential(
            torch.nn.LayerNorm(embedding_size),
            CrossAttentionClassifier(
                output_classes=self.num_atoms,
                embedding_size=128,
                nheads=4,
                dropout=0.2,
                bias=True,
            ),
        )

        self.to(self.device)

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
        action_keys = torch.cat(
            tuple(state.pop(k) for k in self.no_scale_keys),
            dim=1,
        ).to(self.device)
        action_keys = self.actions_mlp(action_keys).unsqueeze(1)

        # Grid inputs
        for key, value in state.items():
            value = value.to(self.device)
            if key == "index":
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

            if key != "examples":
                state[key] = state[key].unsqueeze(1)

        # Concatenate flattned states
        state = torch.cat(
            [*tuple(state.values()), action_keys],
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

        return state.auto_batch_size_()
