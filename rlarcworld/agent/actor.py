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

import torch
import torch.nn as nn
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
                "last_grid": CnnPreTrainedModule(
                    embedding_size=128,
                ),
                "grid": CnnPreTrainedModule(
                    embedding_size=128,
                ),
                "examples": CnnAttention(
                    embedding_size=128, nheads=4, dropout=0.2, bias=True
                ),
                "initial": CnnPreTrainedModule(
                    embedding_size=128,
                ),
                "index": CnnPreTrainedModule(
                    embedding_size=128,
                ),
                # "terminated": torch.nn.Linear(1, 1),
            }
        )

        self.cross_attention_classifier = CrossAttentionClassifier(
            output_classes={
                "x_location": self.size,
                "y_location": self.size,
                "color_values": self.color_values,
                "submit": 2,
            },
            embedding_size=128,
            nheads=4,
            dropout=0.2,
            bias=True,
        )

        self.config = enable_cuda()
        self.device = self.config["device"]

        self.to(self.device)
        # params = torch.tensor(
        #     [[p.numel() * int(p.requires_grad), p.numel()] for p in self.parameters()]
        # )
        # print(
        #     "Actor Network Params:\n {} trainable, {} total".format(
        #         sum(params[:, 0]), sum(params[:, 1])
        #     )
        # )

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
        # Only present in training
        state.pop("terminated", None)
        # Validate input
        self.input_val(state)

        # Brodcast the state
        for key, value in state.items():
            value = value.to(self.device)
            try:
                if key == "index":
                    max_value = torch.max(value)
                    value = value.float() if max_value == 0 else value / max_value
                elif key != "terminated":
                    value = self.scale_arc_grids(value)

                if self.config.get("use_checkpointing"):
                    state[key] = checkpoint.checkpoint(
                        self.inputs_layers[key], value, use_reentrant=False
                    )
                else:
                    state[key] = self.inputs_layers[key](value)

                if key != "examples":
                    state[key] = state[key].unsqueeze(1)
                assert not torch.isnan(state[key]).any(), f"NaN in {key} layer"
            except Exception as e:
                logger.error(
                    f'Error in Actor "{key}" layer, Shape: {value.shape}, and Dtype: {value.dtype}'
                )
                raise e

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

        for key in state.keys():
            assert not torch.isnan(
                state[key]
            ).any(), f"NaN detected in actor output for key {key}!"

        return state.auto_batch_size_()
