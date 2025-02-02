import torch
import torch.nn as nn
from tensordict import TensorDict

import logging

logger = logging.getLogger(__name__)


class ArcActorNetwork(nn.Module):
    def __init__(self, size: int, color_values, test: bool = False):
        super(ArcActorNetwork, self).__init__()
        self.test = test
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
        return x / self.color_values

    def input_val(self, state: TensorDict):
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
        assert isinstance(action, TensorDict), TypeError("Action must be a TensorDict")
        in_keys = {"color_values", "submit", "x_location", "y_location"}
        assert set(action.keys()) == in_keys, ValueError(
            "Action keys must be {}. Keys found {}".format(in_keys, set(action.keys()))
        )

    def forward(self, state: TensorDict):
        """
        Args:
            state (TensorDict): The input state.
        Returns:
            TensorDict: The output actions.
        """
        state = state.clone()
        # Validate input
        self.input_val(state)
        # Brodcast the state
        for key, value in state.items():
            if key == "terminated":
                pass
            if key == "index":
                value = value / torch.max(value)
            else:
                value = self.scale_arc_grids(value)
            state[key] = self.inputs_layers[key](value)
            state[key] = torch.relu(state[key])
            state[key] = state[key].view(state[key].shape[0], -1)

        # Concatenate flattned states
        state = torch.cat(
            tuple(state.values()),
            dim=1,
        )

        # Feed the state to the network
        state = self.linear1(state)
        state, _ = self.gru(state)

        if self.test:
            output = TensorDict(
                {
                    reward_type: torch.softmax(
                        layer(state) * 0.1 + torch.linspace(0, 1, layer.out_features),
                        dim=-1,
                    )
                    for reward_type, layer in self.outputs_layers.items()
                }
            )
        else:
            output = TensorDict(
                {
                    reward_type: torch.softmax(layer(state), dim=-1)
                    for reward_type, layer in self.outputs_layers.items()
                }
            )
        self.output_val(output)
        return output
