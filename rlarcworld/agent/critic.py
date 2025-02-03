import os
from typing import Dict
import torch
from tensordict import TensorDict

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGGING_LEVEL", logging.WARNING))


class ArcCriticNetwork(torch.nn.Module):
    def __init__(
        self,
        size: int,
        color_values: int,
        num_atoms: Dict[str, int],
        v_min: Dict[str, int],
        v_max: Dict[str, int],
        test: bool = False,
    ):
        """
        Args:
            size (int): The size of the grid.
            color_values (int): The number of colors.
            num_atoms (Dict[str, int]): The number of atoms for the categorical distribution type.
        """

        super(ArcCriticNetwork, self).__init__()
        self.test = test
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
                "actions": torch.nn.Linear(4, 1),
                "terminated": torch.nn.Linear(1, 1),
            }
        )
        self.linear1 = torch.nn.Linear(16466, 128)
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

    def output_val(self, distribution: TensorDict):
        assert isinstance(distribution, TensorDict), TypeError(
            "Distribution must be a TensorDict"
        )
        in_keys = {"pixel_wise", "binary"}
        assert set(distribution.keys()) == in_keys, ValueError(
            "Distribution keys must be {}. Keys found {}".format(
                in_keys, set(distribution.keys())
            )
        )

    def forward(self, state: TensorDict, action: torch.Tensor):
        """
        Args:
            state (TensorDict): The input state.
        Returns:
            TensorDict: The output distributions.
        """
        state = state.clone()
        # Validate input
        self.input_val(state)
        state["actions"] = action
        # Brodcast the state
        for key, value in state.items():
            if key == "terminated":
                pass
            if key == "index":
                value = value / torch.max(value)
            elif key == "actions":
                # x_location, y_location, color_values, submit
                value = value / torch.tensor(
                    [self.size, self.size, self.color_values, 1]
                )
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
