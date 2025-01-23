from typing import Dict
import torch
from tensordict import TensorDict

import logging

logger = logging.getLogger(__name__)


class ArcCriticNetwork(torch.nn.Module):
    def __init__(self, size: int, color_values: int, n_atoms: Dict[str, int]):
        """
        Args:
            size (int): The size of the grid.
            color_values (int): The number of colors.
            n_atoms (Dict[str, int]): The number of atoms for the categorical distribution type.
        """

        super(ArcCriticNetwork, self).__init__()
        self.n_atoms = n_atoms
        self.size = size
        self.color_values = color_values
        self.inputs_layers = torch.nn.ModuleDict(
            {
                "last_grid": torch.nn.Conv2d(
                    in_channels=1, out_channels=1, kernel_size=3
                ),
                "grid": torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3),
                "examples": torch.nn.Conv3d(
                    in_channels=10, out_channels=1, kernel_size=3
                ),
                "initial": torch.nn.Conv2d(
                    in_channels=1, out_channels=1, kernel_size=3
                ),
                "index": torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3),
                "actions": torch.nn.Linear(4, 1),
                "terminated": torch.nn.Linear(1, 1),
            }
        )
        self.linear1 = torch.nn.Linear(1000, 128)
        self.gru = torch.nn.GRU(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.outputs_layers = torch.nn.ModuleDict(
            {
                reward_type: torch.nn.Linear(128, n_atoms)
                for reward_type, n_atoms in self.n_atoms.items()
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
            "actions",
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

    def predict(self, input_sample: torch.Tensor):
        self.input_val(input_sample)
        batch_size = input_sample["grid"].shape[0]
        # Catgorical Distribution over returns per action
        return TensorDict(
            {
                reward_type: torch.rand(size=(batch_size, n_atoms))
                for reward_type, n_atoms in self.n_atoms.items()
            }
        )

    def forward(self, state: TensorDict):
        """
        Args:
            state (TensorDict): The input state.
        Returns:
            TensorDict: The output distributions.
        """
        # Validate input
        self.input_val(state)
        # Brodcast the state
        for key, value in state.items():
            if key == "terminated":
                continue
            if key == "index":
                value = value / torch.max(value)
            elif key == "actions":
                # x_location, y_location, color_values, submit
                value = value / [self.size, self.size, self.color_values, 1]
            else:
                value = self.scale_arc_grids(value)
            state[key] = self.inputs_layers[key](value)
            state[key] = torch.relu(state[key])
            state[key] = state[key].view(state[key].shape[0], -1)

        # Concatenate flattned states
        state = torch.cat(
            state.values(),
            dim=1,
        )

        # Feed the state to the network
        state = self.linear1(state)
        state, _ = self.gru(state)
        output = TensorDict(
            {
                reward_type: layer(state)
                for reward_type, layer in self.outputs_layers.items()
            }
        )
        self.output_val(output)
        return output
