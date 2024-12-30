from typing import Dict
import torch
from tensordict import TensorDict

import logging

logger = logging.getLogger(__name__)


class ArcCriticNetwork:
    def __init__(self, n_atoms: Dict[str, int]):
        self.n_atoms = n_atoms

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
            "actions",
        }
        assert set(state.keys()) == in_keys, ValueError(
            "State keys must be {}".format(in_keys)
        )

    def predict(self, input_sample: torch.Tensor):
        self.input_val(input_sample)
        batch_size = input_sample["grid"].shape[0]
        return TensorDict(
            {
                reward_type: torch.rand(size=(batch_size, n_atoms))
                for reward_type, n_atoms in self.n_atoms.items()
            }
        )
