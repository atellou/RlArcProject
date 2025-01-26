import torch
from tensordict import TensorDict

from rlarcworld.agent.actor import ArcActorNetwork
from rlarcworld.agent.critic import ArcCriticNetwork
from rlarcworld.algorithms.d4pg import D4PG
from rlarcworld.utils import categorical_projection

import unittest
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestD4PG(unittest.TestCase):

    def setUp(self):
        self.batch_size = torch.randint(1, 20, size=(1,))
        self.grid_size = 30
        self.color_values = 11
        self.num_atoms = {"pixel_wise": 50, "binary": 3}

        self.d4pg = D4PG()
        self.actor = ArcActorNetwork(
            size=self.grid_size, color_values=self.color_values
        )
        self.critic = ArcCriticNetwork(
            size=self.grid_size, color_values=self.color_values, n_atoms=self.num_atoms
        )

    def test_target_distribution(self):
        reward = {
            "pixel_wise": torch.randint(-40, 2, size=(self.batch_size, 1)),
            "binary": torch.randint(0, 2, size=(self.batch_size, 1)),
        }
        v_min = {"pixel_wise": -40, "binary": 0}
        v_max = {"pixel_wise": 2, "binary": 1}
        done = torch.randint(0, 2, size=(self.batch_size, 1))
        gamma = 0.99

        next_state = TensorDict(
            {
                "last_grid": torch.randn(
                    self.batch_size, 1, self.grid_size, self.grid_size
                ),
                "grid": torch.randn(self.batch_size, 1, self.grid_size, self.grid_size),
                "examples": torch.randn(
                    self.batch_size, 10, 2, self.grid_size, self.grid_size
                ),
                "initial": torch.randn(
                    self.batch_size, 1, self.grid_size, self.grid_size
                ),
                "index": torch.randn(
                    self.batch_size, 1, self.grid_size, self.grid_size
                ),
                "terminated": torch.randn(self.batch_size, 1),
            }
        )
        target_distribution = self.d4pg.compute_critic_target_distribution(
            self.critic,
            self.actor,
            reward,
            next_state,
            done,
            gamma,
            self.num_atoms,
            v_min,
            v_max,
        )
        assert tuple(target_distribution.keys()) == tuple(["pixel_wise", "binary"])
        assert target_distribution["pixel_wise"].shape == (
            self.batch_size,
            self.num_atoms["pixel_wise"],
        )
        assert target_distribution["binary"].shape == (
            self.batch_size,
            self.num_atoms["binary"],
        )


if __name__ == "__main__":
    unittest.main()
