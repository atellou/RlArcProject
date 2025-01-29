import torch
import tensordict
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

    def simmulated_data(self):
        reward = {
            "pixel_wise": torch.randint(-40, 2, size=(self.batch_size, 1)),
            "binary": torch.randint(0, 2, size=(self.batch_size, 1)),
        }
        v_min = {"pixel_wise": -40, "binary": 0}
        v_max = {"pixel_wise": 2, "binary": 1}
        done = torch.randint(0, 2, size=(self.batch_size, 1))
        gamma = 0.99

        state = TensorDict(
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
        return reward, v_min, v_max, done, gamma, state

    def test_target_distribution(self):
        reward, v_min, v_max, done, gamma, next_state = self.simmulated_data()
        target_distribution = self.d4pg.compute_critic_target_distribution(
            self.critic,
            self.actor,
            reward,
            next_state.clone(),
            done,
            gamma,
            self.num_atoms,
            v_min,
            v_max,
        )
        for key, dist in target_distribution.items():
            assert not torch.isnan(
                dist
            ).any(), f"NaN values found in target distribution for key: {key}"

        assert tuple(target_distribution.keys()) == tuple(["pixel_wise", "binary"])
        assert target_distribution["pixel_wise"].shape == (
            self.batch_size,
            self.num_atoms["pixel_wise"],
        )
        assert target_distribution["binary"].shape == (
            self.batch_size,
            self.num_atoms["binary"],
        )
        # Assert probability mass function
        for key, dist in target_distribution.items():
            torch.testing.assert_close(
                torch.sum(dist, dim=1), torch.ones(self.batch_size)
            ), f"Probability mass function not normalized for key: {key}"
            assert torch.all(dist >= 0), f"Negative probability values for key: {key}"
            assert torch.all(
                dist <= 1
            ), f"Probability values greater than 1 for key: {key}"

    def test_critic_loss(self):
        reward, v_min, v_max, done, gamma, next_state = self.simmulated_data()
        target_distribution = self.d4pg.compute_critic_target_distribution(
            self.critic,
            self.actor,
            reward,
            next_state.clone(),
            done,
            gamma,
            self.num_atoms,
            v_min,
            v_max,
        )

        reward, __, __, done, __, state = self.simmulated_data()
        action_probs = self.actor(state)
        # Get best action
        best_action = torch.cat(
            [torch.argmax(x, dim=-1).unsqueeze(-1) for x in action_probs.values()],
            dim=-1,
        )  # Shape: (batch_size, action_space_dim)

        # Backpropagation
        # Define a loss function and an optimizer
        optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.001)
        loss = self.d4pg.compute_critic_loss(
            self.critic, state, best_action, target_distribution
        )
        assert tuple(loss.keys()) == tuple(["pixel_wise", "binary"])
        for key, value in loss.items():
            assert not torch.isnan(
                value
            ).any(), f"NaN values found in loss for key: {key}"
        loss = sum(tuple(loss.values()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    unittest.main()
