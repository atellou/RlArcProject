import os
import torch
from torch.utils.data import DataLoader
import tensordict
from tensordict import TensorDict

from rlarcworld.enviroments.arc_batch_grid_env import ArcBatchGridEnv
from rlarcworld.enviroments.wrappers.rewards import PixelAwareRewardWrapper
from rlarcworld.arc_dataset import ArcDataset, ArcSampleTransformer
from rlarcworld.agent.actor import ArcActorNetwork
from rlarcworld.agent.critic import ArcCriticNetwork
from rlarcworld.algorithms.d4pg import D4PG
from rlarcworld.utils import categorical_projection

import unittest
import logging


logger = logging.getLogger(__name__)


logger.info("Setting up D4PG test")


class TestD4PG(unittest.TestCase):

    def setUp(self):
        self.batch_size = torch.randint(1, 20, size=(1,)).item()
        self.grid_size = 30
        self.color_values = 11
        self.num_atoms = {"pixel_wise": 50, "binary": 3}

        self.actor = ArcActorNetwork(
            size=self.grid_size, color_values=self.color_values, test=True
        )
        v_min = {"pixel_wise": -40, "binary": 0}
        v_max = {"pixel_wise": 2, "binary": 1}
        self.critic = ArcCriticNetwork(
            size=self.grid_size,
            color_values=self.color_values,
            num_atoms=self.num_atoms,
            v_min=v_min,
            v_max=v_max,
            test=True,
        )
        self.d4pg = D4PG(actor=self.actor, critic=self.critic)

    def simmulated_data(self):
        reward = {
            "pixel_wise": torch.randint(-40, 2, size=(self.batch_size, 1)),
            "binary": torch.randint(0, 2, size=(self.batch_size, 1)),
        }

        done = torch.randint(0, 2, size=(self.batch_size, 1))
        gamma = 0.99
        state = TensorDict(
            {
                "last_grid": torch.randint(
                    0,
                    self.color_values,
                    size=(self.batch_size, 1, self.grid_size, self.grid_size),
                ),
                "grid": torch.randint(
                    0,
                    self.color_values,
                    size=(self.batch_size, 1, self.grid_size, self.grid_size),
                ),
                "examples": torch.randint(
                    0,
                    self.color_values,
                    size=(self.batch_size, 10, 2, self.grid_size, self.grid_size),
                ),
                "initial": torch.randint(
                    0,
                    self.color_values,
                    size=(self.batch_size, 1, self.grid_size, self.grid_size),
                ),
                "index": torch.randint(
                    0,
                    self.grid_size,
                    size=(self.batch_size, 1, self.grid_size, self.grid_size),
                ),
                "terminated": torch.randint(0, 2, size=(self.batch_size, 1)).float(),
            }
        )
        return reward, done, gamma, state

    def test_target_distribution(self):
        logger.info("Testing target distribution computation")
        reward, done, gamma, next_state = self.simmulated_data()
        target_distribution = self.d4pg.compute_target_distribution(
            reward,
            next_state.clone(),
            done,
            gamma,
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
        logger.info("Testing critic loss computation")
        reward, done, gamma, next_state = self.simmulated_data()
        target_distribution = self.d4pg.compute_target_distribution(
            reward,
            next_state.clone(),
            done,
            gamma,
        )

        reward, done, __, state = self.simmulated_data()
        action_probs = self.actor(state)
        # Get best action
        best_action = torch.cat(
            [torch.argmax(x, dim=-1).unsqueeze(-1) for x in action_probs.values()],
            dim=-1,
        )  # Shape: (batch_size, action_space_dim)

        # Backpropagation
        # Define a loss function and an optimizer
        optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.001)
        loss = self.d4pg.compute_critic_loss(state, best_action, target_distribution)
        assert tuple(loss.keys()) == tuple(["pixel_wise", "binary"])
        for key, value in loss.items():
            assert not torch.isnan(
                value
            ).any(), f"NaN values found in loss for key: {key}"
        loss = sum(tuple(loss.values()))

        optimizer.zero_grad()
        loss.backward()
        for name, param in self.critic.named_parameters():
            if param.grad is None:
                raise ValueError(
                    f"Gradient not flowing in D4PG ArcActorNetwork for: {name}"
                )
        optimizer.step()

    def test_actor_loss(self):
        logger.info("Testing actor loss computation")
        __, __, __, state = self.simmulated_data()

        # Backpropagation
        # Define a loss function and an optimizer
        optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.001)
        loss = self.d4pg.compute_actor_loss(state)
        assert tuple(loss.keys()) == tuple(["pixel_wise", "binary"])
        for key, value in loss.items():
            assert tuple(value.keys()) == tuple(
                ["x_location", "y_location", "color_values", "submit"]
            )
            for k, v in value.items():
                assert not torch.isnan(
                    v
                ).any(), f"NaN values found in loss for key: [{key}][{k}]"

        loss = sum((g for v in loss.values() for g in v.values()))
        optimizer.zero_grad()
        loss.backward()
        for name, param in self.actor.named_parameters():
            if param.grad is None:
                raise ValueError(
                    f"Gradient not flowing in D4PG ArcActorNetwork for: {name}"
                )
        optimizer.step()

    def test_train_step(self):
        logger.info("Testing train step")
        reward, done, gamma, state = self.simmulated_data()
        __, __, __, next_state = self.simmulated_data()
        action = self.d4pg.actor(state)
        action = torch.cat(
            [torch.argmax(x, dim=-1).unsqueeze(-1) for x in action.values()],
            dim=-1,
        )  # Shape: (batch_size, action_space_dim)
        batch = TensorDict(
            {
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "done": done,
            },
            batch_size=self.batch_size,
        )
        actor_optimizer = torch.optim.Adam(self.d4pg.actor.parameters(), lr=0.001)
        critic_optimizer = torch.optim.Adam(self.d4pg.critic.parameters(), lr=0.001)

        loss_actor, loss_critic = self.d4pg.train_step(
            batch, actor_optimizer, critic_optimizer, gamma
        )

        logger.info(f"Actor loss: {loss_actor}")
        logger.info(f"Critic loss: {loss_critic}")
        # Check that losses are not NaN
        assert not torch.isnan(loss_actor), "Actor loss is NaN"
        assert not torch.isnan(loss_critic), "Critic loss is NaN"

        # Check that gradients are flowing
        for name, param in self.d4pg.actor.named_parameters():
            assert param.grad is not None, f"Gradient not flowing in actor for: {name}"
        for name, param in self.d4pg.critic.named_parameters():
            assert param.grad is not None, f"Gradient not flowing in critic for: {name}"

    def test_train_d4pg(self):
        logger.info("Testing train_d4pg")
        grid_size = 30
        color_values = 11
        n_steps = torch.randint(1, 100, size=(1,)).item()

        # Create an instance of the ArcBatchGridEnv
        logger.info("Creating ArcBatchGridEnv")
        env = ArcBatchGridEnv(grid_size, color_values)
        env = PixelAwareRewardWrapper(env)

        # Create an instance of the ArcDataset
        logger.info("Creating ArcDataset")
        dataset = ArcDataset(
            arc_dataset_dir="rlarcworld/dataset/training",
            keep_in_memory=True,
            transform=ArcSampleTransformer(
                (grid_size, grid_size), examples_stack_dim=10
            ),
        )
        train_samples = DataLoader(dataset=dataset, batch_size=self.batch_size)

        logger.info("Training D4PG")
        self.d4pg.train_d4pg(
            env,
            train_samples,
            batch_size=self.batch_size,
            n_steps=n_steps,
        )


if __name__ == "__main__":
    unittest.main()
