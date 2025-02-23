import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import tensordict
from tensordict import TensorDict
from torchrl.data.replay_buffers import (
    TensorDictReplayBuffer,
    LazyTensorStorage,
    PrioritizedSampler,
)

from rlarcworld.enviroments.arc_batch_grid_env import ArcBatchGridEnv
from rlarcworld.enviroments.wrappers.rewards import PixelAwareRewardWrapper
from rlarcworld.arc_dataset import ArcDataset, ArcSampleTransformer
from rlarcworld.agent.actor import ArcActorNetwork
from rlarcworld.agent.critic import ArcCriticNetwork
from rlarcworld.algorithms.d4pg import D4PG
from rlarcworld.utils import get_nested_ref

import unittest
import logging

from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


logger.info("Setting up D4PG test")


class TestD4PG(unittest.TestCase):

    def setUp(self):
        self.batch_size = torch.randint(1, 20, size=(1,)).item()
        self.grid_size = 30
        self.color_values = 11
        self.num_atoms = {"pixel_wise": 50, "binary": 3}

        self.actor = ArcActorNetwork(
            size=self.grid_size, color_values=self.color_values
        )
        v_min = {"pixel_wise": -40, "binary": 0}
        v_max = {"pixel_wise": 2, "binary": 1}

        env_binary = ArcBatchGridEnv(self.grid_size, self.color_values)
        self.env = PixelAwareRewardWrapper(env_binary)
        dataset = ArcDataset(
            arc_dataset_dir="rlarcworld/dataset/training",
            keep_in_memory=False,
            transform=ArcSampleTransformer(
                (self.grid_size, self.grid_size), examples_stack_dim=10
            ),
        )
        self.train_samples = DataLoader(dataset=dataset, batch_size=len(dataset) // 2)
        self.critic = ArcCriticNetwork(
            size=self.grid_size,
            color_values=self.color_values,
            num_atoms=self.num_atoms,
            v_min=v_min,
            v_max=v_max,
        )
        self.d4pg = D4PG(
            env=self.env,
            actor=self.actor,
            critic=self.critic,
            train_samples=self.train_samples,
            batch_size=self.batch_size,
            target_update_frequency=5,
            n_steps=self.env.n_steps,
            gamma=self.env.gamma,
        )

        self.replay_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(self.batch_size),
            sampler=PrioritizedSampler(
                max_capacity=self.batch_size * 2,
                alpha=1.0,
                beta=1.0,
            ),
        )

        self.d4pg_with_replay_buffer = D4PG(
            env=self.env,
            actor=self.actor,
            critic=self.critic,
            train_samples=self.train_samples,
            batch_size=self.batch_size,
            replay_buffer=self.replay_buffer,
            target_update_frequency=5,
            n_steps=self.env.n_steps,
            gamma=self.env.gamma,
        )

    def simmulated_data(self):
        reward = TensorDict(
            {
                "pixel_wise": torch.randint(-40, 2, size=(self.batch_size, 1)),
                "binary": torch.randint(0, 2, size=(self.batch_size, 1)),
            }
        ).to(self.actor.device)

        done = torch.randint(0, 2, size=(self.batch_size, 1)).to(self.actor.device)
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
                "terminated": torch.randint(0, 2, size=(self.batch_size, 1)),
            }
        ).to(self.actor.device)
        return reward, done, gamma, state

    def test_target_distribution(self):
        logger.info("Testing target distribution computation")
        reward, done, gamma, next_state = self.simmulated_data()
        target_distribution = self.d4pg.compute_target_distribution(
            reward,
            next_state.clone(),
            done,
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
                torch.sum(dist, dim=1),
                torch.ones(self.batch_size).to(self.actor.device),
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
        )

        reward, done, __, state = self.simmulated_data()
        action_probs = self.actor(state)

        # Backpropagation
        # Define a loss function and an optimizer
        optimizer = torch.optim.AdamW(self.critic.parameters(), lr=0.001)
        loss, td_error, q_dist = self.d4pg.compute_critic_loss(
            state, action_probs, target_distribution
        )
        assert tuple(td_error.keys()) == tuple([])

        loss, td_error, q_dist = self.d4pg.compute_critic_loss(
            state, action_probs, target_distribution, compute_td_error=True
        )
        assert tuple(td_error.keys()) == tuple(["pixel_wise", "binary"])
        assert tuple(td_error["pixel_wise"].shape) == tuple(
            [self.batch_size]
        ), "TD Error shape incorrect, expected [{}], got {}".format(
            self.batch_size, td_error["pixel_wise"].shape
        )
        for key, value in td_error.items():
            assert not torch.isnan(
                value
            ).any(), f"NaN values found in TD Error for key: {key}"

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
        optimizer = torch.optim.AdamW(self.actor.parameters(), lr=0.001)
        loss = self.d4pg.compute_actor_loss(state)
        assert tuple(loss.keys()) == tuple(["pixel_wise", "binary"])
        for k, v in loss.items():
            assert not torch.isnan(v).any(), f"NaN values found in loss for key: [{k}]"

        loss = sum(tuple(loss.values()))
        optimizer.zero_grad()
        loss.backward()
        for name, param in self.actor.named_parameters():
            if "base_model" not in name and param.grad is None:
                raise ValueError(
                    f"Gradient not flowing in D4PG ArcActorNetwork for: {name}"
                )
        optimizer.step()

    def test_actor_loss_carsm(self):
        logger.info("Testing actor loss computation CARSM")
        reward, done, gamma, state = self.simmulated_data()
        target_distribution = self.d4pg.compute_target_distribution(
            reward,
            state.clone(),
            done,
        )
        optimizer = torch.optim.AdamW(self.actor.parameters(), lr=0.001)
        d4pg = D4PG(
            env=self.env,
            actor=self.actor,
            critic=self.critic,
            train_samples=self.train_samples,
            batch_size=self.batch_size,
            target_update_frequency=5,
            n_steps=self.env.n_steps,
            gamma=self.env.gamma,
            carsm=True,
        )
        loss = d4pg.compute_actor_loss(state, target_q=target_distribution)
        assert tuple(loss.keys()) == tuple(["pixel_wise", "binary"])
        for k, v in loss.items():
            assert not torch.isnan(v).any(), f"NaN values found in loss for key: [{k}]"

        loss = sum(tuple(loss.values()))
        optimizer.zero_grad()
        loss.backward()
        for name, param in self.actor.named_parameters():
            if "base_model" not in name and param.grad is None:
                raise ValueError(
                    f"Gradient not flowing in D4PG ArcActorNetwork for: {name}"
                )
        optimizer.step()

    def test_train_step(self):
        logger.info("Testing train step")
        reward, done, __, state = self.simmulated_data()
        __, __, __, next_state = self.simmulated_data()
        action = self.d4pg.actor(state)
        batch = TensorDict(
            {
                "state": state,
                "actions": action,
                "reward": reward,
                "next_state": next_state,
                "terminated": done,
            },
            batch_size=self.batch_size,
        ).to(self.actor.device)

        loss_actor, loss_critic = self.d4pg.compute_loss(batch)

        logger.info(f"Actor loss: {loss_actor}")
        logger.info(f"Critic loss: {loss_critic}")
        # Check that losses are not NaN
        assert not torch.isnan(loss_actor), "Actor loss is NaN"
        assert not torch.isnan(loss_critic), "Critic loss is NaN"

        # Check that gradients are flowing
        for name, param in self.d4pg.actor.named_parameters():
            if "base_model" not in name:
                assert (
                    param.grad is not None
                ), f"Gradient not flowing in actor for: {name}"
        for name, param in self.d4pg.critic.named_parameters():
            if "base_model" not in name:
                assert (
                    param.grad is not None
                ), f"Gradient not flowing in critic for: {name}"

    def test_validation_step(self):
        logger.info("Testing validation step")
        reward, done, __, state = self.simmulated_data()
        __, __, __, next_state = self.simmulated_data()
        action_probs = self.d4pg.actor(state)
        batch = TensorDict(
            {
                "state": state,
                "actions": action_probs,
                "reward": reward,
                "next_state": next_state,
                "terminated": done,
            },
            batch_size=self.batch_size,
        ).to(self.actor.device)

        loss_actor, loss_critic = self.d4pg.compute_loss(batch, training=False)

        logger.info(f"Actor loss: {loss_actor}")
        logger.info(f"Critic loss: {loss_critic}")
        # Check that losses are not NaN
        assert not torch.isnan(loss_actor), "Actor loss is NaN"
        assert not torch.isnan(loss_critic), "Critic loss is NaN"

        # Check that gradients are flowing
        for name, param in self.d4pg.actor.named_parameters():
            assert param.grad is None, f"Gradient not flowing in actor for: {name}"
        for name, param in self.d4pg.critic.named_parameters():
            assert param.grad is None, f"Gradient not flowing in critic for: {name}"

    def test_train_d4pg(self):
        max_steps = torch.randint(5, 20, size=(1,)).item()
        logger.info("Testing D4PG without replay buffer for {} steps".format(max_steps))
        self.d4pg.fit(
            max_steps=max_steps,
        )

    def test_train_d4pg_with_replay_buffer(self):
        max_steps = torch.randint(5, 20, size=(1,)).item()
        logger.info("Testing D4PG with replay buffer for {} steps".format(max_steps))
        self.d4pg_with_replay_buffer.fit(
            max_steps=max_steps,
        )
        logger.info("D4PG with replay buffer finished")

    def test_train_d4pg_with_n_step(self):
        grid_size = 30
        color_values = 11
        max_steps = torch.randint(8, 20, size=(1,)).item()
        n_steps = torch.randint(3, max_steps // 2, size=(1,)).item()
        logger.info(
            "Testing train_d4pg with n_steps {} for {} steps".format(n_steps, max_steps)
        )
        gamma = 0.99
        env = ArcBatchGridEnv(grid_size, color_values, n_steps=n_steps, gamma=gamma)
        env = PixelAwareRewardWrapper(env, n_steps=n_steps, gamma=gamma)

        # Create an instance of the ArcDataset
        dataset = ArcDataset(
            arc_dataset_dir="rlarcworld/dataset/training",
            keep_in_memory=False,
            transform=ArcSampleTransformer(
                (grid_size, grid_size), examples_stack_dim=10
            ),
        )
        train_samples = DataLoader(dataset=dataset, batch_size=len(dataset) // 2)
        replay_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(self.batch_size),
            sampler=PrioritizedSampler(
                max_capacity=self.batch_size,
                alpha=1.0,
                beta=1.0,
            ),
        )
        num_atoms = {"pixel_wise": 50, "binary": 3, "n_reward": 50 * n_steps}
        v_min = {"pixel_wise": -40, "binary": 0, "n_reward": -40 * n_steps}
        v_max = {"pixel_wise": 2, "binary": 1, "n_reward": 2 * n_steps}
        critic = ArcCriticNetwork(
            size=self.grid_size,
            color_values=self.color_values,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max,
        )

        d4pg = D4PG(
            env=env,
            actor=self.actor,
            critic=critic,
            train_samples=train_samples,
            batch_size=self.batch_size,
            replay_buffer=replay_buffer,
            target_update_frequency=5,
            n_steps=env.n_steps,
            gamma=env.gamma,
        )
        d4pg.fit(
            max_steps=max_steps,
        )
        assert torch.any(
            env._reward_storage != 0
        ), "Reward Storage is likely not updating, Reward total: {}".format(
            torch.sum(env._reward_storage)
        )

    # def test_validation_d4pg(self):
    #     grid_size = 30
    #     color_values = 11
    #     max_steps = torch.randint(30, 100, size=(1,)).item()
    #     n_steps = torch.randint(3, 20 // 2, size=(1,)).item()
    #     logger.info(
    #         "Testing train_d4pg with n_steps {} for {} steps".format(n_steps, max_steps)
    #     )
    #     gamma = 0.99
    #     env = ArcBatchGridEnv(grid_size, color_values, n_steps=n_steps, gamma=gamma)
    #     env = PixelAwareRewardWrapper(env, n_steps=n_steps, gamma=gamma)

    #     # Create an instance of the ArcDataset
    #     dataset = ArcDataset(
    #         arc_dataset_dir="rlarcworld/dataset/training",
    #         keep_in_memory=False,
    #         transform=ArcSampleTransformer(
    #             (grid_size, grid_size), examples_stack_dim=10
    #         ),
    #     )
    #     train_samples = DataLoader(dataset=dataset, batch_size=len(dataset) // 2)

    #     val_samples = DataLoader(dataset=dataset, batch_size=len(dataset) // 2)
    #     replay_buffer = TensorDictReplayBuffer(
    #         storage=LazyTensorStorage(self.batch_size),
    #         sampler=PrioritizedSampler(
    #             max_capacity=self.batch_size,
    #             alpha=1.0,
    #             beta=1.0,
    #         ),
    #     )

    #     num_atoms = {"pixel_wise": 50, "binary": 3, "n_reward": 50 * n_steps}
    #     v_min = {"pixel_wise": -40, "binary": 0, "n_reward": -40 * n_steps}
    #     v_max = {"pixel_wise": 2, "binary": 1, "n_reward": 2 * n_steps}
    #     critic = ArcCriticNetwork(
    #         size=self.grid_size,
    #         color_values=self.color_values,
    #         num_atoms=num_atoms,
    #         v_min=v_min,
    #         v_max=v_max,
    #     )

    #     d4pg = D4PG(
    #         env=env,
    #         actor=self.actor,
    #         critic=critic,
    #         train_samples=train_samples,
    #         validation_samples=val_samples,
    #         batch_size=self.batch_size,
    #         replay_buffer=replay_buffer,
    #         target_update_frequency=5,
    #         n_steps=env.n_steps,
    #         gamma=env.gamma,
    #         tb_writer=SummaryWriter(log_dir="runs/test_validation_d4pg"),
    #     )
    #     d4pg.fit(
    #         max_steps=max_steps,
    #         validation_steps_frequency=10,
    #         validation_steps_per_train_step=10,
    #         validation_steps_per_episode=max_steps,
    #         logger_frequency=2,
    #         grads_logger_frequency=3,
    #     )

    #     assert os.path.isdir(
    #         "./runs/test_validation_d4pg"
    #     ), "Directory 'runs/test_validation_d4pg' does not exist"

    #     ref, last_key = get_nested_ref(d4pg.history, "Validation/Loss/actor")
    #     assert isinstance(
    #         ref[last_key], np.ndarray
    #     ), "Invalid validation loss history format - expected np.ndarray for actor, got {}".format(
    #         type(ref[last_key])
    #     )

    #     ref, last_key = get_nested_ref(d4pg.history, "Train/Loss/actor")
    #     assert isinstance(
    #         ref[last_key], np.ndarray
    #     ), "Invalid training loss history format - expected np.ndarray for critic, got {}".format(
    #         type(ref[last_key])
    #     )


if __name__ == "__main__":
    unittest.main()
