from typing import Optional, List
import torch
from tensordict import TensorDict
import numpy as np
import gymnasium as gym
import logging

logger = logging.getLogger(__name__)


class ArcBatchGridEnv(gym.Env):

    def __init__(self, size: int, color_values: int):
        # 9 possible values from arc and extras for resizing and no action.
        self.color_values = color_values

        # Size of the grid, assumed to be a MxM grid
        self.size = size

        # Here, the observations will be positions on the grid with a value to set. Used mainly to add stochasticity
        self.observation_space = gym.spaces.Sequence(
            gym.spaces.MultiDiscrete([size, size, color_values]), stack=True
        )

        # We have actions corresponding to "Y Location", "X Location", "Color Value" and "submission"
        self.action_space = gym.spaces.Sequence(
            gym.spaces.MultiDiscrete([size, size, color_values, 2]), stack=True
        )

    def __len__(self):
        return len(self.observations["current"])

    def random_location_generator(self) -> tuple:
        """
        Generate a random location in the grid

        Returns:
            tuple:  x and y location within the observation space defined in init
        """
        return tuple(self.observation_space.sample())

    @property
    def reward_storage(self):
        return self._reward_storage

    @property
    def last_reward(self):
        return self._last_reward

    def reset(self, options: dict, seed: Optional[int] = None) -> tuple:
        """
        Resets the environment to an initial internal state, returning an initial observation and info.
        Args:
            options (dict): A dictionary containing the batch of samples (axis=0) to be used in the environment.
            seed (int, optional): The seed for the random number generator. Defaults to None.
        Returns:
            tuple: A tuple containing the initial observation and info.
        """
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # The sample for every episode is a batch of ARC Samples
        batch = options["batch"]
        batch_in = batch["input"]
        batch_out = batch["output"]
        assert (
            len(batch_in.shape) == 3
            and batch_in.shape[1] == self.size
            and batch_in.shape[2] == self.size
        ), "The batch input shape is not correct. A shape of (batch_size, {size}, {size}) is expected".format(
            self.size
        )
        assert (
            len(batch_out.shape) == 3
            and batch_out.shape[1] == self.size
            and batch_out.shape[2] == self.size
        ), "The batch output (target) shape is not correct. A shape of (batch_size, {size}, {size}) is expected".format(
            self.size
        )
        self.batch_size = batch_in.shape[0]
        # Tergets
        self._target_grids = batch_out
        # Get batch of input grids
        self._initial_grids = batch_in
        self._current_grids = self._initial_grids.clone()
        # State of initial reward
        self._reward_storage = torch.zeros(self.batch_size, dtype=int)
        self._last_reward = self._reward_storage.clone()
        # Index grid: provides information of the order of the modifications to comply with Markov Assumptions
        self._index_grids = self._initial_grids.clone() * 0
        self._timestep = 0

        self.information = TensorDict(
            {
                "initial": self._initial_grids,
                "index": self._index_grids,
            }
        )

        self.observations = TensorDict(
            {"current": self._current_grids, "target": self._target_grids}
        )

        return self.observations, self.information

    def get_difference(self):
        """
        Compute the difference between the current grid and the target grid.

        Returns:
            torch.Tensor: The difference between the current grid and the target grid.
        """
        return self.observations["current"] - self.observations["target"]

    def episode_terminated(self, submission):
        diff = self.get_difference()
        return (torch.sum(torch.abs(diff), dim=(1, 2)) == 0) * torch.as_tensor(
            submission,
            device=diff.device,
            dtype=diff.dtype,
        )

    def reward(
        self,
        grid_diffs: torch.Tensor,
        submission: List[int],
    ):
        """
        Computes the reward for the current state of the environment.

        Args:
            grid_diffs torch.Tensor: Batch of grids representing the difference between current and target.
            submission List(int): Indicates whether the agent indicated a submission to grade.
        Returns:
            float: The reward for the current state of the environment.
        """
        return (
            torch.sum(torch.abs(grid_diffs), dim=(1, 2)) == 0
        ).long() * torch.as_tensor(
            submission,
            device=grid_diffs.device,
            dtype=grid_diffs.dtype,
        )

    def step(self, actions: list):

        if self.action_space.contains(actions.numpy(force=True)):
            logger.debug("Actions are valid")
            # Update the grid with the action.
            y, x, color, submission = (
                actions[:, 0],
                actions[:, 1],
                actions[:, 2],
                actions[:, 3],
            )
            self._timestep += 1
            color = torch.as_tensor(
                color,
                device=self._current_grids.device,
                dtype=self._current_grids.dtype,
            )
            logger.debug(
                "Action performed shapes: Y={}, X={}, Color={}, Submission={}".format(
                    y.shape, x.shape, color.shape, submission.shape
                )
            )
            self.observations["current"][list(range(self.batch_size)), y, x] = color
            self._index_grids[list(range(self.batch_size)), y, x] = self._timestep
        else:
            logger.error(
                "No action performed due to invalid action, sequence of values"
                + " within {} are valid, not inclusive. Given: {}.".format(
                    self.action_space.feature_space.nvec, actions
                )
            )
            raise ValueError(
                "The specified action do not comply with the action space constraints."
                + "Sequence of values within {} are valid, not inclusive. Given: {}.".format(
                    self.action_space.feature_space.nvec, actions
                )
            )

        reward = self.reward(self.get_difference(), submission)
        reward = torch.as_tensor(
            reward,
            device=self._reward_storage.device,
            dtype=self._reward_storage.dtype,
        )
        self._reward_storage += reward
        self._last_reward = reward

        # TODO: No truncate version, evaluate time constraints
        truncated = False
        terminated = self.episode_terminated(submission)
        return (
            self.observations,
            reward,
            torch.sum(terminated) == len(self),
            truncated,
            self.information,
        )


print("Registering gymnasium environment")
gym.envs.registration.register(
    id="ArcBatchGrid-v0",
    entry_point="rlarcworld.enviroments.arc_batch_grid_env:ArcBatchGridEnv",
    nondeterministic=True,
)
