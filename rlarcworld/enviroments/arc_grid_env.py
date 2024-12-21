from typing import Optional
import torch
import numpy as np
import gymnasium as gym
import logging

logger = logging.getLogger(__name__)


class ArcGridEnv(gym.Env):

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

    def _get_obs(self):
        """
        Returns:
            dict: grids with current state and targets
        """
        return {"current": self._current_grids, "target": self._target_grids}

    def _get_info(self):
        """
        Returns:
            dict:
                initial: Grids with the initial state of the problem grids.
                index: Grids with the steps of the modifications performed over the grid.
        """
        return {
            "intial": self._initial_grids,
            "index": self._index_grids,
        }

    def random_location_generator(self) -> tuple:
        """
        Generate a random location in the grid

        Returns:
            tuple:  x and y location within the observation space defined in init
        """
        return tuple(self.observation_space.sample())

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

        # Get batch of input grids
        self._initial_grids = batch_in
        if isinstance(self._initial_grids, torch.Tensor):
            self._current_grids = self._initial_grids.clone()
            # State of initial reward
            self._reward_storage = torch.zeros(self.batch_size, dtype=int)
            self._last_reward = self._reward_storage.clone()
            # Index grid: provides information of the order of the modifications to comply with Markov Assumptions
            self._index_grids = self._initial_grids.clone() * 0
        elif isinstance(self._initial_grids, np.ndarray):
            self._current_grids = self._initial_grids.copy()

            self._reward_storage = np.zeros(self.batch_size, dtype=int)
            self._last_reward = self._reward_storage.copy()

            self._index_grids = self._initial_grids.copy() * 0
        else:
            ValueError("The current grid is not of type torch.Tensor or np.ndarray.")

        self._target_grids = batch_out

        self._timestep = 0
        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def get_difference(self):
        """
        Compute the difference between the current grid and the target grid.

        Returns:
            np.ndarray | torch.Tensor: The difference between the current grid and the target grid.
        """
        return self._current_grids - self._target_grids

    def reward(self, terminated: bool):
        """
        Computes the reward for the current state of the environment.

        Args:
            terminated (bool): Indicates whether the agent indicated a submission.
        Returns:
            float: The reward for the current state of the environment.
        """
        if isinstance(self._current_grids, torch.Tensor):
            return torch.sum(
                torch.abs(self.get_difference()), dim=(1, 2)
            ) * torch.tensor(
                terminated,
                device=self._current_grids.device,
                dtype=self._current_grids.dtype,
            )
        elif isinstance(self._current_grids, np.ndarray):
            return np.sum(np.abs(self.get_difference()), axis=(1, 2)) * terminated
        else:
            raise TypeError(
                "The current grid is not of type torch.Tensor or np.ndarray."
            )

    def step(self, actions: list):

        if self.action_space.contains(actions):
            logger.debug("Actions are valid")
            # Update the grid with the action.
            y, x, color, submission = (
                actions[:, 0],
                actions[:, 1],
                actions[:, 2],
                actions[:, 3],
            )
            self._timestep += 1
            if isinstance(self._current_grids, torch.Tensor) and isinstance(
                color, np.ndarray
            ):
                color = torch.tensor(
                    color,
                    device=self._current_grids.device,
                    dtype=self._current_grids.dtype,
                )
            elif isinstance(self._current_grids, np.ndarray) and isinstance(
                color, torch.Tensor
            ):
                color = color.numpy(force=True)
            else:
                raise TypeError(
                    "The current grid is not of type torch.Tensor or np.ndarray."
                )
            logger.info(
                "Action performed shapes: Y={}, X={}, Color={}, Submission={}".format(
                    y.shape, x.shape, color.shape, submission.shape
                )
            )
            self._current_grids[list(range(self.batch_size)), y, x] = color
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

        terminated = np.all(submission == 1, axis=-1).astype(dtype=int)
        reward = self.reward(terminated)
        if isinstance(self._reward_storage, torch.Tensor) and isinstance(
            reward, np.ndarray
        ):
            reward = torch.tensor(
                reward,
                device=self._reward_storage.device,
                dtype=self._reward_storage.dtype,
            )
        elif isinstance(self._reward_storage, np.ndarray) and isinstance(
            reward, torch.Tensor
        ):
            reward = reward.numpy(force=True)
        self._reward_storage += reward
        self._last_reward = reward

        # TODO: No truncate version, evaluate time constraints
        truncated = False
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info
