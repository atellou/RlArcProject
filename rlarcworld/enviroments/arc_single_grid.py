from typing import Optional
import numpy as np
import gymnasium as gym
import logging

logger = logging.getLogger(__name__)


class ArcSingleGrid(gym.Env):
    # Possibly multiple test grids.
    # Return two samples to evaluate.

    def __init__(self, size: int, color_values: int = 9):
        # 9 possible values from arc and extras for resizing and no action.
        self.color_values = color_values + 2

        # Size of the grid, assumed to be a MxM grid
        self.size = size

        # Here, the observations will be positions on the grid. Used mainly to add stochasticity
        self.observation_space = gym.spaces.Sequence(
            gym.spaces.MultiDiscrete([size, size]), stack=True
        )

        # We have actions corresponding to "X Location", "Y Location", "Color Value" and "submission"
        self.action_space = gym.spaces.Sequence(
            gym.spaces.MultiDiscrete([size, size, color_values + 1, 2]), stack=True
        )

    def _get_obs(self):
        return {"current": self._current_grids, "target": self._target_grids}

    def _get_info(self):
        return {
            "intial": self._initial_grids,
            "index": self._index_grids,
        }

    def random_grid_generator(self):
        """
        Generate a random location in the grid
        """
        return tuple(self.observation_space.sample())

    def reset(self, options: dict, seed: Optional[int] = None) -> tuple:
        """
        Resets the environment to an initial internal state, returning an initial observation and info.
        Args:
            options (dict): A dictionary containing the batch of samples to be used in the environment.
            seed (int, optional): The seed for the random number generator. Defaults to None.
        Returns:
            tuple: A tuple containing the initial observation and info.
        """
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # The sample for every episode is a batch of ARC Samples
        batch = options["batch"]
        n_samples = batch.shape[0]

        # State of initial reward
        self._reward_storage = np.zeros(n_samples, dtype=int)
        self._last_reward = self._reward_storage.copy()

        # Get batch of input grids
        self._initial_grids = batch["input"]
        self._current_grids = self._initial_grids.copy()

        # Index grid: provides information of the order of the modifications to comply with Markov Assumptions
        self._timestep = 0
        self._index_grids = np.zeros((n_samples, self.size, self.size), dtype=np.int32)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_grids = batch["output"]

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def reward(self, terminated: bool):
        return np.array_equal(self._current_grids, self._target_grids) * terminated

    def step(self, actions: list):

        if self.action_space.contains(actions):
            logger.debug("Actions are valid")
            # Update the grid with the action.
            x, y, color, submission = actions
            self._timestep += 1
            self._current_grid[x, y] = color
            self._index_grid[x, y] = self._timestep
        else:
            logger.error(
                "No action performed due to invalid action, sequence of values"
                + " within {} are valid, not inclusive. Given: {}.".format(
                    self.action_space.feature_space.nvec, actions
                )
            )
            raise ValueError(
                "The specified action do not comply with the action space constraints."
            )

        terminated = np.all(submission == 1, axis=-1)
        reward = self.reward(terminated)

        self._reward_storage += reward
        self._last_reward = reward

        # TODO: No truncate version, evaluate time constraints
        truncated = False
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info
