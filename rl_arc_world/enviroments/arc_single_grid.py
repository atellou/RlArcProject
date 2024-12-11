from typing import Optional
import numpy as np
import gymnasium as gym
import logging

logger = logging.getLogger(__name__)


class ArcSingleGrid(gym.Env):

    def __init__(self, arc_dataset_dir: str, size: int = 30, color_values: int = 9):
        self.arc_dir = arc_dataset_dir

        # The maximum size of the square grid
        self.size = size
        # Color value extra for resizing
        self.color_values = color_values + 1

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`-1}^2
        self.observation_space = gym.spaces.Dict(
            {
                "current": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # We have actions corresponding to "X Location", "Y Location", "Color Value" and "submission"
        self.action_space = gym.spaces.MultiDiscrete([size, size, color_values, 1])

    def _get_obs(self):
        return {"current": self._current_grid, "target": self._target_grid}

    def _get_info(self):
        return {
            "intial": self._initial_grid,
            "index": self._index_grid,
        }

    def random_grid_generator(self):
        """
        Generate a random grid within the max size and color values parameters
        """
        # TODO: Get sample from dataset. Currently just random grid.
        return self.np_random.integers(
            0, self.color_values, size=(self.size, self.size), dtype=int
        )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self._reward_storage = 0

        # State of initial reward
        self._last_reward = self._compute_reward()

        # Get initial grid
        # TODO: Get sample from dataset. Currently just random grid.
        self._initial_grid = self.random_grid_generator()
        self._current_grid = self._initial_grid.copy()

        # Index grid: provides information of the order of the modifications to comply with Markov Assumptions
        self._timestep = 0
        self._index_grid = np.zeros((self.size, self.size), dtype=np.int32)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_grid = self.random_grid_generator()
        while np.array_equal(self._current_grid, self._target_grid):
            self._target_grid = self.random_grid_generator()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def _compute_reward(self):
        return -1 * np.sum(np.abs(self._current_grid - self._target_grid))

    def step(self, action):

        if self.action_space.contains(action):
            logger.debug("Action is valid")
            # Update the grid with the action.
            x, y, color, submission = action
            self._timestep += 1
            self._current_grid[x, y] = color
            self._index_grid[x, y] = self._timestep
        else:
            logger.error(
                "The specified action do not comply with the action space constraints."
            )
            logger.warning(
                "No action performed due to invalid action: {}.".format(action)
            )

        # An environment is completed if and only if the grids overlap
        reward = self._compute_reward()
        terminated = reward == 0 and submission
        if terminated:
            logger.info("Successfully completed environment.")
            # Maximum reward: expected to encourage exploration
            reward = abs(self._reward_storage)
        elif reward != 0 and submission:
            logger.warning("Attempting to submit with reward equal {}.".format(reward))
            # Maximum penalization
            reward = -1 * self.size * self.size * self.color_values

        self._reward_storage += reward
        self._last_reward = reward

        # TODO: No truncate version, evaluate time constraints
        truncated = False
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info
