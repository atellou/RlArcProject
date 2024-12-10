from typing import Optional
import numpy as np
import gymnasium as gym


class GridWorldEnv(gym.Env):

    def __init__(self, arc_dataset_dir: str, max_size: int = 30, color_values: int = 9):
        self.arc_dir = arc_dataset_dir

        # The maximum size of the square grid
        self.max_size = max_size

        # Define the current and target grids; chosen in `reset` and updated in `step`.
        self._current_grid = np.zeros((max_size, max_size), dtype=np.int32)
        self._target_grid = np.zeros((max_size, max_size), dtype=np.int32)

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`-1}^2
        self.observation_space = gym.spaces.Dict(
            {
                "current": gym.spaces.Box(0, max_size - 1, shape=(2,), dtype=int),
                "target": gym.spaces.Box(0, max_size - 1, shape=(2,), dtype=int),
            }
        )

        # We have actions corresponding to "X Location", "Y Location", "Color Value"
        self.action_space = gym.spaces.MultiDiscrete([max_size, max_size, 9])

    def _get_obs(self):
        return {"agent": self._current_grid, "target": self._target_grid}

    def _get_info(self):
        return {"grid_differences": self._current_grid - self._target_grid}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
