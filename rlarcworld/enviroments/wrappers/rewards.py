from typing import List, Optional
import numpy as np
import torch
import gymnasium as gym
import logging

logger = logging.getLogger(__name__)


class PixelAwareRewardWrapper(gym.Wrapper):

    def reset(self, options: dict, seed: Optional[int] = None):
        return self.env.reset(options, seed)

    @property
    def observations(self):
        """
        Returns:
            dict: grids with current state and targets
        """
        return self.env.observations.copy()

    def get_difference(self, binary: bool = True):
        """
        Compute the difference between the current grid and the target grid and return ones for differences.

        Returns:
            np.ndarray | torch.Tensor: The difference between the current grid and the target grid.
        """
        obs = self.observations
        current_grids = obs["current"]
        target_grids = obs["target"]
        if binary:
            diff = (current_grids - target_grids) != 0
        else:
            return current_grids - target_grids
        if isinstance(current_grids, torch.Tensor):
            return diff.long()
        elif isinstance(current_grids, np.ndarray):
            return diff.astype(int)
        else:
            raise TypeError(
                "The current grid is not of type torch.Tensor or np.ndarray."
            )

    def __len__(self):
        return len(self.env)

    def max_manality(self):
        """
        Maximum penality for the current grid.
        """
        return -1 * self.env.size**2 * self.env.color_values

    def reward(
        self,
        last_diffs: torch.Tensor | np.ndarray,
        grid_diffs: torch.Tensor | np.ndarray,
        submission: List[int],
    ):
        """
        Return reward based on actions and states

        Cases:
            1. If the agent has submited and exist a difference between the grid and the target, the reward includes the maximum penalty.
            2. If the agent has not submited, the reward do not take into account the maximum penalty.
            3. On each step, the reward is the difference between the last difference and the current difference minus 1 (improvement).
        """
        penalization = submission * self.max_manality()
        if isinstance(grid_diffs, torch.Tensor):
            last_diffs = torch.sum(torch.abs(last_diffs), dim=(1, 2))
            current_diff = torch.sum(torch.abs(grid_diffs), dim=(1, 2))
            if not isinstance(penalization, torch.Tensor):
                penalization = torch.tensor(
                    penalization, device=grid_diffs.device, dtype=grid_diffs.dtype
                )
            penalization = (current_diff > 0).long() * penalization
        elif isinstance(grid_diffs, np.ndarray):
            last_diffs = np.sum(np.abs(last_diffs), axis=(1, 2))
            current_diff = np.sum(np.abs(grid_diffs), axis=(1, 2))
            penalization = (current_diff > 0).astype(int) * penalization
        else:
            raise TypeError(
                "The current grid is not of type torch.Tensor or np.ndarray."
            )
        penalization = current_diff * penalization
        improvement = last_diffs - current_diff
        # The minus one is to avoid the reward to be positive (maximum reward is 0)
        return (improvement - 1) * (current_diff != 0).long() + penalization

    def step(self, actions: list):
        self.last_diffs = self.get_difference()
        obs, _, terminated, truncated, info = self.env.step(actions)
        self.current_diffs = self.get_difference()
        reward = self.reward(self.last_diffs, self.current_diffs, actions[:, 3])
        return obs, reward, terminated, truncated, info
