from typing import List, Optional
import numpy as np
import torch
import gymnasium as gym
import logging

logger = logging.getLogger(__name__)


class ReacherRewardWrapper(gym.Wrapper):

    def get_binary_difference(self):
        """
        Compute the difference between the current grid and the target grid and return ones for differences.

        Returns:
            np.ndarray | torch.Tensor: The difference between the current grid and the target grid.
        """
        obs = self.env.observations
        current_grids = obs["current"]
        target_grids = obs["target"]
        diff = (current_grids - target_grids) != 0
        if isinstance(current_grids, torch.Tensor):
            return diff.long()
        elif isinstance(current_grids, np.ndarray):
            return diff.astype(int)
        else:
            raise TypeError(
                "The current grid is not of type torch.Tensor or np.ndarray."
            )

    def reward(
        self,
        last_diffs: torch.Tensor | np.ndarray,
        grid_diffs: torch.Tensor | np.ndarray,
        terminated: List[int],
    ):
        penalization = terminated * (-1 * self.env.size**2 * self.env.color_values)
        if isinstance(grid_diffs, torch.Tensor):
            current_diff = torch.sum(torch.abs(grid_diffs), dim=(1, 2))
            if not isinstance(penalization, torch.Tensor):
                penalization = torch.tensor(
                    penalization, device=grid_diffs.device, dtype=grid_diffs.dtype
                )
            penalization = (current_diff > 0).long() * penalization
        elif isinstance(grid_diffs, np.ndarray):
            current_diff = np.sum(np.abs(grid_diffs), axis=(1, 2))
            penalization = (current_diff > 0).astype(int) * penalization
        else:
            raise TypeError(
                "The current grid is not of type torch.Tensor or np.ndarray."
            )
        penalization = current_diff * penalization
        improvement = last_diffs - current_diff
        return improvement + penalization

    def step(self, action: list):
        try:
            self.last_diffs = self.current_diffs
        except AttributeError:
            self.last_diffs = self.env.get_difference()
        obs, _, terminated, truncated, info = self.env.step(action)
        self.current_diffs = self.env.get_difference()
        reward = self.reward(self.last_diffs, terminated)
        return obs, reward, terminated, truncated, info
