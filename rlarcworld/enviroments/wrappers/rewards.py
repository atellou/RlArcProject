from typing import List, Optional
import torch
import gymnasium as gym
import logging

logger = logging.getLogger(__name__)


class PixelAwareRewardWrapper(gym.Wrapper):

    def get_difference(self, binary: bool = True):
        """
        Compute the difference between the current grid and the target grid and return ones for differences.

        Returns:
            torch.Tensor: The difference between the current grid and the target grid.
        """
        obs = self.get_wrapper_attr("observations")
        current_grids = obs["current"]
        target_grids = obs["target"]
        if binary:
            diff = (current_grids - target_grids) != 0
        else:
            return current_grids - target_grids
        return diff.long()

    def __len__(self):
        return len(self.env)

    def max_manality(self):
        """
        Maximum penality for the current grid.
        """
        return (
            -1
            * self.get_wrapper_attr("size") ** 2
            * self.get_wrapper_attr("color_values")
        )

    def reward(
        self,
        last_diffs: torch.Tensor,
        grid_diffs: torch.Tensor,
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
        last_diffs = torch.sum(torch.abs(last_diffs), dim=(1, 2))
        current_diff = torch.sum(torch.abs(grid_diffs), dim=(1, 2))
        if not isinstance(penalization, torch.Tensor):
            penalization = torch.tensor(
                penalization, device=grid_diffs.device, dtype=grid_diffs.dtype
            )
        penalization = (current_diff > 0).long() * penalization
        penalization = current_diff * penalization
        improvement = last_diffs - current_diff
        # The minus one is to avoid the reward to be positive (maximum reward is 0)
        return (improvement - 1) * (current_diff != 0).long() + penalization

    def step(self, actions: list):
        self.last_diffs = self.get_difference()
        obs, _, terminated, truncated, info = self.env.step(actions)
        self.current_diffs = self.get_difference()
        reward = self.reward(self.last_diffs, self.current_diffs, actions["submit"])
        return obs, reward, terminated, truncated, info
