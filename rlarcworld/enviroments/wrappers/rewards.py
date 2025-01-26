from typing import List, Optional
import torch
import gymnasium as gym
import logging

logger = logging.getLogger(__name__)


class PixelAwareRewardWrapper(gym.Wrapper):

    def get_difference(self, binary: bool = True):
        """
        Compute the difference between the current grid and the target grid and return ones for differences.

        Args:
            binary (bool): Whether to return a binary difference or the actual difference.
        Returns:
            torch.Tensor: The difference between the current grid and the target grid.
        """
        obs = self.get_wrapper_attr("observations")
        grid_grids = obs["grid"]
        target_grids = obs["target"]
        if binary:
            diff = (grid_grids - target_grids) != 0
        else:
            return grid_grids - target_grids
        return diff.long()

    def __len__(self):
        return len(self.env)

    def max_penality(self):
        """
        Maximum penality for the current grid.
        """
        return -1

    def reward(
        self,
        last_diffs: torch.Tensor,
        grid_diffs: torch.Tensor,
        submission: List[int],
    ):
        """
        Return reward based on actions and grids

        Cases:
            1. If the agent has submited and exist a difference between the grid and the target, the reward includes the maximum penalty.
            2. If the agent has not submited, the reward do not take into account the maximum penalty.
            3. On each step, the reward is the difference between the last difference and the current difference minus 1 (improvement).

        Args:
            last_diffs (torch.Tensor): The difference between the last grid and the target grid.
            grid_diffs (torch.Tensor): The difference between the current grid and the target grid.
            submission (List[int]): Indicates whether the agent indicated a submission to grade.

        Returns:
            torch.Tensor: The reward for the current grid of the environment.
        """
        penalization = submission * self.max_penality()
        last_diffs = torch.sum(torch.abs(last_diffs), dim=(1, 2))
        grid_diff = torch.sum(torch.abs(grid_diffs), dim=(1, 2))
        if not isinstance(penalization, torch.Tensor):
            penalization = torch.tensor(
                penalization, device=grid_diffs.device, dtype=grid_diffs.dtype
            )
        penalization = (grid_diff > 0).long() * penalization
        penalization = grid_diff * penalization
        improvement = last_diffs - grid_diff
        # The minus one is to avoid the reward to be positive (maximum reward is 0)
        return (
            (improvement - 1) * (grid_diff != 0).long()
            + penalization
            + (grid_diff == 0).long() * submission
        )

    def step(self, actions: list):
        """
        Step through the environment and return the observation, reward, terminated, truncated, and info.

        Args:
            actions (list): The actions to take in the environment.

        Returns:
            obs: The observation of the environment.
            reward: The reward of the environment.
            terminated: Whether the episode has been terminated.
            truncated: Whether the episode has been truncated.
            info: Additional information about the environment.
        """
        self.last_diffs = self.get_difference()
        obs, _, terminated, truncated, info = self.env.step(actions)
        self.grid_diffs = self.get_difference()
        reward = self.reward(self.last_diffs, self.grid_diffs, actions["submit"])
        return obs, reward, terminated, truncated, info
