import os
from typing import List, Optional
import torch
import gymnasium as gym
import logging

from rlarcworld.utils import TorchQueue

logger = logging.getLogger(__name__)


class PixelAwareRewardWrapper(gym.Wrapper):

    def __init__(
        self,
        env,
        max_penality: float = -1.0,
        n_steps: int = 1,
        gamma: float = 1.0,
        apply_clamp: bool = False,
        v_min: int = None,
        v_max: int = None,
    ):
        super().__init__(env)
        assert isinstance(max_penality, float), "max_penality must be a float"
        assert (
            isinstance(n_steps, int) and n_steps > 0
        ), "n_steps must be a positive int"
        assert (
            isinstance(gamma, float) and gamma > 0 and gamma <= 1
        ), "gamma must be a positive float lower or equal than 1.0"
        assert isinstance(apply_clamp, bool), "apply_clamp must be a bool"
        if apply_clamp:
            assert v_max is None or isinstance(v_max, int)
            assert v_min is None or isinstance(v_min, int)
            assert v_min < v_max

        self.__max_penality = max_penality
        self.apply_clamp = apply_clamp
        self.v_min = v_min
        self.v_max = v_max
        self.n_steps = n_steps
        self.gamma = gamma

    def reset(self, *, seed=None, options=None):
        reset = super().reset(seed=seed, options=options)
        self.batch_size = self.get_wrapper_attr("batch_size")
        # Reward attributes
        self.discount_factor = torch.ones((self.batch_size, self.n_steps)) * (
            self.gamma ** torch.arange(1, self.n_steps + 1)
        )
        self._reward_storage = TorchQueue(
            torch.zeros((self.batch_size, self.n_steps)), q_size=self.n_steps, q_dim=1
        )
        self._last_reward = torch.zeros(self.batch_size, dtype=int)
        return reset

    def n_step_reward(
        self, apply_clamp: bool = False, v_min: int = None, v_max: int = None
    ):
        """
        Computes the reward for the current grid of the environment.
        """
        assert isinstance(apply_clamp, bool), "apply_clamp must be a bool"
        if apply_clamp:
            return torch.clamp(
                torch.sum(self._reward_storage * self.discount_factor, dim=1),
                min=v_min,
                max=v_max,
            )
        elif self.apply_clamp:
            return torch.clamp(
                torch.sum(self._reward_storage * self.discount_factor, dim=1),
                min=self.v_min,
                max=self.v_max,
            )
        else:
            return torch.sum(self._reward_storage * self.discount_factor, dim=1)

    def get_state(self, **kwargs):
        return self.env.get_state(**kwargs)

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

    @property
    def max_penality(self):
        """
        Maximum penality for the current grid.
        """
        return self.__max_penality

    def reward(
        self,
        last_diffs: torch.Tensor,
        grid_diffs: torch.Tensor,
        submission: List[int],
    ):
        """
        Return reward based on actions and grids

        Notes:
            1. These considerations are for binary differences between the current and target grid.
            2. Consider that the agent can only modify one grid per step and can submit in this same action.
            3. The improvement is considered the change between the last difference and the current difference.
            4. If the improvement is 0, the reward is -1.
            5. If the improvement is positive, the reward is 0.
            6. If the improvement is negative, the reward is -2.
            7. The penalization is proportional to the difference between the current and the target grid.
            8. In a 10x10 grid with all grids different and a max_penality of -1, the penalization would be -100.
            9. If the agent has submitted and there exists a difference between the current and the target grid, the reward is the improvement plus the maximum penalty.
            10. If the agent has submitted and there is no difference between the current and the target grid, the reward is 1.

        Args:
            last_diffs (torch.Tensor): The difference between the last grid and the target grid.
            grid_diffs (torch.Tensor): The difference between the current grid and the target grid.
            submission (List[int]): Indicates whether the agent indicated a submission to grade.

        Returns:
            torch.Tensor: The reward for the current grid of the environment.
        """
        penalization = submission * self.max_penality
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
        reward = (
            (improvement - 1) * (grid_diff != 0).long()
            + penalization
            + (grid_diff == 0).long() * submission
        )
        if self.apply_clamp:
            return torch.clamp(reward, min=self.v_min, max=self.v_max)
        else:
            return reward

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
        self._reward_storage = self._reward_storage.push(reward.unsqueeze(1))
        self._last_reward = reward
        return obs, reward, terminated, truncated, info
