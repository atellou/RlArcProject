"""Reward wrapper utilities for ARC environments."""

import logging
from pathlib import Path
from typing import Any

import gymnasium as gym
import torch
from tensordict import TensorDict

from rlarcworld.common.enums import ActionKeys, StateKeys
from rlarcworld.environments.arc_batch_grid_env import (
    ArcBatchGridEnv,
    ArcEnvRunningMode,
)
from rlarcworld.utils.checkpointing import (
    add_schema_and_class,
    assert_expected_class,
    resolve_load_path,
    resolve_save_path,
)
from rlarcworld.utils.torch_queue import TorchQueue

logger = logging.getLogger(__name__)


class PixelAwareRewardWrapper(gym.Wrapper):
    """Compute pixel-aware rewards and optional n-step returns.

    This wrapper augments an ARC environment with reward processing that can
    clamp values and track n-step discounted returns, while keeping the
    interface compatible with Gym wrappers.
    """

    def __init__(
        self,
        env: ArcBatchGridEnv,
        max_penalty: float = 1.0,
        perfect_no_submit_penalty_coef: float = 1.0,
        zero_improvement_penalty: float = 0.0,
        n_steps: int = 1,
        gamma: float = 0.99,
    ):
        """Initialize the PixelAwareRewardWrapper.

        Args:
            env (gym.Env): Environment to wrap.
            max_penalty (float, optional): Maximum penalty for the reward.
                Defaults to 1.0.
            perfect_no_submit_penalty_coef (float, optional): Coefficient for
                perfect no-submit penalty. Defaults to 1.0.
            zero_improvement_penalty (float, optional): Penalty for zero
                improvement. Defaults to 0.0.
            n_steps (int, optional): Number of steps in the reward computation.
                Defaults to 1.
            gamma (float, optional): Discount factor in the reward
                computation. Defaults to 0.99.
        """
        super().__init__(env)
        assert isinstance(max_penalty, float), "max_penalty must be a float"
        assert (
            isinstance(n_steps, int) and n_steps > 0
        ), "n_steps must be a positive int"
        assert (
            isinstance(gamma, float) and gamma > 0 and gamma <= 1
        ), "gamma must be a positive float lower or equal than 1.0"

        assert (
            isinstance(max_penalty, float) and max_penalty > 0
        ), "max_penalty must be a float greater than 0, got {}".format(max_penalty)
        assert (
            isinstance(perfect_no_submit_penalty_coef, float)
            and perfect_no_submit_penalty_coef >= 0
        ), "perfect_no_submit_penalty_coef must be a float >= 0"
        assert (
            isinstance(zero_improvement_penalty, float)
            and zero_improvement_penalty >= 0
        ), "zero_improvement_penalty must be a float >= 0"
        self.__max_penalty = max_penalty
        self.perfect_no_submit_penalty_coef = perfect_no_submit_penalty_coef
        self.zero_improvement_penalty = zero_improvement_penalty
        self.n_steps = n_steps
        self.gamma = gamma
        self.device = self.get_wrapper_attr("device")
        self.grid_size = self.get_wrapper_attr("grid_size")
        self.color_values = self.get_wrapper_attr("color_values")

    def get_discount_factor(self) -> torch.Tensor:
        """Return the per-step discount factors tensor of shape (B, n_steps)."""
        return torch.ones((self.batch_size, self.n_steps)) * (
            self.gamma ** torch.arange(1, self.n_steps + 1)
        )

    def reset(
        self,
        *,
        options: dict[str, torch.Tensor],
        seed: int | None = None,
    ) -> tuple:
        """Reset the environment and initialize reward-related attributes.

        Args:
            seed (int, optional): Seed for RNG. Defaults to None.
            options (dict, optional): Additional options for reset, including
                the batch of samples. Defaults to None.

        Returns:
            The initial observation and information from the parent
            environment's reset method.
        """
        options["mode"] = ArcEnvRunningMode.TRAIN
        reset = super().reset(seed=seed, options=options)
        self.batch_size: int = options["batch"]["input"].shape[0]
        # Reward attributes - pixel-wise rewards
        self._reward_storage = TorchQueue(
            torch.zeros((self.batch_size, self.n_steps)),
            queue_size=self.n_steps,
            queue_dim=1,
        )
        self.last_reward = torch.zeros(self.batch_size, dtype=torch.float32)
        self.discount_factor: torch.Tensor = self.get_discount_factor().to(self.device)
        self._reward_storage = self._reward_storage.to(self.device)
        self.last_reward = self.last_reward.to(self.device)
        return reset

    def n_step_reward(self) -> torch.Tensor:
        """Compute the pixel-wise n-step reward for the current grid.

        Returns:
            torch.Tensor: The computed n-step pixel-wise reward.
        """
        total = torch.sum(self._reward_storage * self.discount_factor, dim=1)
        return total.to(self.device)

    def get_state(self, **kwargs) -> dict:
        """Return the state of the environment.

        Args:
            **kwargs: Additional keyword arguments passed to the parent
                environment's get_state method.

        Returns:
            The state of the environment, which is a dictionary containing the
            current grid, last grid, initial grid, index, terminated, and
            examples.
        """
        return self.env.get_state(**kwargs)

    def get_difference(self, binary: bool = True) -> torch.Tensor:
        """Compute the difference between current and target grids.

        Args:
            binary (bool, optional): Whether to return the difference as a
                binary tensor (True) or as a numerical tensor (False).
                Defaults to True.

        Returns:
            torch.Tensor: The difference between the current grid and the
            target grid.
        """
        obs = self.get_wrapper_attr("observations")
        grid_grids = obs[StateKeys.GRID.value]
        target_grids = obs[StateKeys.TARGET.value]
        if binary:
            return ((grid_grids - target_grids) != 0).long()
        else:
            return grid_grids - target_grids

    def __len__(self) -> int:
        """Return the number of elements in the underlying environment."""
        return len(self.env)

    @property
    def max_penalty(self):
        """Maximum penalty for the current grid."""
        return self.__max_penalty

    @property
    def reward_storage(self):
        """TorchQueue containing the pixel-wise reward storage."""
        return self._reward_storage

    @reward_storage.setter
    def reward_storage(self, value: torch.Tensor) -> None:
        """Sets the pixel-wise reward storage.

        Args:
            value (torch.Tensor): The reward storage.
        """
        self._reward_storage = TorchQueue(value, queue_size=self.n_steps, queue_dim=1)

    def get_source_env_reward(self) -> torch.Tensor:
        """Returns the last reward from the source environment."""
        q = self.env.reward_storage  # TorchQueue
        dim: int = q.queue_dim  # typically 1
        # build a slice that selects the last element along the queue dimension
        index: list[object] = [slice(None)] * q.ndim
        index[dim] = -1
        res = q[tuple(index)]
        # res is already a tensor; avoid reconstructing via torch.tensor()
        return res.to(self.device)

    def get_source_env_n_step_reward(self) -> torch.Tensor:
        """Returns the last n-step reward from the source environment."""
        return self.env.n_step_reward().to(self.device)

    def reward(
        self,
        last_diffs: torch.Tensor,
        grid_diffs: torch.Tensor,
        submission: list[int] | torch.Tensor,
    ) -> torch.Tensor:
        """Return reward based on actions and grids.

        Notes:
            1. These considerations are for binary differences between the
               current and target grid.
            2. Consider that the agent can only modify one grid per step and
               can submit in this same action.
            3. The improvement is considered the change between the last
               difference and the current difference.
            4. If the improvement is 0, the reward is -1.
            5. If the improvement is positive, the reward is 0.
            6. If the improvement is negative, the reward is -2.
            7. The penalization is proportional to the difference between the
               current and the target grid.
            8. In a 10x10 grid with all grids different and a max_penalty of
               -1, the penalization would be -100.
            9. If the agent has submitted and there exists a difference
               between the current and the target grid, the reward is the
               improvement plus the maximum penalty.
            10. If the agent has submitted and there is no difference between
                the current and the target grid, the reward is 1.

        Args:
            last_diffs (torch.Tensor): The difference between the last grid
                and the target grid.
            grid_diffs (torch.Tensor): The difference between the current grid
                and the target grid.
            submission (List[int] | torch.Tensor): Indicates whether the agent
                indicated a submission to grade.

        Returns:
            torch.Tensor: The reward for the current grid of the environment.
        """
        # Ensure submission is a tensor on the correct device
        if not isinstance(submission, torch.Tensor):
            submission = torch.tensor(submission, device=grid_diffs.device)
        else:
            submission = submission.to(grid_diffs.device)

        # Work in normalized pixel units so the signal stays in a small,
        # meaningful range and doesn't saturate downstream.
        last_diff = torch.sum(torch.abs(last_diffs), dim=(1, 2)).float()
        grid_diff = torch.sum(torch.abs(grid_diffs), dim=(1, 2)).float()

        # Dense shaping: positive when we reduce diff, negative when we make it worse.
        shaping = (last_diff - grid_diff).clamp(-1.0, 1.0)
        if self.zero_improvement_penalty != 0.0:
            shaping = torch.where(
                shaping == 0,
                shaping - float(self.zero_improvement_penalty),
                shaping,
            )

        # Submit grading term: +1 if perfect, else -max_penalty.
        perfect = (grid_diff == 0).float()
        submit_term = submission.float() * (
            perfect * 1.0 + (1.0 - perfect) * (-float(self.max_penalty))
        )
        # If the grid is already perfect but the agent doesn't submit, apply a small
        # penalty to avoid the "stall at 0 reward" local optimum.
        not_submit_term = (
            (1.0 - submission.float())
            * perfect
            * (-float(self.max_penalty))
            * float(self.perfect_no_submit_penalty_coef)
        )

        return shaping + submit_term + not_submit_term, {
            "shaping": shaping,
            "submit_term": submit_term,
            "not_submit_term": not_submit_term,
        }

    def step(
        self, actions: TensorDict | dict[str, torch.Tensor]
    ) -> tuple[TensorDict, torch.Tensor, bool, bool, dict]:
        """Step through the environment and return step outputs.

        Args:
            actions (list): The actions to take in the environment. This is a
                list of length batch_size where each element is a dictionary
                containing the following keys:
                - y_location (int): The y location of the action.
                - x_location (int): The x location of the action.
                - color_values (int): The color values of the action.
                - submit (int): Whether the agent is submitting the current
                  grid for grading.

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
        reward, reward_info = self.reward(
            self.last_diffs, self.grid_diffs, actions[ActionKeys.SUBMIT.value]
        )
        if isinstance(info, TensorDict):
            info.update(reward_info)
        elif isinstance(info, dict):
            info.update(reward_info)

        self._reward_storage = self._reward_storage.push(reward.unsqueeze(-1))
        self.last_reward = reward

        return obs, reward, terminated, truncated, info

    def checkpoint(self) -> dict[str, Any]:
        """Checkpoint the environment.

        Returns:
            dict[str, Any]: The checkpoint.
        """
        try:
            wrapper_payload = {
                "reward_storage": self.reward_storage,
                "last_reward": self.last_reward,
                "max_penalty": self.max_penalty,
                "batch_size": self.batch_size,
                "n_steps": self.n_steps,
                "gamma": self.gamma,
            }
            return {
                "wrapper": add_schema_and_class(
                    wrapper_payload, self.__class__.__name__
                ),
                "env": self.env.checkpoint(),
            }
        except AttributeError as e:
            logger.error(
                "Error checkpointing environment, reset should have been called: %s",
                e,
            )
            raise

    def save_checkpoint(self, path: str | Path):
        """Saves the environment to a checkpoint.

        Args:
            path (str | Path): The path to save the environment to.
        """
        logger.debug(f"Saving environment to {path}")
        try:
            checkpoint_path = resolve_save_path(path, self.__class__.__name__)
            torch.save(self.checkpoint(), checkpoint_path)
        except AttributeError as e:
            logger.error(
                "Error saving environment, reset should have been called by now: %s",
                e,
            )
            raise

    def load(
        self,
        checkpoint: dict[str, Any] | Path | str,
        weights_only: bool = False,
        device: str | None = None,
    ):
        """Load the environment from a checkpoint.

        Args:
            checkpoint (dict[str, Any] | Path | str): The checkpoint to load.
            weights_only (bool, optional): Whether to load only the weights.
                Defaults to False.
            device (str, optional): The device to load the environment on.
                Defaults to None.
        """
        if isinstance(checkpoint, (Path, str)):
            cp_path = resolve_load_path(checkpoint, self.__class__.__name__)
            checkpoint = torch.load(
                cp_path, weights_only=weights_only, map_location=device
            )
        if not isinstance(checkpoint, dict):
            raise ValueError("Checkpoint must be a dictionary or a Path.")
        if "wrapper" not in checkpoint or "env" not in checkpoint:
            raise ValueError("Invalid checkpoint structure for reward wrapper.")
        meta = checkpoint["wrapper"]
        if "class" in meta:
            assert_expected_class(meta.get("class"), self.__class__.__name__)

        # Delegate to wrapped env, forwarding device preference
        self.env.load(checkpoint["env"], weights_only=weights_only, device=device)

        # Restore wrapper state
        self.reward_storage = meta["reward_storage"]
        self.last_reward = meta["last_reward"]
        self.batch_size = meta["batch_size"]

        assert self.max_penalty == meta["max_penalty"]
        assert self.n_steps == meta["n_steps"]
        assert self.gamma == meta["gamma"]

        # Recompute discount factor and ensure tensors are on target device
        target_device = device or self.device
        self.reward_storage = self.reward_storage.to(target_device)
        self.discount_factor = self.get_discount_factor().to(target_device)
