"""Reward wrapper utilities for ARC environments."""

import logging
import os

import gymnasium as gym
import torch
from tensordict import TensorDict

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
        env,
        max_penalty: float = -1.0,
        n_steps: int = 1,
        gamma: float = 1.0,
        apply_clamp: bool = False,
        v_min: int | None = None,
        v_max: int | None = None,
    ):
        """Initialize the PixelAwareRewardWrapper.

        Args:
            env (gym.Env): Environment to wrap.
            max_penalty (float, optional): Maximum penalty for the reward.
                Defaults to -1.0.
            n_steps (int, optional): Number of steps in the reward computation.
                Defaults to 1.
            gamma (float, optional): Discount factor in the reward
                computation. Defaults to 1.0.
            apply_clamp (bool, optional): Whether to apply clamping to the
                reward. Defaults to False.
            v_min (int, optional): Minimum value for clamping. Defaults to
                None.
            v_max (int, optional): Maximum value for clamping. Defaults to
                None.

        Raises:
            AssertionError: If max_penalty is not a float, n_steps is not a
                positive int, gamma is not a positive float less than or equal
                to 1.0, apply_clamp is not a bool, v_max is not None or an
                int, v_min is not None or an int, or v_min >= v_max when
                apply_clamp is True.
        """
        super().__init__(env)
        assert isinstance(max_penalty, float), "max_penalty must be a float"
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
            if isinstance(v_min, int) and isinstance(v_max, int):
                assert v_min < v_max

        self.__max_penalty = max_penalty
        self.apply_clamp = apply_clamp
        self.v_min = v_min
        self.v_max = v_max
        self.n_steps = n_steps
        self.gamma = gamma
        self.device = self.get_wrapper_attr("device")

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
        reset = super().reset(seed=seed, options=options)
        self.batch_size = options["batch"]["input"].shape[0]
        # Reward attributes
        self.discount_factor = self.get_discount_factor().to(self.device)
        self._reward_storage = TorchQueue(
            torch.zeros((self.batch_size, self.n_steps)),
            queue_size=self.n_steps,
            queue_dim=1,
        )
        self.last_reward = torch.zeros(self.batch_size, dtype=int)
        self.discount_factor = self.discount_factor.to(self.device)
        self._reward_storage = self._reward_storage.to(self.device)
        self.last_reward = self.last_reward.to(self.device)
        return reset

    def n_step_reward(
        self, v_min: float | None = None, v_max: float | None = None
    ) -> torch.Tensor:
        """Compute the n-step reward for the current grid of the environment.

        Args:
            apply_clamp (bool, optional): Whether to clamp the reward between
                v_min and v_max. Defaults to False.
            v_min (int, optional): Minimum value for clamping. Defaults to
                None.
            v_max (int, optional): Maximum value for clamping. Defaults to
                None.

        Returns:
            torch.Tensor: The computed reward.
        """
        total = torch.sum(self._reward_storage * self.discount_factor, dim=1)
        if v_min is not None and v_max is not None:
            total = torch.clamp(total, min=v_min, max=v_max)
        elif self.apply_clamp:
            total = torch.clamp(total, min=self.v_min, max=self.v_max)

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
        grid_grids = obs["grid"]
        target_grids = obs["target"]
        if binary:
            diff = (grid_grids - target_grids) != 0
        else:
            return grid_grids - target_grids
        return diff.long()

    def __len__(self) -> int:
        """Return the number of elements in the underlying environment."""
        return len(self.env)

    @property
    def max_penalty(self):
        """Maximum penalty for the current grid."""
        return self.__max_penalty

    @property
    def reward_storage(self):
        """TorchQueue containing the reward storage."""
        return self._reward_storage

    @reward_storage.setter
    def reward_storage(self, value: torch.Tensor) -> None:
        """Sets the reward storage.

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

    def reward(
        self,
        last_diffs: torch.Tensor,
        grid_diffs: torch.Tensor,
        submission: list[int],
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
            submission (List[int]): Indicates whether the agent indicated a
                submission to grade.

        Returns:
            torch.Tensor: The reward for the current grid of the environment.
        """
        penalization = submission * self.max_penalty
        last_diffs = torch.sum(torch.abs(last_diffs), dim=(1, 2))
        grid_diff = torch.sum(torch.abs(grid_diffs), dim=(1, 2))
        if not isinstance(penalization, torch.Tensor):
            penalization = torch.tensor(
                penalization, device=grid_diffs.device, dtype=grid_diffs.dtype
            ).to(self.device)
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
        reward = self.reward(self.last_diffs, self.grid_diffs, actions["submit"])
        self._reward_storage = self._reward_storage.push(reward.unsqueeze(-1))
        self.last_reward = reward
        return obs, reward, terminated, truncated, info

    def save(self, path: str):
        """Saves the environment to a checkpoint.

        Args:
            path (str): The path to save the environment to.
        """
        logger.debug(f"Saving environment to {path}")
        parent_path = os.path.join(path, "parent")
        child_path = os.path.join(path, "child")
        self.env.save(parent_path)
        try:
            os.makedirs(child_path, exist_ok=True)
            torch.save(
                {
                    "reward_storage": self.reward_storage,
                    "last_reward": self.last_reward,
                    "max_penalty": self.max_penalty,
                    "batch_size": self.batch_size,
                    "n_steps": self.n_steps,
                    "gamma": self.gamma,
                    "apply_clamp": self.apply_clamp,
                    "v_min": self.v_min,
                    "v_max": self.v_max,
                },
                os.path.join(child_path, self.__class__.__name__ + ".ptc"),
            )
        except AttributeError as e:
            logger.error(
                "Error saving environment, reset should have been called by now: %s",
                e,
            )
            raise

    def load(
        self,
        parent_path: str,
        child_path: str,
        weights_only: bool = False,
        device: str | None = None,
    ):
        """Load the environment from a checkpoint.

        Args:
            parent_path (str): Path to the parent environment checkpoint.
            child_path (str): Path to the child environment checkpoint.
            weights_only (bool, optional): Whether to load only the weights.
                Defaults to False.
            device (str, optional): The device to load the environment on.
                Defaults to None.
        """
        self.env.load(parent_path, weights_only=weights_only, device=device)
        checkpoint = torch.load(
            child_path, weights_only=weights_only, map_location=device
        )
        self.reward_storage = checkpoint["reward_storage"]
        self.last_reward = checkpoint["last_reward"]
        self.batch_size = checkpoint["batch_size"]

        assert self.max_penalty == checkpoint["max_penalty"]
        assert self.n_steps == checkpoint["n_steps"]
        assert self.gamma == checkpoint["gamma"]
        assert self.apply_clamp == checkpoint["apply_clamp"]
        assert self.v_min == checkpoint["v_min"]
        assert self.v_max == checkpoint["v_max"]

        self.discount_factor = self.get_discount_factor()
        if device is not None:
            self.reward_storage = self.reward_storage.to(device)
            self.discount_factor = self.discount_factor.to(device)
