"""Reward wrapper utilities for ARC environments."""

import logging

import gymnasium as gym
import torch
from tensordict import TensorDict

from rlarcworld.common.enums import ActionKeys, RewardKeys
from rlarcworld.environments.arc_batch_grid_env import (
    ArcBatchGridEnv,
    ArcEnvRunningMode,
)
from rlarcworld.utils.torch_queue import TorchQueue

logger = logging.getLogger(__name__)


class RewardWorldModelWrapper(gym.Wrapper):
    """ARC reward wrapper that delegates reward computation to a world model.

    This wrapper:
    - Wraps an ArcBatchGridEnv and calls a learned reward model on the current
      environment state returned by env.get_state().
    - Expects the model to output a TensorDict with reward heads keyed by
      RewardKeys (e.g. "binary", "pixel_wise").
    - Uses the binary reward head as the scalar per-step reward for the agent,
      applying submit-based logic:
        - submitted AND good: +1
        - submitted AND bad: -env.max_penalty
        - not submitted: 0
    - Maintains a TorchQueue of recent binary rewards and computes n-step
      discounted returns, with optional clamping via v_min/v_max.
    - Tracks a step_horizon and truncates episodes when the horizon is reached,
      in addition to the underlying environment's own termination flags.
    """

    def __init__(
        self,
        env: ArcBatchGridEnv,
        model: torch.nn.Module,
        step_horizon: int = 1,
        n_steps: int = 1,
        gamma: float = 1.0,
        apply_clamp: bool = False,
        v_min: int | None = None,
        v_max: int | None = None,
    ):
        """Initialize the RewardWorldModelWrapper.

        Args:
            env (ArcBatchGridEnv): Environment to wrap.
            model (torch.nn.Module): Reward world model. It is called with
                env.get_state() and must return a TensorDict whose keys match
                RewardKeys (e.g. "binary", "pixel_wise").
            step_horizon (int, optional): Maximum number of environment steps
                before truncation. Defaults to 1.
            n_steps (int, optional): Length of the n-step return window used for
                discounted reward accumulation. Defaults to 1.
            gamma (float, optional): Discount factor used in the n-step return
                computation. Defaults to 1.0.
            apply_clamp (bool, optional): Whether to clamp the n-step return
                between v_min and v_max. Defaults to False.
            v_min (int, optional): Minimum value for clamping the n-step return.
            v_max (int, optional): Maximum value for clamping the n-step return.

        Raises:
            AssertionError: If step_horizon is not a positive int, gamma is not a
                positive float less than or equal to 1.0, apply_clamp is not a bool,
                int, v_min is not None or an int, or v_min >= v_max when
                apply_clamp is True.
        """
        super().__init__(env)
        assert (
            isinstance(step_horizon, int) and step_horizon > 0
        ), "step_horizon must be a positive int"
        assert (
            isinstance(n_steps, int) and n_steps > 0
        ), "n_steps must be a positive int"
        assert (
            isinstance(gamma, float) and gamma > 0 and gamma <= 1
        ), "gamma must be a positive float lower or equal than 1.0"
        assert isinstance(model, torch.nn.Module), "model must be a torch.nn.Module"
        assert isinstance(apply_clamp, bool), "apply_clamp must be a bool"
        if apply_clamp:
            assert v_max is None or isinstance(v_max, int)
            assert v_min is None or isinstance(v_min, int)
            if isinstance(v_min, int) and isinstance(v_max, int):
                assert v_min < v_max

        self.step_horizon = step_horizon
        self.n_steps = n_steps
        self.model = model
        self.gamma = gamma
        self.device = self.get_wrapper_attr("device")
        self.apply_clamp = apply_clamp
        self.v_min, self.v_max = v_min, v_max

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
        self.mode: ArcEnvRunningMode = self.env.mode
        self.batch_size: int = options["batch"]["input"].shape[0]
        # Reward attributes
        self._reward_storage = TorchQueue(
            torch.zeros((self.batch_size, self.n_steps)),
            queue_size=self.n_steps,
            queue_dim=1,
        )
        self.last_reward = torch.zeros(self.batch_size, dtype=torch.float32)
        self.discount_factor: torch.Tensor = self.get_discount_factor().to(self.device)
        self._reward_storage = self._reward_storage.to(self.device)
        self.last_reward = self.last_reward.to(self.device)
        self.__model_reward = TensorDict({}, batch_size=[self.batch_size]).to(
            self.device
        )
        self._remaining_horizon = self.step_horizon
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

    def __len__(self) -> int:
        """Return the number of elements in the underlying environment."""
        return len(self.env)

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

    def reward(
        self,
        reward: torch.Tensor,
        submission: list[int],
    ) -> torch.Tensor:
        """Return the reward of the environment."""
        submission = torch.as_tensor(submission, device=reward.device)

        # Reward logic:
        # - If submitted AND good: +1
        # - If submitted AND bad: -max_penalty
        # - If not submitted: 0
        return submission * (reward - self.env.max_penalty * (1 - reward))

    @property
    def model_reward(self):
        """Model reward."""
        return self.__model_reward

    @property
    def binary_model_reward(self):
        """Binary model reward."""
        return self.__model_reward[RewardKeys.BINARY.value]

    @property
    def pixel_wise_model_reward(self):
        """Pixel-wise model reward."""
        return self.__model_reward[RewardKeys.PIXEL_WISE.value]

    @property
    def n_model_reward(self):
        """N-step model reward."""
        return self.n_step_reward()

    @property
    def remaining_horizon(self):
        """Remaining horizon."""
        return self._remaining_horizon

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
        obs, _, terminated, truncated, info = self.env.step(actions)
        self._remaining_horizon -= 1
        state = self.get_state()
        model_reward = self.model(state)
        submission = obs[ActionKeys.SUBMIT.value]
        self.__model_reward = model_reward
        self._reward_storage = self._reward_storage.push(
            self.binary_model_reward.unsqueeze(-1)
        )

        truncated = truncated or self._remaining_horizon == 0

        return (
            obs,
            self.reward(self.binary_model_reward, submission),
            terminated,
            truncated,
            info,
        )
