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
        grid_diff_epsilon: float = 0.01,
        smooth_transition: bool = True,
        transition_steepness: float = 100.0,
    ):
        """Initialize the RewardWorldModelWrapper.

        Args:
            env (ArcBatchGridEnv): Environment to wrap.
            model (torch.nn.Module): The reward world model.
            step_horizon (int, optional): Maximum number of steps per episode.
                Defaults to 1.
            n_steps (int, optional): Number of steps for n-step returns.
                Defaults to 1.
            gamma (float, optional): Discount factor. Defaults to 1.0.
            apply_clamp (bool, optional): Whether to clamp rewards. Defaults to False.
            v_min (int, optional): Minimum reward value for clamping. Defaults to None.
            v_max (int, optional): Maximum reward value for clamping. Defaults to None.
            grid_diff_epsilon (float, optional): Threshold for considering grid_diff
                as zero. Defaults to 0.01 (1% of pixels).
            smooth_transition (bool, optional): If True, use smooth sigmoid transition
                instead of hard threshold. Defaults to True.
            transition_steepness (float, optional): Controls sharpness of smooth
                transition. Higher values make transition sharper. Defaults to 100.0.
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
        assert (
            isinstance(grid_diff_epsilon, float) and 0 < grid_diff_epsilon < 1
        ), "grid_diff_epsilon must be a float in (0, 1)"
        assert isinstance(smooth_transition, bool), "smooth_transition must be a bool"
        assert (
            isinstance(transition_steepness, float) and transition_steepness > 0
        ), "transition_steepness must be a positive float"

        self.step_horizon = step_horizon
        self.n_steps = n_steps
        self.model = model
        self.gamma = gamma
        self.device = self.get_wrapper_attr("device")
        self.apply_clamp = apply_clamp
        self.v_min, self.v_max = v_min, v_max
        self.max_penalty = self.get_wrapper_attr("max_penalty")
        self.grid_size = self.get_wrapper_attr("grid_size")
        self.color_values = self.get_wrapper_attr("color_values")
        self.grid_diff_epsilon = grid_diff_epsilon
        self.smooth_transition = smooth_transition
        self.transition_steepness = transition_steepness

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
        self._remaining_horizon = self.step_horizon

        # Reward attributes - store both reward types for multi-head n-step returns
        self._reward_storage = TensorDict(
            {
                RewardKeys.BINARY.value: TorchQueue(
                    torch.zeros((self.batch_size, self.n_steps)),
                    queue_size=self.n_steps,
                    queue_dim=1,
                ),
                RewardKeys.PIXEL_WISE.value: TorchQueue(
                    torch.zeros((self.batch_size, self.n_steps)),
                    queue_size=self.n_steps,
                    queue_dim=1,
                ),
            }
        )

        self.last_reward = torch.zeros(self.batch_size, dtype=torch.float32)
        self.discount_factor: torch.Tensor = self.get_discount_factor().to(self.device)
        self._reward_storage = self._reward_storage.to(self.device)
        self.last_reward = self.last_reward.to(self.device)

        # Store raw model outputs and submission for lazy evaluation
        self.__model_reward = TensorDict({}, batch_size=[self.batch_size]).to(
            self.device
        )
        self.__submission = torch.zeros(self.batch_size, dtype=torch.long).to(
            self.device
        )

        # Cache for computed rewards (lazy evaluation)
        self.__binary_reward_cache = None
        self.__pixel_wise_reward_cache = None

        return reset

    def n_step_reward(
        self,
        reward_key: str = RewardKeys.BINARY.value,
        v_min: float | None = None,
        v_max: float | None = None,
    ) -> torch.Tensor:
        """Compute the n-step reward for the current grid of the environment.

        Args:
            reward_key (str, optional): Which reward type to compute n-step return for.
                Defaults to RewardKeys.BINARY.value.
            v_min (float, optional): Minimum value for clamping. Defaults to None.
            v_max (float, optional): Maximum value for clamping. Defaults to None.

        Returns:
            torch.Tensor: The computed n-step discounted return.
        """
        total = torch.sum(
            self._reward_storage[reward_key] * self.discount_factor, dim=1
        )
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
    def reward_storage(self, value: TensorDict) -> None:
        """Sets the reward storage.

        Args:
            value (TensorDict): The reward storage.
        """
        if tuple(value.keys()) != (
            RewardKeys.BINARY.value,
            RewardKeys.PIXEL_WISE.value,
        ):
            raise ValueError(
                "Reward storage must have the same keys as the current reward storage"
            )
        self._reward_storage = TorchQueue(value, queue_size=self.n_steps, queue_dim=1)

    def reward(self) -> torch.Tensor:
        """Return the reward of the environment.

        # Reward logic:
        # - If submitted AND good: +1
        # - If submitted AND bad: -max_penalty
        # - If not submitted: 0
        # Using probability directly for smooth planning gradients
        """
        # Extract P(correct) from [P(incorrect), P(correct)]
        improvement = self.__model_reward[RewardKeys.BINARY.value][..., 1]

        return self.__submission * (improvement - self.max_penalty * (1 - improvement))

    def pixel_wise_reward(self) -> torch.Tensor:
        """Compute the pixel-wise reward for the current grid of the environment.

        The model predicts:
        - binary: P(correct) in [0, 1]
        - pixel_wise: percentage of wrong pixels in [0, 1]

        We interpret pixel_wise as normalized grid_diff (grid_diff / grid_size²).
        """
        submission = self.__submission
        # Model outputs percentage of wrong pixels [0, 1]
        grid_diff_percentage = self.__model_reward[RewardKeys.PIXEL_WISE.value]
        # Extract P(correct) from [P(incorrect), P(correct)]
        improvement = self.__model_reward[RewardKeys.BINARY.value][..., 1]

        # Penalization scaled by percentage of wrong pixels
        penalization = submission * -self.max_penalty
        if not isinstance(penalization, torch.Tensor):
            penalization = torch.tensor(
                penalization,
                device=grid_diff_percentage.device,
                dtype=grid_diff_percentage.dtype,
            ).to(self.device)

        # Compute rewards for imperfect and perfect grids
        penalization = grid_diff_percentage * penalization
        imperfect_reward = (improvement - 1) + penalization
        perfect_reward = submission

        if self.smooth_transition:
            # Smooth sigmoid transition between imperfect and perfect rewards
            # Higher steepness → sharper transition (closer to hard threshold)
            smooth_is_perfect = torch.sigmoid(
                -self.transition_steepness
                * (grid_diff_percentage - self.grid_diff_epsilon)
            )
            reward = (
                smooth_is_perfect * perfect_reward
                + (1 - smooth_is_perfect) * imperfect_reward
            )
        else:
            # Original hard threshold
            is_perfect = (grid_diff_percentage <= self.grid_diff_epsilon).long()
            has_errors = (grid_diff_percentage > self.grid_diff_epsilon).long()
            reward = (
                (improvement - 1) * has_errors + penalization + is_perfect * submission
            )

        if self.apply_clamp:
            return torch.clamp(reward, min=self.v_min, max=self.v_max)
        else:
            return reward

    @property
    def binary_model_reward(self):
        """Binary model reward (lazy evaluation with caching)."""
        if self.__binary_reward_cache is None:
            self.__binary_reward_cache = self.reward()
        return self.__binary_reward_cache

    @property
    def pixel_wise_model_reward(self):
        """Pixel-wise model reward (lazy evaluation with caching)."""
        if self.__pixel_wise_reward_cache is None:
            self.__pixel_wise_reward_cache = self.pixel_wise_reward()
        return self.__pixel_wise_reward_cache

    @property
    def n_binary_reward(self):
        """N-step binary reward."""
        return self.n_step_reward(reward_key=RewardKeys.BINARY.value)

    @property
    def n_pixel_wise_reward(self):
        """N-step pixel-wise reward."""
        return self.n_step_reward(reward_key=RewardKeys.PIXEL_WISE.value)

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
        state = self.get_state(unsqueeze=1)

        # Store raw model output and submission for lazy evaluation
        self.__model_reward = self.model(state).to(self.device)
        self.__submission = torch.as_tensor(
            actions[ActionKeys.SUBMIT.value], device=self.device
        )

        # Invalidate caches (new step = new rewards)
        self.__binary_reward_cache = None
        self.__pixel_wise_reward_cache = None

        # Push both reward types to storage
        self._reward_storage[RewardKeys.BINARY.value] = self._reward_storage[
            RewardKeys.BINARY.value
        ].push(self.binary_model_reward.unsqueeze(-1))
        self._reward_storage[RewardKeys.PIXEL_WISE.value] = self._reward_storage[
            RewardKeys.PIXEL_WISE.value
        ].push(self.pixel_wise_model_reward.unsqueeze(-1))

        truncated = truncated or self._remaining_horizon == 0

        return (
            obs,
            self.binary_model_reward,
            terminated,
            truncated,
            info,
        )
