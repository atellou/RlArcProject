"""ARC batched grid environment and action spaces."""

import logging
from pathlib import Path
from typing import Any, Optional, Union

import gymnasium as gym
import numpy as np
import torch
from tensordict import TensorDict

from rlarcworld.common.enums import ActionKeys, ArcEnvRunningMode, StateKeys
from rlarcworld.utils.checkpointing import (
    add_schema_and_class,
    assert_expected_class,
    resolve_load_path,
    resolve_save_path,
)
from rlarcworld.utils.device import enable_cuda
from rlarcworld.utils.torch_queue import TorchQueue

logger = logging.getLogger(__name__)


class ArcActionSpace:
    """Helper to define Gym action spaces for ARC actions."""

    def __init__(self, grid_size: int, color_values: int):
        """Initialize action spaces.

        Args:
            grid_size (int): The size of the grid.
            color_values (int): The number of color values in the grid
                (non-inclusive upper bound).
        """
        self.grid_size = grid_size
        self.color_values = color_values
        self.gym_space = gym.spaces.Dict(
            {
                ActionKeys.X_LOCATION.value: gym.spaces.Sequence(
                    gym.spaces.Discrete(grid_size), stack=True
                ),
                ActionKeys.Y_LOCATION.value: gym.spaces.Sequence(
                    gym.spaces.Discrete(grid_size), stack=True
                ),
                ActionKeys.COLOR_VALUES.value: gym.spaces.Sequence(
                    gym.spaces.Discrete(color_values), stack=True
                ),
                ActionKeys.SUBMIT.value: gym.spaces.Sequence(
                    gym.spaces.Discrete(2), stack=True
                ),
            }
        )
        self.plain_space = torch.arange(
            self.grid_size * self.grid_size * self.color_values * 2
        )
        self.multi_sequence_space = self.plain_space.reshape(
            self.grid_size, self.grid_size, self.color_values, 2
        )
        self.possible_combinations = np.stack(
            np.meshgrid(
                np.arange(self.grid_size),
                np.arange(self.grid_size),
                np.arange(self.color_values),
                np.arange(2),
            ),
            -1,
        ).reshape(-1, 4)


class ArcBatchGridEnv(gym.Env):
    """Batched ARC environment with discrete actions and pixel grids."""

    def __init__(
        self,
        grid_size: int,
        color_values: int,
        *,
        max_penalty: float = 0.0,
        n_steps: int = 1,
        gamma: float = 1.0,
        device: str | None = None,
    ):
        """Initialize the environment.

        Args:
            grid_size (int): The size of the grid.
            color_values (int): Number of color values in the grid (upper bound).
            max_penalty (float, optional): Maximum penalty for the reward.
                Defaults to 0.
            n_steps (int, optional): Number of steps to look ahead. Defaults to 1.
            gamma (float, optional): Discount factor. Defaults to 1.0.
            device (str, optional): Device to use. Defaults to None.
        """
        assert (
            isinstance(grid_size, int) and grid_size > 0
        ), "grid_size must be a positive int"
        assert (
            isinstance(color_values, int) and color_values > 0
        ), "color_values must be a positive int"
        assert (
            isinstance(n_steps, int) and n_steps > 0
        ), "n_steps must be a positive int"
        assert (
            isinstance(gamma, float) and gamma > 0 and gamma <= 1
        ), "gamma must be a positive float lower or equal than 1.0"

        self.device = device or enable_cuda().get("device")

        # N-step attributes
        assert (
            isinstance(max_penalty, float) and max_penalty >= 0
        ), "max_penalty must be a float greater or equal than 0, got {}".format(
            max_penalty
        )
        self.__max_penalty = max_penalty
        self.n_steps = n_steps
        self.gamma = gamma

        # 9 possible values from arc and extras for resizing and no action.
        self.color_values = color_values

        # Size of the grid, assumed to be a MxM grid
        self.grid_size = grid_size

        # Here, the observations will be positions on the grid with a value to
        # set. Used mainly to add stochasticity.
        self.observation_space = gym.spaces.Dict(
            {
                StateKeys.GRID.value: gym.spaces.Sequence(
                    gym.spaces.Box(0, color_values, shape=(grid_size, grid_size)),
                    stack=True,
                ),
                StateKeys.TARGET.value: gym.spaces.Sequence(
                    gym.spaces.Box(0, color_values, shape=(grid_size, grid_size)),
                    stack=True,
                ),
            }
        )
        self.location_space = gym.spaces.MultiDiscrete(
            [grid_size, grid_size, color_values]
        )

        # We have actions corresponding to Y, X, Color Value and submission
        self.action_space = gym.spaces.Sequence(
            gym.spaces.MultiDiscrete([grid_size, grid_size, color_values, 2]),
            stack=True,
        )
        self.arc_action_space = ArcActionSpace(grid_size, color_values)
        self.action_space = self.arc_action_space.gym_space

    def __len__(self) -> int:
        """Return number of grids in the current batch."""
        return len(self.observations["grid"])

    @property
    def max_penalty(self) -> float:
        """Return the maximum penalty."""
        return self.__max_penalty

    def random_location_generator(self) -> np.ndarray:
        """Generate a random location in the grid.

        Returns:
            ndarray: Batch size ndarray with x and y location within the
                observation space defined in init.
        """
        return np.array([self.location_space.sample() for __ in range(self.batch_size)])

    @property
    def reward_storage(self) -> TorchQueue:
        """Get reward storage.

        Returns:
            TorchQueue: The reward storage.
        """
        return self._reward_storage

    @reward_storage.setter
    def reward_storage(self, value: TorchQueue):
        """Sets the reward storage.

        Args:
            value (TorchQueue): The reward storage.
        """
        self._reward_storage = TorchQueue(value, queue_size=self.n_steps, queue_dim=1)

    def validate_examples(self, examples: torch.Tensor):
        """Validate the examples.

        Args:
            examples (torch.Tensor): The examples to validate.
        """
        assert len(examples.shape) == 5, "Examples grids should be 5D"
        assert examples.shape[0] == self.batch_size, (
            "Examples grids should have the same batch size. Expected {} and "
            "{} passed.".format(self.batch_size, examples.shape[0])
        )
        assert examples.shape[2] == 2, (
            "Examples grids should have input and output grids on third "
            "dimension, examples.shape[2] = {}".format(examples.shape[2])
        )
        assert (
            examples.shape[3] == self.grid_size and examples.shape[3] == self.grid_size
        ), (
            "Examples grids should have the size {}x{}. Example shape 3 and 4: "
            "{}x{}".format(
                self.grid_size, self.grid_size, examples.shape[3], examples.shape[3]
            )
        )

    def get_discount_factor(self) -> torch.Tensor:
        """Return the discount factor tensor for n-step returns.

        Returns:
            torch.Tensor: The discount factor.
        """
        return torch.ones((self.batch_size, self.n_steps)) * (
            self.gamma ** torch.arange(1, self.n_steps + 1)
        )

    def reset(self, *, options: dict, seed: int | None = None) -> tuple:
        """Reset environment and return initial observation and info.

        Args:
            options (dict): Batch of samples (axis=0) to be used in the
                environment.
            seed (int, optional): Seed for RNG. Defaults to None.

        Returns:
            tuple: A tuple containing the initial observation and info.
        """
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # The sample for every episode is a batch of ARC Samples
        batch: dict[str, torch.Tensor] = options["batch"]
        batch_in: torch.Tensor = batch["input"]

        # Require mode to be specified
        mode = options.get("mode")
        if mode is None:
            raise ValueError(
                f"Mode must be specified in options. Modes: {list(ArcEnvRunningMode)}."
            )

        try:
            self.mode = ArcEnvRunningMode(mode)
        except ValueError:
            raise ValueError(
                f"Invalid mode: {mode}. Must be one of {list(ArcEnvRunningMode)}."
            )

        batch_out: torch.Tensor | None = None
        if self.mode.is_train():
            batch_out = batch.get("output")
            if batch_out is None:
                raise ValueError(
                    "In train mode, the batch must contain an 'output' key."
                )
            # Check shape
            assert (
                len(batch_out.shape) == 3
                and batch_out.shape[1] == self.grid_size
                and batch_out.shape[2] == self.grid_size
            ), (
                "The batch output (target) shape is not correct. A shape of "
                "(batch_size, {size}, {size}) is expected".format(size=self.grid_size)
            )

        assert (
            len(batch_in.shape) == 3
            and batch_in.shape[1] == self.grid_size
            and batch_in.shape[2] == self.grid_size
        ), (
            "The batch input shape is not correct. A shape of (batch_size, "
            "{size}, {size}) is expected".format(size=self.grid_size)
        )

        self.batch_size = batch_in.shape[0]

        # grid of initial reward
        self.last_grid = batch_in.clone()

        # Reward attributes
        self.discount_factor = self.get_discount_factor().to(self.device)
        self._reward_storage = TorchQueue(
            torch.zeros((self.batch_size, self.n_steps)),
            queue_size=self.n_steps,
            queue_dim=1,
        ).to(self.device)
        self.last_reward = torch.zeros(self.batch_size, dtype=int).to(self.device)
        self.time_step = 0

        # Validate examples
        self.validate_examples(options["examples"])
        self.information = TensorDict(
            {
                StateKeys.INITIAL.value: batch_in,
                StateKeys.INDEX.value: batch_in.clone() * 0,
                StateKeys.TERMINATED.value: torch.zeros(self.batch_size, dtype=int),
                StateKeys.EXAMPLES.value: options["examples"],
            },
            batch_size=self.batch_size,
        ).to(self.device)

        self.observations = TensorDict(
            {StateKeys.GRID.value: batch_in.clone(), StateKeys.TARGET.value: batch_out},
            batch_size=self.batch_size,
        ).to(self.device)

        self.information = self.information.to(self.device)
        self.observations = self.observations.to(self.device)
        return self.observations, self.information

    def get_difference(self) -> torch.Tensor:
        """Compute the difference between the current grid and the target grid.

        Returns:
            torch.Tensor: The difference between the current grid and the target grid.
        """
        if not self.mode.is_train():
            raise ValueError("get_difference called outside of training mode")
        return (
            self.observations[StateKeys.GRID.value]
            - self.observations[StateKeys.TARGET.value]
        ).to(self.device)

    def episode_terminated(self, submission: torch.Tensor) -> torch.Tensor:
        """Check if the episode has terminated.

        Args:
            submission (torch.Tensor): Whether the agent indicated a submission to
                grade.

        Returns:
            torch.Tensor: The termination status of the episode.
        """
        if not self.mode.is_train():
            raise ValueError("episode_terminated called outside of training mode")

        diff = self.get_difference()
        return (torch.sum(torch.abs(diff), dim=(1, 2)) == 0) * submission

    def reward(
        self,
        grid_diffs: torch.Tensor,
        submission: list[int],
    ) -> torch.Tensor:
        """Compute the reward for the current grid of the environment.

        Reward structure:
        - No submission: 0
        - Successful submission (perfect match): +1
        - Failed submission (any mismatch): -max_penalty

        Args:
            grid_diffs (torch.Tensor): Batch of grids representing the
                difference between the current grid and target.
            submission (list[int]): Indicates whether the agent indicated a
                submission to grade.

        Returns:
            float: The reward for the current grid of the environment.
        """
        submission_tensor = torch.as_tensor(
            submission,
            device=grid_diffs.device,
            dtype=grid_diffs.dtype,
        )

        # Check if grid perfectly matches target
        perfect_match = (torch.sum(torch.abs(grid_diffs), dim=(1, 2)) == 0).long()

        # Reward logic:
        # - If submitted AND perfect: +1
        # - If submitted AND not perfect: -max_penalty
        # - If not submitted: 0
        reward = submission_tensor * (
            perfect_match - self.__max_penalty * (1 - perfect_match)
        )

        return reward

    def n_step_reward(self) -> torch.Tensor:
        """Computes the reward for the current grid of the environment."""
        if isinstance(self._reward_storage, torch.Tensor):
            return torch.sum(self._reward_storage * self.discount_factor, dim=1).to(
                self.device
            )
        return torch.tensor(
            torch.sum(self._reward_storage * self.discount_factor, dim=1)
        ).to(self.device)

    @property
    def state(self) -> TensorDict:
        """Return the state of the environment.

        TensorDict: The state of the environment should contain the following keys:
            - ``grid``: The current state of the grid.
            - ``last_grid``: The state of the grid in the previous step.
            - ``initial``: The initial state of the grid.
            - ``index``: The current index of the batch.
            - ``terminated``: The termination status of the episode.
            - ``examples``: The batch of examples.
        """
        state = TensorDict({}, batch_size=self.batch_size).to(self.device)
        state[StateKeys.GRID.value] = self.observations[StateKeys.GRID.value].clone()
        state[StateKeys.LAST_GRID.value] = self.last_grid.clone()
        state[StateKeys.INITIAL.value] = self.information[StateKeys.INITIAL.value]
        state[StateKeys.INDEX.value] = self.information[StateKeys.INDEX.value].clone()
        state[StateKeys.TERMINATED.value] = self.information[
            StateKeys.TERMINATED.value
        ].clone()
        state[StateKeys.EXAMPLES.value] = self.information[StateKeys.EXAMPLES.value]
        return state

    def get_state(self, unsqueeze: Optional[int] = None) -> TensorDict:
        """Return the current state, optionally adding a batch dim.

        TensorDict: The state of the environment should contain the following keys:
            - ``grid``: The current state of the grid.
            - ``last_grid``: The state of the grid in the previous step.
            - ``initial``: The initial state of the grid.
            - ``index``: The current index of the batch.
            - ``terminated``: The termination status of the episode.
            - ``examples``: The batch of examples.
        """
        if unsqueeze is not None:
            assert isinstance(unsqueeze, int)
            state = self.state
            state.update(
                {
                    StateKeys.GRID.value: state[StateKeys.GRID.value].unsqueeze(
                        unsqueeze
                    ),
                    StateKeys.LAST_GRID.value: state[
                        StateKeys.LAST_GRID.value
                    ].unsqueeze(unsqueeze),
                    StateKeys.INITIAL.value: state[StateKeys.INITIAL.value].unsqueeze(
                        unsqueeze
                    ),
                    StateKeys.INDEX.value: state[StateKeys.INDEX.value].unsqueeze(
                        unsqueeze
                    ),
                    StateKeys.TERMINATED.value: state[
                        StateKeys.TERMINATED.value
                    ].unsqueeze(unsqueeze),
                }
            )
            return state
        return self.state.to(self.device)

    def step(
        self, actions: Union[TensorDict, dict[str, torch.Tensor]]
    ) -> tuple[TensorDict, torch.Tensor, bool, bool, dict]:
        """Take a step and return observation, reward, done flags, and info.

        Args:
            actions (dict): Actions to perform.

        Returns:
            tuple: (observation, reward, terminated, truncated, info).
        """
        # Validate action using gym's space, allowing TensorDict inputs
        try:
            actions_cpu = (
                {k: v.detach().cpu().numpy() for k, v in actions.items()}
                if isinstance(actions, TensorDict)
                else actions
            )
        except Exception:
            actions_cpu = actions

        if self.action_space.contains(actions_cpu):
            self.last_grid = self.observations[StateKeys.GRID.value].clone()
            logger.debug("Actions are valid")
            # Update the grid with the action.
            self.time_step += 1
            logger.debug(
                "Action performed shapes: Y={}, X={}, Color={}, Submission={}".format(
                    actions[ActionKeys.Y_LOCATION.value].shape,
                    actions[ActionKeys.X_LOCATION.value].shape,
                    actions[ActionKeys.COLOR_VALUES.value].shape,
                    actions[ActionKeys.SUBMIT.value].shape,
                )
            )
            # Ensure assigned color values live on the same device as observations
            _grid = self.observations[StateKeys.GRID.value]
            _colors = actions[ActionKeys.COLOR_VALUES.value].to(_grid.device)
            self.observations[StateKeys.GRID.value][
                list(range(self.batch_size)),
                actions[ActionKeys.Y_LOCATION.value],
                actions[ActionKeys.X_LOCATION.value],
            ] = _colors
            self.information[StateKeys.INDEX.value][
                list(range(self.batch_size)),
                actions[ActionKeys.Y_LOCATION.value],
                actions[ActionKeys.X_LOCATION.value],
            ] = self.time_step
        else:
            logger.error(
                (
                    "No action performed due to invalid action; valid ranges: %s; "
                    "given: %s"
                ),
                self.action_space.keys(),
                actions,
            )
            raise ValueError(
                "The specified action does not comply with the action space "
                "constraints. Valid ranges: {}. Given: {}.".format(
                    self.action_space.keys(), actions
                )
            )

        if self.mode.is_train():
            reward = torch.as_tensor(
                self.reward(self.get_difference(), actions[ActionKeys.SUBMIT.value]),
                device=self._reward_storage.device,
                dtype=self._reward_storage.dtype,
            )
        else:
            reward = torch.zeros(
                self.batch_size,
                device=self._reward_storage.device,
                dtype=self._reward_storage.dtype,
            )
        self._reward_storage = self._reward_storage.push(reward.unsqueeze(1))
        self.last_reward = reward

        # TODO: No truncate version, evaluate time constraints
        submit = torch.as_tensor(actions[ActionKeys.SUBMIT.value]).to(self.device)
        truncated = False if self.mode.is_train() else torch.sum(submit) == len(self)
        if self.mode.is_train():
            self.information[StateKeys.TERMINATED.value] = self.episode_terminated(
                submit
            )
        else:
            self.information[StateKeys.TERMINATED.value] = submit
        return (
            self.observations,
            reward,
            torch.sum(self.information[StateKeys.TERMINATED.value]) == len(self),
            truncated,
            self.information,
        )

    def checkpoint(self) -> dict[str, Any]:
        """Checkpoint the environment.

        Returns:
            dict[str, Any]: The checkpoint.
        """
        try:
            payload = {
                "observations": self.observations,
                "information": self.information,
                "last_grid": self.last_grid,
                "last_reward": self.last_reward,
                "time_step": self.time_step,
                "reward_storage": self.reward_storage,
                "mode": self.mode.value,
                "n_steps": self.n_steps,
                "gamma": self.gamma,
                "grid_size": self.grid_size,
                "color_values": self.color_values,
                "batch_size": self.batch_size,
            }
            return add_schema_and_class(payload, self.__class__.__name__)
        except AttributeError as e:
            logger.error(
                "Error checkpointing environment, reset should have been called: %s",
                e,
            )
            raise

    def save_checkpoint(self, path: str | Path) -> None:
        """Saves the environment to a checkpoint.

        Args:
            path (str | Path): The path to save the checkpoint to.
        """
        logger.debug(f"Saving environment to {path}")
        try:
            checkpoint_path = resolve_save_path(path, self.__class__.__name__)
            torch.save(self.checkpoint(), checkpoint_path)
        except AttributeError as e:
            logger.error(
                ("Error saving environment, reset should have been called by now: %s"),
                e,
            )
            raise AttributeError(
                ("Error saving environment, reset should have been called by now: %s")
                % e
            )

    def load(
        self,
        checkpoint: dict[str, Any] | Path | str,
        weights_only: bool = False,
        device: str | None = None,
    ):
        """Loads the environment from a checkpoint.

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
                cp_path,
                weights_only=weights_only,
                map_location=device,
            )
        if not isinstance(checkpoint, dict):
            raise ValueError("Checkpoint must be a dictionary or a Path.")

        if "class" in checkpoint:
            assert_expected_class(checkpoint["class"], self.__class__.__name__)
        self.observations = checkpoint["observations"]
        self.information = checkpoint["information"]
        self.last_grid = checkpoint["last_grid"]
        self.last_reward = checkpoint["last_reward"]
        self.time_step = checkpoint["time_step"]
        self.reward_storage = checkpoint["reward_storage"]
        self.mode = ArcEnvRunningMode(checkpoint["mode"])
        self.batch_size = checkpoint["batch_size"]

        assert self.n_steps == checkpoint["n_steps"]
        assert self.gamma == checkpoint["gamma"]
        assert self.grid_size == checkpoint["grid_size"]
        assert self.color_values == checkpoint["color_values"]

        # Recompute discount factor and ensure all tensors are on env device
        self.discount_factor = self.get_discount_factor()
        target_device = device or self.device
        self.observations = self.observations.to(target_device)
        self.information = self.information.to(target_device)
        self.last_grid = self.last_grid.to(target_device)
        self.reward_storage = self.reward_storage.to(target_device)
        self.discount_factor = self.discount_factor.to(target_device)


logger.info("Registering gymnasium environment")

gym.envs.registration.register(
    id="ArcBatchGrid-v0",
    entry_point="rlarcworld.environments.arc_batch_grid_env:ArcBatchGridEnv",
    nondeterministic=True,
)
