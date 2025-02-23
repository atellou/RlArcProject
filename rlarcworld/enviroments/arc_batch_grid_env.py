import os
from typing import Optional, List
import torch
from tensordict import TensorDict
import numpy as np
import gymnasium as gym
import logging

from rlarcworld.utils import TorchQueue, enable_cuda

logger = logging.getLogger(__name__)


class ArcActionSpace:
    def __init__(self, size: int, color_values: int):
        self.size = size
        self.color_values = color_values
        self.gym_space = gym.spaces.Dict(
            {
                "x_location": gym.spaces.Sequence(
                    gym.spaces.Discrete(size), stack=True
                ),
                "y_location": gym.spaces.Sequence(
                    gym.spaces.Discrete(size), stack=True
                ),
                "color_values": gym.spaces.Sequence(
                    gym.spaces.Discrete(color_values), stack=True
                ),
                "submit": gym.spaces.Sequence(gym.spaces.Discrete(2), stack=True),
            }
        )
        self.plain_space = torch.arange(self.size * self.size * self.color_values * 2)
        self.multi_squence_space = self.plain_space.reshape(
            self.size, self.size, self.color_values, 2
        )
        self.possible_combinations = np.stack(
            np.meshgrid(
                np.arange(self.size),
                np.arange(self.size),
                np.arange(self.color_values),
                np.arange(2),
            ),
            -1,
        ).reshape(-1, 4)


class ArcBatchGridEnv(gym.Env):

    def __init__(
        self, size: int, color_values: int, n_steps: int = 1, gamma: float = 1.0
    ):
        assert isinstance(size, int) and size > 0, "size must be a positive int"
        assert (
            isinstance(color_values, int) and color_values > 0
        ), "color_values must be a positive int"
        assert (
            isinstance(n_steps, int) and n_steps > 0
        ), "n_steps must be a positive int"
        assert (
            isinstance(gamma, float) and gamma > 0 and gamma <= 1
        ), "gamma must be a positive float lower or equal than 1.0"

        self.device = enable_cuda().get("device")

        # N-step attributes
        self.n_steps = n_steps
        self.gamma = gamma

        # 9 possible values from arc and extras for resizing and no action.
        self.color_values = color_values

        # Size of the grid, assumed to be a MxM grid
        self.size = size

        # Here, the observations will be positions on the grid with a value to set. Used mainly to add stochasticity
        self.observation_space = gym.spaces.Dict(
            {
                "grid": gym.spaces.Sequence(
                    gym.spaces.Box(0, color_values, shape=(size, size)), stack=True
                ),
                "target": gym.spaces.Sequence(
                    gym.spaces.Box(0, color_values, shape=(size, size)), stack=True
                ),
            }
        )
        self.location_space = gym.spaces.MultiDiscrete([size, size, color_values])

        # We have actions corresponding to "Y Location", "X Location", "Color Value" and "submission"
        self.action_space = gym.spaces.Sequence(
            gym.spaces.MultiDiscrete([size, size, color_values, 2]), stack=True
        )
        self.arc_action_space = ArcActionSpace(size, color_values)
        self.action_space = self.arc_action_space.gym_space

    def __len__(self):
        return len(self.observations["grid"])

    def random_location_generator(self):
        """
        Generate a random location in the grid

        Returns:
            ndarray: Batch size ndarray with x and y location within the observation space defined in init
        """
        return np.array([self.location_space.sample() for __ in range(self.batch_size)])

    @property
    def reward_storage(self):
        return self._reward_storage

    @property
    def last_reward(self):
        return self._last_reward

    def validate_examples(self, examples: torch.Tensor):
        assert len(examples.shape) == 5, "Examples grids should be 5D"
        assert (
            examples.shape[0] == self.batch_size
        ), "Examples grids should have the same batch size. Expected {} and {} passed.".format(
            self.batch_size, examples.shape[0]
        )
        assert (
            examples.shape[2] == 2
        ), "Examples grids should have input and output grids on third dimesion, examples.shape[2] = {}".format(
            examples.shape[2]
        )
        assert (
            examples.shape[3] == self.size and examples.shape[3] == self.size
        ), "Examples grids should have the size {}x{}. Example shape 3 and 4: {}x{}".format(
            self.size, self.size, examples.shape[3], examples.shape[3]
        )

    def reset(self, *, options: dict, seed: Optional[int] = None) -> tuple:
        """
        Resets the environment to an initial internal grid, returning an initial observation and info.
        Args:
            options (dict): A dictionary containing the batch of samples (axis=0) to be used in the environment.
            seed (int, optional): The seed for the random number generator. Defaults to None.
        Returns:
            tuple: A tuple containing the initial observation and info.
        """
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # The sample for every episode is a batch of ARC Samples
        self.is_train_episode = options.get("is_train", True)
        batch = options["batch"]
        batch_in = batch["input"]
        batch_out = batch["output"]
        assert (
            len(batch_in.shape) == 3
            and batch_in.shape[1] == self.size
            and batch_in.shape[2] == self.size
        ), "The batch input shape is not correct. A shape of (batch_size, {size}, {size}) is expected".format(
            self.size
        )
        assert (
            len(batch_out.shape) == 3
            and batch_out.shape[1] == self.size
            and batch_out.shape[2] == self.size
        ), "The batch output (target) shape is not correct. A shape of (batch_size, {size}, {size}) is expected".format(
            self.size
        )
        self.batch_size = batch_in.shape[0]

        # grid of initial reward
        self.last_grid = batch_in.clone()

        # Reward attributes
        self.discount_factor = torch.ones((self.batch_size, self.n_steps)) * (
            self.gamma ** torch.arange(1, self.n_steps + 1)
        )
        self._reward_storage = TorchQueue(
            torch.zeros((self.batch_size, self.n_steps)),
            queue_size=self.n_steps,
            queue_dim=1,
        )
        self._last_reward = torch.zeros(self.batch_size, dtype=int)
        self._timestep = 0

        # Validate examples
        self.validate_examples(options["examples"])
        self.information = TensorDict(
            {
                "initial": batch_in,
                "index": batch_in.clone() * 0,
                "terminated": torch.zeros(self.batch_size, dtype=int),
                "examples": options["examples"],
            },
            batch_size=self.batch_size,
        ).to(self.device)

        self.observations = TensorDict(
            {"grid": batch_in.clone(), "target": batch_out.clone()},
            batch_size=self.batch_size,
        ).to(self.device)

        self.information = self.information.to(self.device)
        self.observations = self.observations.to(self.device)

        return self.observations, self.information

    def get_difference(self):
        """
        Compute the difference between the current grid and the target grid.

        Returns:
            torch.Tensor: The difference between the current grid and the target grid.
        """
        return (self.observations["grid"] - self.observations["target"]).to(self.device)

    def episode_terminated(self, submission):
        diff = self.get_difference()
        return (torch.sum(torch.abs(diff), dim=(1, 2)) == 0) * torch.as_tensor(
            submission,
            device=diff.device,
            dtype=diff.dtype,
        )

    def reward(
        self,
        grid_diffs: torch.Tensor,
        submission: List[int],
    ):
        """
        Computes the reward for the current grid of the environment.

        Args:
            grid_diffs torch.Tensor: Batch of grids representing the difference between the current grid and target.
            submission List(int): Indicates whether the agent indicated a submission to grade.
        Returns:
            float: The reward for the current grid of the environment.
        """
        return (
            torch.sum(torch.abs(grid_diffs), dim=(1, 2)) == 0
        ).long() * torch.as_tensor(
            submission,
            device=grid_diffs.device,
            dtype=grid_diffs.dtype,
        )

    def n_step_reward(self):
        """
        Computes the reward for the current grid of the environment.
        """
        return torch.tensor(
            torch.sum(self._reward_storage * self.discount_factor, dim=1).cpu().numpy()
        ).to(self.device)

    @property
    def state(self):
        """
        Returns:
            TensorDict: The state of the environment should contain the following keys:
                - ``grid``: The current state of the grid.
                - ``last_grid``: The state of the grid in the previous step.
                - ``initial``: The initial state of the grid.
                - ``index``: The current index of the batch.
                - ``terminated``: The termination status of the episode.
                - ``examples``: The batch of examples.
        """

        state = self.information
        state.update({"grid": self.observations["grid"]})
        state.update({"last_grid": self.last_grid})
        return state.clone()

    def get_state(self, unsqueeze: int = None):
        if unsqueeze is not None:
            assert isinstance(unsqueeze, int)
            state = self.state
            state.update(
                {
                    "grid": state["grid"].unsqueeze(unsqueeze),
                    "last_grid": state["last_grid"].unsqueeze(unsqueeze),
                    "initial": state["initial"].unsqueeze(unsqueeze),
                    "index": state["index"].unsqueeze(unsqueeze),
                    "terminated": state["terminated"].unsqueeze(unsqueeze),
                }
            )
            return state
        return self.state.to(self.device)

    def step(self, actions: list):
        if self.action_space.contains(actions.cpu().numpy()):
            self.last_grid = self.observations["grid"].clone()
            logger.debug("Actions are valid")
            # Update the grid with the action.
            self._timestep += 1
            logger.debug(
                "Action performed shapes: Y={}, X={}, Color={}, Submission={}".format(
                    actions["y_location"].shape,
                    actions["x_location"].shape,
                    actions["color_values"].shape,
                    actions["submit"].shape,
                )
            )
            self.observations["grid"][
                list(range(self.batch_size)),
                actions["y_location"],
                actions["x_location"],
            ] = actions["color_values"]
            self.information["index"][
                list(range(self.batch_size)),
                actions["y_location"],
                actions["x_location"],
            ] = self._timestep
        else:
            logger.error(
                "No action performed due to invalid action, sequence of values"
                + " within {} are valid, not inclusive. Given: {}.".format(
                    self.action_space.keys(), actions
                )
            )
            raise ValueError(
                "The specified action do not comply with the action space constraints."
                + "Sequence of values within {} are valid, not inclusive. Given: {}.".format(
                    self.action_space.keys(), actions
                )
            )

        reward = self.reward(self.get_difference(), actions["submit"])
        reward = torch.as_tensor(
            reward,
            device=self._reward_storage.device,
            dtype=self._reward_storage.dtype,
        )
        self._reward_storage = self._reward_storage.push(reward.unsqueeze(1))
        self._last_reward = reward

        # TODO: No truncate version, evaluate time constraints
        truncated = (
            False
            if self.is_train_episode
            else torch.sum(torch.as_tensor(actions["submit"])) == len(self)
        )
        self.information["terminated"] = self.episode_terminated(actions["submit"])
        return (
            self.observations,
            reward,
            torch.sum(self.information["terminated"]) == len(self),
            truncated,
            self.information,
        )


logger.info("Registering gymnasium environment")
gym.envs.registration.register(
    id="ArcBatchGrid-v0",
    entry_point="rlarcworld.enviroments.arc_batch_grid_env:ArcBatchGridEnv",
    nondeterministic=True,
)
