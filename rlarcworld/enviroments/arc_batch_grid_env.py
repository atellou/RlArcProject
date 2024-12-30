from typing import Optional, List
import torch
from tensordict import TensorDict
import numpy as np
import gymnasium as gym
import logging

logger = logging.getLogger(__name__)


class ArcBatchGridEnv(gym.Env):

    def __init__(self, size: int, color_values: int):
        # 9 possible values from arc and extras for resizing and no action.
        self.color_values = color_values

        # Size of the grid, assumed to be a MxM grid
        self.size = size

        # Here, the observations will be positions on the grid with a value to set. Used mainly to add stochasticity
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Sequence(
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
        self.action_space = gym.spaces.Dict(
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

    def __len__(self):
        return len(self.observations["state"])

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
        Resets the environment to an initial internal state, returning an initial observation and info.
        Args:
            options (dict): A dictionary containing the batch of samples (axis=0) to be used in the environment.
            seed (int, optional): The seed for the random number generator. Defaults to None.
        Returns:
            tuple: A tuple containing the initial observation and info.
        """
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # The sample for every episode is a batch of ARC Samples
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

        # State of initial reward
        self.last_state = batch_in.clone()
        self._reward_storage = torch.zeros(self.batch_size, dtype=int)
        self._last_reward = self._reward_storage.clone()
        self._timestep = 0

        # Validate examples
        self.validate_examples(options["examples"])
        self.information = TensorDict(
            {
                "initial": batch_in,
                "index": batch_in.clone() * 0,
                "terminated": torch.zeros(self.batch_size, dtype=int),
                "examples": options["examples"],
            }
        )

        self.observations = TensorDict(
            {"state": batch_in.clone(), "target": batch_out.clone()}
        )

        return self.observations.to_dict(), self.information.to_dict()

    def get_difference(self):
        """
        Compute the difference between the state grid and the target grid.

        Returns:
            torch.Tensor: The difference between the state grid and the target grid.
        """
        return self.observations["state"] - self.observations["target"]

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
        Computes the reward for the state state of the environment.

        Args:
            grid_diffs torch.Tensor: Batch of grids representing the difference between state and target.
            submission List(int): Indicates whether the agent indicated a submission to grade.
        Returns:
            float: The reward for the state state of the environment.
        """
        return (
            torch.sum(torch.abs(grid_diffs), dim=(1, 2)) == 0
        ).long() * torch.as_tensor(
            submission,
            device=grid_diffs.device,
            dtype=grid_diffs.dtype,
        )

    @property
    def state(self):
        state = self.information
        state.update({"state": self.observations["state"]})
        state.update({"last_state": self.last_state})
        return state

    def step(self, actions: list):

        if self.action_space.contains(actions.numpy()):
            self.last_state = self.observations["state"].clone()
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
            self.observations["state"][
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
        self._reward_storage += reward
        self._last_reward = reward

        # TODO: No truncate version, evaluate time constraints
        truncated = False
        terminated = self.episode_terminated(actions["submit"])
        self.information["terminated"] = terminated
        return (
            self.observations,
            reward,
            bool(torch.sum(terminated) == len(self)),
            truncated,
            self.information,
        )


logger.info("Registering gymnasium environment")
gym.envs.registration.register(
    id="ArcBatchGrid-v0",
    entry_point="rlarcworld.enviroments.arc_batch_grid_env:ArcBatchGridEnv",
    nondeterministic=True,
)
