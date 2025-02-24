import os
from rlarcworld.enviroments.arc_batch_grid_env import ArcBatchGridEnv
from rlarcworld.enviroments.wrappers.rewards import PixelAwareRewardWrapper
import numpy as np
import torch
from tensordict import TensorDict
import unittest
import logging


logger = logging.getLogger(__name__)


class ArcBatchGridsEnv(unittest.TestCase):

    def test_environment(self):
        logger.info("Testing ArcBatchGridEnv")
        batch_size = np.random.randint(2, 4)
        size = np.random.randint(2, 4)
        color_values = np.random.randint(2, 4)

        env = ArcBatchGridEnv(
            size,
            color_values,
            n_steps=np.random.randint(1, 10),
            gamma=1.0,
            device="cpu",
        )
        self.episodes_simulation(batch_size, size, color_values, env)

    def test_wrapper(self):
        logger.info("\nTesting ArcBatchGridEnv with wrapper PixelAwareRewardWrapper")
        batch_size = np.random.randint(2, 4)
        size = np.random.randint(2, 4)
        color_values = np.random.randint(2, 4)

        env = ArcBatchGridEnv(size, color_values, device="cpu")
        env = PixelAwareRewardWrapper(env, n_steps=np.random.randint(1, 10), gamma=1.0)
        self.episodes_simulation(batch_size, size, color_values, env)

    def episodes_simulation(self, batch_size, size, color_values, env):
        logger.info("Batch size: {}".format(batch_size))
        logger.info("Grid size: {}".format(size))
        logger.info("Color values: {}".format(color_values))
        episodes = np.random.randint(5, 10)
        for episode in range(episodes):
            logger.info("Episode: {}/{}".format(episode, episodes))
            dummy_batch = TensorDict(
                {
                    "batch": {
                        "input": torch.randint(
                            0, color_values - 1, size=(batch_size, size, size)
                        ),
                        "output": torch.randint(
                            0, color_values, size=(batch_size, size, size)
                        ),
                    },
                    "examples": torch.randint(
                        0,
                        color_values,
                        size=(batch_size, np.random.randint(2, 5), 2, size, size),
                    ),
                }
            )
            # dummy_batch["batch"]["output"] = dummy_batch["batch"]["input"].clone() + 1
            env.reset(options=dummy_batch)
            initial_diff_grid = env.get_difference()
            last_diff = torch.sum(initial_diff_grid)
            step = 0
            for y_loc in range(size):
                for x_loc in range(size):
                    logger.debug(f"\n>>>>>>>Step {step}/{size**2} <<<<<<<<<<<")
                    target_grids = env.get_wrapper_attr("observations")["target"]
                    is_last_step = int(y_loc == size - 1 and x_loc == size - 1)
                    value = target_grids[:, y_loc, x_loc]
                    action = torch.zeros((batch_size, 4))
                    action[:, 0] = y_loc
                    action[:, 1] = x_loc
                    action[:, 2] = value
                    submission = torch.randint(is_last_step, 2, (batch_size,)).long()
                    action[:, 3] = submission
                    action = action.long()
                    action = self.to_dict_tensors(action, to_torch=True)
                    assert env.action_space.contains(action.numpy()), ValueError(
                        "Action not in action space: {}".format(action)
                    )
                    last_diff = env.get_difference()
                    observation, reward, terminated, truncated, info = env.step(action)
                    grid_diff = env.get_difference()
                    step += 1
                    logger.debug(
                        f"Computing differences in grids for the current step [{step}]"
                    )
                    sum_changed_values, sum_init_diff_values = self.compute_difference(
                        grid_diff,
                        last_diff,
                        initial_diff_grid[:, y_loc, x_loc],
                    )
                    logger.debug(f"Assertions of grids")
                    self.assertions_grids(
                        batch_size,
                        step,
                        x_loc,
                        y_loc,
                        observation,
                        info,
                        sum_changed_values,
                        sum_init_diff_values,
                        dummy_batch,
                    )
                    logger.debug(f"Assertions of termination")
                    grid_diff = torch.sum(torch.abs(grid_diff))
                    if is_last_step:
                        assert terminated, "The last step should terminate the episode"
                        assert (
                            grid_diff == 0
                        ), "The last step should have zero difference"
                    else:
                        assert not terminated or (
                            terminated and grid_diff == 0
                        ), "The episode should not terminate"

                    logger.debug(f"Assertions of rewards")
                    if isinstance(env, PixelAwareRewardWrapper):
                        self.assert_state_property(env.get_wrapper_attr("state"))
                        self.assert_reward_pixel(
                            reward,
                            observation,
                            submission,
                            is_last_step,
                            grid_diff,
                            (
                                dummy_batch["batch"]["input"][:, y_loc, x_loc]
                                - dummy_batch["batch"]["output"][:, y_loc, x_loc]
                                != 0
                            ).long(),
                            env.max_penality,
                        )
                    elif isinstance(env, ArcBatchGridEnv):
                        self.assert_state_property(env.state)
                        self.assert_reward_arc(
                            reward,
                            observation,
                            submission,
                            is_last_step,
                        )

                    # Dummy test n-step computation: Check that it runs
                    n_reward = env.n_step_reward()
                    assert isinstance(n_reward, torch.Tensor), TypeError(
                        "N-step reward must be a Tensor, type {} returned.".format(
                            type(n_reward)
                        )
                    )
                    assert (
                        n_reward.shape[0] == batch_size
                    ), "N-step reward must have batch size {} but has {}".format(
                        batch_size, n_reward.shape[0]
                    )

    def assert_state_property(self, state):
        """
        Assertions for the state of the environment.
        Args:
            state (TensorDict): The state of the environment.
        """
        assert isinstance(state, TensorDict), TypeError(
            "Input state must be a TensorDict, type {} returned.".format(type(state))
        )
        in_keys = {"last_grid", "grid", "examples", "initial", "index", "terminated"}
        assert set(state.keys()) == in_keys, ValueError(
            "State keys must be {}, keys found {}".format(in_keys, set(state.keys()))
        )

    def compute_difference(self, grid_diff, last_diff, initial_diff_grid_loc):
        """
        Compute the difference between the current grid and the target grid to validate differences.

        Args:
            grid_diff (torch.Tensor): The absolute difference between the current grid and the target grid.
            last_diff (torch.Tensor): The last difference (before the last action) between the current grid and the target grid.
            initial_diff_grid_loc (torch.Tensor): The initial difference between the current grid and the target grid in a [X,Y] location.
        Returns:
            torch.Tensor: The change since the last action, improvement in grid differences.
            torch.Tensor: The sum of the initial difference values in the locations changed on the last action.
        """
        sum_changed_values = torch.sum(torch.abs(last_diff - grid_diff))
        sum_init_diff_values = torch.sum(torch.abs(initial_diff_grid_loc))
        return sum_changed_values, sum_init_diff_values

    def assertions_grids(
        self,
        batch_size,
        step,
        x_loc,
        y_loc,
        observation,
        info,
        sum_changed_values,
        sum_init_diff_values,
        dummy_batch,
    ):
        """
        Assertions for the grids.
        Args:
            batch_size (int): The batch size.
            step (int): The current step.
            x_loc (int): The X location.
            y_loc (int): The Y location.
            observation (dict): The observation for the current step.
            info (dict): The information for the current step.
            sum_changed_values (torch.Tensor): The change since the last action, improvement in grid differences.
            sum_init_diff_values (torch.Tensor): The sum of the initial difference values in the locations changed on the last action.
            dummy_batch (dict): The dummy batch for the current episode
        """
        assert (
            sum_changed_values == sum_init_diff_values
        ), "Difference change is not the expected value. The change is: {} and the expected change is: {}".format(
            sum_changed_values, sum_init_diff_values
        )
        torch.testing.assert_close(
            observation["target"],
            dummy_batch["batch"]["output"],
        ), "Target grid should not change"

        torch.testing.assert_close(
            info["initial"], dummy_batch["batch"]["input"]
        ), "Initial grid should not change"
        torch.testing.assert_close(
            info["index"][:, y_loc, x_loc],
            torch.ones(batch_size, dtype=torch.int64) * step,
        ), "Index grid incorrectly updated. Expected: {} in X[{}],Y[{}] locations for every sample.".format(
            step, x_loc, y_loc
        )

        torch.testing.assert_close(
            info["examples"], dummy_batch["examples"]
        ), "Examples grids should not change"
        assert len(info["examples"].shape) == 5, "Examples grids should be 5D"
        assert (
            info["examples"].shape[0] == batch_size
        ), "Examples grids should have the same batch size"
        assert (
            info["examples"].shape[2] == 2
        ), "Examples grids should have input and output grids on third dimesion"

    def assert_reward_arc(
        self,
        reward,
        observation,
        submission,
        is_last_step,
    ):
        """
        Assertions for the termination and rewards.
        Args:
            reward (torch.Tensor): The reward for the current step.
            observation (dict): The observation for the current step.
            submission (torch.Tensor): A tensor indicating the submit action.
            is_last_step (bool): A boolean indicating if the current step is the last step.
        """

        if is_last_step:
            torch.testing.assert_close(
                reward, torch.ones_like(reward)
            ), "The last step should have a reward of 1"
        else:
            torch.testing.assert_close(
                reward,
                (
                    (
                        torch.sum(
                            torch.abs(observation["grid"] - observation["target"]),
                            dim=(1, 2),
                        )
                        == 0
                    )
                    * submission
                ).type(torch.float32),
            ), "The step should have a reward of 0"

    def assert_reward_pixel(
        self,
        reward,
        observation,
        submission,
        is_last_step,
        grid_diff,
        change,
        max_penality,
    ):
        """
        Assertions for the termination and rewards.
        Args:
            reward (torch.Tensor): The reward for the current step.
            observation (dict): The observation for the current step.
            submission (torch.Tensor): A tensor indicating the submit action.
            terminated (bool): A boolean indicating if the env returned termination.
            is_last_step (bool): A boolean indicating if the current step is the last step.
            grid_diff (torch.Tensor): The grid difference between the current grid and the target grid.
            change (int): The total change in the grid.
            max_penality (int): The maximum penality for the current env.
        """
        if is_last_step:
            torch.testing.assert_close(
                reward, torch.ones_like(reward)
            ), "The last step should have a reward of 0"
        else:
            grid_diff = torch.sum(
                ((observation["target"] - observation["grid"]) != 0).long(),
                dim=(1, 2),
            )
            reference = (change - 1) * (grid_diff != 0).long()
            torch.testing.assert_close(
                reward,
                reference
                + (grid_diff * submission * max_penality)
                + ((grid_diff == 0).long() * submission),
            ), "The step do not have the expected reward."

    def to_dict_tensors(self, sample, to_torch: bool = False):
        if to_torch:
            sample = torch.as_tensor(sample)
            return TensorDict(
                {
                    "y_location": sample[:, 0],
                    "x_location": sample[:, 1],
                    "color_values": sample[:, 2],
                    "submit": sample[:, 3],
                }
            )
        else:
            sample = np.array(sample)
            return {
                "y_location": sample[:, 0],
                "x_location": sample[:, 1],
                "color_values": sample[:, 2],
                "submit": sample[:, 3],
            }

    def test_action_space(self):
        batch_size = np.random.randint(5, 100)
        grid_size = np.random.randint(5, 20)
        values = np.random.randint(2, 20)
        logger.debug(
            "Testing Action Space. Grid Size: {}, Color values: {}".format(
                grid_size, values
            )
        )
        env = ArcBatchGridEnv(grid_size=grid_size, color_values=values)
        assert values == env.color_values and grid_size == env.grid_size

        # Valid
        assert env.action_space.contains(self.to_dict_tensors([[0, 0, 0, 0]]))
        assert env.action_space.contains(
            self.to_dict_tensors([[grid_size - 1, grid_size - 1, values - 1, 1]])
        )
        assert env.action_space.contains(
            self.to_dict_tensors([[0, grid_size - 1, values - 1, 0] for __ in range(2)])
        )
        assert env.action_space.contains(
            self.to_dict_tensors([[0, grid_size - 1, 0, 0] for __ in range(3)])
        )
        assert env.action_space.contains(
            self.to_dict_tensors([[grid_size - 1, 0, 0, 0] for __ in range(4)])
        )

        # Not valid
        assert not env.action_space.contains(
            self.to_dict_tensors([[0, 0, 0, 0], [0, grid_size - 1, values - 1, -1]])
        )
        assert not env.action_space.contains(
            self.to_dict_tensors([[0, 0, 0, 0], [0, grid_size - 1, values - 1, 2]])
        )
        assert not env.action_space.contains(
            self.to_dict_tensors([[0, 0, 0, 0], [0, grid_size - 1, values, 1]])
        )
        assert not env.action_space.contains(
            self.to_dict_tensors([[0, 0, 0, 0], [0, grid_size - 1, -1, 1]])
        )
        assert not env.action_space.contains(
            self.to_dict_tensors(
                [
                    [grid_size - 1, grid_size - 1, values - 1, 1],
                    [0, grid_size, values - 1, 1],
                ]
            )
        )
        assert not env.action_space.contains(
            self.to_dict_tensors(
                [
                    [grid_size - 1, grid_size - 1, values - 1, 1],
                    [grid_size, -1, values - 1, 1],
                ]
            )
        )
        assert not env.action_space.contains(
            self.to_dict_tensors(
                [
                    [grid_size - 1, grid_size - 1, values - 1, 1],
                    [-1, grid_size - 1, values - 1, 1],
                ]
            )
        )
        assert not env.action_space.contains(
            self.to_dict_tensors(
                [
                    [grid_size - 1, grid_size - 1, values - 1, 1],
                    [grid_size, grid_size - 1, values - 1, 1],
                ]
            )
        )

        # Not valid format
        with self.assertRaises(IndexError):
            assert not env.action_space.contains(
                self.to_dict_tensors([0, grid_size - 1, values - 1, 2])
            )
        with self.assertRaises(IndexError):
            assert not env.action_space.contains(
                self.to_dict_tensors([[grid_size - 1, values - 1, 2]])
            )


if __name__ == "__main__":
    unittest.main()
