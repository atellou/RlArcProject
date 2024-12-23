from rlarcworld.enviroments.arc_batch_grid_env import ArcBatchGridEnv
from rlarcworld.enviroments.wrappers.rewards import PixelAwareRewardWrapper
import numpy as np
import torch
import unittest
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArcBatchGridsEnv(unittest.TestCase):

    def test_environment(self):
        logger.info("Testing ArcBatchGridEnv")
        batch_size = np.random.randint(20, 50)
        size = np.random.randint(20, 50)
        color_values = np.random.randint(2, 20)

        env = ArcBatchGridEnv(size, color_values)
        self.episodes_simulation(batch_size, size, color_values, env)

    def test_wrapper(self):
        logger.info("\nTesting ArcBatchGridEnv with wrapper PixelAwareRewardWrapper")
        batch_size = np.random.randint(20, 50)
        size = np.random.randint(20, 50)
        color_values = np.random.randint(2, 20)

        env = ArcBatchGridEnv(size, color_values)
        env = PixelAwareRewardWrapper(env)
        self.episodes_simulation(batch_size, size, color_values, env)

    def episodes_simulation(self, batch_size, size, color_values, env):
        logger.info("Batch size: {}".format(batch_size))
        logger.info("Grid size: {}".format(size))
        logger.info("Color values: {}".format(color_values))
        episodes = np.random.randint(5, 10)
        for episode in range(episodes):
            logger.info("Episode: {}/{}".format(episode, episodes))
            dummy_batch = {
                "batch": {
                    "input": torch.randint(
                        0, color_values - 1, size=(batch_size, size, size)
                    ),
                    "output": torch.randint(
                        0, color_values, size=(batch_size, size, size)
                    ),
                }
            }
            # dummy_batch["batch"]["output"] = dummy_batch["batch"]["input"].clone() + 1
            env.reset(dummy_batch)
            initial_diff_grid = env.get_difference()
            last_diff = torch.sum(initial_diff_grid)
            step = 0
            for y_loc in range(size):
                for x_loc in range(size):
                    logger.debug(f"\n>>>>>>>Step {step}/{size**2} <<<<<<<<<<<")
                    target_grids = env.observations["target"]
                    is_last_step = int(y_loc == size - 1 and x_loc == size - 1)
                    value = target_grids[:, y_loc, x_loc].numpy(force=False)
                    action = np.zeros(
                        (batch_size, len(env.action_space.feature_space.nvec))
                    )
                    action[:, 0] = y_loc
                    action[:, 1] = x_loc
                    action[:, 2] = value
                    submission = torch.randint(is_last_step, 2, (batch_size,)).long()
                    action[:, 3] = submission.numpy()
                    assert env.action_space.contains(action), ValueError(
                        "Action not in action space: {}".format(action)
                    )
                    last_diff = env.get_difference()
                    observation, reward, terminated, truncated, info = env.step(action)
                    current_diff = env.get_difference()

                    step += 1
                    logger.debug(
                        f"Computing differences in grids for the current step [{step}]"
                    )
                    sum_changed_values, sum_init_diff_values = self.compute_difference(
                        current_diff,
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
                    current_diff = torch.sum(torch.abs(current_diff))
                    if is_last_step:
                        assert terminated, "The last step should terminate the episode"
                        assert (
                            current_diff == 0
                        ), "The last step should have zero difference"
                    else:
                        assert not terminated or (
                            terminated and current_diff == 0
                        ), "The episode should not terminate"

                    logger.debug(f"Assertions of rewards")
                    if isinstance(env, PixelAwareRewardWrapper):
                        self.assert_reward_pixel(
                            reward,
                            observation,
                            submission,
                            is_last_step,
                            current_diff,
                            (
                                dummy_batch["batch"]["input"][:, y_loc, x_loc]
                                - dummy_batch["batch"]["output"][:, y_loc, x_loc]
                                != 0
                            ).long(),
                            size,
                            color_values,
                        )
                    elif isinstance(env, ArcBatchGridEnv):
                        self.assert_reward_arc(
                            reward,
                            observation,
                            submission,
                            is_last_step,
                        )

    def compute_difference(self, current_diff, last_diff, initial_diff_grid_loc):
        """
        Compute the difference between the current grid and the target grid to validate differences.

        Args:
            current_diff (torch.Tensor): The absolute difference between the current grid and the target grid.
            last_diff (torch.Tensor): The last difference (before the last action) between the current grid and the target grid.
            initial_diff_grid_loc (torch.Tensor): The initial difference between the current grid and the target grid in a [X,Y] location.
        Returns:
            torch.Tensor: The change since the last action, improvement in grid differences.
            torch.Tensor: The sum of the initial difference values in the locations changed on the last action.
        """
        sum_changed_values = torch.sum(torch.abs(last_diff - current_diff))
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
        assert torch.equal(
            observation["target"],
            dummy_batch["batch"]["output"],
        ), "Target grid should not change"

        assert torch.equal(
            info["intial"], dummy_batch["batch"]["input"]
        ), "Initial grid should not change"
        assert np.array_equal(
            info["index"][:, y_loc, x_loc], np.ones(batch_size) * step
        ), "Index grid incorrectly updated. Expected: {} in X[{}],Y[{}] locations for every sample.".format(
            step, x_loc, y_loc
        )

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
            assert torch.equal(
                reward, torch.ones_like(reward)
            ), "The last step should have a reward of 1"
        else:
            assert torch.equal(
                reward,
                (
                    torch.sum(
                        torch.abs(observation["current"] - observation["target"]),
                        dim=(1, 2),
                    )
                    == 0
                ).long()
                * submission,
            ), "The step should have a reward of 0"

    def assert_reward_pixel(
        self,
        reward,
        observation,
        submission,
        is_last_step,
        current_diff,
        change,
        size,
        color_values,
    ):
        """
        Assertions for the termination and rewards.
        Args:
            reward (torch.Tensor): The reward for the current step.
            observation (dict): The observation for the current step.
            submission (torch.Tensor): A tensor indicating the submit action.
            terminated (bool): A boolean indicating if the env returned termination.
            is_last_step (bool): A boolean indicating if the current step is the last step.
            current_diff (torch.Tensor): The current difference between the current grid and the target grid.
            change (int): The total change in the grid.
            size (int): The size of the grid.
            color_values (int): The number of color values.
        """
        if is_last_step:
            assert torch.equal(
                reward, torch.zeros_like(reward)
            ), "The last step should have a reward of 0"
        else:
            current_diff = torch.sum(
                ((observation["target"] - observation["current"]) != 0).long(),
                dim=(1, 2),
            )
            reference = (change - 1) * (current_diff != 0).long()
            assert torch.equal(
                reward,
                reference + (-1 * current_diff * submission * size**2 * color_values),
            ), "The step do not have the expected reward."

    def test_action_space(self):
        grid_size = np.random.randint(5, 20)
        values = np.random.randint(2, 20)
        logger.debug(
            "Testing Action Space. Grid Size: {}, Color values: {}".format(
                grid_size, values
            )
        )
        env = ArcBatchGridEnv(size=grid_size, color_values=values)
        assert values == env.color_values and grid_size == env.size
        # Valid
        assert env.action_space.contains([[0, 0, 0, 0]])
        assert env.action_space.contains(
            [[grid_size - 1, grid_size - 1, values - 1, 1]]
        )
        assert env.action_space.contains(
            [[0, grid_size - 1, values - 1, 0] for __ in range(2)]
        )
        assert env.action_space.contains([[0, grid_size - 1, 0, 0] for __ in range(3)])
        assert env.action_space.contains([[grid_size - 1, 0, 0, 0] for __ in range(4)])
        # Not valid
        assert not env.action_space.contains(
            [[0, 0, 0, 0], [0, grid_size - 1, values - 1, -1]]
        )
        assert not env.action_space.contains(
            [[0, 0, 0, 0], [0, grid_size - 1, values - 1, 2]]
        )
        assert not env.action_space.contains(
            [[0, 0, 0, 0], [0, grid_size - 1, values, 1]]
        )
        assert not env.action_space.contains([[0, 0, 0, 0], [0, grid_size - 1, -1, 1]])
        assert not env.action_space.contains(
            [
                [grid_size - 1, grid_size - 1, values - 1, 1],
                [0, grid_size, values - 1, 1],
            ]
        )
        assert not env.action_space.contains(
            [
                [grid_size - 1, grid_size - 1, values - 1, 1],
                [grid_size, -1, values - 1, 1],
            ]
        )
        assert not env.action_space.contains(
            [
                [grid_size - 1, grid_size - 1, values - 1, 1],
                [-1, grid_size - 1, values - 1, 1],
            ]
        )
        assert not env.action_space.contains(
            [
                [grid_size - 1, grid_size - 1, values - 1, 1],
                [grid_size, grid_size - 1, values - 1, 1],
            ]
        )
        # Not valid shape
        assert not env.action_space.contains([0, grid_size - 1, values - 1, 2])
        assert not env.action_space.contains([[grid_size - 1, values - 1, 2]])


if __name__ == "__main__":
    unittest.main()
