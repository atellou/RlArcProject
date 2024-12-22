from rlarcworld.enviroments.arc_batch_grid_env import ArcBatchGridEnv
import numpy as np
import torch
import unittest
import logging

logger = logging.getLogger(__name__)
logger.info("Testing ArcBatchGridEnv")


class ArcBatchGridsEnv(unittest.TestCase):
    def test_batch(self):
        batch_size = np.random.randint(1, 50)
        size = np.random.randint(1, 50)
        color_values = np.random.randint(1, 50)
        logger.info(
            "Testing Episodes with random batches. Batch size: {}, Grid Size: {}, Color values: {}".format(
                batch_size, size, color_values
            )
        )
        env = ArcBatchGridEnv(size=size, color_values=color_values)
        for episode in range(3):
            logger.info("Episode: {}".format(episode))
            dummy_batch = {
                "batch": {
                    "input": torch.randint(
                        0, color_values, size=(batch_size, size, size)
                    ),
                    "output": torch.randint(
                        0, color_values, size=(batch_size, size, size)
                    ),
                }
            }
            env.reset(dummy_batch)
            initial_diff_grid = torch.abs(
                dummy_batch["batch"]["input"] - dummy_batch["batch"]["output"]
            )
            last_diff = torch.sum(initial_diff_grid)
            step = 0
            for y_loc in range(size):
                for x_loc in range(size):
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
                    last_diff = torch.sum(torch.abs(env.get_difference()))
                    observation, reward, terminated, truncated, info = env.step(action)
                    current_diff = torch.sum(torch.abs(env.get_difference()))
                    step += 1
                    logger.info(
                        f"Computing differences in grids for the current step [{step}]"
                    )
                    sum_changed_values, sum_init_diff_values = self.compute_difference(
                        current_diff,
                        last_diff,
                        initial_diff_grid[:, y_loc, x_loc],
                    )
                    logger.info(f"Assertions of grids")
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
                    logger.info(f"Assertions of rewards and termination")
                    self.assertions_reward_termination(
                        reward,
                        observation,
                        submission,
                        terminated,
                        is_last_step,
                        current_diff,
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
        sum_changed_values = torch.abs(last_diff - current_diff)
        sum_init_diff_values = torch.sum(initial_diff_grid_loc)
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

    def assertions_reward_termination(
        self, reward, observation, submission, terminated, is_last_step, current_diff
    ):
        """
        Assertions for the termination and rewards.
        Args:
            reward (torch.Tensor): The reward for the current step.
            observation (dict): The observation for the current step.
            submission (torch.Tensor): A tensor indicating the submit action.
            terminated (bool): A boolean indicating if the env returned termination.
            is_last_step (bool): A boolean indicating if the current step is the last step.
        """

        if is_last_step:
            assert torch.equal(
                reward, torch.ones_like(reward)
            ), "The last step should have a reward of 1"
            assert terminated, "The last step should terminate the episode"
            assert current_diff == 0, "The last step should have zero difference"
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
            assert not terminated, "The episode should not terminate"

    def test_action_space(self):
        grid_size = np.random.randint(5, 20)
        values = np.random.randint(2, 20)
        logger.info(
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
