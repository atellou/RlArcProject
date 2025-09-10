from rlarcworld.environments.arc_batch_grid_env import ArcBatchGridEnv
from rlarcworld.environments.wrappers.rewards import PixelAwareRewardWrapper
import numpy as np
import torch
from tensordict import TensorDict
import logging
import pytest

logger = logging.getLogger(__name__)


def to_dict_tensors(sample, to_torch: bool = False):
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


def assertions_grids(
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
    reward,
    observation,
    submission,
    is_last_step,
):
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
    reward,
    observation,
    submission,
    is_last_step,
    grid_diff,
    change,
    max_penalty,
):
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
            + (grid_diff * submission * max_penalty)
            + ((grid_diff == 0).long() * submission),
        ), "The step do not have the expected reward."


def episodes_simulation(batch_size, size, color_values, env):
    logger.info("Batch size: {}".format(batch_size))
    logger.info("Grid size: {}".format(size))
    logger.info("Color values: {}".format(color_values))
    episodes = np.random.randint(2, 3)
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
        env.reset(options=dummy_batch)
        initial_diff_grid = env.get_difference()
        last_diff = torch.sum(initial_diff_grid)
        step = 0
        for y_loc in range(size):
            for x_loc in range(size):
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
                action = to_dict_tensors(action, to_torch=True)
                assert env.action_space.contains(action.numpy()), ValueError(
                    "Action not in action space: {}".format(action)
                )
                last_diff = env.get_difference()
                observation, reward, terminated, truncated, info = env.step(action)
                grid_diff = env.get_difference()
                step += 1
                sum_changed_values = torch.sum(torch.abs(last_diff - grid_diff))
                sum_init_diff_values = torch.sum(
                    torch.abs(initial_diff_grid[:, y_loc, x_loc])
                )
                assertions_grids(
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
                grid_diff_abs = torch.sum(torch.abs(grid_diff))
                if is_last_step:
                    assert terminated, "The last step should terminate the episode"
                    assert grid_diff_abs == 0, "The last step should have zero difference"
                else:
                    assert not terminated or (
                        terminated and grid_diff_abs == 0
                    ), "The episode should not terminate"

                if isinstance(env, PixelAwareRewardWrapper):
                    state = env.get_wrapper_attr("state")
                    assert isinstance(state, TensorDict)
                    assert_reward_pixel(
                        reward,
                        observation,
                        submission,
                        is_last_step,
                        grid_diff_abs,
                        (
                            dummy_batch["batch"]["input"][:, y_loc, x_loc]
                            - dummy_batch["batch"]["output"][:, y_loc, x_loc]
                            != 0
                        ).long(),
                        env.max_penalty,
                    )
                elif isinstance(env, ArcBatchGridEnv):
                    state = env.state
                    assert isinstance(state, TensorDict)
                    assert_reward_arc(
                        reward,
                        observation,
                        submission,
                        is_last_step,
                    )

                n_reward = env.n_step_reward()
                assert isinstance(n_reward, torch.Tensor)
                assert n_reward.shape[0] == batch_size


# ------------------------- Tests -------------------------

def test_environment():
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
    episodes_simulation(batch_size, size, color_values, env)


def test_arc_save_and_load(tmp_path):
    color_values = 10
    batch_size = 32
    grid_size = 10
    env = ArcBatchGridEnv(
        grid_size=grid_size,
        color_values=color_values,
        n_steps=10,
        gamma=0.9,
        device="cpu",
    )
    dummy_batch = TensorDict(
        {
            "batch": {
                "input": torch.randint(
                    0, color_values - 1, size=(batch_size, grid_size, grid_size)
                ),
                "output": torch.randint(
                    0, color_values, size=(batch_size, grid_size, grid_size)
                ),
            },
            "examples": torch.randint(
                0,
                color_values,
                size=(batch_size, np.random.randint(2, 5), 2, grid_size, grid_size),
            ),
        }
    )
    env.reset(options=dummy_batch)
    env.observations = torch.randn(32, 10)
    env.information = TensorDict({"key": torch.randn(32, 10)})
    env.last_grid = torch.randn(32, 10)
    env.last_reward = 1.0
    env.time_step = 10
    env.reward_storage = torch.randn(32, 10)
    env.is_train_episode = True

    save_dir = tmp_path / "test_checkpoint"
    env.save(str(save_dir))

    loaded_env = ArcBatchGridEnv(
        grid_size=grid_size, color_values=color_values, n_steps=10, gamma=0.9
    )
    loaded_env.load(
        f"{save_dir}/{loaded_env.__class__.__name__}.ptc", weights_only=False
    )

    torch.testing.assert_close(env.observations, loaded_env.observations)
    torch.testing.assert_close(env.information, loaded_env.information)
    torch.testing.assert_close(env.last_grid, loaded_env.last_grid)
    torch.testing.assert_close(env.last_reward, loaded_env.last_reward)
    torch.testing.assert_close(env.time_step, loaded_env.time_step)
    torch.testing.assert_close(env.reward_storage, loaded_env.reward_storage)
    torch.testing.assert_close(env.discount_factor, loaded_env.discount_factor)
    torch.testing.assert_close(env.is_train_episode, loaded_env.is_train_episode)
    torch.testing.assert_close(env.n_steps, loaded_env.n_steps)
    torch.testing.assert_close(env.gamma, loaded_env.gamma)

    torch.testing.assert_close(env.discount_factor, loaded_env.discount_factor)
    torch.testing.assert_close(env.batch_size, loaded_env.batch_size)
    torch.testing.assert_close(env.n_steps, loaded_env.n_steps)
    torch.testing.assert_close(env.gamma, loaded_env.gamma)


def test_arc_wrapper_save_and_load(tmp_path):
    color_values = 10
    batch_size = 32
    grid_size = 10
    env0 = ArcBatchGridEnv(
        grid_size=grid_size,
        color_values=color_values,
        n_steps=10,
        gamma=0.9,
        device="cpu",
    )
    env1 = PixelAwareRewardWrapper(
        env0, n_steps=10, gamma=0.9, apply_clamp=True, v_min=0, v_max=1
    )
    dummy_batch = TensorDict(
        {
            "batch": {
                "input": torch.randint(
                    0, color_values - 1, size=(batch_size, grid_size, grid_size)
                ),
                "output": torch.randint(
                    0, color_values, size=(batch_size, grid_size, grid_size)
                ),
            },
            "examples": torch.randint(
                0,
                color_values,
                size=(batch_size, np.random.randint(2, 5), 2, grid_size, grid_size),
            ),
        }
    )
    env1.reset(options=dummy_batch)
    env0.observations = torch.randn(32, 10)
    env0.information = TensorDict({"key": torch.randn(32, 10)})
    env0.last_grid = torch.randn(32, 10)
    env0.last_reward = 1.0
    env0.time_step = 10
    env0.reward_storage = torch.randn(32, 10)
    env0.is_train_episode = True

    env1.last_reward = 1.0
    env1.reward_storage = torch.randn(32, 10)

    save_dir = tmp_path / "test_checkpoint"
    env1.save(str(save_dir))

    loaded_env0 = ArcBatchGridEnv(
        grid_size=grid_size, color_values=color_values, n_steps=10, gamma=0.9
    )
    loaded_env1 = PixelAwareRewardWrapper(
        loaded_env0, n_steps=10, gamma=0.9, apply_clamp=True, v_min=0, v_max=1
    )
    loaded_env1.load(
        parent_path=f"{save_dir}/parent/{loaded_env0.__class__.__name__}.ptc",
        child_path=f"{save_dir}/child/{loaded_env1.__class__.__name__}.ptc",
        weights_only=False,
        device="cpu",
    )

    torch.testing.assert_close(env0.observations, loaded_env0.observations)
    torch.testing.assert_close(env0.information, loaded_env0.information)
    torch.testing.assert_close(env0.last_grid, loaded_env0.last_grid)
    torch.testing.assert_close(env0.last_reward, loaded_env0.last_reward)
    torch.testing.assert_close(env0.time_step, loaded_env0.time_step)
    torch.testing.assert_close(env0.reward_storage, loaded_env0.reward_storage)
    torch.testing.assert_close(env0.discount_factor, loaded_env0.discount_factor)
    torch.testing.assert_close(env0.is_train_episode, loaded_env0.is_train_episode)
    torch.testing.assert_close(env0.n_steps, loaded_env0.n_steps)
    torch.testing.assert_close(env0.gamma, loaded_env0.gamma)

    torch.testing.assert_close(env1.last_reward, loaded_env1.last_reward)
    torch.testing.assert_close(env1.reward_storage, loaded_env1.reward_storage)
    torch.testing.assert_close(env1.discount_factor, loaded_env1.discount_factor)

    torch.testing.assert_close(env0.discount_factor, loaded_env0.discount_factor)
    torch.testing.assert_close(env0.batch_size, loaded_env0.batch_size)
    torch.testing.assert_close(env0.n_steps, loaded_env0.n_steps)
    torch.testing.assert_close(env0.gamma, loaded_env0.gamma)

    torch.testing.assert_close(env1.batch_size, loaded_env1.batch_size)
    torch.testing.assert_close(env1.n_steps, loaded_env1.n_steps)
    torch.testing.assert_close(env1.gamma, loaded_env1.gamma)
    torch.testing.assert_close(env1.max_penalty, loaded_env1.max_penalty)
    torch.testing.assert_close(env1.apply_clamp, loaded_env1.apply_clamp)
    torch.testing.assert_close(env1.v_min, loaded_env1.v_min)
    torch.testing.assert_close(env1.v_max, loaded_env1.v_max)


def test_wrapper():
    logger.info("\nTesting ArcBatchGridEnv with wrapper PixelAwareRewardWrapper")
    batch_size = np.random.randint(2, 4)
    size = np.random.randint(2, 4)
    color_values = np.random.randint(2, 4)

    env = ArcBatchGridEnv(size, color_values, device="cpu")
    env = PixelAwareRewardWrapper(env, n_steps=np.random.randint(1, 10), gamma=1.0)
    episodes_simulation(batch_size, size, color_values, env)


def test_action_space():
    grid_size = np.random.randint(5, 20)
    values = np.random.randint(2, 20)
    logger.debug(
        "Testing Action Space. Grid Size: {}, Color values: {}".format(
            grid_size, values
        )
    )
    env = ArcBatchGridEnv(grid_size=grid_size, color_values=values)
    assert values == env.color_values and grid_size == env.grid_size

    assert env.action_space.contains(to_dict_tensors([[0, 0, 0, 0]]))
    assert env.action_space.contains(
        to_dict_tensors([[grid_size - 1, grid_size - 1, values - 1, 1]])
    )
    assert env.action_space.contains(
        to_dict_tensors([[0, grid_size - 1, values - 1, 0] for __ in range(2)])
    )
    assert env.action_space.contains(
        to_dict_tensors([[0, grid_size - 1, 0, 0] for __ in range(3)])
    )
    assert env.action_space.contains(
        to_dict_tensors([[grid_size - 1, 0, 0, 0] for __ in range(4)])
    )

    assert not env.action_space.contains(
        to_dict_tensors([[0, 0, 0, 0], [0, grid_size - 1, values - 1, -1]])
    )
    assert not env.action_space.contains(
        to_dict_tensors([[0, 0, 0, 0], [0, grid_size - 1, values - 1, 2]])
    )
    assert not env.action_space.contains(
        to_dict_tensors([[0, 0, 0, 0], [0, grid_size - 1, values, 1]])
    )
    assert not env.action_space.contains(
        to_dict_tensors([[0, 0, 0, 0], [0, grid_size - 1, -1, 1]])
    )
    assert not env.action_space.contains(
        to_dict_tensors(
            [
                [grid_size - 1, grid_size - 1, values - 1, 1],
                [0, grid_size, values - 1, 1],
            ]
        )
    )
    assert not env.action_space.contains(
        to_dict_tensors(
            [
                [grid_size - 1, grid_size - 1, values - 1, 1],
                [grid_size, -1, values - 1, 1],
            ]
        )
    )
    assert not env.action_space.contains(
        to_dict_tensors(
            [
                [grid_size - 1, grid_size - 1, values - 1, 1],
                [-1, grid_size - 1, values - 1, 1],
            ]
        )
    )
    assert not env.action_space.contains(
        to_dict_tensors(
            [
                [grid_size - 1, grid_size - 1, values - 1, 1],
                [grid_size, grid_size - 1, values - 1, 1],
            ]
        )
    )

    with pytest.raises(IndexError):
        assert not env.action_space.contains(
            to_dict_tensors([0, grid_size - 1, values - 1, 2])
        )
    with pytest.raises(IndexError):
        assert not env.action_space.contains(
            to_dict_tensors([[grid_size - 1, values - 1, 2]])
        )
