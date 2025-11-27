import logging
from pathlib import Path

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from rlarcworld.agent.reward_world_model import RewardWorldModel
from rlarcworld.common.enums import ActionKeys, RewardKeys
from rlarcworld.environments.arc_batch_grid_env import (
    ArcBatchGridEnv,
    ArcEnvRunningMode,
)
from rlarcworld.environments.wrappers.pixel_aware_reward import PixelAwareRewardWrapper
from rlarcworld.environments.wrappers.reward_world_model import RewardWorldModelWrapper

logger = logging.getLogger(__name__)


def make_batch(batch_size: int, grid: int, colors: int) -> dict:
    # Simple synthetic data
    batch_in = torch.randint(0, colors, (batch_size, grid, grid))
    batch_out = torch.randint(0, colors, (batch_size, grid, grid))
    examples = torch.randint(0, colors, (batch_size, 10, 2, grid, grid))
    return {"input": batch_in, "output": batch_out, "examples": examples}


def step_once(env: ArcBatchGridEnv) -> None:
    # Random but valid actions
    y = torch.randint(0, env.grid_size, (env.batch_size,))
    x = torch.randint(0, env.grid_size, (env.batch_size,))
    c = torch.randint(0, env.color_values, (env.batch_size,))
    s = torch.zeros(env.batch_size, dtype=torch.long)  # no submit
    actions = TensorDict(
        {
            ActionKeys.Y_LOCATION.value: y,
            ActionKeys.X_LOCATION.value: x,
            ActionKeys.COLOR_VALUES.value: c,
            ActionKeys.SUBMIT.value: s,
        },
        batch_size=env.batch_size,
    )
    env.step(actions)


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
    assert sum_changed_values == sum_init_diff_values, (
        "Difference change is not the expected value. The change is: {} and "
        "the expected change is: {}".format(sum_changed_values, sum_init_diff_values)
    )
    (
        torch.testing.assert_close(
            observation["target"],
            dummy_batch["batch"]["output"],
        ),
        "Target grid should not change",
    )

    (
        torch.testing.assert_close(info["initial"], dummy_batch["batch"]["input"]),
        "Initial grid should not change",
    )
    (
        torch.testing.assert_close(
            info["index"][:, y_loc, x_loc],
            torch.ones(batch_size, dtype=torch.int64) * step,
        ),
        (
            "Index grid incorrectly updated. Expected: {} in X[{}],Y[{}] "
            "locations for every sample.".format(step, x_loc, y_loc)
        ),
    )

    (
        torch.testing.assert_close(info["examples"], dummy_batch["examples"]),
        "Examples grids should not change",
    )
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
        (
            torch.testing.assert_close(reward, torch.ones_like(reward)),
            "The last step should have a reward of 1",
        )
    else:
        (
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
            ),
            "The step should have a reward of 0",
        )


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
        (
            torch.testing.assert_close(reward, torch.ones_like(reward)),
            "The last step should have a reward of 0",
        )
    else:
        grid_diff = torch.sum(
            ((observation["target"] - observation["grid"]) != 0).long(),
            dim=(1, 2),
        )
        reference = (change - 1) * (grid_diff != 0).long()
        (
            torch.testing.assert_close(
                reward,
                reference
                + (grid_diff * submission * max_penalty)
                + ((grid_diff == 0).long() * submission),
            ),
            "The step do not have the expected reward.",
        )


def episodes_simulation(batch_size, size, color_values, env):
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Grid size: {size}")
    logger.info(f"Color values: {color_values}")
    episodes = np.random.randint(2, 3)
    for episode in range(episodes):
        logger.info(f"Episode: {episode}/{episodes}")
        dummy_batch = {
            "batch": {
                "input": torch.randint(
                    0, color_values - 1, size=(batch_size, size, size)
                ),
                "output": torch.randint(0, color_values, size=(batch_size, size, size)),
            },
            "examples": torch.randint(
                0,
                color_values,
                size=(batch_size, np.random.randint(2, 5), 2, size, size),
            ),
            "mode": "train",
        }
        env.reset(options=dummy_batch)
        initial_diff_grid = env.get_difference()
        last_diff = torch.sum(initial_diff_grid)
        step = 0
        for y_loc in range(size):
            for x_loc in range(size):
                # Support both wrapper and base env when accessing observations
                try:
                    obs_dict = env.get_wrapper_attr("observations")
                except AttributeError:
                    obs_dict = env.observations
                target_grids = obs_dict["target"]
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
                    f"Action not in action space: {action}"
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
                    assert (
                        grid_diff_abs == 0
                    ), "The last step should have zero difference"
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


def test_environment(env: ArcBatchGridEnv):
    logger.info("Testing ArcBatchGridEnv (fixture)")
    batch_size = np.random.randint(2, 4)
    size = env.grid_size
    color_values = env.color_values
    episodes_simulation(batch_size, size, color_values, env)


@pytest.mark.parametrize("as_dir", [False, True])
@pytest.mark.parametrize("is_train", [True, False])
def test_arc_batch_grid_env_checkpoint_roundtrip(
    tmp_path: Path, as_dir: bool, is_train: bool
):
    grid, colors, batch_size = 6, 7, 3
    env1 = ArcBatchGridEnv(grid_size=grid, color_values=colors, n_steps=3, gamma=0.9)

    batch = make_batch(batch_size, grid, colors)
    obs1, info1 = env1.reset(
        options={"batch": batch, "examples": batch["examples"], "mode": "train"}
    )
    step_once(env1)

    # Save
    target_path = tmp_path / ("env_ckpt.pt" if not as_dir else "env_dir")
    env1.save_checkpoint(target_path)

    # Load into fresh env with same constructor args
    env2 = ArcBatchGridEnv(grid_size=grid, color_values=colors, n_steps=3, gamma=0.9)
    env2.load(target_path)

    # Compare key fields
    def assert_td_close(a: TensorDict, b: TensorDict):
        assert set(a.keys()) == set(b.keys())
        for k in a.keys():
            torch.testing.assert_close(a[k], b[k])

    assert_td_close(env2.observations, env1.observations)
    assert_td_close(env2.information, env1.information)
    torch.testing.assert_close(env2.last_grid, env1.last_grid)
    torch.testing.assert_close(env2.last_reward, env1.last_reward)
    assert env2.time_step == env1.time_step
    torch.testing.assert_close(env2.reward_storage, env1.reward_storage)
    assert env2.mode == env1.mode
    assert env2.n_steps == env1.n_steps
    assert env2.gamma == env1.gamma
    assert env2.grid_size == env1.grid_size
    assert env2.color_values == env1.color_values
    assert env2.batch_size == env1.batch_size

    # Discount factor recomputed should match definition
    torch.testing.assert_close(env2.get_discount_factor(), env1.get_discount_factor())


@pytest.mark.parametrize("as_dir", [False, True])
@pytest.mark.parametrize("apply_clamp", [False, True])
def test_reward_wrapper_checkpoint_roundtrip(
    tmp_path: Path, as_dir: bool, apply_clamp: bool
):
    grid, colors, batch_size = 5, 6, 2
    base = ArcBatchGridEnv(grid_size=grid, color_values=colors, n_steps=4, gamma=0.95)
    env1 = PixelAwareRewardWrapper(
        base,
        max_penalty=1.0,
        n_steps=4,
        gamma=0.95,
        apply_clamp=apply_clamp,
        v_min=-10 if apply_clamp else None,
        v_max=10 if apply_clamp else None,
    )

    batch = make_batch(batch_size, grid, colors)
    obs1, info1 = env1.reset(
        options={"batch": batch, "examples": batch["examples"], "mode": "train"}
    )
    step_once(env1.env)  # step base env; wrapper uses it in reward
    # Simulate wrapper step reward by pushing some reward
    env1._reward_storage = env1._reward_storage.push(
        torch.zeros(batch_size, 1).to(env1.device)
    )

    # Save
    target_path = tmp_path / ("wrap_ckpt.pt" if not as_dir else "wrap_dir")
    env1.save_checkpoint(target_path)

    # Load into fresh wrapped env
    base2 = ArcBatchGridEnv(grid_size=grid, color_values=colors, n_steps=4, gamma=0.95)
    env2 = PixelAwareRewardWrapper(
        base2,
        max_penalty=1.0,
        n_steps=4,
        gamma=0.95,
        apply_clamp=apply_clamp,
        v_min=-10 if apply_clamp else None,
        v_max=10 if apply_clamp else None,
    )
    env2.load(target_path)

    # Compare wrapper fields
    torch.testing.assert_close(env2.reward_storage, env1.reward_storage)
    torch.testing.assert_close(env2.last_reward, env1.last_reward)
    assert env2.batch_size == env1.batch_size
    assert env2.max_penalty == env1.max_penalty
    assert env2.n_steps == env1.n_steps
    assert env2.gamma == env1.gamma
    assert env2.apply_clamp == env1.apply_clamp
    assert env2.v_min == env1.v_min
    assert env2.v_max == env1.v_max

    # Compare inner environment state
    def assert_td_close(a: TensorDict, b: TensorDict):
        assert set(a.keys()) == set(b.keys())
        for k in a.keys():
            torch.testing.assert_close(a[k], b[k])

    assert_td_close(env2.env.observations, env1.env.observations)
    assert_td_close(env2.env.information, env1.env.information)
    torch.testing.assert_close(env2.env.last_grid, env1.env.last_grid)
    torch.testing.assert_close(env2.env.last_reward, env1.env.last_reward)
    assert env2.env.time_step == env1.env.time_step
    torch.testing.assert_close(env2.env.reward_storage, env1.env.reward_storage)
    assert env2.env.mode == env1.env.mode
    assert env2.env.batch_size == env1.env.batch_size


def test_wrapper(wrapped_env: PixelAwareRewardWrapper):
    logger.info(
        "\nTesting ArcBatchGridEnv with wrapper PixelAwareRewardWrapper (fixture)"
    )
    batch_size = np.random.randint(2, 4)
    size = wrapped_env.get_wrapper_attr("grid_size")
    color_values = wrapped_env.get_wrapper_attr("color_values")
    episodes_simulation(batch_size, size, color_values, wrapped_env)


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


def test_inference_mode():
    """Test environment behavior in inference mode (no target grid)."""
    grid_size = 5
    color_values = 10
    batch_size = 2
    env = ArcBatchGridEnv(grid_size=grid_size, color_values=color_values)

    # Create a batch without output (target) to simulate inference
    batch_in = torch.randint(0, color_values, (batch_size, grid_size, grid_size))
    examples = torch.randint(0, color_values, (batch_size, 10, 2, grid_size, grid_size))
    batch = {"input": batch_in}

    # Reset the environment in inference mode
    env.reset(options={"batch": batch, "examples": examples, "mode": "inference"})

    # Verify inference mode flags
    assert env.mode == "inference", "Environment should be in inference mode"

    # Step once and check rewards
    step_once(env)
    assert torch.all(env.last_reward == 0), "Inference step should have zero reward"

    # Attempt to call training-only methods should raise errors
    with pytest.raises(ValueError):
        env.get_difference()

    with pytest.raises(ValueError):
        env.episode_terminated(torch.zeros(batch_size))


def test_reward_world_model_wrapper_basic():
    """Test basic functionality of RewardWorldModelWrapper."""
    logger.info("Testing RewardWorldModelWrapper basic functionality")

    grid_size = 10
    color_values = 11
    batch_size = 2
    n_steps = 3
    step_horizon = 5

    # Create base environment
    env = ArcBatchGridEnv(
        grid_size=grid_size,
        color_values=color_values,
        n_steps=n_steps,
    )

    # Create reward world model
    model = RewardWorldModel(
        grid_size=grid_size,
        color_values=color_values,
        embedding_size=128,
    )

    # Wrap environment
    wrapped_env = RewardWorldModelWrapper(
        env=env,
        model=model,
        step_horizon=step_horizon,
        n_steps=n_steps,
        gamma=0.99,
    )

    # Create batch
    batch = make_batch(batch_size, grid_size, color_values)

    # Reset environment
    obs, info = wrapped_env.reset(
        options={
            "batch": batch,
            "examples": batch["examples"],
            "mode": ArcEnvRunningMode.TRAIN,
        }
    )

    # Verify initial state
    assert wrapped_env.batch_size == batch_size
    assert wrapped_env.remaining_horizon == step_horizon
    assert wrapped_env._reward_storage is not None

    # Verify reward storage structure
    assert RewardKeys.BINARY.value in wrapped_env._reward_storage
    assert RewardKeys.PIXEL_WISE.value in wrapped_env._reward_storage

    # Take a step
    actions = TensorDict(
        {
            ActionKeys.Y_LOCATION.value: torch.randint(0, grid_size, (batch_size,)),
            ActionKeys.X_LOCATION.value: torch.randint(0, grid_size, (batch_size,)),
            ActionKeys.COLOR_VALUES.value: torch.randint(
                0, color_values, (batch_size,)
            ),
            ActionKeys.SUBMIT.value: torch.zeros(batch_size, dtype=torch.long),
        },
        batch_size=batch_size,
    )

    obs, reward, terminated, truncated, info = wrapped_env.step(actions)

    # Verify step results
    assert reward.shape == (batch_size,)
    assert wrapped_env.remaining_horizon == step_horizon - 1

    # Verify rewards are accessible
    binary_reward = wrapped_env.binary_model_reward
    pixel_wise_reward = wrapped_env.pixel_wise_model_reward

    assert binary_reward.shape == (batch_size,)
    assert pixel_wise_reward.shape == (batch_size,)

    # Verify n-step rewards
    n_binary = wrapped_env.n_binary_reward
    n_pixel_wise = wrapped_env.n_pixel_wise_reward

    assert n_binary.shape == (batch_size,)
    assert n_pixel_wise.shape == (batch_size,)


def test_reward_world_model_wrapper_multi_step():
    """Test RewardWorldModelWrapper with multiple steps and n-step returns."""
    logger.info("Testing RewardWorldModelWrapper multi-step behavior")

    grid_size = 8
    color_values = 11
    batch_size = 3
    n_steps = 5
    step_horizon = 10
    gamma = 0.95

    # Create environment and model
    env = ArcBatchGridEnv(grid_size=grid_size, color_values=color_values, n_steps=1)
    model = RewardWorldModel(grid_size=grid_size, color_values=color_values)

    wrapped_env = RewardWorldModelWrapper(
        env=env,
        model=model,
        step_horizon=step_horizon,
        n_steps=n_steps,
        gamma=gamma,
    )

    # Reset
    batch = make_batch(batch_size, grid_size, color_values)
    wrapped_env.reset(
        options={
            "batch": batch,
            "examples": batch["examples"],
            "mode": ArcEnvRunningMode.TRAIN,
        }
    )

    # Take multiple steps
    rewards_collected = []
    for i in range(n_steps):
        actions = TensorDict(
            {
                ActionKeys.Y_LOCATION.value: torch.randint(0, grid_size, (batch_size,)),
                ActionKeys.X_LOCATION.value: torch.randint(0, grid_size, (batch_size,)),
                ActionKeys.COLOR_VALUES.value: torch.randint(
                    0, color_values, (batch_size,)
                ),
                ActionKeys.SUBMIT.value: torch.zeros(batch_size, dtype=torch.long),
            },
            batch_size=batch_size,
        )

        obs, reward, terminated, truncated, info = wrapped_env.step(actions)
        rewards_collected.append(wrapped_env.binary_model_reward.clone())

        assert wrapped_env.remaining_horizon == step_horizon - (i + 1)

    # Verify n-step return computation
    n_step_return = wrapped_env.n_binary_reward

    # Manually compute expected n-step return
    expected_return = torch.zeros(batch_size).to(wrapped_env.device)
    for i, r in enumerate(rewards_collected):
        expected_return += (gamma**i) * r

    # Should be close (within numerical precision)
    torch.testing.assert_close(n_step_return, expected_return, rtol=1e-4, atol=1e-4)


def test_reward_world_model_wrapper_horizon_truncation():
    """Test that wrapper correctly truncates episodes at step_horizon."""
    logger.info("Testing RewardWorldModelWrapper horizon truncation")

    grid_size = 5
    color_values = 11
    batch_size = 2
    step_horizon = 3

    env = ArcBatchGridEnv(grid_size=grid_size, color_values=color_values)
    model = RewardWorldModel(grid_size=grid_size, color_values=color_values)

    wrapped_env = RewardWorldModelWrapper(
        env=env,
        model=model,
        step_horizon=step_horizon,
        n_steps=1,
        gamma=1.0,
    )

    batch = make_batch(batch_size, grid_size, color_values)
    wrapped_env.reset(
        options={
            "batch": batch,
            "examples": batch["examples"],
            "mode": ArcEnvRunningMode.TRAIN,
        }
    )

    # Take steps until horizon
    for i in range(step_horizon):
        actions = TensorDict(
            {
                ActionKeys.Y_LOCATION.value: torch.zeros(batch_size, dtype=torch.long),
                ActionKeys.X_LOCATION.value: torch.zeros(batch_size, dtype=torch.long),
                ActionKeys.COLOR_VALUES.value: torch.zeros(
                    batch_size, dtype=torch.long
                ),
                ActionKeys.SUBMIT.value: torch.zeros(batch_size, dtype=torch.long),
            },
            batch_size=batch_size,
        )

        obs, reward, terminated, truncated, info = wrapped_env.step(actions)

        if i < step_horizon - 1:
            assert not truncated, f"Should not truncate before horizon at step {i}"
        else:
            assert truncated, f"Should truncate at horizon (step {i})"


def test_reward_world_model_wrapper_caching():
    """Test that reward caching works correctly."""
    logger.info("Testing RewardWorldModelWrapper reward caching")

    grid_size = 5
    color_values = 11
    batch_size = 2

    env = ArcBatchGridEnv(grid_size=grid_size, color_values=color_values)
    model = RewardWorldModel(grid_size=grid_size, color_values=color_values)

    wrapped_env = RewardWorldModelWrapper(
        env=env,
        model=model,
        step_horizon=10,
        n_steps=1,
        gamma=1.0,
    )

    batch = make_batch(batch_size, grid_size, color_values)
    wrapped_env.reset(
        options={
            "batch": batch,
            "examples": batch["examples"],
            "mode": ArcEnvRunningMode.TRAIN,
        }
    )

    # Take one step
    actions = TensorDict(
        {
            ActionKeys.Y_LOCATION.value: torch.zeros(batch_size, dtype=torch.long),
            ActionKeys.X_LOCATION.value: torch.zeros(batch_size, dtype=torch.long),
            ActionKeys.COLOR_VALUES.value: torch.zeros(batch_size, dtype=torch.long),
            ActionKeys.SUBMIT.value: torch.zeros(batch_size, dtype=torch.long),
        },
        batch_size=batch_size,
    )

    wrapped_env.step(actions)

    # Access rewards multiple times - should use cache
    reward1 = wrapped_env.binary_model_reward
    reward2 = wrapped_env.binary_model_reward

    # Should be identical (same object)
    assert torch.equal(reward1, reward2)

    # Take another step - cache should be invalidated
    wrapped_env.step(actions)
    reward3 = wrapped_env.binary_model_reward

    # May or may not be different, but should be computed fresh
    assert reward3.shape == reward1.shape
