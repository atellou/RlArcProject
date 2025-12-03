import os

import torch
from tensordict import TensorDict

from rlarcworld.common.configs import NStepMinVMaxConfig
from rlarcworld.common.enums import RewardKeys
from rlarcworld.environments.arc_batch_grid_env import ArcBatchGridEnv
from rlarcworld.environments.steppers.arc_stepper import PixelWiseArcStepper
from rlarcworld.environments.steppers.core import EnvStepper
from rlarcworld.environments.wrappers.pixel_aware_reward import PixelAwareRewardWrapper
from rlarcworld.models.actor import ArcActorNetwork
from rlarcworld.utils.logging_utils import configure_logger

configure_logger(level=os.environ.get("LOG_LEVEL", "INFO").upper(), json_format=False)


def test_env_stepper_one_step(device):
    grid_size, color_values = 16, 7
    env = PixelAwareRewardWrapper(
        ArcBatchGridEnv(grid_size, color_values, device=device)
    )
    actor = ArcActorNetwork(
        grid_size=grid_size,
        color_values=color_values,
        config={"device": device, "amp_enabled": False, "use_checkpointing": False},
    )
    actor.eval()

    # Create a simple sampler that yields one sample
    def simple_sampler():
        batch_in = torch.randint(0, color_values, (1, grid_size, grid_size))
        batch_out = torch.randint(0, color_values, (1, grid_size, grid_size))
        examples = torch.randint(0, color_values, (1, 10, 2, grid_size, grid_size))
        yield {
            "batch": {"input": batch_in, "output": batch_out},
            "examples": examples,
        }

    def env_reset_fn(env, sample, seed=None):
        env.reset(options={**sample, "mode": "train"}, seed=seed)

    def on_step_fn(
        env, state, action_probs, reward, observation, terminated, truncated, info
    ):
        return (
            env,
            state,
            action_probs,
            {"reward": reward},
            observation,
            terminated,
            truncated,
            info,
        )

    stepper = EnvStepper(
        actor=actor,
        device=device,
        env=env,
        sampler=simple_sampler(),
        env_sample_reset=env_reset_fn,
        amp_enabled=False,
        on_step=on_step_fn,
    )

    # Reset env with a trivial batch to satisfy interface
    batch_in = torch.randint(0, color_values, (1, grid_size, grid_size))
    batch_out = torch.randint(0, color_values, (1, grid_size, grid_size))
    examples = torch.randint(0, color_values, (1, 10, 2, grid_size, grid_size))
    env.reset(
        options={
            "batch": {"input": batch_in, "output": batch_out},
            "examples": examples,
            "mode": "train",
        }
    )

    step = stepper.step(env)

    assert isinstance(step, TensorDict)
    for key in ["state", "actions", "reward", "next_state", "terminated", "truncated"]:
        assert key in step

    # Check shapes minimally
    assert isinstance(step["state"], TensorDict)
    assert isinstance(step["next_state"], TensorDict)
    assert isinstance(step["actions"], TensorDict)
    assert isinstance(step["reward"], TensorDict)


def test_env_stepper_one_step_with_on_step(device):
    grid_size, color_values = 16, 7
    env = PixelAwareRewardWrapper(
        ArcBatchGridEnv(grid_size, color_values, device=device)
    )
    actor = ArcActorNetwork(
        grid_size=grid_size,
        color_values=color_values,
        config={"device": device, "amp_enabled": False, "use_checkpointing": False},
    )
    actor.eval()

    # Create a simple sampler that yields one sample
    def simple_sampler():
        batch_in = torch.randint(0, color_values, (1, grid_size, grid_size))
        batch_out = torch.randint(0, color_values, (1, grid_size, grid_size))
        examples = torch.randint(0, color_values, (1, 10, 2, grid_size, grid_size))
        yield {
            "batch": {"input": batch_in, "output": batch_out},
            "examples": examples,
        }

    def env_reset_fn(env, sample, seed=None):
        env.reset(options={**sample, "mode": "train"}, seed=seed)

    # Use on_step callback to process rewards
    def process_rewards(
        env, state, action_probs, reward, observation, terminated, truncated, info
    ):
        reward = TensorDict(
            {"pixel_wise": reward, "binary": env.get_source_env_reward()}
        )
        return (
            env,
            state,
            action_probs,
            reward,
            observation,
            terminated,
            truncated,
            info,
        )

    stepper = EnvStepper(
        actor=actor,
        device=device,
        env=env,
        sampler=simple_sampler(),
        env_sample_reset=env_reset_fn,
        amp_enabled=False,
        on_step=process_rewards,
    )

    # Reset env with a trivial batch to satisfy interface
    batch_in = torch.randint(0, color_values, (1, grid_size, grid_size))
    batch_out = torch.randint(0, color_values, (1, grid_size, grid_size))
    examples = torch.randint(0, color_values, (1, 10, 2, grid_size, grid_size))
    env.reset(
        options={
            "batch": {"input": batch_in, "output": batch_out},
            "examples": examples,
            "mode": "train",
        }
    )

    step = stepper.step(env)

    assert isinstance(step, TensorDict)
    for key in ["state", "actions", "reward", "next_state", "terminated", "truncated"]:
        assert key in step

    # Reward should be a TensorDict with at least pixel_wise and binary keys
    assert isinstance(step["reward"], TensorDict)
    assert "pixel_wise" in step["reward"]
    assert "binary" in step["reward"]


def test_env_stepper_multiple_steps_with_on_step(device):
    grid_size, color_values = 16, 7
    env = PixelAwareRewardWrapper(
        ArcBatchGridEnv(grid_size, color_values, n_steps=3, device=device),
        n_steps=3,
    )
    actor = ArcActorNetwork(
        grid_size=grid_size,
        color_values=color_values,
        config={"device": device, "amp_enabled": False, "use_checkpointing": False},
    )
    actor.eval()

    # Create a simple sampler that yields one sample
    def simple_sampler():
        batch_in = torch.randint(0, color_values, (1, grid_size, grid_size))
        batch_out = torch.randint(0, color_values, (1, grid_size, grid_size))
        examples = torch.randint(0, color_values, (1, 10, 2, grid_size, grid_size))
        yield {
            "batch": {"input": batch_in, "output": batch_out},
            "examples": examples,
        }

    def env_reset_fn(env, sample, seed=None):
        env.reset(options={**sample, "mode": "train"}, seed=seed)

    stepper = PixelWiseArcStepper(
        max_steps_per_epoch=2,
        dataset_path="tests/test_data/",
        grid_size=grid_size,
        color_values=color_values,
        actor=actor,
        config={"device": device, "amp_enabled": False, "use_checkpointing": False},
        n_step_config=NStepMinVMaxConfig(
            n_steps=2,
            pixel_wise_min_v=-10,
            pixel_wise_max_v=1,
            binary_min_v=-10,
            binary_max_v=1,
        ),
    )

    # Reset env
    batch_in = torch.randint(0, color_values, (1, grid_size, grid_size))
    batch_out = torch.randint(0, color_values, (1, grid_size, grid_size))
    examples = torch.randint(0, color_values, (1, 10, 2, grid_size, grid_size))
    env.reset(
        options={
            "batch": {"input": batch_in, "output": batch_out},
            "examples": examples,
            "mode": "train",
        }
    )

    # Take multiple steps
    steps = []
    for _ in range(3):
        step = stepper.core.step(env)
        steps.append(step)
        # Check if we should break early
        if step["terminated"].item() or step["truncated"].item():
            break

    # Verify we got at least one step
    assert len(steps) >= 1

    # Check the structure of each step
    for step in steps:
        assert isinstance(step, TensorDict)
        for key in [
            "state",
            "actions",
            "reward",
            "next_state",
            "terminated",
            "truncated",
        ]:
            assert key in step

        # Reward should be a TensorDict with at least pixel_wise and binary keys
        assert isinstance(step["reward"], TensorDict)
        assert "pixel_wise" in step["reward"]
        assert "binary" in step["reward"]
        assert set(RewardKeys.get_n_step_keys()).issubset(step["reward"].keys())


def test_env_stepper_episode_stream(device):
    """Test the episode stream functionality."""
    grid_size, color_values = 8, 5
    env = PixelAwareRewardWrapper(
        ArcBatchGridEnv(grid_size, color_values, n_steps=2, device=device),
        n_steps=2,
    )

    actor = ArcActorNetwork(
        grid_size=grid_size,
        color_values=color_values,
        config={"device": device, "amp_enabled": False, "use_checkpointing": False},
    )
    actor.eval()

    # Create a simple sampler that yields one sample
    def simple_sampler():
        batch_in = torch.randint(0, color_values, (1, grid_size, grid_size))
        batch_out = torch.randint(0, color_values, (1, grid_size, grid_size))
        examples = torch.randint(0, color_values, (1, 4, 2, grid_size, grid_size))
        yield {
            "batch": {"input": batch_in, "output": batch_out},
            "examples": examples,
        }

    def env_reset_fn(env, sample, seed=None):
        env.reset(options={**sample, "mode": "train"}, seed=seed)

    def on_step(
        env, state, action_probs, reward, observation, terminated, truncated, info
    ):
        reward = TensorDict(
            {"pixel_wise": reward, "binary": env.get_source_env_reward()}
        )
        return (
            env,
            state,
            action_probs,
            reward,
            observation,
            terminated,
            truncated,
            info,
        )

    stepper = EnvStepper(
        actor=actor,
        device=device,
        env=env,
        sampler=simple_sampler(),
        env_sample_reset=env_reset_fn,
        max_steps=3,
        amp_enabled=False,
        on_step=on_step,
    )

    # Reset env
    batch_in = torch.randint(0, color_values, (1, grid_size, grid_size))
    batch_out = torch.randint(0, color_values, (1, grid_size, grid_size))
    examples = torch.randint(0, color_values, (1, 4, 2, grid_size, grid_size))
    env.reset(
        options={
            "batch": {"input": batch_in, "output": batch_out},
            "examples": examples,
            "mode": "train",
        }
    )

    # Create episode stream and collect all steps
    episode_steps = list(stepper.create_episode_stream())

    # Verify we got at least one step
    assert len(episode_steps) >= 1

    # Check the structure of each tuple (done, step_number, step_data)
    for i, (done, step_number, step_data) in enumerate(episode_steps):
        assert isinstance(done, bool)
        assert isinstance(step_number, int)
        assert step_number == i
        assert isinstance(step_data, TensorDict)

        # Check step_data structure
        for key in [
            "state",
            "actions",
            "reward",
            "next_state",
            "terminated",
            "truncated",
        ]:
            assert key in step_data

        assert isinstance(step_data["reward"], TensorDict)


def test_env_stepper_step_stream_from_samples(device):
    """Test the step stream from samples functionality."""
    grid_size, color_values = 8, 5

    actor = ArcActorNetwork(
        grid_size=grid_size,
        color_values=color_values,
        config={"device": device, "amp_enabled": False, "use_checkpointing": False},
    )
    actor.eval()

    stepper = PixelWiseArcStepper(
        max_steps_per_epoch=2,
        dataset_path="tests/test_data/",
        grid_size=grid_size,
        color_values=color_values,
        actor=actor,
        config={"device": device, "amp_enabled": False, "use_checkpointing": False},
        n_step_config=NStepMinVMaxConfig(
            n_steps=2,
            pixel_wise_min_v=-10,
            pixel_wise_max_v=1,
            binary_min_v=-10,
            binary_max_v=1,
        ),
    )

    # Create step stream from samples and collect steps
    stream_steps = []
    for done, step_number, step_data in stepper.create_step_stream_from_samples():
        stream_steps.append((done, step_number, step_data))
        if len(stream_steps) >= 4:  # Limit to avoid infinite loop
            break

    # Verify we got at least some steps
    assert len(stream_steps) >= 2

    # Check the structure of each tuple (done, step_number, step_data)
    for done, step_number, step_data in stream_steps:
        assert isinstance(done, bool)
        assert isinstance(step_number, int)
        assert isinstance(step_data, TensorDict)

        # Check step_data structure
        for key in [
            "state",
            "actions",
            "reward",
            "next_state",
            "terminated",
            "truncated",
        ]:
            assert key in step_data

        assert isinstance(step_data["state"], TensorDict)
        assert isinstance(step_data["next_state"], TensorDict)
        assert isinstance(step_data["actions"], TensorDict)
        assert isinstance(step_data["reward"], TensorDict)
        # When n_steps>1 and wrapper supports n-step, expect n_reward in reward dict
        assert set(RewardKeys.get_n_step_keys()).issubset(step_data["reward"].keys())
