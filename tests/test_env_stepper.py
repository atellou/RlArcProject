import os

import torch
from tensordict import TensorDict

from rlarcworld.agent.actor import ArcActorNetwork
from rlarcworld.environments.arc_batch_grid_env import ArcBatchGridEnv
from rlarcworld.environments.reward_processor import RewardProcessor
from rlarcworld.environments.stepper import EnvStepper
from rlarcworld.environments.wrappers.pixel_aware_reward import PixelAwareRewardWrapper
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
        env.reset(options=sample, seed=seed)

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
        env.reset(options=sample, seed=seed)

    # Use on_step callback to process rewards
    on_step = RewardProcessor(n_steps=1, device=device, v_min_n_step=0, v_max_n_step=1)
    stepper = EnvStepper(
        actor=actor,
        device=device,
        env=env,
        sampler=simple_sampler(),
        env_sample_reset=env_reset_fn,
        amp_enabled=False,
        on_step=on_step.get_on_step_rewards(),
    )

    # Reset env with a trivial batch to satisfy interface
    batch_in = torch.randint(0, color_values, (1, grid_size, grid_size))
    batch_out = torch.randint(0, color_values, (1, grid_size, grid_size))
    examples = torch.randint(0, color_values, (1, 10, 2, grid_size, grid_size))
    env.reset(
        options={
            "batch": {"input": batch_in, "output": batch_out},
            "examples": examples,
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
        env.reset(options=sample, seed=seed)

    on_step = RewardProcessor(n_steps=3, device=device, v_min_n_step=0, v_max_n_step=1)
    stepper = EnvStepper(
        actor=actor,
        device=device,
        env=env,
        sampler=simple_sampler(),
        env_sample_reset=env_reset_fn,
        max_steps=3,
        amp_enabled=False,
        on_step=on_step.get_on_step_rewards(),
    )

    # Reset env
    batch_in = torch.randint(0, color_values, (1, grid_size, grid_size))
    batch_out = torch.randint(0, color_values, (1, grid_size, grid_size))
    examples = torch.randint(0, color_values, (1, 10, 2, grid_size, grid_size))
    env.reset(
        options={
            "batch": {"input": batch_in, "output": batch_out},
            "examples": examples,
        }
    )

    # Take multiple steps
    steps = []
    for _ in range(3):
        step = stepper.step(env)
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
        assert "n_reward" in step["reward"]


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
        env.reset(options=sample, seed=seed)

    on_step = RewardProcessor(n_steps=2, device=device, v_min_n_step=0, v_max_n_step=1)
    stepper = EnvStepper(
        actor=actor,
        device=device,
        env=env,
        sampler=simple_sampler(),
        env_sample_reset=env_reset_fn,
        max_steps=3,
        amp_enabled=False,
        on_step=on_step.get_on_step_rewards(),
    )

    # Reset env
    batch_in = torch.randint(0, color_values, (1, grid_size, grid_size))
    batch_out = torch.randint(0, color_values, (1, grid_size, grid_size))
    examples = torch.randint(0, color_values, (1, 4, 2, grid_size, grid_size))
    env.reset(
        options={
            "batch": {"input": batch_in, "output": batch_out},
            "examples": examples,
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
    # Wrap env with n_steps=2 to exercise n-step paths in stream
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

    # Create a sampler that yields multiple samples
    def multi_sampler():
        for _ in range(2):  # Two episodes
            batch_in = torch.randint(0, color_values, (1, grid_size, grid_size))
            batch_out = torch.randint(0, color_values, (1, grid_size, grid_size))
            examples = torch.randint(0, color_values, (1, 4, 2, grid_size, grid_size))
            yield {
                "batch": {"input": batch_in, "output": batch_out},
                "examples": examples,
            }

    def env_reset_fn(env, sample, seed=None):
        env.reset(options=sample, seed=seed)

    on_step = RewardProcessor(n_steps=2, device=device, v_min_n_step=0, v_max_n_step=1)
    stepper = EnvStepper(
        actor=actor,
        device=device,
        env=env,
        sampler=multi_sampler(),
        env_sample_reset=env_reset_fn,
        max_steps=2,
        amp_enabled=False,
        on_step=on_step.get_on_step_rewards(),
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
        assert "n_reward" in step_data["reward"]
