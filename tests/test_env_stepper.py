import os

import torch
from tensordict import TensorDict
from torch.utils.data import DataLoader

from rlarcworld.agent.actor import ArcActorNetwork
from rlarcworld.environments.arc_batch_grid_env import ArcBatchGridEnv
from rlarcworld.environments.wrappers.rewards import PixelAwareRewardWrapper
from rlarcworld.environments.reward_processor import RewardProcessor
from rlarcworld.environments.stepper import EnvStepper
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

    stepper = EnvStepper(actor=actor, device=device, amp_enabled=False, on_step=None)

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

    # Use on_step callback to process rewards
    on_step = RewardProcessor(n_steps=1, device=device, v_min_n_step=0, v_max_n_step=1)
    stepper = EnvStepper(
        actor=actor,
        device=device,
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

    on_step = RewardProcessor(n_steps=3, device=device, v_min_n_step=0, v_max_n_step=1)
    stepper = EnvStepper(
        actor=actor,
        device=device,
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

    # Take multiple steps and validate structure each time
    steps = []
    for _ in range(3):
        step = stepper.step(env)
        steps.append(step)

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


def test_env_stepper_step_stream_unbounded(device):
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

    on_step = RewardProcessor(n_steps=2, device=device, v_min_n_step=0, v_max_n_step=1)
    stepper = EnvStepper(
        actor=actor,
        device=device,
        amp_enabled=False,
        on_step=on_step.get_on_step_rewards(),
    )

    # Dataset
    batch_in = torch.randint(0, color_values, (grid_size, grid_size))
    batch_out = torch.randint(0, color_values, (grid_size, grid_size))
    examples = torch.randint(0, color_values, (4, 2, grid_size, grid_size))
    sample = {"task": {"input": batch_in, "output": batch_out}, "examples": examples}
    samples = DataLoader([sample], batch_size=1)

    # Unbounded stream, manually limit to N iterations
    N = 5
    results = []
    for i, (step_num, step_data) in enumerate(
        stepper.create_step_stream(env=env, samples=samples, max_steps=-1, seed=0)
    ):
        results.append((step_num, step_data))
        if i + 1 >= N:
            break

    assert len(results) == N
    for i, (step_num, step) in enumerate(results):
        assert step_num == i
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
        assert isinstance(step["reward"], TensorDict)


def test_env_stepper_step_stream(device):
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

    on_step = RewardProcessor(n_steps=2, device=device, v_min_n_step=0, v_max_n_step=1)
    stepper = EnvStepper(
        actor=actor,
        device=device,
        amp_enabled=False,
        on_step=on_step.get_on_step_rewards(),
    )

    # Build a tiny dataset and DataLoader yielding the expected structure
    # Provide unbatched tensors per sample; DataLoader will add the batch dimension
    batch_in = torch.randint(0, color_values, (grid_size, grid_size))
    batch_out = torch.randint(0, color_values, (grid_size, grid_size))
    examples = torch.randint(0, color_values, (4, 2, grid_size, grid_size))
    sample = {"task": {"input": batch_in, "output": batch_out}, "examples": examples}
    samples = DataLoader([sample], batch_size=1)

    # Iterate the step stream for a few steps
    max_steps = 3
    stream = stepper.create_step_stream(
        env=env, samples=samples, max_steps=max_steps, seed=0
    )
    results = []
    for step_num, step_data in stream:
        results.append((step_num, step_data))

    # Validate results length and structure
    assert len(results) == max_steps
    for i, (step_num, step) in enumerate(results):
        assert step_num == i
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
        assert isinstance(step["state"], TensorDict)
        assert isinstance(step["next_state"], TensorDict)
        assert isinstance(step["actions"], TensorDict)
        assert isinstance(step["reward"], TensorDict)
        # When n_steps>1 and wrapper supports n-step, expect n_reward in reward dict
        assert "n_reward" in step["reward"]
