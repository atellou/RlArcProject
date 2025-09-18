"""Environment Stepper for RL training loops.

This module provides a class to handle environment interactions in a clean way,
separating environment stepping logic from the training loop.
"""

from collections.abc import Callable, Generator, Iterator
from typing import Any

import torch
from gymnasium import Env
from tensordict import TensorDict

from rlarcworld.agent.actor import ArcActorNetwork
from rlarcworld.utils.device import AmpScaler

# Step callback type: receives
# (env, state, action_probs, reward, observation, terminated, truncated, info)
# and returns the same tuple possibly modified
StepArgs = tuple[
    Env,
    TensorDict,
    dict[str, torch.Tensor],
    torch.Tensor,
    TensorDict,
    bool,
    bool,
    dict,
]
OnStepFn = Callable[
    [
        Env,
        TensorDict,
        dict[str, torch.Tensor],
        torch.Tensor,
        TensorDict,
        bool,
        bool,
        dict,
    ],
    StepArgs,
]

# Object that the sampler iterator returns
ResetSample = Any


class EnvStepper:
    """Handles environment stepping and episode simulation for RL training.

    This class encapsulates the environment interaction logic, including:
    - Single-step execution
    - Multi-step simulation with n-step returns
    - Episode simulation with proper reset and termination handling
    """

    def __init__(
        self,
        actor: ArcActorNetwork,
        device: str | torch.device,
        env: Env,
        sampler: Iterator[ResetSample],
        max_steps: int = -1,
        env_sample_reset: Callable[[Env, ResetSample, int | None], Env] | None = None,
        amp_enabled: bool = True,
        on_step: OnStepFn | None = None,
    ):
        """Initialize the environment stepper.

        Args:
            actor: The actor network used to select actions
            device: Device to use for tensors
            env (Env): The environment to use for training.
            sampler (Iterator[ResetSample]): Iterator yielding batches of objects
                to reset the environment with, if env_sample_reset is provided.
            max_steps (int): Maximum number of steps per episode (-1 for no limit)
            env_sample_reset (Optional[Callable[[Env, ResetSample, int | None], Env]]):
                Callable to reset the environment with a batch of objects.
            amp_enabled: Whether to use automatic mixed precision
            on_step: Optional callback function to be called after each step
        """
        self.actor = actor
        self.device = device
        self.env = env
        self.max_steps = max_steps
        self.sampler = sampler
        self.env_sample_reset = env_sample_reset
        self.amp_enabled = amp_enabled
        self.on_step = on_step
        self.loaded_checkpoint = False
        # Shared AMP context manager (no-op on CPU or when disabled)
        self.amp_cm = AmpScaler(amp_enabled=self.amp_enabled, device=self.device)

    def step(self, env: Env, state: TensorDict | None = None) -> TensorDict:
        """Take a single step in the environment.

        Args:
            env: The environment to step in
            state: Current state (if None, will get from env)

        Returns:
            Tuple of:
            - state: Current state
            - action_probs: Action probabilities from actor
            - reward: Rewards from environment
            - next_state: Next state
            - done: Whether episode is done
            - truncated: Whether episode was truncated
        """
        with torch.no_grad():
            if state is None:
                state = env.get_state(unsqueeze=1).to(self.device)

            # Get action from actor (AMP context is a no-op if disabled/CPU)
            with self.amp_cm:
                action_probs = self.actor(state)

            # Convert to discrete actions for environment (ensure 1D per batch)
            actions = TensorDict(
                {
                    key: torch.argmax(x, dim=-1).to(self.device)
                    for key, x in action_probs.items()
                }
            ).to(self.device)

            # Step environment
            observation, reward, terminated, truncated, info = env.step(actions)

            if self.on_step:
                (
                    env,
                    state,
                    action_probs,
                    reward,
                    observation,
                    terminated,
                    truncated,
                    info,
                ) = self.on_step(
                    env,
                    state,
                    action_probs,
                    reward,
                    observation,
                    terminated,
                    truncated,
                    info,
                )

            # Get next state
            next_state = env.get_state(unsqueeze=1)

            # Ensure reward is a TensorDict
            if isinstance(reward, TensorDict):
                reward_td = reward.auto_batch_size_()
            elif isinstance(reward, dict):
                reward_td = TensorDict(reward, batch_size=[]).auto_batch_size_()
            else:
                reward_td = TensorDict(
                    {"value": reward}, batch_size=[]
                ).auto_batch_size_()

            # Format output
            return TensorDict(
                {
                    "state": state.auto_batch_size_(),
                    "reward": reward_td,
                    "actions": action_probs.auto_batch_size_(),
                    "next_state": next_state.auto_batch_size_(),
                    "terminated": torch.tensor([terminated], device=self.device),
                    "truncated": torch.tensor([truncated], device=self.device),
                },
                batch_size=[],
            ).to(self.device)

    def create_episode_stream(
        self,
    ) -> Generator[tuple[bool, int, TensorDict], None, None]:
        """Create a generator that yields (done, step_num, step_data) tuples.

        Yields:
            Tuples of (done, step_number, step_data) where step_data is a TensorDict
            with keys: state, reward, actions, next_state, terminated, truncated
        """
        step_number = 0
        done = False

        while not done and (self.max_steps <= -1 or step_number < self.max_steps):
            step_data = self.step(self.env)
            # Ensure Python bool type for loop control
            term = bool(step_data["terminated"].item())
            trunc = bool(step_data["truncated"].item())
            done = term or trunc
            yield (
                done,
                step_number,
                step_data,
            )
            step_number += 1

    def _reset_env(self, sample: Any, seed: int | None = None):
        """Reset the environment with a sample.

        Args:
            sample: Sample to reset the environment with
            seed: Random seed for reproducibility
        """
        if self.env_sample_reset is not None:
            self.env_sample_reset(self.env, sample, seed)
        else:
            self.env.reset(seed=seed)

    def create_step_stream_from_samples(
        self,
        seed: int | None = None,
    ) -> Generator[tuple[bool, int, TensorDict], None, None]:
        """Create a generator that yields step data from samples.

        Args:
            seed: Random seed for reproducibility

        Yields:
            Tuples of (done, step_number, step_data) where step_data is a TensorDict
            with keys: state, reward, actions, next_state, terminated, truncated
        """
        for sample in self.sampler:
            self._reset_env(sample, seed=seed)
            for done, step_number, step_data in self.create_episode_stream():
                yield done, step_number, step_data
