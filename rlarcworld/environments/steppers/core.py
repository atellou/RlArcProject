"""Environment Stepper for RL training loops.

This module provides a class to handle environment interactions in a clean way,
separating environment stepping logic from the training loop.
"""

import logging
from collections.abc import Callable, Generator, Iterator
from typing import Any

import torch
from gymnasium import Env
from tensordict import TensorDict

from rlarcworld.models.actor import ViTARCActor
from rlarcworld.utils.checkpointing import add_schema_and_class
from rlarcworld.utils.device import AmpScaler

logger = logging.getLogger(__name__)

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
        actor: ViTARCActor,
        device: str | torch.device,
        env: Env,
        sampler: Iterator[ResetSample] | None = None,
        max_steps: int = -1,
        env_sample_reset: Callable[..., None] | None = None,
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
            env_sample_reset (Optional[Callable[..., None]]):
                Callable to reset the environment with a batch of objects and kwargs.
            amp_enabled: Whether to use automatic mixed precision
            on_step: Optional callback function to be called after each step
        """  # noqa: E501
        self.actor = actor
        self.device = device
        self.env = env
        self.max_steps = max_steps
        self.sampler = sampler
        self.env_sample_reset = env_sample_reset
        self.amp_enabled = amp_enabled
        self.on_step = on_step
        self.loaded_checkpoint = False
        # Shared AMP ctx manager (no-op on CPU or when disabled)
        self.amp_cm = AmpScaler(amp_enabled=self.amp_enabled, device=self.device)
        # Progress tracking for checkpointing
        self._samples_processed: int = 0
        self._episode_step_idx: int = 0

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

            # Get action from actor (AMP ctx is a no-op if disabled/CPU)
            # Force eval mode temporarily to avoid BatchNorm errors with small batches
            prev_training = self.actor.training
            try:
                self.actor.eval()
                with self.amp_cm:
                    action_probs = self.actor(state)
            finally:
                if prev_training:
                    self.actor.train(True)

            # Convert to discrete actions for environment (ensure 1D per batch)
            actions = TensorDict(
                {
                    key: torch.argmax(x, dim=-1).to(self.device)
                    for key, x in action_probs.items()
                }
            ).to(self.device)

            # Step environment
            env_device = getattr(env, "device", None)
            if env_device is not None:
                actions_env = actions.to(env_device)
            else:
                actions_env = actions
            observation, reward, terminated, truncated, info = env.step(actions_env)

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
            next_state = env.get_state(unsqueeze=1).to(self.device)

            # Determine batch size (B) for this transition
            batch_size = None
            try:
                batch_size = int(reward.shape[0])
            except Exception:
                try:
                    batch_size = int(state.batch_size[0])
                except Exception:
                    batch_size = None
            if batch_size is None:
                batch_size = 1

            # Ensure reward is a TensorDict matching critic heads
            if isinstance(reward, TensorDict):
                reward_td = reward.auto_batch_size_()
            elif isinstance(reward, dict):
                reward_td = TensorDict(
                    reward, batch_size=[batch_size]
                ).auto_batch_size_()
            else:
                raise ValueError(
                    "Reward must be a TensorDict or dict. Got: {}".format(type(reward))
                )

            # Build per-element terminated/truncated vectors for batched envs
            terminated_vec = None
            if isinstance(info, TensorDict):
                try:
                    if "terminated" in info.keys():
                        terminated_vec = info["terminated"]
                except Exception:
                    terminated_vec = None
            if terminated_vec is not None:
                terminated_tensor = terminated_vec.to(self.device).float().reshape(-1)
            else:
                terminated_tensor = torch.full(
                    (batch_size,),
                    float(bool(terminated)),
                    device=self.device,
                    dtype=torch.float32,
                )

            truncated_tensor = torch.full(
                (batch_size,),
                float(bool(truncated)),
                device=self.device,
                dtype=torch.float32,
            )

            # Format output
            return TensorDict(
                {
                    "state": state.auto_batch_size_(),
                    "reward": reward_td,
                    "actions": action_probs.auto_batch_size_(),
                    "next_state": next_state.auto_batch_size_(),
                    "terminated": terminated_tensor,
                    "truncated": truncated_tensor,
                },
                batch_size=[batch_size],
            ).to(self.device)

    def create_episode_stream(
        self,
    ) -> Generator[tuple[bool, int, TensorDict], None, None]:
        """Create a generator that yields (done, step_num, step_data) tuples.

        Yields:
            Tuples of (done, step_number, step_data) where step_data is a TensorDict
            with keys: state, reward, actions, next_state, terminated, truncated
        """
        step_number = int(self._episode_step_idx)
        done = False

        while not done and (self.max_steps <= -1 or step_number < self.max_steps):
            step_data = self.step(self.env)
            # Ensure Python bool type for loop control
            term = bool(torch.all(step_data["terminated"]).item())
            trunc = bool(torch.all(step_data["truncated"]).item())
            done = term or trunc
            yield (
                done,
                step_number,
                step_data,
            )
            step_number += 1
            self._episode_step_idx = int(step_number)
        logger.info(f"Episode finished with {step_number} steps")

    def _reset_env(self, sample, **kwargs) -> None:
        """Reset the environment with a sample.

        Args:
            sample: Sample to reset the environment with
            **kwargs: Additional arguments passed to reset
        """
        if self.env_sample_reset is not None:
            self.env_sample_reset(self.env, sample, **kwargs)
        else:
            self.env.reset(**kwargs)

    def create_step_stream_from_samples(
        self, **kwargs
    ) -> Generator[tuple[bool, int, TensorDict], None, None]:
        """Create a generator that yields step data from samples.

        Args:
            **kwargs: Additional arguments passed to reset

        Yields:
            Tuples of (done, step_number, step_data) where step_data is a TensorDict
            with keys: state, reward, actions, next_state, terminated, truncated
        """
        if self.sampler is None:
            raise ValueError("Sampler is not initialized")
        for sample in self.sampler:
            yield from self.stream_on_reset(sample, **kwargs)
            # count one sample processed after finishing its episode
            self._samples_processed += 1

    def stream_on_reset(self, sample, **kwargs):
        """Stream on reset callback."""
        self._episode_step_idx = 0
        self._reset_env(sample, **kwargs)
        for done, step_number, step_data in self.create_episode_stream():
            yield done, step_number, step_data

    # ---------------- Checkpoint API ----------------
    def checkpoint(self) -> dict:
        """Return a minimal checkpoint for stepper progress.

        Note: We intentionally do not serialize the sampler iterable itself.
        Consumers should restore sampler deterministically (e.g., via RNG state)
        and use ``_samples_processed`` to advance if needed.
        """
        payload = {
            "state": {
                "max_steps": self.max_steps,
                "samples_processed": self._samples_processed,
                "episode_step_idx": self._episode_step_idx,
                "amp_enabled": self.amp_enabled,
                "loaded_checkpoint": self.loaded_checkpoint,
            }
        }
        return add_schema_and_class(payload, self.__class__.__name__)

    def load(self, checkpoint: dict) -> None:
        """Load stepper progress from a checkpoint dict.

        This does not restore the sampler or RNG states.
        """
        if not isinstance(checkpoint, dict):
            raise ValueError("Stepper checkpoint must be a dict")
        state = checkpoint.get("state", {})
        self.max_steps = int(state.get("max_steps", self.max_steps))
        self._samples_processed = int(state.get("samples_processed", 0))
        self._episode_step_idx = int(state.get("episode_step_idx", 0))
        self.amp_enabled = bool(state.get("amp_enabled", self.amp_enabled))
        self.loaded_checkpoint = bool(state.get("loaded_checkpoint", False))
