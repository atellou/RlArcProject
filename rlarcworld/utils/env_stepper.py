"""
Environment Stepper for RL training loops.

This module provides a class to handle environment interactions in a clean way,
separating environment stepping logic from the training loop.
"""

from typing import Callable, Dict, Generator, Optional, Tuple, Union

import torch
from tensordict import TensorDict
from torch.utils.data import DataLoader

from gymnasium import Env

from rlarcworld.agent.actor import ArcActorNetwork

StepArgs = [
    Env,
    TensorDict,
    Dict[str, torch.Tensor],
    torch.Tensor,
    TensorDict,
    bool,
    bool,
    Dict,
]
OnStepFn = Callable[
    StepArgs,
    StepArgs,
]


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
        device: Union[str, torch.device],
        amp_enabled: bool = True,
        on_step: Optional[OnStepFn] = None,
    ):
        """Initialize the environment stepper.

        Args:
            actor: The actor network used to select actions
            device: Device to use for tensors
            amp_enabled: Whether to use automatic mixed precision
            on_step: Optional callback function to be called after each step
        """
        self.actor = actor
        self.device = device
        self.amp_enabled = amp_enabled
        self.on_step = on_step
        self.loaded_checkpoint = False

    def step(
        self, env, state: Optional[TensorDict] = None
    ) -> Tuple[TensorDict, TensorDict, bool, bool, Dict]:
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

            # Get action from actor
            with torch.amp.autocast(device_type="cuda" if self.amp_enabled else None):
                action_probs = self.actor(state)

            # Convert to discrete actions for environment
            actions = {k: torch.argmax(p, dim=-1) for k, p in action_probs.items()}

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

            # Format output
            step_data = {
                "state": state.auto_batch_size_(),
                "reward": TensorDict(reward, batch_size=[]).auto_batch_size_(),
                "actions": action_probs.auto_batch_size_(),
                "next_state": next_state.auto_batch_size_(),
                "terminated": torch.tensor([terminated], device=self.device),
                "truncated": torch.tensor([truncated], device=self.device),
            }

            return TensorDict(step_data, batch_size=[]).to(self.device)

    def create_step_stream(
        self,
        env,
        samples: DataLoader,
        max_steps: int = -1,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Generator[Tuple[int, TensorDict], None, None]:
        """Create a generator that yields (step_num, step_data) tuples.

        Args:
            env: The environment to interact with
            samples: DataLoader yielding batches of tasks/examples
            max_steps: Maximum number of steps per episode (-1 for no limit)
            seed: Random seed for environment
            **kwargs: Additional arguments to pass to env.reset()

        Yields:
            Tuples of (step_number, step_data) where step_data is a TensorDict
            with keys: state, reward, actions, next_state, terminated, truncated
        """
        # Reset environment with first batch
        if not self.loaded_checkpoint:
            first_batch = next(iter(samples))
            kwargs.update(
                {"batch": first_batch["task"], "examples": first_batch["examples"]}
            )
            env.reset(options=kwargs, seed=seed)
            self.loaded_checkpoint = True

        # Run episode
        step_number = 0
        done = False

        while not done and (max_steps <= -1 or step_number < max_steps):
            step_data = self.step(env)
            yield step_number, step_data

            done = step_data["terminated"] or step_data["truncated"]
            step_number += 1
