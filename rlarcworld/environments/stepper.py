"""Environment Stepper for RL training loops.

This module provides a class to handle environment interactions in a clean way,
separating environment stepping logic from the training loop.
"""

from collections.abc import Callable, Generator

import torch
from gymnasium import Env
from tensordict import TensorDict
from torch.utils.data import DataLoader

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
        amp_enabled: bool = True,
        on_step: OnStepFn | None = None,
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

    def create_step_stream(
        self,
        env,
        samples: DataLoader,
        max_steps: int = -1,
        seed: int | None = None,
        **kwargs,
    ) -> Generator[tuple[int, TensorDict], None, None]:
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

            # Ensure Python bool type for loop control
            term = bool(step_data["terminated"].item())
            trunc = bool(step_data["truncated"].item())
            done = term or trunc
            step_number += 1
