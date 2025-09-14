"""Trainer wrapper around D4PGCore with pluggable components.

This module contains a lightweight trainer that orchestrates sampling,
updates, target network sync, and optional callbacks for logging,
validation, and checkpointing.
"""

from collections.abc import Callable, Iterator
from typing import Any, Protocol

import torch
from tensordict import TensorDict

from .d4pg import D4PGCore


class ValidateFn(Protocol):
    """Validation function signature."""

    def __call__(self, epoch: int, step: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return validation metrics for the given epoch and step."""
        ...


class CheckpointPayloadFn(Protocol):
    """Checkpoint payload function signature."""

    def __call__(
        self, epoch: int, step: int, actor: torch.nn.Module, critic: torch.nn.Module
    ) -> dict[str, Any]:
        """Build a checkpoint payload from trainer state and models."""
        ...


class ReplayBufferProto(Protocol):
    """A minimal protocol for prioritized or uniform replay buffers.

    The implementing buffer may be prioritized or uniform and should expose
    extend/sample/update_priority and an optional ``storage`` attribute.
    """

    def extend(self, batch: torch.Tensor) -> Any:
        """Append a batch of transitions and return indices/handles."""
        ...

    def sample(self, batch_size: int) -> TensorDict:
        """Sample a batch of transitions as a TensorDict."""
        ...

    def update_priority(self, indices: Any, priorities: torch.Tensor) -> None:
        """Update sampling priorities for the given indices."""
        ...

    @property
    def storage(self) -> Any:  # noqa: D401 - property described in class doc
        """Underlying storage implementation (buffer-specific)."""
        ...


# Callback/hook signatures (epoch, step, loss_actor, loss_critic, batch)
OnStepFn = Callable[
    [int, int, torch.Tensor | None, torch.Tensor | None, TensorDict | None], None
]
OnValidateFn = Callable[[int, int, torch.Tensor, torch.Tensor, TensorDict], None]
OnCheckpointFn = Callable[[dict[str, Any]], None]


class D4PGTrainer:
    """Decoupled trainer/wrapper for D4PGCore.

    This wrapper is designed to be environment- and logging-agnostic. It
    relies on dependency injection through callables:
      - step_stream_factory: returns an iterator that yields per-step
        TensorDicts with keys expected by the core ('state', 'actions',
        'reward', 'next_state', 'terminated').
      - replay_buffer (optional): any object satisfying ReplayBufferProto.
      - on_step/on_validate/on_checkpoint (optional): callbacks for
        logging/traceability/checkpointing.

    The wrapper only orchestrates sampling, training steps, target updates,
    and calling hooks.
    """

    def __init__(
        self,
        *,
        core: D4PGCore,
        batch_size: int,
        step_stream_factory: Callable[[int], Iterator[TensorDict]],
        replay_buffer: ReplayBufferProto | None = None,
        warmup_buffer_ratio: float = 0.2,
        target_update_frequency: int = 10,
        on_step: OnStepFn | None = None,
        on_validate: OnValidateFn | None = None,
        on_checkpoint: OnCheckpointFn | None = None,
        device: str | None = None,
    ) -> None:
        """Initialize the trainer with the provided components.

        Args:
            core (D4PGCore): The core D4PG agent to train.
            batch_size (int): The batch size for training.
            step_stream_factory (Optional[Callable[[int], Iterator[TensorDict]]]):
                Factory returning an iterator that yields per-step TensorDicts
                with keys expected by the core ('state', 'actions', 'reward',
                'next_state', 'terminated').
            replay_buffer (Optional[ReplayBufferProto]): A replay buffer to
                store transitions for experience replay.
            warmup_buffer_ratio (float): Ratio of buffer to fill before training.
            target_update_frequency (int): Frequency of target updates.
            on_step (Optional[OnStepFn]): Callback for logging/tracing.
            on_validate (Optional[OnValidateFn]): Callback for validation.
            on_checkpoint (Optional[OnCheckpointFn]): Callback for checkpointing.
            device (Optional[str]): The device to use for training.
        """
        self.core = core
        self.step_stream_factory = step_stream_factory
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.warmup_buffer_ratio = warmup_buffer_ratio
        self.target_update_frequency = target_update_frequency
        self.on_step = on_step
        self.on_validate = on_validate
        self.on_checkpoint = on_checkpoint
        self.device = (
            device or getattr(core, "device", None) or str(torch.device("cpu"))
        )

        if self.replay_buffer is None and self.batch_size is None:
            raise ValueError("batch_size must be provided when replay_buffer is None")

    # ---------------- Orchestration helpers ----------------
    def _maybe_store_and_sample(self, step_state: TensorDict) -> TensorDict | None:
        """Store the transition in the replay buffer and return a batch if ready.

        Args:
            step_state: Dictionary containing the current step's data

        Returns:
            Optional batch of transitions if buffer is warmed up, else None
        """
        # Single-step batch mode (no replay): keep (B, ...) shape without extra time dim
        if self.replay_buffer is None:
            return step_state.to(self.device)

        # Replay buffer mode (append transition and sample)
        batch_size = step_state.batch_size[0]

        # Use max priority from buffer if available, else 1.0
        maxp = getattr(
            getattr(self.replay_buffer, "sampler", None), "_max_priority", 1.0
        )
        # Normalize to python float for torch.full fill_value
        try:
            if isinstance(maxp, torch.Tensor):
                # some samplers keep a tensor buffer, take max scalar
                maxp = maxp.max().item()
            else:
                maxp = float(maxp)
        except Exception:
            maxp = 1.0
        priority = torch.full((batch_size,), maxp, device=self.device)

        # Add to replay buffer without introducing an extra leading time dimension
        indices = self.replay_buffer.extend(step_state)

        # Update priorities if using prioritized replay
        try:
            self.replay_buffer.update_priority(indices, priority)
        except Exception:
            # allow non-PER buffers
            pass

        # Check if we've collected enough samples to start training
        if hasattr(self.replay_buffer, "storage") and hasattr(
            self.replay_buffer.storage, "max_size"
        ):
            storage_len = len(self.replay_buffer.storage)
            storage_cap = self.replay_buffer.storage.max_size
            if storage_len > storage_cap * self.warmup_buffer_ratio:
                return self.replay_buffer.sample(self.batch_size).to(self.device)

        return None

    @staticmethod
    def _filter_completed_state(batch: TensorDict) -> TensorDict:
        mask = batch["state"]["terminated"] == 0
        selected = mask.nonzero(as_tuple=True)[0].cpu()
        return batch[selected]

    # ---------------- Public API ----------------
    def train(
        self,
        *,
        epochs: int = 1,
        max_steps_per_epoch: int = -1,
        validate_every_steps: int = -1,
        validate_fn: None | ValidateFn = None,
        checkpoint_every_steps: int = -1,
        checkpoint_payload_fn: None | CheckpointPayloadFn = None,
    ) -> None:
        """Run a training loop using the provided step stream and hooks.

        Args:
            epochs (int): Number of epochs to train for.
            max_steps_per_epoch (int): Max steps per epoch (-1 for no limit).
            validate_every_steps (int): Run validation every N steps (-1 to
                disable).
            validate_fn (ValidateFn): Validation function receiving (epoch, step) and
                returning a tuple of (actor_metric, critic_metric) tensors.
            checkpoint_every_steps (int): Save checkpoint every N steps (-1 to
                disable).
            checkpoint_payload_fn (CheckpointPayloadFn): Function that builds a
             checkpoint payload dict from (epoch, step, actor_module, critic_module).
        """
        for epoch in range(epochs):
            step_stream = self.step_stream_factory(epoch)

            loss_actor, loss_critic = torch.tensor(0.0), torch.tensor(0.0)

            for step_idx, step_state in enumerate(step_stream):
                iteration = epoch * max_steps_per_epoch + step_idx

                # Store transition and maybe sample batch
                batch = self._maybe_store_and_sample(step_state)

                # Update networks if we have a batch
                if batch is not None:
                    loss_actor, loss_critic = self.core.compute_loss(
                        batch,
                        training=True,
                        update_actor=True,
                        update_critic=True,
                        compute_td_error=True,  # Enable TD error computation for PER
                    )

                # Call step callback
                if self.on_step is not None and batch is not None:
                    la = loss_actor if isinstance(loss_actor, torch.Tensor) else None
                    lc = loss_critic if isinstance(loss_critic, torch.Tensor) else None
                    self.on_step(epoch, step_idx, la, lc, batch)

                # Update target networks periodically
                if batch is not None and iteration % self.target_update_frequency == 0:
                    self.core.update_target_networks()

                # Validation
                if (
                    validate_every_steps > 0
                    and iteration % validate_every_steps == 0
                    and validate_fn is not None
                ):
                    with torch.no_grad():
                        v_actor, v_critic = validate_fn(epoch, step_idx)
                        ctx = TensorDict({}, batch_size=[])
                    if self.on_validate is not None:
                        self.on_validate(epoch, step_idx, v_actor, v_critic, ctx)

                # Checkpointing
                if (
                    checkpoint_every_steps > 0
                    and iteration % checkpoint_every_steps == 0
                ):
                    if checkpoint_payload_fn is not None:
                        payload = checkpoint_payload_fn(
                            epoch, step_idx, loss_actor, loss_critic
                        )
                    else:
                        payload = {
                            "epoch": epoch,
                            "step": step_idx,
                            "loss_actor": (
                                loss_actor.item() if loss_actor is not None else None
                            ),
                            "loss_critic": (
                                loss_critic.item() if loss_critic is not None else None
                            ),
                        }
                    if self.on_checkpoint is not None:
                        self.on_checkpoint(payload)

                # Early termination
                if max_steps_per_epoch > 0 and step_idx >= max_steps_per_epoch - 1:
                    break
