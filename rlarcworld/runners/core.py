"""Trainer wrapper around a Core algorithm with pluggable components.

This module contains a lightweight trainer that orchestrates sampling,
updates, target network sync, and optional callbacks for logging,
validation, and checkpointing.
"""

import logging
from collections import OrderedDict
from collections.abc import Callable, Iterator
from typing import Any, Protocol

import torch
from tensordict import TensorDict

logger = logging.getLogger(__name__)


class CoreProto(Protocol):
    """Minimal protocol a learning core must implement to work with the trainer."""

    actor: torch.nn.Module
    critic: torch.nn.Module

    def compute_loss(
        self,
        batch: TensorDict,
        *,
        training: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the loss for the actor and critic networks."""
        ...

    def update_target_networks(self) -> None:
        """Update the target networks."""
        ...

    def hyperparameters(self) -> dict:
        """Returns the hyperparameters of the algorithm."""
        ...


class OnStepContext:
    """Context object containing all information about a training step."""

    def __init__(
        self,
        epoch: int,
        step_idx: int,
        iteration: int,
        batch: TensorDict | None,
        loss_actor: torch.Tensor | None,
        loss_critic: torch.Tensor | None,
        core: CoreProto,
        hyperparameters: dict,
    ):
        """Initialize step context.

        Args:
            epoch: Current epoch number
            step_idx: Current step index within epoch
            iteration: Global iteration number
            batch: Training batch (None if no batch available)
            loss_actor: Actor loss (None if no training occurred)
            loss_critic: Critic loss (None if no training occurred)
            core: Algorithm core containing networks and methods
            hyperparameters: Hyperparameters of the algorithm
        """
        self.epoch = epoch
        self.step_idx = step_idx
        self.iteration = iteration
        self.batch = batch
        self.loss_actor = loss_actor
        self.loss_critic = loss_critic
        self.core = core
        self.hyperparameters = hyperparameters


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


class Trainer:
    """Decoupled trainer/wrapper for CoreProto                      .

    This wrapper is designed to be environment- and logging-agnostic. It
    relies on dependency injection through callables:
      - step_stream_factory: returns an iterator that yields per-step
        TensorDicts with keys expected by the core ('state', 'actions',
        'reward', 'next_state', 'terminated').
      - replay_buffer (optional): any object satisfying ReplayBufferProto.
      - on_step: callback for logging/traceability/validation/checkpointing.

    The wrapper only orchestrates sampling, training steps, target updates,
    and calling hooks.
    """

    def __init__(
        self,
        *,
        core: CoreProto,
        batch_size: int,
        step_stream_factory: Callable[[], Iterator[tuple[bool, int, TensorDict]]],
        replay_buffer: ReplayBufferProto | None = None,
        warmup_buffer_ratio: float = 0.2,
        on_step: OrderedDict[str, Callable[[OnStepContext], None]] | None = None,
        on_train_exception: OrderedDict[str, Callable[[Exception, OnStepContext], None]]
        | None = None,
        device: str | None = None,
    ) -> None:
        """Initialize the trainer with the provided components.

        Args:
            core (CoreProto): The algorithm core to train.
            batch_size (int): The batch size for training.
            step_stream_factory (Callable[[], Iterator[tuple[bool, int, TensorDict]]]):
                Factory returning an iterator that yields per-step TensorDicts
                with keys expected by the core ('state', 'actions', 'reward',
                'next_state', 'terminated').
            replay_buffer (Optional[ReplayBufferProto]): A replay buffer to
                store transitions for experience replay.
            warmup_buffer_ratio (float): Ratio of buffer to fill before training.
            on_step (Optional[OrderedDict[str, Callable[[OnStepContext], None]]]):
                Callback for logging/tracing/validation/checkpointing.
            on_train_exception (Optional[OrderedDict[str, Callable[[Exception, OnStepContext], None]]]):
                Callback for training exceptions.
            device (Optional[str]): The device to use for training.
        """  # noqa: E501
        self.core = core
        self.step_stream_factory = step_stream_factory
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.warmup_buffer_ratio = warmup_buffer_ratio
        self.on_step = on_step
        self.on_train_exception = on_train_exception
        self.device = (
            device or getattr(core, "device", None) or str(torch.device("cpu"))
        )

        if self.replay_buffer is None and self.batch_size is None:
            raise ValueError("batch_size must be provided when replay_buffer is None")

    @property
    def hyperparameters(self):
        """Returns the hyperparameters of the algorithm."""
        return {
            "batch_size": self.batch_size,
            "replay_buffer": {
                "alpha": self.replay_buffer.sampler.alpha
                if self.replay_buffer is not None
                and hasattr(self.replay_buffer, "sampler")
                and hasattr(self.replay_buffer.sampler, "alpha")
                else None,
                "beta": self.replay_buffer.sampler.beta
                if self.replay_buffer is not None
                and hasattr(self.replay_buffer, "sampler")
                and hasattr(self.replay_buffer.sampler, "beta")
                else None,
                "warmup_buffer_ratio": self.warmup_buffer_ratio,
            },
            "device": self.device,
            "core": self.core.hyperparameters,
        }

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
    ) -> None:
        """Run a training loop using the provided step stream and hooks.

        Args:
            epochs (int): Number of epochs to train for.
            max_steps_per_epoch (int): Max steps per epoch (-1 for no limit).
        """
        try:
            iteration = 0
            for epoch in range(epochs):
                loss_actor, loss_critic = torch.tensor(0.0), torch.tensor(0.0)

                for done, step_idx, step_state in self.step_stream_factory():
                    iteration += 1

                    # Store transition and maybe sample batch
                    batch = self._maybe_store_and_sample(step_state)

                    # Update networks if we have a batch
                    if batch is not None:
                        loss_actor, loss_critic = self.core.compute_loss(
                            batch,
                            training=True,
                        )
                    else:
                        loss_actor, loss_critic = None, None

                    # Create step context
                    context = OnStepContext(
                        epoch=epoch,
                        step_idx=step_idx,
                        iteration=iteration,
                        batch=batch,
                        loss_actor=loss_actor,
                        loss_critic=loss_critic,
                        core=self.core,
                        hyperparameters=self.hyperparameters,
                    )

                    # Call step callback
                    if self.on_step and batch is not None:
                        for callback_name, callback_step in self.on_step.items():
                            logger.debug(f"Calling on_step {callback_name}")
                            callback_step(context)

                    # Early termination
                    if max_steps_per_epoch > 0 and step_idx >= max_steps_per_epoch - 1:
                        break
        except Exception as e:
            context = OnStepContext(
                epoch=epoch,
                step_idx=step_idx,
                iteration=iteration,
                batch=batch,
                loss_actor=loss_actor,
                loss_critic=loss_critic,
                core=self.core,
                hyperparameters=self.hyperparameters,
            )
            if self.on_train_exception is not None:
                for (
                    callback_name,
                    callback_exception,
                ) in self.on_train_exception.items():
                    logger.debug(f"Calling on_train_exception {callback_name}")
                    callback_exception(e, context)
            raise e
