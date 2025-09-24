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

from rlarcworld.utils.checkpointing import add_schema_and_class

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

    def __len__(self) -> int:
        """Return the number of transitions in the buffer."""
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
        min_train_batch_size: int = 2,
        train_step_stream_factory: Callable[[], Iterator[tuple[bool, int, TensorDict]]],
        validation_step_stream_factory: Callable[
            [], Iterator[tuple[bool, int, TensorDict]]
        ]
        | None = None,
        replay_buffer: ReplayBufferProto | None = None,
        warmup_buffer_ratio: float = 0.2,
        on_train_step: OrderedDict[str, Callable[[OnStepContext], None]] | None = None,
        on_validation_step: OrderedDict[str, Callable[[OnStepContext], None]]
        | None = None,
        on_exception: OrderedDict[str, Callable[[Exception, OnStepContext], None]]
        | None = None,
        device: str | None = None,
    ) -> None:
        """Initialize the trainer with the provided components.

        Args:
            core (CoreProto): The algorithm core to train.
            batch_size (int): The batch size for training.
            min_train_batch_size (int): The minimum batch size for training.
            train_step_stream_factory (Callable[[], Iterator[tuple[bool, int, TensorDict]]]):
                Factory returning an iterator that yields per-step TensorDicts
                with keys expected by the core ('state', 'actions', 'reward',
                'next_state', 'terminated').
            validation_step_stream_factory (Callable[[], Iterator[tuple[bool, int, TensorDict]]]):
                Factory returning an iterator that yields per-step TensorDicts
                with keys expected by the core ('state', 'actions', 'reward',
                'next_state', 'terminated').
            replay_buffer (Optional[ReplayBufferProto]): A replay buffer to
                store transitions for experience replay.
            warmup_buffer_ratio (float): Ratio of buffer to fill before training.
            on_train_step (Optional[OrderedDict[str, Callable[[OnStepContext], None]]]):
                Callback for logging/tracing/validation/checkpointing.
            on_validation_step (Optional[OrderedDict[str, Callable[[OnStepContext], None]]]):
                Callback for logging/tracing/validation/checkpointing.
            on_exception (Optional[OrderedDict[str, Callable[[Exception, OnStepContext], None]]]):
                Callback for training exceptions.
            device (Optional[str]): The device to use for training.
        """  # noqa: E501
        self.core = core
        self.train_step_stream_factory = train_step_stream_factory
        self.validation_step_stream_factory = validation_step_stream_factory
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.min_train_batch_size = int(min_train_batch_size)
        self.warmup_buffer_ratio = warmup_buffer_ratio
        self.on_train_step = on_train_step
        self.on_validation_step = on_validation_step
        self.on_exception = on_exception
        self.device = (
            device or getattr(core, "device", None) or str(torch.device("cpu"))
        )

        # Internal progress tracking for checkpointing
        self._epoch: int = 0
        self._iteration: int = 0

        if self.replay_buffer is None and self.batch_size is None:
            raise ValueError("batch_size must be provided when replay_buffer is None")

    @property
    def hyperparameters(self):
        """Returns the hyperparameters of the algorithm."""
        return {
            "batch_size": self.batch_size,
            "min_train_batch_size": self.min_train_batch_size,
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
        try:
            batch_size = int(step_state["terminated"].shape[0])
        except Exception:
            try:
                batch_size = int(step_state.batch_size[0])
            except Exception:
                batch_size = 1

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

        if len(step_state.batch_size) == 0:
            step_state = step_state.auto_batch_size_()

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
    def _run(
        self,
        epochs: int,
        step_stream_factory: Callable[[], Iterator[tuple[bool, int, TensorDict]]],
        *,
        on_step: OrderedDict[str, Callable[[OnStepContext], None]] | None = None,
        training: bool = True,
        validation_epochs_frequency: int = 1,
        validation_steps_frequency: int = -1,
        max_steps_per_epoch: int = -1,
    ) -> None:
        """Run a training loop using the provided step stream and hooks.

        Args:
            epochs (int): Number of epochs to train for.
            step_stream_factory (Callable[[], Iterator[tuple[bool, int, TensorDict]]]):
                Factory returning an iterator that yields per-step TensorDicts
                with keys expected by the core ('state', 'actions', 'reward',
                'next_state', 'terminated').
            on_step (OrderedDict[str, Callable[[OnStepContext], None]] | None):
                Callback for logging/tracing/validation/checkpointing.
            training (bool): Whether to run in training mode.
            validation_epochs_frequency (int): Frequency of validation epochs.
            validation_steps_frequency (int): Frequency of validation steps.
            max_steps_per_epoch (int): Max steps per epoch (-1 for no limit).
        """
        # Initialize variables for exception safety
        epoch = -1
        step_idx = -1
        iteration = self._iteration
        batch = None
        loss_actor = None
        loss_critic = None
        # Track and set module modes for validation
        prev_actor_mode = getattr(self.core.actor, "training", True)
        prev_critic_mode = getattr(self.core.critic, "training", True)
        if not training:
            try:
                self.core.actor.eval()
                self.core.critic.eval()
            except Exception:
                pass
        try:
            # continue from stored iteration when resuming
            if (
                self.validation_step_stream_factory is not None
                and epochs > 0
                and validation_epochs_frequency > 0
                and epochs % validation_epochs_frequency == 0
                and training
            ):
                self.validate(validation_epochs_frequency)
            for epoch in range(self._epoch, self._epoch + epochs):
                loss_actor, loss_critic = torch.tensor(0.0), torch.tensor(0.0)

                for done, step_idx, step_state in step_stream_factory():
                    iteration += 1

                    # Store transition and maybe sample batch
                    if training:
                        batch = self._maybe_store_and_sample(step_state)
                    else:
                        batch = step_state.to(self.device)

                    # Update networks if we have a batch
                    if batch is not None:
                        # Filter out completed transitions to avoid training on
                        # states that won't evolve further and can create
                        # degenerate micro-batches.
                        try:
                            batch = self._filter_completed_state(batch)
                        except Exception:
                            # If structure is different, proceed without filtering.
                            pass

                        # Enforce minimum batch size for stable updates
                        effective_bs = None
                        try:
                            # Prefer explicit leading dim from a known tensor
                            effective_bs = int(batch["state"]["terminated"].shape[0])
                        except Exception:
                            try:
                                effective_bs = int(batch.batch_size[0])
                            except Exception:
                                effective_bs = None
                        if effective_bs is None:
                            effective_bs = int(self.batch_size)

                        if training and effective_bs < self.min_train_batch_size:
                            logger.info(
                                "Skipping rest of epoch %d due to small batch(%d < %d)",
                                epoch,
                                effective_bs,
                                self.min_train_batch_size,
                            )
                            # Skip remaining steps in this epoch
                            break

                        loss_actor, loss_critic = self.core.compute_loss(
                            batch,
                            training=training,
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
                    if on_step and batch is not None:
                        for callback_name, callback_step in on_step.items():
                            logger.debug(
                                "Calling {} on_step {}".format(
                                    "training" if training else "validation",
                                    callback_name,
                                )
                            )
                            callback_step(context)

                    if (
                        self.validation_step_stream_factory is not None
                        and validation_steps_frequency > 0
                        and iteration % validation_steps_frequency == 0
                        and training
                    ):
                        self.validate(validation_epochs_frequency)

                    # Early termination
                    if max_steps_per_epoch > 0 and step_idx >= max_steps_per_epoch - 1:
                        break
                # update stored epoch at end of epoch loop iteration
                self._epoch = epoch + 1
            # persist iteration counter
            self._iteration = iteration
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
            if self.on_exception is not None:
                for (
                    callback_name,
                    callback_exception,
                ) in self.on_exception.items():
                    logger.debug(
                        "Calling {} on_exception {}".format(
                            "training" if training else "validation",
                            callback_name,
                        )
                    )
                    callback_exception(e, context)
            raise e
        finally:
            # Restore previous modes if altered
            if not training:
                try:
                    if prev_actor_mode:
                        self.core.actor.train(True)
                    else:
                        self.core.actor.eval()
                    if prev_critic_mode:
                        self.core.critic.train(True)
                    else:
                        self.core.critic.eval()
                except Exception:
                    pass

    # ---------------- Checkpoint API ----------------
    def checkpoint(self) -> dict:
        """Return a minimal checkpoint for trainer progress.

        This intentionally avoids serializing callbacks. It captures only
        counters in order to support resuming training.
        """
        return add_schema_and_class(
            {
                "state": {
                    "epoch": self._epoch,
                    "iteration": self._iteration,
                    "batch_size": self.batch_size,
                    "warmup_buffer_ratio": self.warmup_buffer_ratio,
                    "min_train_batch_size": self.min_train_batch_size,
                },
            },
            self.__class__.__name__,
        )

    def load(self, checkpoint: dict) -> None:
        """Load trainer progress from a checkpoint dict.

        Note: This does not restore callbacks or replay buffer contents.
        """
        if not isinstance(checkpoint, dict):
            raise ValueError("Trainer checkpoint must be a dict")
        state = checkpoint.get("state", {})
        self._epoch = int(state.get("epoch", 0))
        self._iteration = int(state.get("iteration", 0))
        # Optionally restore basic config
        self.batch_size = int(state.get("batch_size", self.batch_size))
        self.warmup_buffer_ratio = float(
            state.get("warmup_buffer_ratio", self.warmup_buffer_ratio)
        )
        self.min_train_batch_size = int(
            state.get("min_train_batch_size", self.min_train_batch_size)
        )

    def train(
        self,
        epochs,
        validation_epochs_frequency=1,
        validation_steps_frequency=-1,
        max_steps_per_epoch=-1,
    ) -> None:
        """Run the training loop.

        Args:
            epochs (int): Number of epochs to train for.
            validation_epochs_frequency (int): Frequency of validation epochs.
            validation_steps_frequency (int): Frequency of validation steps.
            max_steps_per_epoch (int): Max steps per epoch (-1 for no limit).
        """
        self._run(
            step_stream_factory=self.train_step_stream_factory,
            on_step=self.on_train_step,
            training=True,
            epochs=epochs,
            validation_epochs_frequency=validation_epochs_frequency,
            validation_steps_frequency=validation_steps_frequency,
            max_steps_per_epoch=max_steps_per_epoch,
        )

    def validate(self, epochs, max_steps_per_epoch=-1) -> None:
        """Run the validation loop.

        Args:
            epochs (int): Number of epochs to validate for.
            max_steps_per_epoch (int): Max steps per epoch (-1 for no limit).
        """
        if self.validation_step_stream_factory is None:
            raise ValueError("Validation step stream factory is not set")
        self._run(
            step_stream_factory=self.validation_step_stream_factory,
            on_step=self.on_validation_step,
            training=False,
            epochs=epochs,
            max_steps_per_epoch=max_steps_per_epoch,
        )
