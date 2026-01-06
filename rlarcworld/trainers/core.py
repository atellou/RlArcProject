"""Trainer wrapper around a Core algorithm with pluggable components.

This module contains a lightweight trainer that orchestrates sampling,
updates, target network sync, and optional callbacks for logging,
validation, and checkpointing.
"""

import logging
from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Tuple

import torch
from tensordict import TensorDict

from rlarcworld.common.protocols import ActorCriticCoreProto, ReplayBufferProto
from rlarcworld.utils.checkpointing import add_schema_and_class

logger = logging.getLogger(__name__)


@dataclass
class OnStepContext:
    """ctx object containing all information about a training step."""

    def __init__(
        self,
        epoch: int,
        step_idx: int,
        iteration: int,
        batch: TensorDict | None,
        loss_actor: torch.Tensor | None,
        loss_critic: torch.Tensor | None,
        td_error: TensorDict | None,
        core: ActorCriticCoreProto,
        hyperparameters: dict,
        warmup: bool,
        buffer_warmup: float | None = None,
        run_end: bool = False,
    ):
        """Initialize step ctx.

        Args:
            epoch: Current epoch number
            step_idx: Current step index within epoch
            iteration: Global iteration number
            batch: Training batch (None if no batch available)
            loss_actor: Actor loss (None if no training occurred)
            loss_critic: Critic loss (None if no training occurred)
            td_error: TD error (None if no training occurred)
            core: Algorithm core containing networks and methods
            hyperparameters: Hyperparameters of the algorithm
            warmup: Whether the training is in warmup phase
            buffer_warmup: Buffer warmup percentage (None if not applicable)
            run_end: Indicates whether the call is made at the end of _run call
        """
        self.epoch = epoch
        self.step_idx = step_idx
        self.iteration = iteration
        self.batch = batch
        self.loss_actor = loss_actor
        self.loss_critic = loss_critic
        self.td_error = td_error
        self.core = core
        self.hyperparameters = hyperparameters
        self.warmup = warmup
        self.buffer_warmup = buffer_warmup
        self.run_end = run_end


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
        core: ActorCriticCoreProto,
        batch_size: int,
        min_train_batch_size: int = 2,
        train_step_stream_factory: Callable[[], Iterator[tuple[bool, int, TensorDict]]],
        validation_step_stream_factory: Callable[
            [], Iterator[tuple[bool, int, TensorDict]]
        ]
        | None = None,
        replay_buffer: ReplayBufferProto | None = None,
        replay_buffer_update_priority: Callable[
            [OnStepContext, dict[str, torch.Tensor], ReplayBufferProto], None
        ]
        | None = None,
        warmup_buffer_ratio: float = 0.2,
        on_train_step: OrderedDict[str, Callable[[OnStepContext], None]] | None = None,
        on_validation_step: OrderedDict[str, Callable[[OnStepContext], None]]
        | None = None,
        on_exception: OrderedDict[str, Callable[[Exception, OnStepContext], None]]
        | None = None,
        device: str | None = None,
        priority_init_value: float = 1_000_000,
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
            replay_buffer_update_priority (Optional[Callable[[OnStepContext, dict[str, torch.Tensor], ReplayBufferProto], None]]):
                A function to update the priority of the replay buffer.
            warmup_buffer_ratio (float): Ratio of buffer to fill before training.
            on_train_step (Optional[OrderedDict[str, Callable[[OnStepContext], None]]]):
                Callback for logging/tracing/validation/checkpointing.
            on_validation_step (Optional[OrderedDict[str, Callable[[OnStepContext], None]]]):
                Callback for logging/tracing/validation/checkpointing.
            on_exception (Optional[OrderedDict[str, Callable[[Exception, OnStepContext], None]]]):
                Callback for training exceptions.
            device (Optional[str]): The device to use for training.
            priority_init_value (float): The initial priority value for the replay buffer.
        """  # noqa: E501
        self.core = core
        self.train_step_stream_factory = train_step_stream_factory
        self.validation_step_stream_factory = validation_step_stream_factory
        self.replay_buffer = replay_buffer
        self.replay_buffer_update_priority = replay_buffer_update_priority
        self.batch_size = batch_size
        self.min_train_batch_size = int(min_train_batch_size)
        self.warmup_buffer_ratio = warmup_buffer_ratio
        self.on_train_step = on_train_step
        self.on_validation_step = on_validation_step
        self.on_exception = on_exception
        self.device = (
            device or getattr(core, "device", None) or str(torch.device("cpu"))
        )
        self.priority_init_value = priority_init_value

        # Internal progress tracking for checkpointing
        self._train_epoch: int = 0
        self._train_iteration: int = 0

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
    def _maybe_store_and_sample(
        self, step_state: TensorDict
    ) -> Tuple[TensorDict | None, dict[str, torch.Tensor] | None]:
        """Store the transition in the replay buffer and return a batch if ready.

        Args:
            step_state: Dictionary containing the current step's data

        Returns:
            Optional batch of transitions if buffer is warmed up, else None
        """
        # Single-step batch mode (no replay): keep (B, ...) shape without extra time dim
        if self.replay_buffer is None:
            return step_state.to(self.device), None

        if len(step_state.batch_size) == 0:
            step_state = step_state.auto_batch_size_()

        # Add to replay buffer without introducing an extra leading time dimension
        indices = self.replay_buffer.extend(step_state)
        storage_len = len(self.replay_buffer.storage)
        storage_cap = self.replay_buffer.storage.max_size

        if storage_len <= storage_cap * self.warmup_buffer_ratio:
            priority = torch.full(
                (self.batch_size,), self.priority_init_value, device=self.device
            )
            try:
                self.replay_buffer.update_priority(indices, priority)
            except Exception:
                # allow non-PER buffers
                pass

        # Check if we've collected enough samples to start training
        if storage_len > storage_cap * self.warmup_buffer_ratio:
            # Sample with return_info=True to get index metadata for priority update
            batch, info = self.replay_buffer.sample(self.batch_size, return_info=True)
            return batch.to(self.device), info

        return None, None

    def warmup_info(self) -> tuple[float, bool]:
        """Get warmup information including phase status and percentage."""
        if self.replay_buffer is None:
            return 0.0, False
        storage_len = len(self.replay_buffer.storage)
        storage_cap = self.replay_buffer.storage.max_size
        warmup_size = int(storage_cap * float(self.warmup_buffer_ratio))
        # Guard against nonsensical configs (e.g. ratio=0)
        if warmup_size <= 0:
            return 100.0, False
        is_warmup = storage_len < warmup_size
        percentage = (float(storage_len) / float(warmup_size)) * 100.0
        # Keep the log readable once we cross warmup.
        percentage = min(percentage, 100.0)
        return percentage, is_warmup

    @property
    def train_epoch(self) -> int:
        """Get the current training epoch."""
        return self._train_epoch

    @property
    def train_iteration(self) -> int:
        """Get the current training iteration."""
        return self._train_iteration

    # ---------------- Public API ----------------
    def _run(
        self,
        epochs: int,
        step_stream_factory: Callable[[], Iterator[tuple[bool, int, TensorDict]]],
        *,
        on_step: OrderedDict[str, Callable[[OnStepContext], None]] | None = None,
        training: bool = True,
        max_steps_per_epoch: int = -1,
        validation_epochs: int = 1,
        validation_steps_per_epoch: int = -1,
        validation_steps_frequency: int = -1,
    ) -> None:
        """Run a batch loops using the provided step stream and hooks.

        Args:
            epochs (int): Total number of epochs to run. Resuming from a checkpoint,
                training continues from current epoch until total epoch count reached.
            step_stream_factory (Callable[[], Iterator[tuple[bool, int, TensorDict]]]):
                Factory returning an iterator that yields per-step TensorDicts
                with keys expected by the core ('state', 'actions', 'reward',
                'next_state', 'terminated').
            on_step (OrderedDict[str, Callable[[OnStepContext], None]] | None):
                Callback for logging/tracing/validation/checkpointing.
            training (bool): Whether to run in training mode.
            max_steps_per_epoch (int): Max steps per epoch (-1 for no limit).
            validation_epochs (int): Number of epochs to validate for.
            validation_steps_per_epoch (int):
                Max steps per epoch on validation (-1 for no limit).
            validation_steps_frequency (int): Frequency of validation steps.
        """
        # Initialize variables for exception safety
        step_idx = -1
        batch = None
        loss_actor = None
        loss_critic = None
        td_error = None
        # Track and set module modes for validation
        prev_actor_mode = getattr(self.core.actor, "training", True)
        prev_critic_mode = getattr(self.core.critic, "training", True)
        if training:
            epoch = self._train_epoch
            iteration = self._train_iteration
            if epoch > 0 or iteration > 0:
                logger.info(
                    f"Resuming {self.__class__.__name__} training from epoch "
                    f"{epoch}/{epochs} and iteration {iteration}"
                )
        else:
            epoch = 0
            iteration = 0
            try:
                self.core.actor.eval()
                self.core.critic.eval()
            except Exception:
                pass
        try:
            for current_epoch in range(epoch, epochs):
                loss_actor, loss_critic, td_error = (
                    torch.tensor(0.0),
                    torch.tensor(0.0),
                    None,
                )

                for _done, step_idx, step_state in step_stream_factory():
                    # Store transition and maybe sample batch
                    if training:
                        batch, info = self._maybe_store_and_sample(step_state)
                    else:
                        batch, info = step_state.to(self.device), None

                    # Update networks if we have a batch
                    if batch is not None:
                        # Filter out completed transitions to avoid training on
                        # states that won't evolve further and can create
                        # degenerate micro-batches.
                        try:
                            mask = batch["state"]["terminated"] == 0
                            selected_indices = mask.nonzero(as_tuple=True)[0]
                            batch = batch[selected_indices]
                            # Also filter info indices to match the filtered batch
                            if info is not None and "index" in info:
                                info["index"] = info["index"][selected_indices]
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
                            logger.debug(
                                "Skipping rest of epoch %d due to small batch(%d < %d)",
                                current_epoch,
                                effective_bs,
                                self.min_train_batch_size,
                            )
                            # Skip remaining steps in this epoch
                            break

                        loss_actor, loss_critic, td_error = self.core.compute_loss(
                            batch,
                            training=training,
                        )
                    else:
                        loss_actor, loss_critic, td_error = None, None, None

                    # Create step ctx
                    buffer_warmup, warmup = self.warmup_info()
                    ctx = OnStepContext(
                        epoch=current_epoch,
                        step_idx=step_idx,
                        iteration=iteration,
                        batch=batch,
                        loss_actor=loss_actor,
                        loss_critic=loss_critic,
                        td_error=td_error,
                        core=self.core,
                        hyperparameters=self.hyperparameters,
                        warmup=warmup,
                        buffer_warmup=buffer_warmup,
                    )

                    if (
                        self.replay_buffer is not None
                        and self.replay_buffer_update_priority is not None
                        and batch is not None
                        and training
                    ):
                        if info is None:
                            raise ValueError(
                                "Info is None - cannot update replay buffer priorities"
                            )
                        self.replay_buffer_update_priority(
                            ctx, info, self.replay_buffer
                        )

                    # Call step callback
                    if on_step is not None:
                        for callback_name, callback_step in on_step.items():
                            callback_step(ctx)

                    if (
                        self.validation_step_stream_factory is not None
                        and validation_steps_frequency > 0
                        and iteration % validation_steps_frequency == 0
                        and not warmup
                        and training
                    ):
                        self.validate(validation_epochs, validation_steps_per_epoch)

                    # Early termination
                    if max_steps_per_epoch > 0 and step_idx >= max_steps_per_epoch - 1:
                        break

                    iteration += 1

                    # persist iteration counter
                    if training:
                        self._train_iteration = iteration
                # update stored epoch at end of epoch loop iteration
                if training:
                    self._train_epoch += 1
        except Exception as e:
            logger.exception(
                f"Exception during training step iteration {iteration}: {e}"
            )
            ctx = OnStepContext(
                epoch=epoch,
                step_idx=step_idx,
                iteration=iteration,
                batch=batch,
                loss_actor=loss_actor,
                loss_critic=loss_critic,
                td_error=td_error,
                core=self.core,
                hyperparameters=self.hyperparameters,
                warmup=warmup,
                buffer_warmup=buffer_warmup,
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
                    callback_exception(e, ctx)
            raise e
        finally:
            # Restore previous modes if altered
            ctx = OnStepContext(
                epoch=epoch,
                step_idx=step_idx,
                iteration=iteration,
                batch=batch,
                loss_actor=loss_actor,
                loss_critic=loss_critic,
                td_error=td_error,
                core=self.core,
                hyperparameters=self.hyperparameters,
                warmup=warmup,
                buffer_warmup=buffer_warmup,
                run_end=True,
            )
            if on_step is not None:
                for callback_name, callback in on_step.items():
                    logger.debug(
                        "Calling {} on_step {}".format(
                            "training" if training else "validation",
                            callback_name,
                        )
                    )
                    callback(ctx)
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

    def _validate_train_buffer(self, epochs: int, max_steps_per_epoch: int) -> None:
        """Validate that the replay buffer has enough data for training."""
        if self.replay_buffer is not None and self.warmup_buffer_ratio > 0:
            storage_cap = self.replay_buffer.storage.max_size
            warmup_buffer_size = int(self.warmup_buffer_ratio * storage_cap)
            logger.info(f"The replay buffer warmup size is {warmup_buffer_size}")
            if epochs <= 0:
                raise ValueError("epochs must be greater than 0")
            if max_steps_per_epoch == 0:
                raise ValueError("max_steps_per_epoch must be different from 0")
            training_steps_required = max_steps_per_epoch * epochs
            if max_steps_per_epoch > 0 and training_steps_required < warmup_buffer_size:
                raise ValueError(
                    f"Warmup size ({warmup_buffer_size}) is greater than "
                    f"required training duration ({training_steps_required} steps). "
                    "Increase training steps or reduce warmup_buffer_ratio."
                )

    def train(
        self,
        epochs: int,
        max_steps_per_epoch: int = -1,
        validation_epochs: int = 1,
        validation_steps_per_epoch: int = -1,
        validation_steps_frequency: int = -1,
    ) -> None:
        """Run the training loop.

        Args:
            epochs (int): Number of epochs to train for.
            max_steps_per_epoch (int): Max steps per epoch (-1 for no limit).
            validation_epochs (int): Number of epochs to validate for.
            validation_steps_per_epoch (int): Max steps per epoch (-1 for no limit).
            validation_steps_frequency (int): Frequency of validation steps.
        """
        self._validate_train_buffer(epochs, max_steps_per_epoch)

        self._run(
            step_stream_factory=self.train_step_stream_factory,
            on_step=self.on_train_step,
            training=True,
            epochs=epochs,
            max_steps_per_epoch=max_steps_per_epoch,
            validation_epochs=validation_epochs,
            validation_steps_per_epoch=validation_steps_per_epoch,
            validation_steps_frequency=validation_steps_frequency,
        )

    def validate(self, epochs: int, max_steps_per_epoch: int = -1) -> None:
        """Run the validation loop.

        Args:
            epochs (int): Number of epochs to validate for.
            max_steps_per_epoch (int): Max steps per epoch (-1 for no limit).
        """
        if self.validation_step_stream_factory is None:
            raise ValueError("Validation step stream factory is not set")
        logger.info(
            "Running validation loop, epochs: {}, max_steps_per_epoch: {}".format(
                epochs, max_steps_per_epoch
            )
        )
        self._run(
            step_stream_factory=self.validation_step_stream_factory,
            on_step=self.on_validation_step,
            training=False,
            epochs=epochs,
            max_steps_per_epoch=max_steps_per_epoch,
        )

    # ---------------- Checkpoint API ----------------
    def checkpoint(self) -> dict:
        """Return a minimal checkpoint for trainer progress.

        This intentionally avoids serializing callbacks. It captures only
        counters in order to support resuming training.
        """
        return add_schema_and_class(
            {
                "state": {
                    "epoch": self._train_epoch,
                    "iteration": self._train_iteration,
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
        self._train_epoch = int(state.get("epoch", 0))
        self._train_iteration = int(state.get("iteration", 0))
        # Optionally restore basic config
        self.batch_size = int(state.get("batch_size", self.batch_size))
        self.warmup_buffer_ratio = float(
            state.get("warmup_buffer_ratio", self.warmup_buffer_ratio)
        )
        self.min_train_batch_size = int(
            state.get("min_train_batch_size", self.min_train_batch_size)
        )
