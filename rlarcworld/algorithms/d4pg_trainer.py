from typing import Callable, Dict, Iterator, Optional, Tuple, Protocol, Any

import torch
from tensordict import TensorDict

from .d4pg import D4PGCore


class ReplayBufferProto(Protocol):
    """A minimal protocol for prioritized or uniform replay buffers."""

    def extend(self, batch: torch.Tensor) -> Any: ...

    def sample(self, batch_size: int) -> TensorDict: ...

    def update_priority(self, indices: Any, priorities: torch.Tensor) -> None: ...

    @property
    def storage(self) -> Any: ...


# Callback/hook signatures
OnStepFn = Callable[[int, int, TensorDict, torch.Tensor, torch.Tensor], None]
OnValidateFn = Callable[[int, int, torch.Tensor, torch.Tensor, TensorDict], None]
OnCheckpointFn = Callable[[Dict[str, Any]], None]


class D4PGTrainer:
    """
    Decoupled trainer/wrapper for D4PGCore.

    This wrapper is designed to be environment- and logging-agnostic. It relies on
    dependency injection through callables:
      - step_stream_factory: returns an iterator/generator that yields per-step TensorDicts
        with the keys expected by the core ('state', 'actions', 'reward', 'next_state', 'terminated').
      - replay_buffer (optional): any object satisfying ReplayBufferProto.
      - on_step/on_validate/on_checkpoint (optional): callbacks to plug logging/traceability/checkpointing.

    The wrapper only orchestrates: sampling, training steps, target updates, and hooks.
    """

    def __init__(
        self,
        *,
        core: D4PGCore,
        step_stream_factory: Optional[Callable[[int], Iterator[TensorDict]]] = None,
        replay_buffer: Optional[ReplayBufferProto] = None,
        batch_size: Optional[int] = None,
        warmup_buffer_ratio: float = 0.2,
        target_update_frequency: int = 10,
        on_step: Optional[OnStepFn] = None,
        on_validate: Optional[OnValidateFn] = None,
        on_checkpoint: Optional[OnCheckpointFn] = None,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize the trainer with the provided components.

        Args:
            core (D4PGCore): The core D4PG agent to train.
            step_stream_factory (Optional[Callable[[int], Iterator[TensorDict]]]): A factory function that returns an iterator/generator that yields per-step TensorDicts with the keys expected by the core ('state', 'actions', 'reward', 'next_state', 'terminated'). Defaults to None.
            replay_buffer (Optional[ReplayBufferProto]): A replay buffer to store transitions for experience replay. Defaults to None.
            batch_size (Optional[int]): The batch size for training. Defaults to None.
            warmup_buffer_ratio (float): The ratio of buffer to fill before training. Defaults to 0.2.
            target_update_frequency (int): The frequency of target updates. Defaults to 10.
            on_step (Optional[OnStepFn]): A callback function to plug logging/traceability/checkpointing. Defaults to None.
            on_validate (Optional[OnValidateFn]): A callback function to plug logging/traceability/checkpointing. Defaults to None.
            on_checkpoint (Optional[OnCheckpointFn]): A callback function to plug logging/traceability/checkpointing. Defaults to None.
            device (Optional[str]): The device to use for training. Defaults to None.
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
        self.device = device or next(core.parameters()).device

        if self.replay_buffer is None and self.batch_size is None:
            raise ValueError("batch_size must be provided when replay_buffer is None")

    # ---------------- Orchestration helpers ----------------
    def _maybe_store_and_sample(self, step_state: TensorDict) -> Optional[TensorDict]:
        """Store the transition in the replay buffer and return a batch if ready.

        Args:
            step_state: Dictionary containing the current step's data

        Returns:
            Optional batch of transitions if buffer is warmed up, else None
        """
        # Single-step batch mode (no replay)
        if self.replay_buffer is None:
            return step_state.unsqueeze(0).to(self.device)

        # Replay buffer mode (append transition and sample)
        batch_size = step_state.batch_size[0]

        # Use max priority from buffer if available, else 1.0
        priority = torch.full(
            (batch_size,),
            getattr(getattr(self.replay_buffer, "sampler", None), "_max_priority", 1.0),
            device=self.device,
        )

        # Add to replay buffer
        indices = self.replay_buffer.extend(step_state.unsqueeze(0))

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
        validate_fn: Optional[
            Callable[[int, int], Tuple[torch.Tensor, torch.Tensor, TensorDict]]
        ] = None,
        checkpoint_every_steps: int = -1,
        checkpoint_payload_fn: Optional[
            Callable[[int, int, torch.Tensor, torch.Tensor], Dict[str, Any]]
        ] = None,
    ) -> None:
        """Run a training loop using the provided step stream and hooks.

        Args:
            epochs (int): Number of epochs to train for
            max_steps_per_epoch (int): Maximum number of steps per epoch (-1 for no limit)
            validate_every_steps (int): Run validation every N steps (-1 to disable)
            validate_fn (Optional[Callable[[int, int], Tuple[torch.Tensor, torch.Tensor, TensorDict]]]): Validation function. Defaults to None.
            checkpoint_every_steps (int): Save checkpoint every N steps (-1 to disable)
            checkpoint_payload_fn (Optional[Callable[[int, int, torch.Tensor, torch.Tensor], Dict[str, Any]]]): Checkpoint payload function. Defaults to None.
        """

        for epoch in range(epochs):
            step_stream = self.step_stream_factory(epoch)

            loss_actor, loss_critic = torch.tensor(0.0), torch.tensor(0.0)

            for step_idx, step_state in enumerate(step_stream):
                iteration = epoch * max_steps_per_epoch + step_idx
                decay_flag = False

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
                    decay_flag = True

                # Call step callback
                if self.on_step is not None:
                    self.on_step(
                        epoch=epoch,
                        step=step_idx,
                        loss_actor=(
                            loss_actor.item() if loss_actor is not None else None
                        ),
                        loss_critic=(
                            loss_critic.item() if loss_critic is not None else None
                        ),
                        batch=batch,
                        decay_flag=decay_flag,
                    )

                # Update target networks periodically
                if decay_flag and iteration % self.target_update_frequency == 0:
                    self.core.update_target_networks()

                # Validation
                if (
                    validate_every_steps > 0
                    and iteration % validate_every_steps == 0
                    and validate_fn is not None
                ):
                    with torch.no_grad():
                        v_actor, v_critic, ctx = validate_fn(epoch, step_idx)
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
