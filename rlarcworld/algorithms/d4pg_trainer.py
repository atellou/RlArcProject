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
        step_stream_factory: Callable[[int], Iterator[TensorDict]],
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
            step_stream_factory (Callable[[int], Iterator[TensorDict]]): A factory function that returns an iterator/generator that yields per-step TensorDicts with the keys expected by the core ('state', 'actions', 'reward', 'next_state', 'terminated').
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
        self.device = device or core.device

        if self.replay_buffer is None and self.batch_size is None:
            raise ValueError("batch_size must be provided when replay_buffer is None")

    # ---------------- Orchestration helpers ----------------
    def _maybe_store_and_sample(self, step_state: TensorDict) -> Optional[TensorDict]:
        """Store transition and sample a batch (if buffer warmup satisfied).

        If no replay_buffer is provided, return the single-step batch directly.
        """
        # Single-step batch mode (no replay)
        if self.replay_buffer is None:
            batch = TensorDict(
                {
                    "state": step_state["state"],
                    "actions": step_state["actions"],
                    "reward": step_state["reward"],
                    "next_state": step_state["next_state"],
                    "terminated": step_state["next_state"]["terminated"],
                }
            ).to(self.device)
            return batch

        # Replay buffer mode (append transition and sample)
        sampled_batch_size = step_state["next_state"]["terminated"].shape[0]
        # Use max priority from buffer if available, else 1.0
        priority = torch.full(
            (sampled_batch_size,),
            getattr(getattr(self.replay_buffer, "sampler", {}), "_max_priority", 1.0),
            device=self.device,
        )
        # Pack per-item TensorDicts as done in the original code
        items = []
        for i in range(sampled_batch_size):
            items.append(
                TensorDict(
                    {
                        "state": step_state["state"][i],
                        "actions": step_state["actions"][i],
                        "reward": step_state["reward"][i],
                        "next_state": step_state["next_state"][i],
                        "terminated": step_state["next_state"]["terminated"][i],
                    }
                ).auto_batch_size_()
            )
        stacked = torch.stack(items)
        indices = self.replay_buffer.extend(stacked)
        try:
            self.replay_buffer.update_priority(indices, priority)
        except Exception:
            # allow non-PER buffers
            pass

        # Warmup logic
        try:
            storage_len = len(self.replay_buffer.storage)
            storage_cap = self.replay_buffer.storage.max_size
        except Exception:
            storage_len, storage_cap = 0, 0
        if storage_cap and storage_len > storage_cap * self.warmup_buffer_ratio:
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
        logger_frequency: int = 1000,
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

        - step_stream_factory(epoch) should yield per-step TensorDicts with required keys.
        - validate_fn, if provided, is called as validate_fn(epoch, step) and must
          return (val_actor_loss, val_critic_loss, context_tensor_dict).
        - checkpoint_payload_fn, if provided, should build a dict with any payload to be saved.
        """
        iteration = 0
        decay_flag = False

        for epoch in range(epochs):
            step_stream = self.step_stream_factory(epoch)
            for step_idx, step_state in enumerate(step_stream):
                if 0 <= max_steps_per_epoch < step_idx:
                    break

                # Orchestrate experience and sampling
                batch = self._maybe_store_and_sample(step_state)
                if batch is None:
                    continue

                # Remove completed states
                batch = self._filter_completed_state(batch)

                # Compute and apply losses
                loss_actor, loss_critic = self.core.compute_loss(batch, training=True)

                # Post-step hook
                if self.on_step is not None and iteration % logger_frequency == 0:
                    self.on_step(epoch, step_idx, step_state, loss_actor, loss_critic)

                # Target updates and decay
                iteration += 1
                if decay_flag:
                    self.core.step_schedulers_and_decay()
                if step_idx % self.target_update_frequency == 0:
                    self.core.update_target_networks()
                    decay_flag = True

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
                            "iteration": iteration,
                            "loss_actor": float(loss_actor.detach().cpu().item()),
                            "loss_critic": float(loss_critic.detach().cpu().item()),
                        }
                    if self.on_checkpoint is not None:
                        self.on_checkpoint(payload)
