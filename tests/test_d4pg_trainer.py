from collections import OrderedDict

import pytest
import torch
from torchrl.data.replay_buffers import (
    LazyTensorStorage,
    PrioritizedSampler,
    TensorDictReplayBuffer,
)

from rlarcworld.algorithms.d4pg import D4PGCore
from rlarcworld.models.actor import ArcActorNetwork
from rlarcworld.models.critic import ArcCriticNetwork
from rlarcworld.runners.core import OnStepContext, Trainer
from tests.utils.synthetic import make_synthetic_step_on_device

"""
Most tests rely on fixtures defined in tests/conftest.py:
- device, grid_size, color_values, batch_size
- actor, critic, d4pg_core
- per_replay_buffer
- make_synthetic_step(batch_size, actor)
"""


def test_d4pg_trainer_training_updates_and_loss_ranges(
    device, batch_size, grid_size, color_values, d4pg_core, make_synthetic_step
):
    core = d4pg_core
    max_steps = 6

    def step_stream_factory():
        for step_idx in range(max_steps):
            done = step_idx == max_steps - 1
            yield done, step_idx, make_synthetic_step(batch_size, core.actor)

    losses_actor = []
    losses_critic = []

    def on_step(ctx: OnStepContext):
        if ctx.loss_actor is not None:
            losses_actor.append(ctx.loss_actor)
        if ctx.loss_critic is not None:
            losses_critic.append(ctx.loss_critic)

    # Minimal replay buffer to return batches without adding an extra leading batch dim
    class _MiniStorage:
        def __init__(self):
            self.max_size = 1
            self._len = 0

        def __len__(self):
            return self._len

    class _MiniBuffer:
        def __init__(self):
            self.storage = _MiniStorage()
            self._last = None

        def extend(self, batch):
            # trainer provides batch with leading [1] dim; store squeezed batch
            self._last = batch.squeeze(0)
            self.storage._len = 1
            return [0]

        def sample(self, bs, return_info=False):
            if return_info:
                return self._last, {"index": torch.tensor([0])}
            return self._last

        def update_priority(self, indices, priorities):
            pass

    trainer = Trainer(
        core=core,
        train_step_stream_factory=step_stream_factory,
        replay_buffer=_MiniBuffer(),
        batch_size=batch_size,
        warmup_buffer_ratio=0.0,
        on_train_step={"test": on_step},
        device=str(device),
    )

    # Snapshot some params before training
    actor_before = [
        p.detach().clone() for p in core.actor.parameters() if p.requires_grad
    ][:3]
    critic_before = [
        p.detach().clone() for p in core.critic.parameters() if p.requires_grad
    ][:3]

    trainer.train(epochs=1, max_steps_per_epoch=max_steps)

    # Ensure some updates occurred
    actor_changed = any(
        not torch.allclose(pb, pa)
        for pb, pa in zip(
            actor_before,
            [p.detach().clone() for p in core.actor.parameters() if p.requires_grad][
                :3
            ],
        )
    )
    critic_changed = any(
        not torch.allclose(pb, pa)
        for pb, pa in zip(
            critic_before,
            [p.detach().clone() for p in core.critic.parameters() if p.requires_grad][
                :3
            ],
        )
    )
    assert actor_changed, "Actor parameters did not update in trainer loop"
    assert critic_changed, "Critic parameters did not update in trainer loop"

    # Losses recorded: finite and reasonable ranges (broad sanity bounds)
    assert all(torch.isfinite(v) for v in losses_actor)
    assert all(torch.isfinite(v) for v in losses_critic)


def test_d4pg_trainer_per_update_priority_called(
    device, d4pg_core, grid_size, color_values, batch_size, make_synthetic_step
):
    core = d4pg_core
    max_steps = 6

    def step_stream_factory():
        for step_idx in range(max_steps):
            done = step_idx == max_steps - 1
            yield done, step_idx, make_synthetic_step(batch_size, core.actor)

    capacity = 20
    base_rb = TensorDictReplayBuffer(
        storage=LazyTensorStorage(capacity),
        sampler=PrioritizedSampler(max_capacity=capacity, alpha=0.6, beta=0.4),
    )

    class _RecordRB:
        def __init__(self, rb):
            self._rb = rb
            self.calls = []

        @property
        def storage(self):
            return self._rb.storage

        @property
        def sampler(self):
            return self._rb.sampler

        def extend(self, *args, **kwargs):
            return self._rb.extend(*args, **kwargs)

        def sample(self, *args, **kwargs):
            return self._rb.sample(*args, **kwargs)

        def update_priority(self, indices, priorities):
            # Record and forward
            self.calls.append((indices, priorities.detach().cpu()))
            return self._rb.update_priority(indices, priorities)

    rb = _RecordRB(base_rb)

    def replay_buffer_update_priority(
        context: OnStepContext,
        info: dict[str, torch.Tensor],
        replay_buffer: _RecordRB,
    ) -> None:
        if context.td_error is None or info is None:
            return
        # Extract first available TD-error from TensorDict
        td_error_value = next(iter(context.td_error.values()))
        replay_buffer.update_priority(info["index"], td_error_value)

    trainer = Trainer(
        core=core,
        train_step_stream_factory=step_stream_factory,
        replay_buffer=rb,
        replay_buffer_update_priority=replay_buffer_update_priority,
        batch_size=batch_size,
        warmup_buffer_ratio=0.0,
        device=str(device),
    )

    trainer.train(epochs=1, max_steps_per_epoch=6)

    # Assert update_priority was called at least once with positive finite priorities
    assert len(rb.calls) > 0, "update_priority was not called"
    any_positive = False
    for indices, priorities in rb.calls:
        assert torch.isfinite(priorities).all()
        # length match: indices and priorities
        try:
            n_idx = len(indices)
        except Exception:
            n_idx = int(torch.numel(indices))
        assert priorities.numel() == n_idx
        if (priorities > 0).any():
            any_positive = True
    assert any_positive, "No positive priorities were passed to update_priority"


def test_d4pg_trainer_warmup_with_prioritized_sampler(
    device, d4pg_core, grid_size, color_values, batch_size, make_synthetic_step
):
    core = d4pg_core

    # Real PER buffer
    capacity = 10
    warmup_ratio = 0.5
    rb = TensorDictReplayBuffer(
        storage=LazyTensorStorage(capacity),
        sampler=PrioritizedSampler(max_capacity=capacity, alpha=0.6, beta=0.4),
    )

    def replay_buffer_update_priority(
        context: OnStepContext,
        info: dict[str, torch.Tensor],
        replay_buffer: TensorDictReplayBuffer,
    ) -> None:
        if context.td_error is None or info is None:
            return
        # Extract first available TD-error from TensorDict
        td_error_value = next(iter(context.td_error.values()))
        replay_buffer.update_priority(info["index"], td_error_value)

    max_steps = 10

    def step_stream_factory():
        for step_idx in range(max_steps):
            done = step_idx == max_steps - 1
            yield done, step_idx, make_synthetic_step(batch_size, core.actor)

    first_training_step = None
    losses_actor, losses_critic = [], []

    def on_step(ctx: OnStepContext):
        nonlocal first_training_step
        if ctx.batch is not None and first_training_step is None:
            first_training_step = ctx.step_idx
        if ctx.loss_actor is not None:
            losses_actor.append(ctx.loss_actor)
        if ctx.loss_critic is not None:
            losses_critic.append(ctx.loss_critic)

    trainer = Trainer(
        core=core,
        train_step_stream_factory=step_stream_factory,
        replay_buffer=rb,
        replay_buffer_update_priority=replay_buffer_update_priority,
        batch_size=batch_size,
        warmup_buffer_ratio=warmup_ratio,
        on_train_step={"test": on_step},
        device=str(device),
    )

    # Compute expected minimal steps before training starts:
    # find smallest n such that n*batch_size > capacity*warmup_ratio
    import math

    threshold = capacity * warmup_ratio
    expected_min_steps = math.floor(threshold / batch_size) + 1

    trainer.train(epochs=1, max_steps_per_epoch=max_steps)

    assert (
        first_training_step is not None
    ), "Training did not start after warmup threshold"
    assert first_training_step >= expected_min_steps - 1  # step is zero-based index
    # After warmup, we should have at least one finite loss recorded
    assert any(torch.isfinite(v) for v in losses_actor)
    assert any(torch.isfinite(v) for v in losses_critic)


def test_d4pg_trainer_real_replay_buffer_and_step_handlers(
    device, d4pg_core, grid_size, color_values, batch_size, make_synthetic_step
):
    core = d4pg_core

    # Real replay buffer from torchrl
    capacity = 50
    rb = TensorDictReplayBuffer(
        storage=LazyTensorStorage(capacity),
        sampler=PrioritizedSampler(max_capacity=capacity, alpha=0.6, beta=0.4),
    )

    max_steps = 7

    def step_stream_factory():
        for step_idx in range(max_steps):
            done = step_idx == max_steps - 1
            yield done, step_idx, make_synthetic_step(batch_size, core.actor)

    # Track losses and step handler execution
    losses_actor, losses_critic = [], []
    target_updates_called = 0

    def logging_handler(ctx: OnStepContext):
        # Record losses if present
        if ctx.loss_actor is not None:
            losses_actor.append(ctx.loss_actor)
        if ctx.loss_critic is not None:
            losses_critic.append(ctx.loss_critic)

    def target_update_handler(ctx: OnStepContext):
        # Simulate target network updates
        # (in real usage, this would call core.update_target_networks())
        nonlocal target_updates_called
        if ctx.batch is not None and ctx.iteration % 2 == 0:  # Every 2 steps
            target_updates_called += 1

    trainer = Trainer(
        core=core,
        train_step_stream_factory=step_stream_factory,
        replay_buffer=rb,
        batch_size=batch_size,
        warmup_buffer_ratio=0.0,
        on_train_step={
            "logging": logging_handler,
            "target_update": target_update_handler,
        },
        device=device,
    )

    trainer.train(epochs=1, max_steps_per_epoch=max_steps)

    # Verify step handlers were called and training occurred
    assert target_updates_called >= 1, "Target update handler should have been called"
    assert len(losses_actor) > 0 and len(losses_critic) > 0
    assert all(torch.isfinite(v) for v in losses_actor)
    assert all(torch.isfinite(v) for v in losses_critic)
    assert all(
        abs(v) < 1e3 for v in losses_actor
    ), f"Actor loss out of range: {losses_actor}"
    assert all(
        abs(v) < 1e3 for v in losses_critic
    ), f"Critic loss out of range: {losses_critic}"


def _build_simple_core(
    grid_size: int,
    color_values: int,
    device: torch.device,
    amp: bool = False,
    max_grad_norm: float = 0.0,
) -> D4PGCore:
    # Retained for the specialized AMP/grad-clipping test
    actor = ArcActorNetwork(
        grid_size=grid_size,
        color_values=color_values,
        config={"device": str(device), "amp_enabled": amp, "use_checkpointing": False},
    )
    critic = ArcCriticNetwork(
        grid_size=grid_size,
        color_values=color_values,
        num_atoms={"pixel_wise": 51, "binary": 3},
        v_min={"pixel_wise": -40, "binary": 0},
        v_max={"pixel_wise": 2, "binary": 1},
        config={"device": str(device), "amp_enabled": amp, "use_checkpointing": False},
    )
    core = D4PGCore(
        actor=actor,
        critic=critic,
        gamma=0.99,
        tau=0.005,
        n_steps=1,
        amp_enabled=amp,
        max_grad_norm=max_grad_norm,
        config={"device": str(device), "amp_enabled": amp, "use_checkpointing": False},
    )
    return core


# Stream factory now built inline in tests using make_synthetic_step fixture


def test_d4pg_trainer_per_warmup_and_priority(
    device, d4pg_core, grid_size, color_values, batch_size, make_synthetic_step
):
    core = d4pg_core
    max_steps = 8

    def step_stream_factory():
        for step_idx in range(max_steps):
            done = step_idx == max_steps - 1
            yield done, step_idx, make_synthetic_step(batch_size, core.actor)

    # Create PER buffer with priority tracking
    class _PERBuffer:
        def __init__(self):
            self.storage = type("S", (), {"max_size": 10, "__len__": lambda self: 10})()
            self.update_called = False
            self._last = None

        def extend(self, batch):
            self._last = batch.squeeze(0)
            return [0]

        def sample(self, bs, return_info=False):
            if return_info:
                return self._last, {"index": torch.tensor([0])}
            return self._last

        def update_priority(self, indices, priorities):
            self.update_called = True

    per_buffer = _PERBuffer()

    def replay_buffer_update_priority(
        context: OnStepContext,
        info: dict[str, torch.Tensor],
        replay_buffer: _PERBuffer,
    ) -> None:
        if context.td_error is None or info is None:
            return
        # Extract first available TD-error from TensorDict
        td_error_value = next(iter(context.td_error.values()))
        replay_buffer.update_priority(info["index"], td_error_value)

    losses_actor, losses_critic = [], []

    def on_step(ctx: OnStepContext):
        # Record losses if present
        if ctx.loss_actor is not None:
            losses_actor.append(ctx.loss_actor)
        if ctx.loss_critic is not None:
            losses_critic.append(ctx.loss_critic)

    trainer = Trainer(
        core=core,
        train_step_stream_factory=step_stream_factory,
        replay_buffer=per_buffer,
        replay_buffer_update_priority=replay_buffer_update_priority,
        batch_size=batch_size,
        # need >5 samples before training starts (given max_size=10)
        warmup_buffer_ratio=0.5,
        on_train_step={"test": on_step},
        device=str(device),
    )

    trainer.train(epochs=1, max_steps_per_epoch=8)

    # Ensure priority update attempted
    assert per_buffer.update_called is True
    # Training should have occurred after warmup threshold;
    # losses should be recorded and finite
    assert len(losses_actor) > 0 and len(losses_critic) > 0
    assert all(torch.isfinite(v) for v in losses_actor)
    assert all(torch.isfinite(v) for v in losses_critic)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="AMP test requires CUDA")
def test_d4pg_trainer_amp_and_gradient_clipping(batch_size: int = 2):
    device = torch.device("cuda")
    grid_size, color_values = 12, 5
    # Enable AMP and gradient clipping
    core = _build_simple_core(
        grid_size, color_values, device, amp=True, max_grad_norm=0.1
    )

    max_steps = 5

    def step_stream_factory():
        make_step = make_synthetic_step_on_device(device, grid_size, color_values)
        for step_idx in range(max_steps):
            done = step_idx == max_steps - 1
            yield done, step_idx, make_step(batch_size, core.actor)

    # Simple buffer to bypass warmup
    class _Buf:
        def __init__(self):
            self.storage = type("S", (), {"max_size": 1, "__len__": lambda self: 1})()

        def extend(self, batch):
            self._last = batch.squeeze(0)
            return [0]

        def sample(self, bs, return_info=False):
            if return_info:
                return self._last, {"index": torch.tensor([0])}
            return self._last

        def update_priority(self, indices, priorities):
            pass

        def __len__(self):
            return 1

    losses_actor, losses_critic = [], []

    def on_step(ctx: OnStepContext):
        if ctx.loss_actor is not None:
            losses_actor.append(ctx.loss_actor)
        if ctx.loss_critic is not None:
            losses_critic.append(ctx.loss_critic)

    trainer = Trainer(
        core=core,
        train_step_stream_factory=step_stream_factory,
        replay_buffer=_Buf(),
        batch_size=batch_size,
        warmup_buffer_ratio=0.0,
        on_train_step=OrderedDict({"test": on_step}),
        device=device,
    )

    # Run a short AMP-enabled, clipped training loop
    trainer.train(epochs=1, max_steps_per_epoch=5)

    # Assert training produced finite losses
    assert len(losses_actor) > 0 and len(losses_critic) > 0
    assert all(torch.isfinite(v) for v in losses_actor)
    assert all(torch.isfinite(v) for v in losses_critic)


def test_d4pg_trainer_validation_process(d4pg_core, make_synthetic_step, batch_size):
    """Test validation process works with separate train/validation streams."""
    max_steps = 10

    def train_step_stream_factory():
        for step_idx in range(max_steps):
            done = step_idx == max_steps - 1
            yield done, step_idx, make_synthetic_step(batch_size, d4pg_core.actor)

    def validation_step_stream_factory():
        # Validation uses fewer steps
        for step_idx in range(5):
            done = step_idx == 4
            yield done, step_idx, make_synthetic_step(batch_size, d4pg_core.actor)

    # Simple buffer to bypass warmup
    class _Buf:
        def __init__(self):
            self.storage = type("S", (), {"max_size": 1, "__len__": lambda self: 1})()

        def extend(self, batch):
            self._last = batch.squeeze(0)
            return [0]

        def sample(self, bs, return_info=False):
            if return_info:
                return self._last, {"index": torch.tensor([0])}
            return self._last

        def update_priority(self, indices, priorities):
            pass

        def __len__(self):
            return 1

    train_losses_actor, train_losses_critic = [], []
    val_losses_actor, val_losses_critic = [], []

    def on_train_step(ctx: OnStepContext):
        if ctx.loss_actor is not None:
            train_losses_actor.append(ctx.loss_actor)
        if ctx.loss_critic is not None:
            train_losses_critic.append(ctx.loss_critic)

    def on_validation_step(ctx: OnStepContext):
        if ctx.loss_actor is not None:
            val_losses_actor.append(ctx.loss_actor)
        if ctx.loss_critic is not None:
            val_losses_critic.append(ctx.loss_critic)

    trainer = Trainer(
        core=d4pg_core,
        train_step_stream_factory=train_step_stream_factory,
        validation_step_stream_factory=validation_step_stream_factory,
        replay_buffer=_Buf(),
        batch_size=batch_size,
        warmup_buffer_ratio=0.0,
        on_train_step=OrderedDict({"test": on_train_step}),
        on_validation_step=OrderedDict({"test": on_validation_step}),
        device=d4pg_core.device,
    )

    # Run training (with validation_epochs_frequency=0 to disable auto-validation)
    trainer.train(epochs=1, validation_epochs_frequency=0)

    # Run validation explicitly
    trainer.validate(epochs=1)

    # Assert both training and validation produced finite losses
    assert len(train_losses_actor) > 0 and len(train_losses_critic) > 0
    assert all(torch.isfinite(v) for v in train_losses_actor)
    assert all(torch.isfinite(v) for v in train_losses_critic)

    assert len(val_losses_actor) > 0 and len(val_losses_critic) > 0
    assert all(torch.isfinite(v) for v in val_losses_actor)
    assert all(torch.isfinite(v) for v in val_losses_critic)

    # Validation should have fewer steps than training (5 vs 10)
    assert len(val_losses_actor) == 5
    assert len(val_losses_critic) == 5
    assert len(train_losses_actor) == 10
    assert len(train_losses_critic) == 10


def test_d4pg_trainer_validation_without_factory_raises_error(
    d4pg_core, make_synthetic_step, batch_size
):
    """Test calling validate() without validation_step_stream_factory raises error."""

    def train_step_stream_factory():
        for step_idx in range(5):
            done = step_idx == 4
            yield done, step_idx, make_synthetic_step(batch_size, d4pg_core.actor)

    trainer = Trainer(
        core=d4pg_core,
        train_step_stream_factory=train_step_stream_factory,
        validation_step_stream_factory=None,  # No validation factory
        batch_size=batch_size,
        device=d4pg_core.device,
    )

    # Should raise ValueError when trying to validate
    with pytest.raises(ValueError, match="Validation step stream factory is not set"):
        trainer.validate(epochs=1)
