import torch
import pytest
from tensordict import TensorDict

from rlarcworld.agent.actor import ArcActorNetwork
from rlarcworld.agent.critic import ArcCriticNetwork
from rlarcworld.algorithms.d4pg import D4PGCore
from rlarcworld.algorithms.d4pg_trainer import D4PGTrainer
from torchrl.data.replay_buffers import TensorDictReplayBuffer, LazyTensorStorage, PrioritizedSampler


def _make_synthetic_step(batch_size: int, grid_size: int, color_values: int, device: torch.device, actor: ArcActorNetwork) -> TensorDict:
    state = TensorDict(
        {
            "last_grid": torch.randint(0, color_values, (batch_size, 1, grid_size, grid_size), device=device),
            "grid": torch.randint(0, color_values, (batch_size, 1, grid_size, grid_size), device=device),
            "examples": torch.randint(0, color_values, (batch_size, 10, 2, grid_size, grid_size), device=device),
            "initial": torch.randint(0, color_values, (batch_size, 1, grid_size, grid_size), device=device),
            "index": torch.randint(0, grid_size, (batch_size, 1, grid_size, grid_size), device=device),
            "terminated": torch.zeros((batch_size, 1), device=device, dtype=torch.long),
        }
    ).to(device)

    next_state = state.clone()

    reward = TensorDict(
        {
            "pixel_wise": torch.randint(-40, 2, size=(batch_size, 1), device=device).to(torch.float32),
            "binary": torch.randint(0, 2, size=(batch_size, 1), device=device).to(torch.float32),
        }
    )

    with torch.no_grad():
        actions = actor(state)

    step_state = TensorDict(
        {
            "state": state,
            "actions": actions,
            "reward": reward,
            "next_state": next_state,
            "terminated": torch.zeros((batch_size, 1), device=device),
        },
        batch_size=[batch_size],
    ).to(device)
    return step_state


@pytest.mark.parametrize("batch_size", [2])
def test_d4pg_trainer_training_updates_and_loss_ranges(device, batch_size):
    grid_size = 16
    color_values = 7

    actor = ArcActorNetwork(
        grid_size=grid_size,
        color_values=color_values,
        config={"device": str(device), "amp_enabled": False, "use_checkpointing": False},
    )
    critic = ArcCriticNetwork(
        grid_size=grid_size,
        color_values=color_values,
        num_atoms={"pixel_wise": 51, "binary": 3},
        v_min={"pixel_wise": -40, "binary": 0},
        v_max={"pixel_wise": 2, "binary": 1},
        config={"device": str(device), "amp_enabled": False, "use_checkpointing": False},
    )

    core = D4PGCore(
        actor=actor,
        critic=critic,
        gamma=0.99,
        tau=0.005,
        n_steps=1,
        amp_enabled=False,
        config={"device": str(device), "amp_enabled": False, "use_checkpointing": False},
    )

    # Simple step stream that yields a fixed number of synthetic steps
    max_steps = 6
    def step_stream_factory(epoch: int):
        for _ in range(max_steps):
            yield _make_synthetic_step(batch_size, grid_size, color_values, device, actor)

    losses_actor = []
    losses_critic = []

    def on_step(epoch, step, loss_actor, loss_critic, batch, decay_flag):
        if loss_actor is not None:
            losses_actor.append(loss_actor)
        if loss_critic is not None:
            losses_critic.append(loss_critic)

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
        def sample(self, bs):
            return self._last
        def update_priority(self, indices, priorities):
            pass

    trainer = D4PGTrainer(
        core=core,
        step_stream_factory=step_stream_factory,
        replay_buffer=_MiniBuffer(),
        batch_size=batch_size,
        warmup_buffer_ratio=0.0,
        target_update_frequency=2,
        on_step=on_step,
        device=str(device),
    )

    # Snapshot some params before training
    actor_before = [p.detach().clone() for p in core.actor.parameters() if p.requires_grad][:3]
    critic_before = [p.detach().clone() for p in core.critic.parameters() if p.requires_grad][:3]

    trainer.train(epochs=1, max_steps_per_epoch=max_steps)

    # Ensure some updates occurred
    actor_changed = any(
        not torch.allclose(pb, pa)
        for pb, pa in zip(actor_before, [p.detach().clone() for p in core.actor.parameters() if p.requires_grad][:3])
    )
    critic_changed = any(
        not torch.allclose(pb, pa)
        for pb, pa in zip(critic_before, [p.detach().clone() for p in core.critic.parameters() if p.requires_grad][:3])
    )
    assert actor_changed, "Actor parameters did not update in trainer loop"
    assert critic_changed, "Critic parameters did not update in trainer loop"

    # Losses recorded: finite and reasonable ranges (broad sanity bounds)
    assert all(torch.isfinite(torch.tensor(v)) for v in losses_actor)
    assert all(torch.isfinite(torch.tensor(v)) for v in losses_critic)


def _snapshot_params(module: torch.nn.Module, limit: int = 5):
    return [p.detach().clone().cpu() for i, p in enumerate(module.parameters()) if p.requires_grad and i < limit]


def _params_changed(before, module: torch.nn.Module) -> bool:
    after = [p.detach().clone().cpu() for i, p in enumerate(module.parameters()) if p.requires_grad and i < len(before)]
    return any(not torch.allclose(b, a) for b, a in zip(before, after))


def test_d4pg_trainer_real_replay_buffer_and_target_updates(device):
    grid_size, color_values, batch_size = 10, 5, 2
    core = _build_simple_core(grid_size, color_values, device, amp=False)

    # Real replay buffer from torchrl
    capacity = 50
    rb = TensorDictReplayBuffer(
        storage=LazyTensorStorage(capacity=capacity),
        sampler=PrioritizedSampler(max_capacity=capacity, alpha=0.6, beta=0.4),
    )

    max_steps = 7
    step_stream_factory = _make_stream_factory(max_steps, batch_size, grid_size, color_values, device, core.actor)

    target_update_frequency = 2
    updates_detected = {"actor": 0, "critic": 0}
    last_checked_step = {"actor": -1, "critic": -1}
    # Snapshots to compare across expected update boundaries
    actor_tgt_prev = _snapshot_params(core.actor_target)
    critic_tgt_prev = _snapshot_params(core.critic_target)
    # Track losses too
    losses_actor, losses_critic = [], []

    def on_step(epoch, step, loss_actor, loss_critic, batch, decay_flag):
        # Only check when a training step happened
        if not decay_flag:
            return
        # Record losses
        if loss_actor is not None:
            losses_actor.append(loss_actor)
        if loss_critic is not None:
            losses_critic.append(loss_critic)
        # Expected update boundary
        if step % target_update_frequency == 0:
            # detect change
            nonlocal actor_tgt_prev, critic_tgt_prev
            if _params_changed(actor_tgt_prev, core.actor_target):
                updates_detected["actor"] += 1
                last_checked_step["actor"] = step
                actor_tgt_prev = _snapshot_params(core.actor_target)
            if _params_changed(critic_tgt_prev, core.critic_target):
                updates_detected["critic"] += 1
                last_checked_step["critic"] = step
                critic_tgt_prev = _snapshot_params(core.critic_target)

    trainer = D4PGTrainer(
        core=core,
        step_stream_factory=step_stream_factory,
        replay_buffer=rb,
        batch_size=batch_size,
        warmup_buffer_ratio=0.0,
        target_update_frequency=target_update_frequency,
        on_step=on_step,
        device=str(device),
    )

    trainer.train(epochs=1, max_steps_per_epoch=max_steps)

    # We expect at least 1-2 target updates over 7 steps with freq=2 (at steps 0,2,4,6)
    assert updates_detected["actor"] >= 1
    assert updates_detected["critic"] >= 1
    # Sanity ranges to catch divergence; these are intentionally broad
    assert len(losses_actor) > 0 and len(losses_critic) > 0
    assert all(torch.isfinite(torch.tensor(v)) for v in losses_actor)
    assert all(torch.isfinite(torch.tensor(v)) for v in losses_critic)
    assert all(abs(v) < 1e3 for v in losses_actor), f"Actor loss out of range: {losses_actor}"
    assert all(abs(v) < 1e3 for v in losses_critic), f"Critic loss out of range: {losses_critic}"


def _make_per_buffer(max_size: int):
    class _Storage:
        def __init__(self):
            self.max_size = max_size
            self._len = 0
        def __len__(self):
            return self._len
    class _Sampler:
        def __init__(self):
            self._max_priority = 1.0
    class _PERBuffer:
        def __init__(self):
            self.storage = _Storage()
            self.sampler = _Sampler()
            self._last = None
            self.update_called = False
        def extend(self, batch):
            self._last = batch.squeeze(0)
            self.storage._len += 1
            return [self.storage._len - 1]
        def sample(self, bs):
            return self._last
        def update_priority(self, indices, priorities):
            self.update_called = True
    return _PERBuffer()


def _build_simple_core(grid_size: int, color_values: int, device: torch.device, amp: bool = False, max_grad_norm: float = 0.0) -> D4PGCore:
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


def _make_stream_factory(max_steps: int, batch_size: int, grid_size: int, color_values: int, device: torch.device, actor: ArcActorNetwork):
    def _factory(epoch: int):
        for _ in range(max_steps):
            yield _make_synthetic_step(batch_size, grid_size, color_values, device, actor)
    return _factory


def test_d4pg_trainer_per_warmup_and_priority(device):
    grid_size, color_values, batch_size = 12, 5, 2
    core = _build_simple_core(grid_size, color_values, device, amp=False)
    step_stream_factory = _make_stream_factory(8, batch_size, grid_size, color_values, device, core.actor)

    per_buffer = _make_per_buffer(max_size=10)
    losses_actor, losses_critic = [], []
    def on_step(epoch, step, loss_actor, loss_critic, batch, decay_flag):
        # Record losses if present
        if loss_actor is not None:
            losses_actor.append(loss_actor)
        if loss_critic is not None:
            losses_critic.append(loss_critic)

    trainer = D4PGTrainer(
        core=core,
        step_stream_factory=step_stream_factory,
        replay_buffer=per_buffer,
        batch_size=batch_size,
        warmup_buffer_ratio=0.5,  # need >5 samples before training starts (given max_size=10)
        target_update_frequency=2,
        on_step=on_step,
        device=str(device),
    )

    trainer.train(epochs=1, max_steps_per_epoch=8)

    # Ensure priority update attempted
    assert per_buffer.update_called is True
    # Training should have occurred after warmup threshold; losses should be recorded and finite
    assert len(losses_actor) > 0 and len(losses_critic) > 0
    assert all(torch.isfinite(torch.tensor(v)) for v in losses_actor)
    assert all(torch.isfinite(torch.tensor(v)) for v in losses_critic)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="AMP test requires CUDA")
def test_d4pg_trainer_amp_and_gradient_clipping(batch_size: int = 2):
    device = torch.device("cuda")
    grid_size, color_values = 12, 5
    # Enable AMP and gradient clipping
    core = _build_simple_core(grid_size, color_values, device, amp=True, max_grad_norm=0.1)

    step_stream_factory = _make_stream_factory(5, batch_size, grid_size, color_values, device, core.actor)

    # Simple buffer to bypass warmup
    class _Buf:
        def __init__(self):
            self.storage = type("S", (), {"max_size": 1, "__len__": lambda self: 1})()
        def extend(self, batch):
            self._last = batch.squeeze(0)
            return [0]
        def sample(self, bs):
            return self._last
        def update_priority(self, indices, priorities):
            pass

    losses_actor, losses_critic = [], []
    def on_step(epoch, step, loss_actor, loss_critic, batch, decay_flag):
        if loss_actor is not None:
            losses_actor.append(loss_actor)
        if loss_critic is not None:
            losses_critic.append(loss_critic)

    trainer = D4PGTrainer(
        core=core,
        step_stream_factory=step_stream_factory,
        replay_buffer=_Buf(),
        batch_size=batch_size,
        warmup_buffer_ratio=0.0,
        target_update_frequency=1,
        on_step=on_step,
        device=str(device),
    )

    # Run a short AMP-enabled, clipped training loop
    trainer.train(epochs=1, max_steps_per_epoch=5)

    # Assert training produced finite losses
    assert len(losses_actor) > 0 and len(losses_critic) > 0
    assert all(torch.isfinite(torch.tensor(v)) for v in losses_actor)
    assert all(torch.isfinite(torch.tensor(v)) for v in losses_critic)
