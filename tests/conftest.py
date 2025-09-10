import os
import pytest
import torch

from rlarcworld.agent.actor import ArcActorNetwork
from rlarcworld.agent.critic import ArcCriticNetwork
from rlarcworld.algorithms.d4pg import D4PGCore
from rlarcworld.algorithms.d4pg_trainer import D4PGTrainer
from torchrl.data.replay_buffers import TensorDictReplayBuffer, LazyTensorStorage, PrioritizedSampler
from tests.utils.synthetic import make_synthetic_step_on_device
from rlarcworld.environments.arc_batch_grid_env import ArcBatchGridEnv
from rlarcworld.environments.wrappers.rewards import PixelAwareRewardWrapper
from rlarcworld.arc_dataset import ArcDataset, ArcSampleTransformer
from torch.utils.data import DataLoader


@pytest.fixture(scope="session")
def device():
    # Force CPU for stable tests across environments during migration
    return torch.device("cpu")


@pytest.fixture(autouse=True)
def _set_log_level_env(monkeypatch):
    # Ensure consistent logging level during tests
    monkeypatch.setenv("LOG_LEVEL", os.environ.get("LOG_LEVEL", "INFO"))
    yield


# ---------- Basic dimension fixtures ----------

@pytest.fixture(scope="session")
def grid_size() -> int:
    return 16


@pytest.fixture(scope="session")
def color_values() -> int:
    return 11


@pytest.fixture(scope="session")
def batch_size() -> int:
    return 2


# ---------- Model fixtures ----------

@pytest.fixture()
def actor(device, grid_size, color_values) -> ArcActorNetwork:
    return ArcActorNetwork(
        grid_size=grid_size,
        color_values=color_values,
        config={"device": str(device), "amp_enabled": False, "use_checkpointing": False},
    )


@pytest.fixture()
def critic(device, grid_size, color_values) -> ArcCriticNetwork:
    num_atoms = {"pixel_wise": 51, "binary": 3}
    v_min = {"pixel_wise": -40, "binary": 0}
    v_max = {"pixel_wise": 2, "binary": 1}
    return ArcCriticNetwork(
        grid_size=grid_size,
        color_values=color_values,
        num_atoms=num_atoms,
        v_min=v_min,
        v_max=v_max,
        config={"device": str(device), "amp_enabled": False, "use_checkpointing": False},
    )


@pytest.fixture()
def d4pg_core(actor: ArcActorNetwork, critic: ArcCriticNetwork, device) -> D4PGCore:
    return D4PGCore(
        actor=actor,
        critic=critic,
        gamma=0.99,
        tau=0.005,
        n_steps=1,
        amp_enabled=False,
        config={"device": str(device), "amp_enabled": False, "use_checkpointing": False},
    )


# ---------- Replay buffer fixtures ----------

@pytest.fixture()
def per_replay_buffer() -> TensorDictReplayBuffer:
    capacity = 50
    return TensorDictReplayBuffer(
        storage=LazyTensorStorage(capacity),
        sampler=PrioritizedSampler(max_capacity=capacity, alpha=0.6, beta=0.4),
    )


# ---------- Utility: synthetic step -----------

@pytest.fixture()
def make_synthetic_step(device, grid_size, color_values):
    """Return a callable generating a synthetic step on the current test device."""
    return make_synthetic_step_on_device(device, grid_size, color_values)


@pytest.fixture()
def trainer_factory():
    """Factory to construct a D4PGTrainer with common defaults for tests.

    Usage:
        trainer = trainer_factory(
            core=core,
            step_stream_factory=stream_fn,
            replay_buffer=rb,
            batch_size=bs,
            warmup_buffer_ratio=0.0,
            target_update_frequency=1,
            on_step=None,
            device="cpu",
        )
    """
    def _make(
        *,
        core: D4PGCore,
        step_stream_factory,
        replay_buffer,
        batch_size: int,
        warmup_buffer_ratio: float = 0.0,
        target_update_frequency: int = 1,
        on_step=None,
        device: str | torch.device = "cpu",
    ) -> D4PGTrainer:
        return D4PGTrainer(
            core=core,
            step_stream_factory=step_stream_factory,
            replay_buffer=replay_buffer,
            batch_size=batch_size,
            warmup_buffer_ratio=warmup_buffer_ratio,
            target_update_frequency=target_update_frequency,
            on_step=on_step,
            device=str(device),
        )

    return _make


# ---------- Environment fixtures (parametrized over n_steps) ----------

@pytest.fixture(params=[1, 3])
def n_steps_param(request) -> int:
    return int(request.param)


@pytest.fixture()
def env(grid_size, color_values, n_steps_param, device) -> ArcBatchGridEnv:
    return ArcBatchGridEnv(
        grid_size=grid_size,
        color_values=color_values,
        n_steps=n_steps_param,
        gamma=0.99,
        device=str(device),
    )


@pytest.fixture()
def wrapped_env(env: ArcBatchGridEnv, n_steps_param: int) -> PixelAwareRewardWrapper:
    return PixelAwareRewardWrapper(
        env,
        n_steps=n_steps_param,
        gamma=0.99,
        apply_clamp=False,
        v_min=0,
        v_max=1,
    )


# ---------- Dataset and DataLoader fixtures ----------

@pytest.fixture()
def dataset() -> ArcDataset:
    # Raw dataset without transform
    return ArcDataset(arc_dataset_dir="tests/test_data", keep_in_memory=False)


@pytest.fixture()
def transformed_dataset(grid_size) -> ArcDataset:
    # Build a simple transform with square output using grid_size
    out_size = (grid_size, grid_size)
    n_examples = 5
    transform = ArcSampleTransformer(out_size, n_examples, zero_based_correction=0)
    return ArcDataset(
        arc_dataset_dir="tests/test_data",
        keep_in_memory=False,
        transform=transform,
    )


@pytest.fixture()
def train_loader(transformed_dataset, batch_size) -> DataLoader:
    return DataLoader(transformed_dataset, batch_size=batch_size, shuffle=True, num_workers=0)


@pytest.fixture()
def val_loader(transformed_dataset) -> DataLoader:
    # Deterministic evaluation loader
    return DataLoader(transformed_dataset, batch_size=1, shuffle=False, num_workers=0)
