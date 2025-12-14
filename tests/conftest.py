import os

import pytest
import torch
from torch.utils.data import DataLoader
from torchrl.data.replay_buffers import (
    LazyTensorStorage,
    PrioritizedSampler,
    TensorDictReplayBuffer,
)

from rlarcworld.algorithms.d4pg import D4PGCore
from rlarcworld.arc_dataset import ArcDataset, ArcSampleTransformer
from rlarcworld.common.configs import RlArcWorldViTARCConfig
from rlarcworld.common.enums import RewardKeys, StateKeys
from rlarcworld.environments.arc_batch_grid_env import ArcBatchGridEnv
from rlarcworld.environments.wrappers.pixel_aware_reward import PixelAwareRewardWrapper
from rlarcworld.models.actor import ViTARCActor
from rlarcworld.models.critic import VitARCCritic
from rlarcworld.runners.core import Trainer
from tests.utils.synthetic import make_synthetic_step_on_device


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


@pytest.fixture(scope="session")
def rl_vit_arc_config(grid_size) -> RlArcWorldViTARCConfig:
    """Shared ViTARC config for actor/critic tests."""
    return RlArcWorldViTARCConfig(
        grid_size=grid_size,
        d_model=64,
        d_ff=128,
        num_layers=2,
        num_heads=4,
        use_ope=False,
        grids_to_encode=[StateKeys.GRID.value],
    )


@pytest.fixture()
def actor(device, color_values, rl_vit_arc_config) -> ViTARCActor:
    return ViTARCActor(
        color_values=color_values,
        rl_vit_arc_config=rl_vit_arc_config,
        config={
            "device": str(device),
            "amp_enabled": False,
            "use_checkpointing": False,
        },
    )


@pytest.fixture()
def critic(device, color_values, rl_vit_arc_config) -> VitARCCritic:
    num_atoms = {RewardKeys.PIXEL_WISE.value: 51, RewardKeys.BINARY.value: 3}
    v_min = {RewardKeys.PIXEL_WISE.value: -40, RewardKeys.BINARY.value: 0}
    v_max = {RewardKeys.PIXEL_WISE.value: 2, RewardKeys.BINARY.value: 1}
    return VitARCCritic(
        color_values=color_values,
        rl_vit_arc_config=rl_vit_arc_config,
        num_atoms=num_atoms,
        v_min=v_min,
        v_max=v_max,
        config={
            "device": str(device),
            "amp_enabled": False,
            "use_checkpointing": False,
        },
    )


@pytest.fixture()
def d4pg_core(actor: ViTARCActor, critic: VitARCCritic, device) -> D4PGCore:
    return D4PGCore(
        actor=actor,
        critic=critic,
        gamma=0.99,
        tau=0.005,
        n_steps=1,
        amp_enabled=False,
        config={
            "device": str(device),
            "amp_enabled": False,
            "use_checkpointing": False,
        },
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
            train_step_stream_factorystream_fn,
            replay_buffer=rb,
            batch_size=bs,
            warmup_buffer_ratio=0.0,
            on_train_step=None,
            on_exception=None,
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
        on_train_step=None,
        on_exception=None,
        device: str | torch.device = "cpu",
    ) -> Trainer:
        return Trainer(
            core=core,
            train_step_stream_factory=step_stream_factory,
            replay_buffer=replay_buffer,
            batch_size=batch_size,
            warmup_buffer_ratio=warmup_buffer_ratio,
            on_train_step=on_train_step,
            on_exception=on_exception,
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
    return PixelAwareRewardWrapper(env, n_steps=n_steps_param, gamma=0.99)


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
    return DataLoader(
        transformed_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )


@pytest.fixture()
def val_loader(transformed_dataset) -> DataLoader:
    # Deterministic evaluation loader
    return DataLoader(transformed_dataset, batch_size=1, shuffle=False, num_workers=0)
