import os
from pathlib import Path

import pytest
import torch

from rlarcworld.agent.actor import ArcActorNetwork
from rlarcworld.agent.critic import ArcCriticNetwork
from rlarcworld.algorithms.d4pg import D4PGCore
from rlarcworld.utils.device import enable_cuda
from rlarcworld.utils.logging_utils import configure_logger

# Configure logger for test session
configure_logger(level=os.environ.get("LOG_LEVEL", "INFO").upper(), json_format=False)


def build_core(grid_size: int = 16, color_values: int = 7) -> D4PGCore:
    config = enable_cuda(amp_enabled=False)
    device = torch.device(config["device"])  # cpu in CI typically

    actor = ArcActorNetwork(
        grid_size=grid_size,
        color_values=color_values,
        config={
            "device": str(device),
            "amp_enabled": config["amp_enabled"],
            "use_checkpointing": False,
        },
    ).to(device)
    actor.eval()

    num_atoms = {"pixel_wise": 50, "binary": 3}
    v_min = {"pixel_wise": -40, "binary": 0}
    v_max = {"pixel_wise": 2, "binary": 1}
    critic = ArcCriticNetwork(
        grid_size=grid_size,
        color_values=color_values,
        num_atoms=num_atoms,
        v_min=v_min,
        v_max=v_max,
        config={
            "device": str(device),
            "amp_enabled": config["amp_enabled"],
            "use_checkpointing": False,
        },
    ).to(device)
    critic.eval()
    critic.z_atoms = critic.z_atoms.to(device)

    core = D4PGCore(
        actor=actor,
        critic=critic,
        gamma=0.95,
        tau=0.01,
        n_steps=1,
        carsm=False,
        entropy_coef=0.0,
        entropy_coef_decay=1.0,
        max_grad_norm=0.0,
        amp_enabled=config["amp_enabled"],
        config=config,
    )
    return core


@pytest.mark.parametrize("from_path", [False, True])
def test_checkpoint_and_load(from_path: bool, tmp_path: Path):
    # Build original core and create a checkpoint
    core1 = build_core()
    ckpt = core1.checkpoint()

    # Sanity: expected keys exist
    assert set(ckpt.keys()) == {
        "actor",
        "critic",
        "actor_optimizer",
        "critic_optimizer",
        "actor_scheduler",
        "critic_scheduler",
        "hyperparameters",
    }

    # Optionally save to disk and prepare load input
    load_input: dict | Path
    if from_path:
        path = tmp_path / "d4pg_core_ckpt.pt"
        core1.save_checkpoint(path)
        load_input = path
    else:
        load_input = ckpt

    # Build a fresh core and load
    core2 = build_core()

    # Target networks should be refreshed to equal online networks
    def state_dict_equal(m1: torch.nn.Module, m2: torch.nn.Module) -> bool:
        for (k1, v1), (k2, v2) in zip(m1.state_dict().items(), m2.state_dict().items()):
            if k1 != k2:
                return False
            try:
                torch.testing.assert_close(v1, v2)
            except AssertionError:
                return False
        return True

    # Verify parameters are different before loading (likely true for random init)
    def first_n_params(module, n=3):
        return [p.detach().clone() for p in module.parameters() if p.requires_grad][:n]

    assert not state_dict_equal(core1.actor, core2.actor)
    assert not state_dict_equal(core1.critic, core2.critic)

    core2.load(load_input)

    # After load, params should match core1's
    actor_after = first_n_params(core2.actor)
    critic_after = first_n_params(core2.critic)
    actor_ref = first_n_params(core1.actor)
    critic_ref = first_n_params(core1.critic)

    for a, b in zip(actor_after, actor_ref):
        torch.testing.assert_close(a, b)
    for a, b in zip(critic_after, critic_ref):
        torch.testing.assert_close(a, b)

    assert state_dict_equal(core2.actor, core1.actor)
    assert state_dict_equal(core2.critic, core1.critic)
    assert state_dict_equal(core2.actor, core2.actor_target)
    assert state_dict_equal(core2.critic, core2.critic_target)

    # Hyperparameters restored: check a subset
    hp = ckpt["hyperparameters"]
    assert pytest.approx(core2.gamma) == hp["gamma"]
    assert pytest.approx(core2.tau) == hp["tau"]
    assert pytest.approx(core2.entropy_coef) == hp["entropy_coef"]
    assert core2.carsm == hp["carsm"]

    # Learning rates restored
    assert pytest.approx(core2.actor_optimizer.param_groups[0]["lr"]) == hp["actor_lr"]
    assert (
        pytest.approx(core2.critic_optimizer.param_groups[0]["lr"]) == hp["critic_lr"]
    )
