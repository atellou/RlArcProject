import os
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict

from rlarcworld.algorithms.d4pg import D4PGCore
from rlarcworld.common.enums import StateKeys
from rlarcworld.models.actor import ViTARCActor
from rlarcworld.models.critic import ArcCriticNetwork
from rlarcworld.nn_modules import RlArcWorldViTARCConfig
from rlarcworld.utils.device import enable_cuda
from rlarcworld.utils.logging_utils import configure_logger

# Configure logger for test session
configure_logger(level=os.environ.get("LOG_LEVEL", "INFO").upper(), json_format=False)


@pytest.fixture(params=[False, True])
def core_setup(request):
    amp_flag = bool(request.param)
    config = enable_cuda(amp_enabled=amp_flag)
    device = torch.device(
        config["device"]
    )  # "cuda" when available and amp enabled, else cpu
    grid_size = 30
    color_values = 11
    batch_size = 4

    actor_config = RlArcWorldViTARCConfig(
        grid_size=grid_size,
        d_model=64,
        d_ff=128,
        num_layers=2,
        num_heads=4,
        use_ope=False,
        grids_to_encode=[StateKeys.GRID.value],
    )
    actor = ViTARCActor(
        color_values=color_values,
        rl_vit_arc_config=actor_config,
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
    )
    critic.eval()
    # Ensure critic.z_atoms lives on the same device
    critic.z_atoms = critic.z_atoms.to(device)

    core = D4PGCore(
        actor=actor,
        critic=critic,
        gamma=0.99,
        tau=0.005,
        n_steps=1,
        carsm=False,
        entropy_coef=0.0,
        entropy_coef_decay=1.0,
        max_grad_norm=0.0,
        amp_enabled=config["amp_enabled"],
        config=config,
    )

    # Create synthetic batch
    state = TensorDict(
        {
            "last_grid": torch.randint(
                0, color_values, (batch_size, 1, grid_size, grid_size), device=device
            ),
            "grid": torch.randint(
                0, color_values, (batch_size, 1, grid_size, grid_size), device=device
            ),
            "examples": torch.randint(
                0,
                color_values,
                (batch_size, 10, 2, grid_size, grid_size),
                device=device,
            ),
            "initial": torch.randint(
                0, color_values, (batch_size, 1, grid_size, grid_size), device=device
            ),
            "index": torch.randint(
                0, grid_size, (batch_size, 1, grid_size, grid_size), device=device
            ),
            "terminated": torch.zeros((batch_size, 1), device=device, dtype=torch.long),
        }
    ).to(device)

    next_state = state.clone()

    reward = TensorDict(
        {
            "pixel_wise": torch.randint(-40, 2, size=(batch_size, 1), device=device).to(
                torch.float32
            ),
            "binary": torch.randint(0, 2, size=(batch_size, 1), device=device).to(
                torch.float32
            ),
        }
    )
    done = torch.zeros((batch_size, 1), device=device)

    return core, state, next_state, reward, done, batch_size, num_atoms


def build_core(grid_size: int = 16, color_values: int = 7) -> D4PGCore:
    config = enable_cuda(amp_enabled=False)
    device = torch.device(config["device"])  # cpu in CI typically

    actor_config = RlArcWorldViTARCConfig(
        grid_size=grid_size,
        d_model=64,
        d_ff=128,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        ape_type="SinusoidalAPE2D",
        rpe_type="Two-slope-Alibi",
        use_ope=False,
        ape_mixer="default",
    )

    actor = ViTARCActor(
        color_values=color_values,
        rl_vit_arc_config=actor_config,
    ).to(config["device"])
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
        "schema_version",
        "class",
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


def test_compute_target_distribution(core_setup):
    core, state, next_state, reward, done, batch_size, num_atoms = core_setup

    target = core.compute_target_distribution(reward, next_state, done)

    # Keys and shapes
    assert set(target.keys()) == {"pixel_wise", "binary"}  # expected keys
    assert target["pixel_wise"].shape == (
        batch_size,
        num_atoms["pixel_wise"],
    )  # atoms dim
    assert target["binary"].shape == (batch_size, num_atoms["binary"])  # atoms dim

    # Valid distribution: sum to 1 and within [0,1]
    for k, dist in target.items():
        torch.testing.assert_close(
            torch.sum(dist, dim=1), torch.ones(batch_size, device=dist.device)
        )
        assert torch.all(dist >= 0)
        assert torch.all(dist <= 1)


def test_compute_losses(core_setup):
    core, state, next_state, reward, done, batch_size, _ = core_setup

    # Actions from actor
    with torch.no_grad():
        action_probs = core.actor(state)

    # Target dist
    target = core.compute_target_distribution(reward, next_state, done)

    # Critic loss
    loss_td, td_err, q_dist = core.compute_critic_loss(
        state, action_probs, target, compute_td_error=True
    )
    assert set(loss_td.keys()) == {"pixel_wise", "binary"}  # both heads present
    # Losses are finite
    for k, v in loss_td.items():
        assert torch.isfinite(v).all(), f"Critic loss for {k} not finite: {v}"
    for k, dist in q_dist.items():
        assert torch.isfinite(
            dist
        ).all(), f"Q distribution for {k} has non-finite values"
    # td_err is a TensorDict; check each underlying tensor
    for k, v in td_err.items():
        assert torch.isfinite(v).all(), f"TD error for {k} has non-finite values"

    # Actor loss
    loss_actor_td = core.compute_actor_loss(state, target_q=target)
    assert set(loss_actor_td.keys()) == {"pixel_wise", "binary"}  # both heads present
    for k, v in loss_actor_td.items():
        assert torch.isfinite(v).all(), f"Actor loss for {k} not finite: {v}"

    # Backward one step end-to-end
    loss_actor = sum(loss_actor_td.values()) / 2
    loss_critic = sum(loss_td.values()) / 2

    core.actor_optimizer.zero_grad()
    loss_actor.backward(retain_graph=True)
    # Gradients should flow to actor params
    actor_total_grad = 0.0
    for p in core.actor.parameters():
        if p.grad is not None:
            actor_total_grad += float(p.grad.abs().sum().item())
    assert actor_total_grad > 0.0, "No gradients flowed to actor parameters"
    core.actor_optimizer.step()

    core.critic_optimizer.zero_grad()
    loss_critic.backward()
    # Gradients should flow to critic params
    critic_total_grad = 0.0
    for p in core.critic.parameters():
        if p.grad is not None:
            critic_total_grad += float(p.grad.abs().sum().item())
    assert critic_total_grad > 0.0, "No gradients flowed to critic parameters"
    core.critic_optimizer.step()


def test_compute_loss_training_updates_and_finite(core_setup):
    core, state, next_state, reward, done, batch_size, _ = core_setup

    # Prepare batch with actions from actor
    with torch.no_grad():
        actions = core.actor(state)

    batch = TensorDict(
        {
            "state": state,
            "actions": actions,
            "reward": reward,
            "next_state": next_state,
            "terminated": done,
        }
    ).to(core.device)

    # Snapshot a couple of parameters to detect updates
    actor_params_before = [
        p.detach().clone() for p in core.actor.parameters() if p.requires_grad
    ][:3]
    critic_params_before = [
        p.detach().clone() for p in core.critic.parameters() if p.requires_grad
    ][:3]

    loss_actor, loss_critic, td_error = core.compute_loss(
        batch,
        training=True,
    )

    # Losses are finite
    assert torch.isfinite(loss_actor).all()
    assert torch.isfinite(loss_critic).all()

    # At least one parameter changed for actor and critic
    actor_changed = False
    for p_before, p_after in zip(
        actor_params_before,
        [p.detach().clone() for p in core.actor.parameters() if p.requires_grad][:3],
    ):
        if not torch.allclose(p_before, p_after):
            actor_changed = True
            break
    critic_changed = False
    for p_before, p_after in zip(
        critic_params_before,
        [p.detach().clone() for p in core.critic.parameters() if p.requires_grad][:3],
    ):
        if not torch.allclose(p_before, p_after):
            critic_changed = True
            break
    assert (
        actor_changed
    ), "Actor parameters did not update during compute_loss(training=True)"
    assert (
        critic_changed
    ), "Critic parameters did not update during compute_loss(training=True)"


def test_compute_loss_eval_no_update(core_setup):
    core, state, next_state, reward, done, _, _ = core_setup

    # Prepare batch with actions from actor
    with torch.no_grad():
        actions = core.actor(state)

    batch = TensorDict(
        {
            "state": state,
            "actions": actions,
            "reward": reward,
            "next_state": next_state,
            "terminated": done,
        }
    ).to(core.device)

    # Snapshot parameters to ensure no update
    actor_params_before = [
        p.detach().clone() for p in core.actor.parameters() if p.requires_grad
    ][:3]
    critic_params_before = [
        p.detach().clone() for p in core.critic.parameters() if p.requires_grad
    ][:3]

    loss_actor, loss_critic, td_error = core.compute_loss(
        batch,
        training=False,
    )

    # Losses are finite
    assert torch.isfinite(loss_actor).all()
    assert torch.isfinite(loss_critic).all()

    # Parameters should not change
    for p_before, p_after in zip(
        actor_params_before,
        [p.detach().clone() for p in core.actor.parameters() if p.requires_grad][:3],
    ):
        torch.testing.assert_close(p_before, p_after)
    for p_before, p_after in zip(
        critic_params_before,
        [p.detach().clone() for p in core.critic.parameters() if p.requires_grad][:3],
    ):
        torch.testing.assert_close(p_before, p_after)
