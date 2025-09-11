import os
import torch
import pytest
from tensordict import TensorDict

from rlarcworld.agent.actor import ArcActorNetwork
from rlarcworld.agent.critic import ArcCriticNetwork
from rlarcworld.algorithms.d4pg import D4PGCore
from rlarcworld.utils.configure_logger import configure_logger
from rlarcworld.utils.device import enable_cuda


# Configure logger for test session
configure_logger(level=os.environ.get("LOG_LEVEL", "INFO").upper(), json_format=False)


@pytest.fixture(params=[False, True])
def core_setup(request):
    amp_flag = bool(request.param)
    config = enable_cuda(amp_enabled=amp_flag)
    device = torch.device(config["device"])  # "cuda" when available and amp enabled, else cpu
    grid_size = 30
    color_values = 11
    batch_size = 4

    actor = ArcActorNetwork(
        grid_size=grid_size,
        color_values=color_values,
        config={"device": str(device), "amp_enabled": config["amp_enabled"], "use_checkpointing": False},
    )
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
        config={"device": str(device), "amp_enabled": config["amp_enabled"], "use_checkpointing": False},
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
    done = torch.zeros((batch_size, 1), device=device)

    return core, state, next_state, reward, done, batch_size, num_atoms


def test_compute_target_distribution(core_setup):
    core, state, next_state, reward, done, batch_size, num_atoms = core_setup

    target = core.compute_target_distribution(reward, next_state, done)

    # Keys and shapes
    assert set(target.keys()) == set(["pixel_wise", "binary"])  # expected keys
    assert target["pixel_wise"].shape == (batch_size, num_atoms["pixel_wise"])  # atoms dim
    assert target["binary"].shape == (batch_size, num_atoms["binary"])  # atoms dim

    # Valid distribution: sum to 1 and within [0,1]
    for k, dist in target.items():
        torch.testing.assert_close(torch.sum(dist, dim=1), torch.ones(batch_size, device=dist.device))
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
    loss_td, td_err, q_dist = core.compute_critic_loss(state, action_probs, target, compute_td_error=True)
    assert set(loss_td.keys()) == set(["pixel_wise", "binary"])  # both heads present
    # Losses are finite
    for k, v in loss_td.items():
        assert torch.isfinite(v).all(), f"Critic loss for {k} not finite: {v}"
    for k, dist in q_dist.items():
        assert torch.isfinite(dist).all(), f"Q distribution for {k} has non-finite values"
    # td_err is a TensorDict; check each underlying tensor
    for k, v in td_err.items():
        assert torch.isfinite(v).all(), f"TD error for {k} has non-finite values"

    # Actor loss
    loss_actor_td = core.compute_actor_loss(state, target_q=target)
    assert set(loss_actor_td.keys()) == set(["pixel_wise", "binary"])  # both heads present
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
    actor_params_before = [p.detach().clone() for p in core.actor.parameters() if p.requires_grad][:3]
    critic_params_before = [p.detach().clone() for p in core.critic.parameters() if p.requires_grad][:3]

    loss_actor, loss_critic = core.compute_loss(
        batch, compute_td_error=True, training=True, update_actor=True, update_critic=True
    )

    # Losses are finite
    assert torch.isfinite(loss_actor).all()
    assert torch.isfinite(loss_critic).all()

    # At least one parameter changed for actor and critic
    actor_changed = False
    for p_before, p_after in zip(actor_params_before, [p.detach().clone() for p in core.actor.parameters() if p.requires_grad][:3]):
        if not torch.allclose(p_before, p_after):
            actor_changed = True
            break
    critic_changed = False
    for p_before, p_after in zip(critic_params_before, [p.detach().clone() for p in core.critic.parameters() if p.requires_grad][:3]):
        if not torch.allclose(p_before, p_after):
            critic_changed = True
            break
    assert actor_changed, "Actor parameters did not update during compute_loss(training=True)"
    assert critic_changed, "Critic parameters did not update during compute_loss(training=True)"


def test_compute_loss_eval_no_update(core_setup):
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

    # Snapshot parameters to ensure no update
    actor_params_before = [p.detach().clone() for p in core.actor.parameters() if p.requires_grad][:3]
    critic_params_before = [p.detach().clone() for p in core.critic.parameters() if p.requires_grad][:3]

    loss_actor, loss_critic = core.compute_loss(
        batch, compute_td_error=False, training=False, update_actor=False, update_critic=False
    )

    # Losses are finite
    assert torch.isfinite(loss_actor).all()
    assert torch.isfinite(loss_critic).all()

    # Parameters should not change
    for p_before, p_after in zip(actor_params_before, [p.detach().clone() for p in core.actor.parameters() if p.requires_grad][:3]):
        torch.testing.assert_close(p_before, p_after)
    for p_before, p_after in zip(critic_params_before, [p.detach().clone() for p in core.critic.parameters() if p.requires_grad][:3]):
        torch.testing.assert_close(p_before, p_after)
