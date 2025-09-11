import pytest
import torch

from rlarcworld.utils.distributions import categorical_projection


def make_support(num_atoms=5, v_min=-2.0, v_max=2.0, device="cpu"):
    # categorical_projection expects z_atoms to broadcast with (batch, 1), so shape (1, num_atoms)
    z_atoms = torch.linspace(v_min, v_max, steps=num_atoms, device=device).view(1, -1)
    return z_atoms, v_min, v_max, num_atoms


def test_shapes_and_normalization():
    device = torch.device("cpu")
    batch_size = 4
    z_atoms, v_min, v_max, num_atoms = make_support(num_atoms=7, v_min=-3.0, v_max=3.0, device=device)

    # uniform next_q_dist
    next_q_dist = torch.full((batch_size, num_atoms), 1.0 / num_atoms, device=device)
    rewards = torch.zeros((batch_size, 1), device=device)
    terminated = torch.zeros((batch_size, 1), device=device)
    gamma = 0.99

    proj = categorical_projection(
        next_q_dist,
        rewards,
        terminated,
        gamma,
        z_atoms,
        v_min,
        v_max,
        num_atoms,
        apply_softmax=True,
        n_steps=1,
    )

    assert proj.shape == (batch_size, num_atoms)
    # Proper distribution
    torch.testing.assert_close(proj.sum(dim=1), torch.ones(batch_size, device=device))
    assert torch.all(proj >= 0)
    assert torch.all(proj <= 1)


@pytest.mark.parametrize(
    "num_atoms,v_min,v_max",
    [
        (5, -2.0, 2.0),
        (7, -3.0, 3.0),
        (11, -1.0, 1.0),
        (9, -4.0, 4.0),
    ],
)
def test_shapes_and_normalization_parametrized(num_atoms, v_min, v_max):
    device = torch.device("cpu")
    batch_size = 3
    z_atoms, v_min, v_max, num_atoms = make_support(
        num_atoms=num_atoms, v_min=v_min, v_max=v_max, device=device
    )

    # random but normalized next_q_dist
    next_q_dist = torch.rand((batch_size, num_atoms), device=device)
    next_q_dist = next_q_dist / next_q_dist.sum(dim=1, keepdim=True)
    rewards = torch.randn((batch_size, 1), device=device).clamp(v_min, v_max)
    terminated = torch.zeros((batch_size, 1), device=device)
    gamma = 0.95

    proj = categorical_projection(
        next_q_dist,
        rewards,
        terminated,
        gamma,
        z_atoms,
        v_min,
        v_max,
        num_atoms,
        apply_softmax=True,
        n_steps=2,
    )

    assert proj.shape == (batch_size, num_atoms)
    torch.testing.assert_close(proj.sum(dim=1), torch.ones(batch_size, device=device))
    assert torch.all(proj >= 0)
    assert torch.all(proj <= 1)


@pytest.mark.parametrize("n_steps", [1, 3])
@pytest.mark.parametrize("term_flag", [0, 1])
def test_terminated_independent_of_nsteps(n_steps, term_flag):
    device = torch.device("cpu")
    batch_size = 3
    z_atoms, v_min, v_max, num_atoms = make_support(num_atoms=9, v_min=-4.0, v_max=4.0, device=device)

    # next_q_dist peaked at center atom to make mapping clearer
    next_q_dist = torch.zeros((batch_size, num_atoms), device=device)
    center = num_atoms // 2
    next_q_dist[:, center] = 1.0

    # set a reward that is within the support
    rewards = torch.tensor([[1.0], [0.0], [-1.5]], device=device)
    terminated = torch.full((batch_size, 1), float(term_flag), device=device)
    gamma = 0.9

    proj = categorical_projection(
        next_q_dist,
        rewards,
        terminated,
        gamma,
        z_atoms,
        v_min,
        v_max,
        num_atoms,
        apply_softmax=True,
        n_steps=n_steps,
    )

    # For terminated=1, projection should ignore next_q_dist shift (gamma term) regardless of n_steps
    if term_flag == 1:
        proj_n1 = categorical_projection(
            next_q_dist, rewards, torch.ones_like(terminated), gamma, z_atoms, v_min, v_max, num_atoms, True, 1
        )
        torch.testing.assert_close(proj, proj_n1)

    # Always a proper distribution
    torch.testing.assert_close(proj.sum(dim=1), torch.ones(batch_size, device=device))
    assert torch.all(proj >= 0)
    assert torch.all(proj <= 1)


def test_gamma_zero_behavior():
    device = torch.device("cpu")
    batch_size = 2
    z_atoms, v_min, v_max, num_atoms = make_support(num_atoms=5, v_min=-2.0, v_max=2.0, device=device)

    # arbitrary next distribution
    next_q_dist = torch.rand((batch_size, num_atoms), device=device)
    next_q_dist = next_q_dist / next_q_dist.sum(dim=1, keepdim=True)

    rewards = torch.tensor([[0.5], [-1.25]], device=device)
    terminated = torch.zeros((batch_size, 1), device=device)

    proj = categorical_projection(
        next_q_dist,
        rewards,
        terminated,
        gamma=0.0,  # should ignore next_q_dist support and project purely around rewards
        z_atoms=z_atoms,
        v_min=v_min,
        v_max=v_max,
        num_atoms=num_atoms,
        apply_softmax=True,
        n_steps=3,
    )

    # Proper distribution
    torch.testing.assert_close(proj.sum(dim=1), torch.ones(batch_size, device=device))
    assert torch.all(proj >= 0)
    assert torch.all(proj <= 1)
