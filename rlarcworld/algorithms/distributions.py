"""Distributions to be used in the algorithms."""

import torch


def categorical_projection(
    next_q_dist: torch.Tensor,
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    gamma: float,
    z_atoms: torch.Tensor,
    v_min: float,
    v_max: float,
    num_atoms: int,
    apply_softmax: bool = True,
    n_steps: int = 1,
) -> torch.Tensor:
    """Projects the target distribution using the Bellman update.

    Args:
        next_q_dist (torch.Tensor): Next state Q-distribution (batch_size, num_atoms).
        rewards (torch.Tensor): Rewards from the batch (batch_size, 1).
        terminated (torch.Tensor): Done flags from the batch (batch_size, 1).
        gamma (float): Discount factor.
        z_atoms (torch.Tensor): Z reference distribution Atom values (num_atoms, 1).
        v_min (float): Minimum value for value distribution.
        v_max (float): Maximum value for value distribution.
        num_atoms (int): Number of atoms in the distribution.
        apply_softmax (bool): Whether to apply softmax to the projected distribution.
        n_steps (int): Number of steps in the Bellman update.

    Returns:
        torch.Tensor: Projected target distribution (batch_size, num_atoms).
    """
    delta_z = (v_max - v_min) / (num_atoms - 1)

    # Compute the target distribution support
    tz = rewards + (gamma**n_steps) * (1 - terminated) * z_atoms
    tz = tz.clamp(v_min, v_max)

    # Map values to categorical bins
    bins = (tz - v_min) / delta_z
    lower, upper = bins.floor().long(), bins.ceil().long()
    lower = lower.clamp(0, num_atoms - 1)
    upper = upper.clamp(0, num_atoms - 1)

    # Distribute probability mass
    projected_dist = torch.zeros_like(next_q_dist).to(next_q_dist.device)
    projected_dist.scatter_add_(
        dim=-1, index=lower, src=next_q_dist * (upper.float() - bins)
    )
    projected_dist.scatter_add_(
        dim=-1, index=upper, src=next_q_dist * (bins - lower.float())
    )

    return torch.softmax(projected_dist, dim=-1) if apply_softmax else projected_dist
