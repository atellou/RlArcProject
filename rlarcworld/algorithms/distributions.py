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
    normalize_projection: bool = True,
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
        normalize_projection (bool): Whether to normalize the projected distribution.
        n_steps (int): Number of steps in the Bellman update.

    Returns:
        torch.Tensor: Projected target distribution (batch_size, num_atoms).
    """
    # Under AMP, next_q_dist can be float16/bfloat16. The projection involves
    # small probability masses and normalization that can underflow in fp16.
    # Do the projection math in float32 to avoid NaNs and keep the result as
    # a valid probability distribution.
    next_q_dist = next_q_dist.float()
    rewards = rewards.float()
    terminated = terminated.float()
    z_atoms = z_atoms.float()

    delta_z = (v_max - v_min) / (num_atoms - 1)

    # Ensure proper shapes for broadcasting
    if rewards.dim() == 1:
        rewards = rewards.unsqueeze(-1)
    if terminated.dim() == 1:
        terminated = terminated.unsqueeze(-1)
    if z_atoms.dim() == 1:
        z_atoms = z_atoms.unsqueeze(0)

    # Compute the target distribution support
    # (batch_size, 1) + scalar * (batch_size, 1) * (1, num_atoms)
    # -> (batch_size, num_atoms)
    tz = rewards + (gamma**n_steps) * (1 - terminated) * z_atoms
    tz = tz.clamp(v_min, v_max)

    # Map values to categorical bins
    bins = (tz - v_min) / delta_z
    lower, upper = bins.floor().long(), bins.ceil().long()
    lower = lower.clamp(0, num_atoms - 1)
    upper = upper.clamp(0, num_atoms - 1)

    # Distribute probability mass
    projected_dist = torch.zeros_like(next_q_dist).to(next_q_dist.device)
    w_lower = upper.float() - bins
    w_upper = bins - lower.float()

    # If bins is exactly an integer, floor == ceil and both weights become 0.
    # In that case, assign all probability mass to that atom.
    eq_mask = upper.eq(lower)
    w_lower = torch.where(eq_mask, torch.ones_like(w_lower), w_lower)
    w_upper = torch.where(eq_mask, torch.zeros_like(w_upper), w_upper)
    projected_dist.scatter_add_(dim=-1, index=lower, src=next_q_dist * w_lower)
    projected_dist.scatter_add_(dim=-1, index=upper, src=next_q_dist * w_upper)

    if not normalize_projection:
        return projected_dist

    # projected_dist is already a probability distribution in expectation.
    # Renormalize to guard against numerical drift (instead of softmax, which
    # would distort the distribution).
    denom = projected_dist.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return projected_dist / denom
