import torch


def categorical_projection(
    next_q_dist,
    rewards,
    terminated,
    gamma,
    z_atoms,
    v_min,
    v_max,
    num_atoms,
    apply_softmax=True,
):
    """
    Projects the target distribution using the Bellman update.

    Args:
        next_q_dist (torch.Tensor): Next state Q-distribution (batch_size, num_atoms).
        rewards (torch.Tensor): Rewards from the batch (batch_size, 1).
        terminated (torch.Tensor): Done flags from the batch (batch_size, 1).
        gamma (float): Discount factor.
        z_atoms (torch.Tensor): Z reference distribution Atom values (num_atoms, 1).
        v_min (float): Minimum value for value distribution.
        v_max (float): Maximum value for value distribution.
        num_atoms (int): Number of atoms in the distribution.

    Returns:
        torch.Tensor: Projected target distribution (batch_size, num_atoms).
    """
    delta_z = (v_max - v_min) / (num_atoms - 1)

    # Compute the target distribution support
    tz = rewards + gamma * (1 - terminated) * z_atoms
    tz = tz.clamp(v_min, v_max)

    # Map values to categorical bins
    b = (tz - v_min) / delta_z
    l, u = b.floor().long(), b.ceil().long()
    l = l.clamp(0, num_atoms - 1)
    u = u.clamp(0, num_atoms - 1)

    # Distribute probability mass
    projected_dist = torch.zeros_like(next_q_dist)
    projected_dist.scatter_add_(dim=-1, index=l, src=next_q_dist * (u.float() - b))
    projected_dist.scatter_add_(dim=-1, index=u, src=next_q_dist * (b - l.float()))

    return torch.softmax(projected_dist, dim=-1) if apply_softmax else projected_dist
