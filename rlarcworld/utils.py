import torch


def categorical_projection(next_q_dist, rewards, dones, gamma, v_min, v_max, num_atoms):
    """
    Projects the target distribution using the Bellman update.

    Args:
        next_q_dist (torch.Tensor): Next state Q-distribution (batch_size, num_atoms).
        rewards (torch.Tensor): Rewards from the batch (batch_size, 1).
        dones (torch.Tensor): Done flags from the batch (batch_size, 1).
        gamma (float): Discount factor.
        v_min (float): Minimum value for value distribution.
        v_max (float): Maximum value for value distribution.
        num_atoms (int): Number of atoms in the distribution.

    Returns:
        torch.Tensor: Projected target distribution (batch_size, num_atoms).
    """
    delta_z = (v_max - v_min) / (num_atoms - 1)
    z = torch.linspace(v_min, v_max, num_atoms).to(rewards.device)  # Atom values

    # Compute the target distribution support
    tz = rewards.unsqueeze(-1) + gamma * (1 - dones.unsqueeze(-1)) * z.unsqueeze(0)
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

    return projected_dist
