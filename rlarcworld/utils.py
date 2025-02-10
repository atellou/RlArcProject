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


class TorchQueue(torch.Tensor):
    _q_size: tuple
    _q_dim: int

    def __new__(cls, data, q_size: int, q_dim: int = 0, *args, **kwargs):
        return super().__new__(cls, data, *args, **kwargs)

    def __init__(self, data, q_size: int, q_dim: int = 0):
        assert (
            isinstance(q_size, int) and q_size > 0
        ), "q_size must be a positive integer."
        assert isinstance(q_dim, int), "q_dim must be an integer."
        assert data.shape[q_dim] <= q_size, "Data shape is {}, q_size is {}".format(
            data.shape[q_dim], q_size
        )
        self._q_size = q_size
        self._q_dim = q_dim
        self.__reversed_indices = torch.arange(-1 - q_size, 0)

    def clone(self, *args, **kwargs):
        return TorchQueue(
            super().clone(*args, **kwargs),
            data=self,
            q_size=self._q_size,
            q_dim=self._q_dim,
        )

    def to(self, *args, **kwargs):
        new_obj = super().to(*args, **kwargs)
        if new_obj is self:
            return self
        return TorchQueue(new_obj, q_size=self._q_size, q_dim=self._q_dim)

    @property
    def q_size(self):
        return self._q_size

    @property
    def q_dim(self):
        return self._q_dim

    @property
    def is_full(self):
        return self.shape[self._q_dim] == self._q_size

    def push(self, item):
        """
        Adds an item to the queue. If the queue is full, the oldest item is removed first.

        Args:
            item (torch.Tensor): The item to be added to the queue.

        Returns:
            torch.Tensor: The item that was added to the queue.
        """
        item = torch.cat([self, item], dim=self._q_dim)
        if item.shape[self._q_dim] > self._q_size:
            item = self.__correct_q_size(item)
            assert item.shape[self._q_dim] == self._q_size
        assert (
            item.shape[self._q_dim] <= self._q_size
        ), "Item shape is {}, q_size is {}".format(
            item.shape[self._q_dim], self._q_size
        )
        return TorchQueue(item, q_size=self._q_size, q_dim=self._q_dim)

    def __correct_q_size(self, item=None):
        """
        Correct the size of the queue along the specified dimension when full.
        """
        if item is None:
            item = self
        return TorchQueue(
            torch.index_select(
                item, self._q_dim, self.__reversed_indices[1:] + self._q_size + 1
            ),
            q_size=self._q_size,
            q_dim=self._q_dim,
        )

    def pop(self):
        """
        Pops the oldest elements along the specified dimension.

        Returns:
            torch.Tensor: The oldest item in the queue.
            TorchQueue: The updated queue without the oldest item.
        """
        return self[0], TorchQueue(self[1:], q_size=self._q_size, q_dim=self._q_dim)
