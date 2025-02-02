import copy
import torch
import torch.nn.functional as F
import torch.optim as optim

from tensordict import TensorDict

from rlarcworld.utils import categorical_projection


class D4PG:

    def __init__(self, actor, critic):
        self.actor = actor
        self.critic = critic

        # Create target networks as copies of the main networks
        self.actor_target = copy.deepcopy(actor)
        self.critic_target = copy.deepcopy(critic)

        # Ensure target networks start with the same weights
        self.actor_target.load_state_dict(actor.state_dict())
        self.critic_target.load_state_dict(critic.state_dict())

        # Set target networks to evaluation mode (no need for gradients)
        self.actor_target.eval()
        self.critic_target.eval()

        self.critic_criterion = torch.nn.KLDivLoss(reduction="batchmean")

    def compute_target_distribution(
        self,
        reward: torch.Tensor,
        next_state: TensorDict,
        done: torch.Tensor,
        gamma: float,
    ):
        """
        Computes the target distribution for the critic network.

        Args:
            reward (torch.Tensor): Rewards from the batch.
            next_state (TensorDict): TensorDict of next states.
            done (torch.Tensor): Done flags from the batch.
            gamma (float): Discount factor.

        Returns:
            torch.Tensor: Projected target distribution (batch_size, num_atoms).
        """
        with torch.no_grad():
            # Use the target actor to get the best action for each next state
            action_probs = self.actor_target(next_state)

            # Get best action
            best_next_action = torch.cat(
                [torch.argmax(x, dim=-1).unsqueeze(-1) for x in action_probs.values()],
                dim=-1,
            )  # Shape: (batch_size, action_space_dim)

            # Get the Q-distribution for the best action from the target critic
            best_next_q_dist = self.critic_target(next_state, best_next_action)

        # Project the distribution using the categorical projection
        target_dist = TensorDict(
            {
                key: categorical_projection(
                    best_next_q_dist[key],
                    reward[key],
                    done,
                    gamma,
                    self.critic_target.z_atoms[key],
                    self.critic_target.v_min[key],
                    self.critic_target.v_max[key],
                    self.critic_target.num_atoms[key],
                    apply_softmax=True,
                )
                for key in best_next_q_dist.keys()
            }
        )

        return target_dist

    def compute_critic_loss(self, state, action, target_dist):
        """
        Computes the critic loss using KL divergence.

        Args:
            state (TensorDict): TensorDict of current states.
            action (torch.Tensor): Actions taken in the batch.
            target_dist (torch.Tensor): Target distribution computed by Bellman backup.

        Returns:
            torch.Tensor: Critic loss.
        """
        # Get state-action value distribution from the critic
        q_dist = self.critic(state, action)  # Shape: (batch_size, num_atoms)
        # KL Divergence (TdError)
        loss = TensorDict(
            {
                key: self.critic_criterion(q_dist[key], target_dist[key])
                for key in q_dist.keys()
            }
        )

        return loss

    def compute_actor_loss(self, state):
        """
        Compute the loss for the actor network.

        Args:
            state (TensorDict): TensorDict of current states.

        Returns:
            torch.Tensor: Actor loss (negative expected Q-value).
        """
        # Get action probabilities from the actor
        action_probs = self.actor(state)
        # Get best action
        best_next_action = torch.cat(
            [torch.argmax(x, dim=-1).unsqueeze(-1) for x in action_probs.values()],
            dim=-1,
        ).float()  # Shape: (batch_size, action_space_dim [4])

        # Compute gradient of expected Q-value w.r.t. actions
        best_next_action.requires_grad = True

        probs = self.critic(state, best_next_action)
        loss = TensorDict({})
        for key in probs.keys():
            Q = (probs[key] * self.critic.z_atoms[key].to(probs[key].device)).sum(
                dim=-1
            )
            grad = torch.autograd.grad(Q.sum(), best_next_action, retain_graph=True)[0]
            # Policy gradient update
            loss[key] = {
                k: -torch.sum(x * grad[: grad.shape[0], i].view(-1, 1))
                for i, (k, x) in enumerate(action_probs.items())
            }

        return loss

    def update_target_networks(self, tau=0.005):
        """Performs a soft update on the target networks."""
        for target_param, param in zip(
            self.actor_target.parameters(), self.actor.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for target_param, param in zip(
            self.critic_target.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def train_step(self, batch, actor_optimizer, critic_optimizer, gamma):
        """Performs one training step for the actor and critic.

        Args:
            batch (TensorDict): Batch of data.
            actor_optimizer (torch.optim.Optimizer): Optimizer for the actor.
            critic_optimizer (torch.optim.Optimizer): Optimizer for the critic.
            gamma (float): Discount factor.

        Returns:
            Tuple[float, float]: Actor and critic losses.
        """

        # Compute target distribution
        target_probs = self.compute_target_distribution(
            batch["reward"], batch["next_state"], batch["done"], gamma
        )

        # Critic update
        critic_optimizer.zero_grad()
        loss_critic = self.compute_critic_loss(
            batch["state"], batch["action"], target_probs
        )
        loss_critic = sum(tuple(loss_critic.values()))
        loss_critic.backward()
        critic_optimizer.step()

        # Actor update
        actor_optimizer.zero_grad()
        loss_actor = self.compute_actor_loss(batch["state"])
        loss_actor = sum((g for v in loss_actor.values() for g in v.values()))
        loss_actor.backward()
        actor_optimizer.step()

        self.update_target_networks(tau=0.005)
        return loss_actor, loss_critic

    def train_d4pg(
        self,
        replay_buffer,
        actor_optimizer,
        critic_optimizer,
        gamma,
        num_atoms,
        v_min,
        v_max,
        batch_size,
        target_update_freq,
        steps,
    ):
        """
        Full training loop for D4PG with categorical actions.

        Args:
            replay_buffer (ReplayBuffer): Experience replay buffer.
            actor_optimizer (torch.optim.Optimizer): Optimizer for the actor.
            critic_optimizer (torch.optim.Optimizer): Optimizer for the critic.
            gamma (float): Discount factor.
            num_atoms (int): Number of atoms for critic distribution.
            v_min (float): Minimum value of the value distribution.
            v_max (float): Maximum value of the value distribution.
            batch_size (int): Batch size for updates.
            target_update_freq (int): Steps between target network updates.
            steps (int): Total number of training steps.
        """

        for step in range(steps):
            # Sample a batch from the replay buffer
            batch = replay_buffer.sample(batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            # Convert to tensors
            states = torch.stack(states)
            actions = torch.tensor(actions, dtype=torch.long)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            next_states = torch.stack(next_states)
            dones = torch
