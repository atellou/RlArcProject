import torch
import torch.nn.functional as F
import torch.optim as optim

from tensordict import TensorDict

from rlarcworld.utils import categorical_projection


class D4PG:

    def __init__(self):
        self.critic_target = None
        self.actor_target = None
        self.criterion = torch.nn.KLDivLoss(reduction="batchmean")

    def compute_critic_target_distribution(
        self,
        critic_target,
        actor_target,
        reward,
        next_state,
        done,
        gamma,
    ):
        """
        Computes the target distribution for the critic network.

        Args:
            critic_target (nn.Module): Target critic network.
            actor_target (nn.Module): Target actor network.
            reward (torch.Tensor): Rewards from the batch.
            next_state (TensorDict): TensorDict of next states.
            done (torch.Tensor): Done flags from the batch.
            gamma (float): Discount factor.

        Returns:
            torch.Tensor: Projected target distribution (batch_size, num_atoms).
        """
        with torch.no_grad():
            # Use the target actor to get the best action for each next state
            action_probs = actor_target(next_state)
            # Get best action
            best_next_action = torch.cat(
                [torch.argmax(x, dim=-1).unsqueeze(-1) for x in action_probs.values()],
                dim=-1,
            )  # Shape: (batch_size, action_space_dim)

            # Get the Q-distribution for the best action from the target critic
            best_next_q_dist = critic_target(next_state, best_next_action)

        # Assert probability mass function
        for key, dist in best_next_q_dist.items():
            torch.testing.assert_close(
                torch.sum(dist, dim=1), torch.ones(dist.shape[0])
            ), f"Probability mass function not normalized for key: {key}"
            assert torch.all(dist >= 0), f"Negative probability values for key: {key}"
            assert torch.all(
                dist <= 1
            ), f"Probability values greater than 1 for key: {key}"

        # Project the distribution using the categorical projection
        target_dist = TensorDict(
            {
                key: categorical_projection(
                    best_next_q_dist[key],
                    reward[key],
                    done,
                    gamma,
                    critic_target.z_atoms[key],
                    critic_target.v_min[key],
                    critic_target.v_max[key],
                    critic_target.num_atoms[key],
                    apply_softmax=True,
                )
                for key in best_next_q_dist.keys()
            }
        )

        return target_dist

    def compute_critic_loss(self, critic, state, action, target_dist):
        """
        Computes the critic loss using KL divergence.

        Args:
            critic (nn.Module): Critic network.
            state (TensorDict): TensorDict of current states.
            action (torch.Tensor): Actions taken in the batch.
            target_dist (torch.Tensor): Target distribution computed by Bellman backup.

        Returns:
            torch.Tensor: Critic loss.
        """
        # Get state-action value distribution from the critic
        q_dist = critic(state, action)  # Shape: (batch_size, num_atoms)
        # KL Divergence (TdError)
        loss = TensorDict(
            {
                key: self.criterion(q_dist[key], target_dist[key])
                for key in q_dist.keys()
            }
        )

        return loss

    def compute_actor_loss(self, actor, critic, state):
        """
        Compute the loss for the actor network.

        Args:
            actor (nn.Module): Actor network.
            critic (nn.Module): Critic network.
            state (TensorDict): TensorDict of current states.

        Returns:
            torch.Tensor: Actor loss (negative expected Q-value).
        """
        # Get action probabilities from the actor
        action_probs = actor(state)
        # Get best action
        best_next_action = torch.cat(
            [torch.argmax(x, dim=-1).unsqueeze(-1) for x in action_probs.values()],
            dim=-1,
        ).float()  # Shape: (batch_size, action_space_dim [4])

        # Compute gradient of expected Q-value w.r.t. actions
        best_next_action.requires_grad = True

        probs = critic(state, best_next_action)
        loss = TensorDict({})
        for key in probs.keys():
            Q = (probs[key] * critic.z_atoms[key].to(probs[key].device)).sum(dim=-1)
            grad = torch.autograd.grad(Q.sum(), best_next_action, retain_graph=True)[0]
            # Policy gradient update
            loss[key] = {
                k: -torch.sum(x * grad[: grad.shape[0], i].view(-1, 1))
                for i, (k, x) in enumerate(action_probs.items())
            }

        return loss

    def train_d4pg(
        self,
        actor,
        critic,
        actor_target,
        critic_target,
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
            actor (nn.Module): Actor network.
            critic (nn.Module): Critic network.
            actor_target (nn.Module): Target actor network.
            critic_target (nn.Module): Target critic network.
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
            dones = torch.tensor(dones, dtype=torch.float32)

            # --- Critic Update ---
            # Compute the target distribution
            target_dist = self.compute_critic_target_distribution(
                critic_target,
                actor_target,
                rewards,
                next_states,
                dones,
                gamma,
                num_atoms,
                v_min,
                v_max,
            )

            # Compute the critic loss
            critic_loss = self.compute_critic_loss(critic, states, actions, target_dist)

            # Backpropagate and update the critic
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # --- Actor Update ---
            # Compute the actor loss
            actor_loss = self.compute_actor_loss(actor, critic, states)

            # Backpropagate and update the actor
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # --- Target Network Update ---
            if step % target_update_freq == 0:
                for target_param, param in zip(
                    actor_target.parameters(), actor.parameters()
                ):
                    target_param.data.copy_(param.data)
                for target_param, param in zip(
                    critic_target.parameters(), critic.parameters()
                ):
                    target_param.data.copy_(param.data)

            # Log progress (optional)
            if step % 100 == 0:
                print(
                    f"Step {step}/{steps}, Critic Loss: {critic_loss.item():.4f}, Actor Loss: {actor_loss.item():.4f}"
                )
