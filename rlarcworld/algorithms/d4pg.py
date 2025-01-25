import torch
import torch.nn.functional as F
import torch.optim as optim
from rlarcworld.utils import categorical_projection


class D4PG:
    def compute_critic_target_distribution(
        critic_target,
        reward,
        next_state,
        done,
        gamma,
        num_atoms,
        v_min,
        v_max,
    ):
        """
        Computes the target distribution for the critic network.

        Args:
            critic_target (nn.Module): Target critic network.
            reward (torch.Tensor): Rewards from the batch.
            next_state (TensorDict): TensorDict of next states.
            done (torch.Tensor): Done flags from the batch.
            gamma (float): Discount factor.
            num_atoms (int): Number of atoms for the categorical distribution.
            v_min (float): Minimum value for value distribution.
            v_max (float): Maximum value for value distribution.

        Returns:
            torch.Tensor: Projected target distribution (batch_size, num_atoms).
        """
        z = torch.linspace(v_min, v_max, num_atoms).to(reward.device)  # Atom values

        # Get next-state Q-distribution from the target critic
        next_q_dist = critic_target(next_state).get(
            "q_dist"
        )  # Shape: (batch_size, num_actions, num_atoms)

        # Compute the expected Q-values for each action
        next_q_values = (next_q_dist * z).sum(
            dim=-1
        )  # Shape: (batch_size, num_actions)

        # Select the best action for the next state
        best_next_action = torch.argmax(next_q_values, dim=-1)  # Shape: (batch_size,)

        # Get the Q-distribution corresponding to the best next action
        best_next_q_dist = next_q_dist[
            torch.arange(next_q_dist.size(0)), best_next_action
        ]  # Shape: (batch_size, num_atoms)

        # Project the distribution using the categorical projection
        target_dist = categorical_projection(
            best_next_q_dist, reward, done, gamma, v_min, v_max, num_atoms
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
        q_dist = critic(state).get(
            "q_dist"
        )  # Shape: (batch_size, num_actions, num_atoms)
        action_q_dist = q_dist[
            torch.arange(q_dist.size(0)), action
        ]  # Shape: (batch_size, num_atoms)
        loss = -torch.sum(
            target_dist * torch.log(action_q_dist + 1e-8), dim=-1
        ).mean()  # KL Divergence
        return loss

    def compute_actor_loss(self, actor, critic, state, v_min, v_max):
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
        action_probs = actor(state).get(
            "action_probs"
        )  # Shape: (batch_size, num_actions)

        # Get Q-value distributions from the critic
        q_dist = critic(state).get(
            "q_dist"
        )  # Shape: (batch_size, num_actions, num_atoms)
        z = torch.linspace(v_min, v_max, q_dist.size(-1)).to(
            q_dist.device
        )  # Atom values
        q_values = (q_dist * z).sum(dim=-1)  # Convert distribution to expected Q-values

        # Compute expected Q-value weighted by action probabilities
        expected_q = torch.sum(action_probs * q_values, dim=-1)  # Shape: (batch_size,)

        # Loss is negative expected Q-value (maximize expected Q-value)
        actor_loss = -expected_q.mean()

        return actor_loss

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
