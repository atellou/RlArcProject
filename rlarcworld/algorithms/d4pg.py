import os
import copy
import torch
import torch.nn.functional as F
import torch.optim as optim

from tensordict import TensorDict

from rlarcworld.utils import categorical_projection

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGGING_LEVEL", logging.WARNING))


class D4PG:

    def __init__(
        self,
        actor,
        critic,
        replay_buffer=None,
        gamma=0.99,
        tau=0.001,
        target_update_frequency=10,
    ):
        logger.info("Initializing D4PG...")

        # Store parameters
        self.global_step = 0
        self.global_episode = 0
        self.gamma = gamma
        self.tau = tau
        self.target_update_frequency = target_update_frequency
        self.replay_buffer = replay_buffer
        if self.replay_buffer is not None:
            logger.info("Replay buffer initialized")
        else:
            logger.warning("Replay buffer not set. Skipping experience storage.")

        # Store networks
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

        # Create optimizers
        self.actor_optimizer = optim.Adam(actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(critic.parameters(), lr=3e-4)

        # Create loss criterion
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
            target_update_frequency (int): Frequency of target network updates.

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

        if self.global_step % self.target_update_frequency == 0:
            self.update_target_networks(tau=0.005)
        return loss_actor, loss_critic

    def train_d4pg(
        self, env, train_samples, batch_size, n_steps=-1, warmup_buffer_ratio=0.2
    ):
        """Performs one full training step for the actor and critic in D4PG."""
        assert (
            self.replay_buffer is None or self.replay_buffer.max_size >= batch_size
        ), "Replay buffer size is too small for the given batch size. Must grater or equal to batch_size"
        self.global_step = 0
        self.global_episode = 0
        for episode, samples in enumerate(train_samples):
            self.global_episode += 1
            episode_reward = 0
            state = env.reset(
                options={"batch": samples["task"], "examples": samples["examples"]},
                seed=episode,
            )
            done = False

            while not done and (n_steps <= -1 or n_steps > 0):
                with torch.no_grad():
                    self.global_step += 1
                    n_steps -= 1
                    state = env.get_wrapper_attr("state")
                    actions = self.actor(state)
                    __, reward, terminated, truncated, __ = env.step(
                        self.actor.get_discrete_actions(actions)
                    )
                    next_state = env.get_wrapper_attr("state")
                    episode_reward += reward

                # Store the experience in the replay buffer
                if self.replay_buffer is not None:
                    priority = (
                        1.0
                        if len(self.replay_buffer.buffer) == 0
                        else max(self.replay_buffer.priorities)
                    )
                    self.replay_buffer.store(
                        state, actions, reward, next_state, terminated, priority
                    )
                    if (
                        len(self.replay_buffer.buffer)
                        > self.replay_buffer.max_size * warmup_buffer_ratio
                        or len(self.replay_buffer.buffer) > batch_size
                    ):
                        batch = self.replay_buffer.sample(batch_size)
                    else:
                        batch = None
                else:
                    batch = TensorDict(
                        {
                            "state": state,
                            "action": actions,
                            "reward": reward,
                            "next_state": next_state,
                            "done": terminated,
                        },
                        batch_size=batch_size,
                    )
                if batch is not None:
                    loss_actor, loss_critic = self.train_step(
                        batch=batch,
                        actor_optimizer=self.actor_optimizer,
                        critic_optimizer=self.critic_optimizer,
                        gamma=self.gamma,
                    )
                done = terminated or truncated

            logger.info(
                "Episode {}, Reward: {:.2f}, Actor Loss: {:.4f}, Critic Loss: {:.4f}".format(
                    episode, episode_reward, loss_actor, loss_critic
                )
            )
