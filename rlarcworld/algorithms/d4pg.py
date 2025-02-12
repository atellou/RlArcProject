import os
import copy
import torch
import torch.nn.functional as F
import torch.optim as optim

from tensordict import TensorDict

from rlarcworld.utils import categorical_projection

import logging

logger = logging.getLogger(__name__)


class D4PG:

    def __init__(
        self,
        actor,
        critic,
        replay_buffer=None,
        n_steps=1,
        gamma=0.99,
        tau=0.001,
        target_update_frequency=10,
    ):
        """
        Initialize a D4PG instance.

        Parameters
        ----------
        actor : torch.nn.Module
            Actor network.
        critic : torch.nn.Module
            Critic network.
        replay_buffer : rlarcworld.data.replay_buffers.TensorDictReplayBuffer, optional
            Replay buffer to store experiences in. If not provided, experiences won't
            be stored and the algorithm will not be able to learn.
        n_steps : int, optional
            Number of steps to take before updating the target networks. Defaults to 1.
        gamma : float, optional
            Discount factor for future rewards. Defaults to 0.99.
        tau : float, optional
            Soft update parameter for target networks. Defaults to 0.001.
        target_update_frequency : int, optional
            Frequency of target network updates. Defaults to 10.
        """
        logger.debug("Initializing D4PG...")

        # Store parameters
        self.n_steps = n_steps
        self.gamma = gamma
        self.tau = tau
        self.target_update_frequency = target_update_frequency
        self.replay_buffer = replay_buffer
        if self.replay_buffer is not None:
            logger.debug("Replay buffer initialized")
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

    def categorize_actions(self, actions, as_dict=False):
        """
        Convert a probability distribution action to discrete action.

        Args:
            actions (TensorDict or torch.Tensor): Distribution over actions.
            as_dict (bool, optional): Whether to return a TensorDict or a single
                tensor. Defaults to False.

        Returns:
            TensorDict or torch.Tensor: The categorical action.
        """

        if as_dict:
            return TensorDict(
                {key: torch.argmax(x, dim=-1) for key, x in actions.items()}
            )
        return torch.cat(
            [torch.argmax(x, dim=-1).unsqueeze(-1) for x in actions.values()],
            dim=-1,
        )

    def compute_target_distribution(
        self,
        reward: torch.Tensor,
        next_state: TensorDict,
        terminated: torch.Tensor,
        gamma: float,
    ):
        """
        Computes the target distribution for the critic network.

        Args:
            reward (torch.Tensor): Rewards from the batch.
            next_state (TensorDict): TensorDict of next states.
            terminated (torch.Tensor): terminated flags from the batch.
            gamma (float): Discount factor.

        Returns:
            torch.Tensor: Projected target distribution (batch_size, num_atoms).
        """
        with torch.no_grad():
            # Use the target actor to get the best action for each next state
            action_probs = self.actor_target(next_state)

            # Get best action
            best_next_action = self.categorize_actions(
                action_probs
            )  # Shape: (batch_size, action_space_dim)

            # Get the Q-distribution for the best action from the target critic
            best_next_q_dist = self.critic_target(next_state, best_next_action)

        # Project the distribution using the categorical projection
        assert best_next_q_dist.keys() == reward.keys(), ValueError(
            "Keys of best_next_q_dist and reward must be the same"
        )
        target_dist = TensorDict(
            {
                key: categorical_projection(
                    best_next_q_dist[key],
                    reward[key],
                    terminated,
                    gamma,
                    self.critic_target.z_atoms[key],
                    self.critic_target.v_min[key],
                    self.critic_target.v_max[key],
                    self.critic_target.num_atoms[key],
                    apply_softmax=True,
                    n_steps=self.n_steps if key == "n_reward" else 1,
                )
                for key in best_next_q_dist.keys()
            }
        )

        return target_dist

    def compute_critic_loss(self, state, action, target_dist, compute_td_error=False):
        """
        Computes the loss for the critic network.

        Args:
            state (TensorDict): TensorDict of current states.
            action (torch.Tensor): Tensor of actions.
            target_dist (TensorDict): Target distribution from the target critic.
            compute_td_error (bool, optional): Whether to compute the TD error. Defaults to False.

        Returns:
            Tuple[TensorDict, TensorDict]: Critic loss and TD error.
        """
        # Get state-action value distribution from the critic
        q_dist = self.critic(state, action)  # Shape: (batch_size, num_atoms)
        # TD Error
        loss = TensorDict({})
        td_error = TensorDict({})
        for key in q_dist.keys():
            if compute_td_error:
                td_error[key] = torch.abs(
                    torch.sum(q_dist[key] * self.critic.z_atoms[key], dim=-1)
                    - torch.sum(target_dist[key] * self.critic.z_atoms[key], dim=-1)
                )
            # KL Divergence
            loss[key] = self.critic_criterion(q_dist[key], target_dist[key])

        return loss, td_error

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
        best_next_action = self.categorize_actions(
            action_probs
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
        """
        Updates the target networks using the Polyak averaging method.

        Args:
            tau (float, optional): The update rate. Defaults to 0.005.
        """
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
            batch["reward"], batch["next_state"], batch["terminated"], gamma
        )

        # Critic update
        critic_optimizer.zero_grad()
        loss_critic, td_error = self.compute_critic_loss(
            batch["state"],
            batch["action"],
            target_probs,
            compute_td_error=self.replay_buffer is not None,
        )
        loss_critic = tuple(loss_critic.values())
        loss_critic = sum(loss_critic) / len(loss_critic)
        loss_critic.backward()
        critic_optimizer.step()

        # Actor update
        actor_optimizer.zero_grad()
        loss_actor = self.compute_actor_loss(batch["state"])
        loss_actor = tuple(g for v in loss_actor.values() for g in v.values())
        loss_actor = sum(loss_actor) / len(loss_actor)
        loss_actor.backward()
        actor_optimizer.step()

        if self.replay_buffer is not None:
            td_error = tuple(td_error.values())
            td_error = sum(td_error) / len(td_error)
            self.replay_buffer.update_priority(batch["index"], td_error + 1e-6)

        return loss_actor, loss_critic

    def step(self, env):
        """
        Performs a single step in the environment using the current policy.

        Args:
            env: The environment to interact with.

        Returns:
            Tuple containing:
                - state (TensorDict): The current state of the environment.
                - reward (TensorDict): The rewards received after taking the action.
                - actions (torch.Tensor): The actions taken by the actor.
                - next_state (TensorDict): The state of the environment after the action.
                - done (bool): Whether the episode has terminated.
                - truncated (bool): Whether the episode has been truncated.
        """

        with torch.no_grad():
            state = env.get_state(unsqueeze=1)
            actions = self.actor(state)
            actions = self.categorize_actions(actions, as_dict=True)
            __, reward, done, truncated, __ = env.step(actions)
            actions = torch.cat([x.unsqueeze(-1) for x in actions.values()], dim=-1)
            reward = TensorDict(
                {
                    "pixel_wise": reward.unsqueeze(-1),
                    "binary": env.get_wrapper_attr("last_reward").unsqueeze(-1),
                }
            ).auto_batch_size_()
            next_state = env.get_state(unsqueeze=1)
            if self.n_steps > 1:
                reward["n_reward"] = env.n_step_reward().unsqueeze(-1)
        return state, reward, actions, next_state, done, truncated

    def replay_buffer_step(
        self,
        batch_size,
        state,
        actions,
        reward,
        next_state,
        warmup_buffer_ratio=0.2,
    ):
        """
        Stores a batch of transitions in the replay buffer and samples a new batch of size `batch_size`.

        Args:
            batch_size (int): The size of the batch to sample.
            state (TensorDict): The state of the environment.
            actions (torch.Tensor): The actions taken by the actor.
            reward (TensorDict): The rewards received from the environment.
            next_state (TensorDict): The next state of the environment.
            warmup_buffer_ratio (float, optional): The ratio of the replay buffer to fill before sampling. Defaults to 0.2.

        Returns:
            TensorDict or None: A batch of transitions if the replay buffer is full, otherwise None.
        """
        priority = (
            torch.ones(batch_size)
            if len(self.replay_buffer.storage) == 0
            else torch.ones(batch_size) * max(self.replay_buffer.sampler._max_priority)
        )
        batch = torch.stack(
            [
                TensorDict(
                    {
                        "state": state[i],
                        "action": actions[i],
                        "reward": reward[i],
                        "next_state": next_state[i],
                        "terminated": next_state["terminated"][i],
                    }
                ).auto_batch_size_()
                for i in range(batch_size)
            ]
        )
        indices = self.replay_buffer.extend(batch)
        self.replay_buffer.update_priority(indices, priority)
        if (
            len(self.replay_buffer.storage)
            > self.replay_buffer.storage.max_size * warmup_buffer_ratio
            or len(self.replay_buffer.storage) > batch_size
        ):
            batch = self.replay_buffer.sample(batch_size)
        else:
            batch = None
        return batch

    def fileter_compleated_state(
        self,
        batch,
    ):
        """
        Filters out the completed states from a batch of transitions.

        Args:
            batch (TensorDict): A batch of transitions.

        Returns:
            TensorDict: A filtered batch of transitions where the states are not completed.
        """
        mask = batch["state"]["terminated"] == 0
        selected_indices = mask.nonzero(as_tuple=True)[0]
        batch = batch[selected_indices]
        return batch

    def validate_d4pg(self, env, validation_samples, batch_size):
        """
        Validates the D4PG algorithm using the given enviroment and validation samples.

        Args:
            env (ArcBatchGridEnv): The enviroment to use for validation.
            validation_samples (DataLoader): A DataLoader containing the validation samples.
            batch_size (int): The size of the batch to sample from the validation samples.
        """

    def train_d4pg(
        self,
        env,
        train_samples,
        batch_size,
        max_steps=-1,
        warmup_buffer_ratio=0.2,
        validation_frequency=100,
    ):
        """
        Trains the D4PG algorithm using the given enviroment and train samples.

        Args:
            env (ArcBatchGridEnv): The enviroment to use for training.
            train_samples (DataLoader): A DataLoader containing the training samples.
            batch_size (int): The size of the batch to sample from the replay buffer.
            max_steps (int, optional): The maximum number of steps to take in the enviroment. Defaults to -1 (no limit).
            warmup_buffer_ratio (float, optional): The ratio of the replay buffer to fill before sampling. Defaults to 0.2.
            validation_frequency (int, optional): The frequency of validation steps. Defaults to 100.
        """
        assert (
            env.n_steps == self.n_steps
        ), "n-steps in enviroment ({}) is different from n-steps in algorithm ({})".format(
            env.n_steps, self.n_steps
        )
        assert (
            self.n_steps <= max_steps or max_steps < 0
        ), "max_steps must be greater or equal to n_steps"
        assert (
            self.replay_buffer is None
            or self.replay_buffer.storage.max_size >= batch_size
        ), "Replay buffer size is too small for the given batch size. Must grater or equal to batch_size"
        global_step = 0
        global_episode = 0
        for episode, samples in enumerate(train_samples):
            global_episode += 1
            episode_reward = {"pixel_wise": 0.0, "binary": 0.0}
            env.reset(
                options={"batch": samples["task"], "examples": samples["examples"]},
                seed=episode,
            )
            done = False
            episode_step = 0
            while not done and (max_steps <= -1 or max_steps > 0):
                episode_step += 1
                global_step += 1
                max_steps -= 1
                state, reward, actions, next_state, done, truncated = self.step(env)

                episode_reward["pixel_wise"] += torch.sum(reward["pixel_wise"])
                episode_reward["binary"] += torch.sum(reward["binary"])

                # Store the experience in the replay buffer
                if self.replay_buffer is not None and episode_step >= self.n_steps:
                    batch = self.replay_buffer_step(
                        batch_size,
                        state,
                        actions,
                        reward,
                        next_state,
                        warmup_buffer_ratio,
                    )
                elif self.replay_buffer is None:
                    batch = TensorDict(
                        {
                            "state": state,
                            "action": actions,
                            "reward": reward,
                            "next_state": next_state,
                            "terminated": next_state["terminated"],
                        },
                    ).auto_batch_size_()
                else:
                    batch = None

                if batch is not None:
                    batch = self.fileter_compleated_state(batch)
                    loss_actor, loss_critic = self.train_step(
                        batch=batch,
                        actor_optimizer=self.actor_optimizer,
                        critic_optimizer=self.critic_optimizer,
                        gamma=self.gamma,
                    )

                    if global_step % self.target_update_frequency == 0:
                        self.update_target_networks(tau=self.tau)
                done = done or truncated

            logger.debug(
                "Episode {}, Reward['pixel_wise']: {:.2f} Reward['binary']: {:.2f}, Actor Loss: {:.4f}, Critic Loss: {:.4f}".format(
                    episode,
                    episode_reward["pixel_wise"],
                    episode_reward["binary"],
                    loss_actor,
                    loss_critic,
                )
            )
