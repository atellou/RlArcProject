import copy
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from tensordict import TensorDict
from torchrl.data import ReplayBuffer

from rlarcworld.enviroments.wrappers.rewards import PixelAwareRewardWrapper
from rlarcworld.arc_dataset import ArcDataset
from rlarcworld.utils import categorical_projection

import logging

logger = logging.getLogger(__name__)


class D4PG:

    def __init__(
        self,
        env: PixelAwareRewardWrapper,
        train_samples: DataLoader,
        actor: torch.nn.Module,
        critic: torch.nn.Module,
        batch_size: int,
        validation_samples: DataLoader = None,
        replay_buffer: ReplayBuffer = None,
        warmup_buffer_ratio: float = 0.2,
        n_steps: int = 1,
        gamma: float = 0.99,
        tau: float = 0.001,
        target_update_frequency: int = 10,
    ):
        assert (
            env.n_steps == n_steps
        ), "n-steps in enviroment ({}) is different from n-steps in algorithm ({})".format(
            env.n_steps, n_steps
        )
        assert (
            replay_buffer is None or replay_buffer.storage.max_size >= batch_size
        ), "Replay buffer size is too small for the given batch size. Must grater or equal to batch_size"

        # Store parameters
        self.n_steps = n_steps
        self.gamma = gamma
        self.tau = tau
        self.target_update_frequency = target_update_frequency
        self.replay_buffer = replay_buffer
        self.warmup_buffer_ratio = warmup_buffer_ratio

        if self.replay_buffer is None:
            self.batch_size = train_samples.batch_size
            logger.warning(
                "Replay buffer not set. The environment will use the samples given by the dataloades: {}".format(
                    self.batch_size
                )
            )
        else:
            self.batch_size = batch_size

        if self.replay_buffer is not None:
            logger.debug("Replay buffer initialized")
        else:
            logger.warning("Replay buffer not set. Skipping experience storage.")

        # Store enviroment and samples
        self.train_env = env
        self.train_samples = train_samples

        if validation_samples is not None:
            self.validation_env = copy.deepcopy(env)
            self.validation_samples = validation_samples

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

            # Get the Q-distribution for the best action from the target critic
            best_next_q_dist = self.critic_target(next_state, action_probs)

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
                    self.gamma,
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
            action (TensorDict): TensorDict of actions.
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

        return loss, td_error, q_dist

    def compute_actor_loss(self, state, action_probs=None, critic_probs=None):
        """
        Compute the loss for the actor network.

        Args:
            state (TensorDict): TensorDict of current states.

        Returns:
            torch.Tensor: Actor loss (negative expected Q-value).
        """
        # Get action probabilities from the actor
        if action_probs is None:
            action_probs = self.actor(state)
        else:
            self.actor.output_val(action_probs)

        # Compute gradient of expected Q-value w.r.t. actions
        if critic_probs is None:
            critic_probs = self.critic(state, action_probs)
        else:
            self.critic.output_val(critic_probs)

        # Compute Q-Values
        loss = TensorDict({})
        for key in critic_probs.keys():
            loss[key] = -torch.sum(
                critic_probs[key]
                * self.critic.z_atoms[key].to(critic_probs[key].device),
                dim=-1,
                keepdim=True,
            ).mean()

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

    def replay_buffer_step(
        self,
        state,
        actions,
        reward,
        next_state,
    ):
        """
        Stores a batch of transitions in the replay buffer and samples a new batch of size `batch_size`.

        Args:
            batch_size (int): The size of the batch to sample.
            state (TensorDict): The state of the environment.
            actions (torch.Tensor): The actions taken by the actor.
            reward (TensorDict): The rewards received from the environment.
            next_state (TensorDict): The next state of the environment.

        Returns:
            TensorDict or None: A batch of transitions if the replay buffer is full, otherwise None.
        """
        sampled_batch_size = self.train_samples.batch_size
        priority = (
            torch.ones(sampled_batch_size)
            if len(self.replay_buffer.storage) == 0
            else torch.ones(sampled_batch_size)
            * max(self.replay_buffer.sampler._max_priority)
        )
        batch = torch.stack(
            [
                TensorDict(
                    {
                        "state": state[i],
                        "actions": actions[i],
                        "reward": reward[i],
                        "next_state": next_state[i],
                        "terminated": next_state["terminated"][i],
                    }
                ).auto_batch_size_()
                for i in range(sampled_batch_size)
            ]
        )
        indices = self.replay_buffer.extend(batch)
        self.replay_buffer.update_priority(indices, priority)
        if (
            len(self.replay_buffer.storage)
            > self.replay_buffer.storage.max_size * self.warmup_buffer_ratio
        ):
            batch = self.replay_buffer.sample(self.batch_size)
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

    def compute_loss(self, batch, training=True):
        """
        Perform a training step for the D4PG algorithm.

        Args:
            step (int): The current step in the enviroment.
            batch_size (int): The size of the batch to sample from the replay buffer.
            state (TensorDict): The current state of the enviroment.
            actions (TensorDict): The actions taken by the actor in the current state.
            reward (TensorDict): The rewards received from the enviroment.
            next_state (TensorDict): The next state of the enviroment.

        Returns:
            tuple: The actor and critic losses as a tuple.
        """
        grading_method = "enable_grad" if training else "no_grad"
        with getattr(torch, grading_method)():
            # Compute target distribution
            target_probs = self.compute_target_distribution(
                batch["reward"], batch["next_state"], batch["terminated"]
            )

            # Critic update
            loss_critic, td_error, q_dist = self.compute_critic_loss(
                batch["state"],
                batch["actions"],
                target_probs,
                compute_td_error=self.replay_buffer is not None,
            )
            loss_critic = tuple(loss_critic.values())
            loss_critic = sum(loss_critic) / len(loss_critic)

            if training:
                self.critic_optimizer.zero_grad()
                loss_critic.backward()
                self.critic_optimizer.step()
                # Actor update
                loss_actor = self.compute_actor_loss(batch["state"])
            else:
                loss_actor = self.compute_actor_loss(
                    batch["state"],
                    action_probs=batch["actions"],
                    critic_probs=q_dist,
                )
            loss_actor = tuple(loss_actor.values())
            loss_actor = sum(loss_actor) / len(loss_actor)

            if training:
                self.actor_optimizer.zero_grad()
                loss_actor.backward()
                self.actor_optimizer.step()

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
            action_probs = self.actor(state)
            actions = self.categorize_actions(action_probs, as_dict=True)
            __, reward, done, truncated, __ = env.step(actions)
            reward = TensorDict(
                {
                    "pixel_wise": reward.unsqueeze(-1),
                    "binary": env.get_wrapper_attr("last_reward").unsqueeze(-1),
                }
            ).auto_batch_size_()
            next_state = env.get_state(unsqueeze=1)
            if self.n_steps > 1:
                reward["n_reward"] = env.n_step_reward().unsqueeze(-1)

        return TensorDict(
            {
                "state": state,
                "reward": reward,
                "actions": action_probs,
                "next_state": next_state,
                "done": done,
                "truncated": truncated,
            }
        ).auto_batch_size_()

    def episodes_simulation(self, env, samples, max_steps: int = -1, seed: int = None):
        assert (
            env.n_steps == self.n_steps
        ), "n-steps in enviroment ({}) is different from n-steps in algorithm ({})".format(
            env.n_steps, self.n_steps
        )
        assert (
            self.n_steps <= max_steps or max_steps < 0
        ), "max_steps must be greater or equal to n_steps"
        env.reset(
            options={"batch": samples["task"], "examples": samples["examples"]},
            seed=seed,
        )
        done = False
        step_number = 0
        while not done and (max_steps <= -1 or max_steps >= step_number):
            step_state = self.step(env)
            yield step_number, step_state
            done = step_state["done"] or step_state["truncated"]
            step_number += 1

    def env_simulation(
        self,
        env,
        samples,
        max_steps=-1,
    ):
        """
        Trains the D4PG algorithm using the given enviroment and train samples.

        Args:
            env (ArcBatchGridEnv): The enviroment to use for training.
            train_samples (DataLoader): A DataLoader containing the training samples.
            batch_size (int): The size of the batch to sample from the replay buffer.
            max_steps (int, optional): The maximum number of steps to take in the enviroment. Defaults to -1 (no limit).

        Yields:
            Tuple containing:
                - state (TensorDict): The current state of the environment.
                - reward (TensorDict): The rewards received after taking the action.
                - actions (torch.Tensor): The actions taken by the actor.
                - next_state (TensorDict): The state of the environment after the action.
                - done (bool): Whether the episode has terminated.
                - truncated (bool): Whether the episode has been truncated.
        """
        assert (
            self.n_steps <= max_steps or max_steps < 0
        ), "max_steps must be greater or equal to n_steps"

        for episode_number, sample_batch in enumerate(samples):

            # episode_reward = {"pixel_wise": 0.0, "binary": 0.0}
            env.reset(
                options={
                    "batch": sample_batch["task"],
                    "examples": sample_batch["examples"],
                },
                seed=episode_number,
            )
            for step_number, step_state in self.episodes_simulation(
                env, sample_batch, max_steps, seed=episode_number
            ):
                yield episode_number, step_state

    def validation_process(self, max_steps=-1):
        with torch.no_grad():
            for step_number, (episode_number, step_state) in enumerate(
                self.env_simulation(
                    self.validation_env,
                    self.validation_samples,
                    max_steps=max_steps,
                )
            ):
                step_state["terminated"] = step_state["next_state"]["terminated"]
                loss_actor, loss_critic = self.compute_loss(step_state, training=False)
                yield episode_number, step_number, loss_actor, loss_critic, step_state

    def fit(
        self,
        epochs=1,
        max_steps=-1,
        validation_steps_frequency=-1,
        validation_steps_per_episode=-1,
    ):
        for epoch in range(epochs):
            for step_number, (episode_number, step_state) in enumerate(
                self.env_simulation(
                    self.train_env,
                    self.train_samples,
                    max_steps=max_steps,
                )
            ):
                # Store the experience in the replay buffer
                if self.replay_buffer is not None:
                    batch = self.replay_buffer_step(
                        step_state["state"],
                        step_state["actions"],
                        step_state["reward"],
                        step_state["next_state"],
                    )
                elif self.replay_buffer is None:
                    batch = TensorDict(
                        {
                            "state": step_state["state"],
                            "actions": step_state["actions"],
                            "reward": step_state["reward"],
                            "next_state": step_state["next_state"],
                            "terminated": step_state["next_state"]["terminated"],
                        },
                    ).auto_batch_size_()
                else:
                    return

                if batch is None:
                    continue

                batch = self.fileter_compleated_state(batch)
                loss_actor, loss_critic = self.compute_loss(batch, training=True)

                if (
                    validation_steps_frequency > 0
                    and step_number % validation_steps_frequency == 0
                    and batch is not None
                ):
                    if not hasattr(
                        self, "running_validation_process"
                    ) or initial_episode_number == (len(self.validation_samples) - 1):
                        if validation_steps_per_episode < 0:
                            logger.warning(
                                "VALIDATION process could take a long time,"
                                " it will run indeterminately until the end of the environment."
                                " Meaning, that all grids should be completed to end the process."
                            )
                        initial_episode_number = 0
                        self.running_validation_process = self.validation_process(
                            max_steps=validation_steps_per_episode
                        )
                    for (
                        val_episode_number,
                        val_step_number,
                        val_loss_actor,
                        val_loss_critic,
                        val_step_state,
                    ) in self.running_validation_process:
                        if val_episode_number > initial_episode_number:
                            initial_episode_number = copy.copy(val_episode_number)
                            break

                if step_number % self.target_update_frequency == 0:
                    self.update_target_networks(tau=self.tau)
