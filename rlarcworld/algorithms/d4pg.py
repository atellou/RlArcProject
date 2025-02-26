import os
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from tensordict import TensorDict
from torchrl.data import ReplayBuffer

from rlarcworld.enviroments.wrappers.rewards import PixelAwareRewardWrapper
from rlarcworld.arc_dataset import ArcDataset
from rlarcworld.utils import (
    categorical_projection,
    get_nested_ref,
    enable_cuda,
    BetaScheduler,
)
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
        policy_lr: float = 3e-4,
        critic_lr: float = 3e-3,
        lr_scheduler_kwargs: dict = None,
        validation_samples: DataLoader = None,
        replay_buffer: ReplayBuffer = None,
        warmup_buffer_ratio: float = 0.2,
        beta_scheduler: BetaScheduler = None,
        n_steps: int = 1,
        gamma: float = 0.99,
        carsm=False,
        entropy_coef: float = 0.01,
        entropy_coef_decay: float = 0.995,
        tau: float = 0.001,
        target_update_frequency: int = 10,
        tb_writer: SummaryWriter = None,
        history_file: str = None,
        extras_hparams: dict = None,
        save_path: str = None,
        config=None,
    ):
        """
        D4PG (Distributed Distributional Deep Deterministic Policy Gradient) implementation.

        This class implements the D4PG algorithm for reinforcement learning with the following key features:
        - Distributional critic that models value distributions rather than expectations
        - N-step returns for more accurate value estimation
        - Prioritized experience replay for more efficient learning
        - Target networks with soft updates for stability
        - Support for both pixel-wise and binary rewards

        Args:
            env (PixelAwareRewardWrapper): Environment wrapper that provides pixel-aware rewards
            train_samples (DataLoader): DataLoader providing training samples
            actor (torch.nn.Module): Actor network that outputs action probabilities
            critic (torch.nn.Module): Critic network that outputs value distributions
            batch_size (int): Size of batches for training
            policy_lr (float, optional): Learning rate for actor network. Defaults to 3e-4
            critic_lr (float, optional): Learning rate for critic network. Defaults to 3e-3
            lr_scheduler_kwargs (dict, optional): Keyword arguments for learning rate scheduler StepLR. Defaults to None
            validation_samples (DataLoader, optional): DataLoader for validation samples
            replay_buffer (ReplayBuffer, optional): Buffer for experience replay
            warmup_buffer_ratio (float, optional): Ratio of buffer to fill before training. Defaults to 0.2
            beta_scheduler (BetaScheduler, optional): Scheduler for beta values in prioritized experience replay. Defaults to None
            n_steps (int, optional): Number of steps for n-step returns. Defaults to 1
            gamma (float, optional): Discount factor. Defaults to 0.99
            carsam (bool, optional): Whether to use CARSAM actor computation of loss. Defaults to False
            entropy_coef (float, optional): Initial coefficient for entropy regularization. Higher values encourage more exploration. Defaults to 0.01
            entropy_coef_decay (float, optional): Rate at which entropy coefficient decays over time. Values closer to 1.0 decay more slowly. Defaults to 0.995
            tau (float, optional): Target network update rate. Defaults to 0.001
            target_update_frequency (int, optional): Steps between target updates. Defaults to 10
            tb_writer (SummaryWriter, optional): TensorBoard writer for logging
            history_file (str, optional): Path to save training history
            extras_hparams (dict, optional): Dictionary of extra hyperparameters to log
            save_path (str, optional): Path to save model checkpoints
            config (dict, optional): Configuration dictionary with device, and weather to use AMP and checkpointing

        Methods:
            step(): Performs a single environment step using current policy
            compute_loss(): Computes actor and critic losses for a batch of transitions
            fit(): Main training loop that handles training and validation
            validation_process(): Runs validation episodes
            update_target_networks(): Updates target network weights
            compute_target_distribution(): Computes target distribution for critic training
            compute_critic_loss(): Computes critic loss given current states/actions
            compute_actor_loss(): Computes actor loss for policy improvement
            replay_buffer_step(): Handles experience storage and sampling
            episodes_simulation(): Simulates episodes in the environment
            env_simulation(): Handles environment simulation with logging
            categorize_actions(): Converts action probabilities to discrete actions
            history_add(): Adds values to training history
            fileter_compleated_state(): Filters out completed states from batches
        """
        assert isinstance(
            train_samples, DataLoader
        ), "train_samples must be an instance of DataLoader"
        assert isinstance(
            env, PixelAwareRewardWrapper
        ), "Environment must be an instance of PixelAwareRewardWrapper"
        assert (
            env.n_steps == n_steps
        ), "n-steps in enviroment ({}) is different from n-steps in algorithm ({})".format(
            env.n_steps, n_steps
        )
        assert (
            replay_buffer is None or replay_buffer.storage.max_size >= batch_size
        ), "Replay buffer size is too small for the given batch size. Must grater or equal to batch_size"

        # Store parameters
        self.learning_rate_actor = policy_lr
        self.learning_rate_critic = critic_lr
        self.n_steps = n_steps
        self.gamma = gamma
        self.carsm = carsm
        self.entropy_coef = entropy_coef
        self.entropy_coef_decay = entropy_coef_decay
        self.tau = tau
        self.target_update_frequency = target_update_frequency
        self.replay_buffer = replay_buffer
        self.warmup_buffer_ratio = warmup_buffer_ratio
        self.beta_scheduler = beta_scheduler
        self.tb_writer = tb_writer
        self.config = enable_cuda() if config is None else config
        self.device = self.config["device"]
        self.amp_scaler = (
            torch.amp.GradScaler(device=self.device)
            if self.config.get("use_amp", False)
            else None
        )

        self.history = {}
        if extras_hparams is None:
            self.extras_hparams = {}

        if self.tb_writer is not None:
            self.writer_base_dir = tb_writer.log_dir
            if save_path is None:
                save_path = self.writer_base_dir
        else:
            self.writer_base_dir = None

        if save_path is not None:
            self.save_path = save_path
        else:
            self.save_path = None

        self.history_file = history_file

        if self.replay_buffer is None:
            self.batch_size = train_samples.batch_size
        else:
            self.batch_size = batch_size

        if self.replay_buffer is not None:
            logger.debug("Replay buffer initialized")

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
        self.actor_optimizer = optim.AdamW(actor.parameters(), lr=policy_lr)
        self.critic_optimizer = optim.AdamW(critic.parameters(), lr=critic_lr)

        # Create schedulers
        if lr_scheduler_kwargs is None:
            self.actor_scheduler = None
            self.critic_scheduler = None
        else:
            self.actor_scheduler = optim.lr_scheduler.StepLR(
                self.actor_optimizer, **lr_scheduler_kwargs
            )
            self.critic_scheduler = optim.lr_scheduler.StepLR(
                self.critic_optimizer, **lr_scheduler_kwargs
            )

        # Create loss criterion
        self.critic_criterion = torch.nn.KLDivLoss(reduction="batchmean")

        # Move networks and optimizers to device
        self.actor = self.actor.to(self.device)
        self.critic = self.critic.to(self.device)
        self.actor_target = self.actor_target.to(self.device)
        self.critic_target = self.critic_target.to(self.device)

    def apply_decay(self):
        """
        Applies entropy coefficient decay.
        """
        if self.actor_scheduler is not None:
            self.actor_scheduler.step()
        if self.critic_scheduler is not None:
            self.critic_scheduler.step()
        self.entropy_coef *= self.entropy_coef_decay
        if self.replay_buffer is not None and self.beta_scheduler is not None:
            self.replay_buffer.sampler.beta = self.beta_scheduler.step()

    def log_parameters(self, step):
        """
        Logs the current entropy coefficient to TensorBoard.

        Args:
            step (int): The current training step.
        """
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(
                "Parameters/EntropyCoef", self.entropy_coef, global_step=step
            )
            self.tb_writer.add_scalar("Parameters/Gamma", self.gamma, global_step=step)
            self.tb_writer.add_scalar(
                "Parameters/TargetUpdateFrequency",
                self.target_update_frequency,
                global_step=step,
            )
            self.tb_writer.add_scalars(
                "Parameters/LearningRate",
                {
                    "Critic": self.critic_optimizer.param_groups[0]["lr"],
                    "Actor": self.actor_optimizer.param_groups[0]["lr"],
                },
                global_step=step,
            )
            self.tb_writer.add_scalars(
                "Parameters/ReplayBuffer",
                {
                    "Alpha": (
                        self.replay_buffer.sampler.alpha
                        if self.replay_buffer is not None
                        else 0
                    ),
                    "Beta": (
                        self.replay_buffer.sampler.beta
                        if self.replay_buffer is not None
                        else 0
                    ),
                },
                global_step=step,
            )

    def history_add(self, key, value):
        """
        Adds a value to the training history.

        Args:
            key (str): The key for the value to be added.
            value (any): The value to be added.
        """
        ref, last_key = get_nested_ref(self.history, key, default=np.array([]))
        ref[last_key] = np.append(ref[last_key], value)

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
            ).to(self.device)
        return torch.cat(
            [torch.argmax(x, dim=-1).unsqueeze(-1) for x in actions.values()],
            dim=-1,
        ).to(self.device)

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

        Returns:
            torch.Tensor: Projected target distribution (batch_size, num_atoms).
        """
        with torch.no_grad():
            # Use the target actor to get the best action for each next state
            if self.amp_scaler is not None:
                with torch.amp.autocast(device_type="cuda"):
                    action_probs = self.actor_target(next_state)
            else:
                action_probs = self.actor_target(next_state)

            # Get the Q-distribution for the best action from the target critic
            if self.amp_scaler is not None:
                with torch.amp.autocast(device_type="cuda"):
                    best_next_q_dist = self.critic_target(next_state, action_probs)
            else:
                best_next_q_dist = self.critic_target(next_state, action_probs)

        # Project the distribution using the categorical projection
        assert best_next_q_dist.keys() == reward.keys(), ValueError(
            "Keys of best_next_q_dist and reward must be the same"
        )
        target_dist = TensorDict(
            {
                key: categorical_projection(
                    best_next_q_dist[key],  # next_q_dist
                    reward[key],  # rewards
                    terminated,  # terminated
                    self.gamma,  # gamma
                    self.critic_target.z_atoms[key],  # z_atoms
                    self.critic_target.v_min[key],  # v_min
                    self.critic_target.v_max[key],  # v_max
                    self.critic_target.num_atoms[key],  # num_atoms
                    apply_softmax=True,  # apply_softmax
                    n_steps=self.n_steps if key == "n_reward" else 1,  # n_steps
                )
                for key in best_next_q_dist.keys()
            }
        ).to(self.device)

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
        if self.amp_scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                q_dist = self.critic(state, action)
        else:
            q_dist = self.critic(state, action)
        # TD Error
        loss = TensorDict({}).to(self.device)
        td_error = TensorDict({}).to(self.device)
        for key in q_dist.keys():
            if compute_td_error:
                td_error[key] = torch.abs(
                    torch.sum(q_dist[key] * self.critic.z_atoms[key], dim=-1)
                    - torch.sum(target_dist[key] * self.critic.z_atoms[key], dim=-1)
                )
            # KL Divergence
            loss[key] = self.critic_criterion(q_dist[key], target_dist[key])

        return loss, td_error, q_dist

    def compute_actor_loss(
        self, state, target_q=None, action_probs=None, critic_probs=None
    ):
        """
        Compute the loss for the actor network.

        Args:
            state (TensorDict): TensorDict of current states.
            target_q (TensorDict): TensorDict of target Q-values.
            action_probs (TensorDict, optional): Pre-computed action probabilities. If None, will be computed using actor network.
            critic_probs (TensorDict, optional): Pre-computed critic probabilities. If None, will be computed using critic network.

        Returns:
            torch.Tensor: Actor loss (negative expected Q-value).
        """
        assert (
            not self.carsm or target_q is not None
        ), "target_q must be provided when usin CARSM loss."
        # Get action probabilities from the actor
        if action_probs is None:
            if self.amp_scaler is not None:
                with torch.amp.autocast(device_type="cuda"):
                    action_probs = self.actor(state)
            else:
                action_probs = self.actor(state)
        else:
            self.actor.output_val(action_probs)

        # Compute gradient of expected Q-value w.r.t. actions
        if critic_probs is None:
            if self.amp_scaler is not None:
                with torch.amp.autocast(device_type="cuda"):
                    critic_probs = self.critic(state, action_probs)
            else:
                critic_probs = self.critic(state, action_probs)
        else:
            self.critic.output_val(critic_probs)

        if self.entropy_coef != 0 or self.carsm:
            entropy = 0
            log_probs = []
            action = self.categorize_actions(action_probs, as_dict=True)
            for key in action_probs.keys():
                if self.carsm:
                    dist = torch.distributions.Categorical(probs=action_probs[key])
                    log_probs.append(dist.log_prob(action[key]))
                if self.entropy_coef != 0:
                    entropy += -torch.sum(
                        action_probs[key] * torch.log(action_probs[key] + 1e-10), dim=-1
                    ).mean()

            entropy /= len(action_probs.keys()) / len(critic_probs.keys())
            if self.carsm:
                log_probs = torch.sum(torch.stack(log_probs, dim=-1), dim=-1).unsqueeze(
                    -1
                ) / len(action_probs.keys())

        # Compute Q-Values
        loss = TensorDict({})
        for key in critic_probs.keys():
            loss[key] = torch.sum(
                critic_probs[key]
                * self.critic.z_atoms[key].to(critic_probs[key].device),
                dim=-1,
                keepdim=True,
            )
            if self.carsm:
                loss[key] = target_q[key] - loss[key]
                loss[key] = loss[key] * log_probs
            loss[key] = -torch.mean(loss[key])

            if self.entropy_coef != 0:
                loss[key] = loss[key] - self.entropy_coef * entropy

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
            actions (TensorDict): The actions taken by the actor.
            reward (TensorDict): The rewards received from the environment.
            next_state (TensorDict): The next state of the environment.

        Returns:
            TensorDict or None: A batch of transitions if the replay buffer is full, otherwise None.
        """
        sampled_batch_size = next_state["terminated"].shape[0]
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
            batch = self.replay_buffer.sample(self.batch_size).to(self.device)
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
        selected_indices = mask.nonzero(as_tuple=True)[0].cpu()
        batch = batch[selected_indices]
        return batch

    def compute_loss(self, batch, training=True, tb_writer_tag=None, global_step=None):
        """
        Computes actor and critic losses for a batch of transitions.

        Args:
            batch (TensorDict): Batch of transitions containing:
                - state (TensorDict): Current states
                - actions (TensorDict): Actions taken
                - reward (TensorDict): Rewards received
                - next_state (TensorDict): Next states
                - terminated (torch.Tensor): Terminal flags
            training (bool, optional): Whether to compute gradients and update networks. Defaults to True.
            tb_writer_tag (str, optional): Tag for TensorBoard logging. Defaults to None.
            global_step (int, optional): Global step for TensorBoard logging. Required if tb_writer_tag is provided.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Actor loss and critic loss
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
                if self.amp_scaler is not None:
                    with torch.amp.autocast(device_type="cuda"):
                        self.amp_scaler.scale(loss_critic).backward()
                else:
                    loss_critic.backward()
                # NOTE: Generating errors in tensorboard
                # if tb_writer_tag is not None and self.tb_writer is not None:
                #     assert global_step is not None, "global_step must be provided"
                #     for name, param in self.critic.named_parameters():
                #         if param.grad is not None:
                #             try:
                #                 self.tb_writer.add_histogram(
                #                     os.path.join(tb_writer_tag, "/Grads/critic/", name),
                #                     param.grad,
                #                     global_step,
                #                 )
                #             except Exception as e:
                #                 logger.error(
                #                     f"Failed to add histogram for {name} of critic params with shape {param.shape} in {tb_writer_tag}. Skipping..."
                #                 )
                #                 logger.error(e)

                if self.amp_scaler is not None:
                    with torch.amp.autocast(device_type="cuda"):
                        self.amp_scaler.step(self.critic_optimizer)
                        self.amp_scaler.update()
                else:
                    self.critic_optimizer.step()
                # Actor update
                loss_actor = self.compute_actor_loss(batch["state"], target_probs)
            else:
                loss_actor = self.compute_actor_loss(
                    batch["state"],
                    target_probs,
                    action_probs=batch["actions"],
                    critic_probs=q_dist,
                )
            loss_actor = tuple(loss_actor.values())
            loss_actor = sum(loss_actor) / len(loss_actor)

            if training:
                self.actor_optimizer.zero_grad()
                if self.amp_scaler is not None:
                    with torch.amp.autocast(device_type="cuda"):
                        self.amp_scaler.scale(loss_actor).backward()
                else:
                    loss_actor.backward()
                # NOTE: Generating errors in tensorboard
                # if tb_writer_tag is not None and self.tb_writer is not None:
                #     assert global_step is not None, "global_step must be provided"
                #     for name, param in self.actor.named_parameters():
                #         try:
                #             if param.grad is not None:
                #                 self.tb_writer.add_histogram(
                #                     os.path.join(tb_writer_tag, "/Grads/actor/", name),
                #                     param.grad,
                #                     global_step,
                #                 )
                #         except Exception as e:
                #             logger.error(
                #                 f"Failed to add histogram for {name} of actor params with shape {param.shape} in {tb_writer_tag}. Skipping..."
                #             )
                #             logger.error(e)
                if self.amp_scaler is not None:
                    with torch.amp.autocast(device_type="cuda"):
                        self.amp_scaler.step(self.actor_optimizer)
                        self.amp_scaler.update()
                else:
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
            state = env.get_state(unsqueeze=1).to(self.device)
            if self.amp_scaler is not None:
                with torch.amp.autocast(device_type="cuda"):
                    action_probs = self.actor(state)
            else:
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
                if self.critic.v_min.get("n_reward") is not None:
                    reward["n_reward"] = (
                        env.n_step_reward(
                            v_min=self.critic.v_min.get("n_reward"),
                            v_max=self.critic.v_max.get("n_reward"),
                        )
                        .unsqueeze(-1)
                        .to(self.device)
                    )
                else:
                    reward["n_reward"] = (
                        env.n_step_reward().unsqueeze(-1).to(self.device)
                    )

        return (
            TensorDict(
                {
                    "state": state.auto_batch_size_(),
                    "reward": reward.auto_batch_size_(),
                    "actions": action_probs.auto_batch_size_(),
                    "next_state": next_state.auto_batch_size_(),
                    "done": done,
                    "truncated": truncated,
                }
            )
            .auto_batch_size_()
            .to(self.device)
        )

    def episodes_simulation(self, env, samples, max_steps: int = -1, seed: int = None):
        """
        Simulates episodes in the environment using the current policy.

        Args:
            env: The environment to interact with.
            samples: Samples containing task and examples data for the environment.
            max_steps (int, optional): Maximum number of steps per episode. -1 means no limit. Defaults to -1.
            seed (int, optional): Random seed for reproducibility. Defaults to None.

        Yields:
            Tuple containing:
                - step_number (int): Current step number in the episode
                - step_state (TensorDict): Dictionary containing:
                    - state: Current environment state
                    - reward: Rewards received from the environment
                    - actions: Action probabilities from the actor
                    - next_state: Next environment state
                    - done: Whether episode is done
                    - truncated: Whether episode was truncated
        """
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
            logger.debug(f"Episode simulation step: {step_number}")

    def env_simulation(
        self,
        env,
        samples,
        max_steps=-1,
        tb_writer_tag=None,
        merge_graphs=True,
        **kwargs,
    ):
        """
        Simulates episodes in the environment and handles logging.

        Args:
            env: The environment to interact with
            samples: Samples containing task and examples data for the environment
            max_steps (int, optional): Maximum number of steps per episode. -1 means no limit. Defaults to -1
            tb_writer_tag (str, optional): Tag for TensorBoard logging. Defaults to None
            merge_graphs (bool, optional): Whether to merge graphs in TensorBoard. Defaults to True

        Yields:
            Tuple containing:
                - episode_number (int): Current episode number
                - step_state (TensorDict): Dictionary containing:
                    - state: Current environment state
                    - reward: Rewards received from the environment
                    - actions: Action probabilities from the actor
                    - next_state: Next environment state
                    - done: Whether episode is done
                    - truncated: Whether episode was truncated

        Raises:
            AssertionError: If max_steps is less than n_steps when max_steps is not -1
        """
        assert (
            self.n_steps <= max_steps or max_steps < 0
        ), "max_steps must be greater or equal to n_steps"

        for episode_number, sample_batch in enumerate(samples):
            sample_batch = TensorDict(sample_batch).to(self.device)
            kwargs.update(
                {
                    "batch": sample_batch["task"],
                    "examples": sample_batch["examples"],
                }
            )
            env.reset(
                options=kwargs,
                seed=episode_number,
            )
            for step_number, step_state in self.episodes_simulation(
                env, sample_batch, max_steps, seed=episode_number
            ):
                yield episode_number, step_state

                if tb_writer_tag is not None and self.tb_writer is not None:
                    tag = os.path.join(
                        tb_writer_tag,
                        f"Episode/{episode_number}",
                    )
                    mask = step_state["state"]["terminated"] == 0
                    selected_indices = mask.nonzero(as_tuple=True)[0].cpu()
                    rewards = step_state["reward"][selected_indices]
                    for k, v in rewards.items():
                        value = torch.mean(v).item()
                        path = os.path.join(tag, f"Reward/{k}")
                        self.history_add(path, value)
                        if merge_graphs:
                            path = f"Reward/{k}"
                            value = {tag: value}
                            self.tb_writer.add_scalars(
                                path,
                                value,
                                step_number,
                            )
                        else:
                            self.tb_writer.add_scalar(
                                path,
                                value,
                                step_number,
                            )

                    compleatition_percentage = (
                        torch.count_nonzero(step_state["state"]["terminated"])
                        / mask.shape[0]
                    ).item()
                    path = os.path.join(tag, f"Metric/Completition[%]")
                    self.history_add(path, value)
                    if merge_graphs:
                        path = f"Metric/Completition[%]"
                        compleatition_percentage = {tag: compleatition_percentage}
                        self.tb_writer.add_scalars(
                            path,
                            compleatition_percentage,
                            step_number,
                        )
                    else:
                        self.tb_writer.add_scalar(
                            path,
                            compleatition_percentage,
                            step_number,
                        )

    def validation_process(
        self,
        max_steps=-1,
        tb_writer_tag="Validation",
        logger_frequency=1000,
        merge_graphs=True,
    ):
        """
        Performs validation on the validation environment.

        Args:
            max_steps (int, optional): Maximum number of steps per episode. -1 means no limit. Defaults to -1
            tb_writer_tag (str, optional): Tag for TensorBoard logging. Defaults to "Validation"
            logger_frequency (int, optional): Frequency of logging. Defaults to 1000
            merge_graphs (bool, optional): Whether to merge graphs in TensorBoard. Defaults to True

        Returns:
            Generator yielding:
                - episode_number (int): Current episode number
                - step_number (int): Current step number in the episode
                - loss_actor (torch.Tensor): Actor loss
                - loss_critic (torch.Tensor): Critic loss
                - step_state (TensorDict): Dictionary containing:
                    - state: Current environment state
                    - reward: Rewards received from the environment
                    - actions: Action probabilities from the actor
                    - next_state: Next environment state
                    - done: Whether episode is done
                    - truncated: Whether episode was truncated
        """
        with torch.no_grad():
            for step_number, (episode_number, step_state) in enumerate(
                self.env_simulation(
                    self.validation_env,
                    self.validation_samples,
                    max_steps=max_steps,
                    tb_writer_tag=tb_writer_tag,
                    merge_graphs=merge_graphs,
                    is_train=False,
                )
            ):
                step_state["terminated"] = step_state["next_state"]["terminated"]
                mask = step_state["state"]["terminated"] == 0
                selected_indices = mask.nonzero(as_tuple=True)[0].cpu()
                for key in step_state.keys():
                    if isinstance(step_state[key], TensorDict):
                        step_state[key] = step_state[key][selected_indices]
                loss_actor, loss_critic = self.compute_loss(step_state, training=False)
                yield episode_number, step_number, loss_actor, loss_critic, step_state

                if self.tb_writer is not None and step_number % logger_frequency == 0:
                    batch_size = step_state["reward"].batch_size[0]
                    value = loss_actor.item() / batch_size
                    path = os.path.join(tb_writer_tag, "Loss/actor")
                    self.history_add(path, value)
                    if merge_graphs:
                        path = "Loss/actor"
                        value = {tb_writer_tag: value}
                        self.tb_writer.add_scalars(
                            path,
                            value,
                            step_number,
                        )
                    else:
                        self.tb_writer.add_scalar(
                            path,
                            value,
                            step_number,
                        )
                    value = loss_critic.item() / batch_size
                    path = os.path.join(tb_writer_tag, "Loss/critic")
                    self.history_add(path, value)
                    if merge_graphs:
                        path = "Loss/critic"
                        value = {tb_writer_tag: value}
                        self.tb_writer.add_scalars(
                            path,
                            value,
                            step_number,
                        )
                    else:
                        self.tb_writer.add_scalar(
                            path,
                            value,
                            step_number,
                        )

    def fit(
        self,
        epochs=1,
        max_steps=-1,
        validation_steps_frequency=-1,
        validation_steps_per_train_step=-1,
        validation_steps_per_episode=-1,
        logger_frequency=1000,
        grads_logger_frequency=1000000,
        tb_writer_tag="Train",
        validation_tb_writer_tag="Validation",
        merge_graphs=True,
    ):
        """
        Train the model using the provided samples.

        Args:
            epochs (int, optional): Number of epochs to train. Defaults to 1.
            max_steps (int, optional): Maximum number of steps per episode. -1 means no limit. Defaults to -1
            validation_steps_frequency (int, optional): Frequency of validation steps. -1 means no validation. Defaults to -1
            validation_steps_per_train_step (int, optional): Number of validation steps per train step. -1 means all steps. Defaults to -1
            validation_steps_per_episode (int, optional): Number of validation steps per episode. -1 means all steps. Defaults to -1
            logger_frequency (int, optional): Logging frequency. Defaults to 1000
            grads_logger_frequency (int, optional): Gradient logging frequency. Defaults to 1000000
            tb_writer_tag (str, optional): Tag for TensorBoard logging. Defaults to "Train"
            validation_tb_writer_tag (str, optional): Tag for TensorBoard validation logging. Defaults to "Validation"
            merge_graphs (bool, optional): Whether to merge graphs in TensorBoard. Defaults to True

        """
        self.iteration = 0
        decay_flag = False
        for epoch_n in range(epochs):
            logger.info(f"Epoch: {epoch_n}, Step runned: {self.iteration}")
            for step_number, (episode_number, step_state) in enumerate(
                self.env_simulation(
                    self.train_env,
                    self.train_samples,
                    max_steps=max_steps,
                    tb_writer_tag=tb_writer_tag,
                )
            ):
                logger.debug(
                    f"Epoch: {epoch_n}, Step: {step_number}, Episode: {episode_number}"
                )
                self.log_parameters(self.iteration)
                # Store the experience in the replay buffer
                if self.replay_buffer is not None:
                    batch = self.replay_buffer_step(
                        step_state["state"],
                        step_state["actions"],
                        step_state["reward"],
                        step_state["next_state"],
                    )
                elif self.replay_buffer is None:
                    batch = (
                        TensorDict(
                            {
                                "state": step_state["state"],
                                "actions": step_state["actions"],
                                "reward": step_state["reward"],
                                "next_state": step_state["next_state"],
                                "terminated": step_state["next_state"]["terminated"],
                            },
                        )
                        .auto_batch_size_()
                        .to(self.device)
                    )
                else:
                    batch = None

                if batch is None:
                    continue

                batch = self.fileter_compleated_state(batch)

                if (
                    self.tb_writer is not None
                    and self.iteration % grads_logger_frequency == 0
                ):
                    loss_actor, loss_critic = self.compute_loss(
                        batch,
                        training=True,
                        tb_writer_tag=tb_writer_tag,
                        global_step=self.iteration,
                    )
                    logger.info(
                        f"Epoch: {epoch_n}, Step: {step_number}, Episode: {episode_number}, Loss actor: {loss_actor}, Loss critic: {loss_critic}"
                    )
                else:
                    loss_actor, loss_critic = self.compute_loss(batch, training=True)
                if (
                    self.tb_writer is not None
                    and self.iteration % logger_frequency == 0
                ):
                    batch_size = step_state["reward"].batch_size[0]
                    value = loss_actor.item() / batch_size
                    path = os.path.join(tb_writer_tag, "Loss/actor")
                    self.history_add(path, value)
                    if merge_graphs:
                        path = "Loss/actor"
                        value = {tb_writer_tag: value}
                        self.tb_writer.add_scalars(path, value, self.iteration)
                    else:
                        self.tb_writer.add_scalar(path, value, self.iteration)

                    value = loss_critic.item() / batch_size
                    path = os.path.join(tb_writer_tag, "Loss/critic")
                    self.history_add(path, value)
                    if merge_graphs:
                        path = "Loss/critic"
                        value = {tb_writer_tag: value}
                        self.tb_writer.add_scalars(path, value, self.iteration)
                    else:
                        self.tb_writer.add_scalar(path, value, self.iteration)
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
                            max_steps=validation_steps_per_episode,
                            tb_writer_tag=validation_tb_writer_tag,
                            merge_graphs=merge_graphs,
                        )
                    for (
                        val_episode_number,
                        val_step_number,
                        val_loss_actor,
                        val_loss_critic,
                        val_step_state,
                    ) in self.running_validation_process:
                        if val_episode_number > initial_episode_number or (
                            validation_steps_per_train_step > 0
                            and val_step_number % validation_steps_per_train_step == 0
                        ):
                            initial_episode_number = copy.copy(val_episode_number)
                            logger.info(
                                f"Epoch: {epoch_n}, Step: {step_number}, Episode: {episode_number}, Validation loss actor: {val_loss_actor}, Validation loss critic: {val_loss_critic}"
                            )
                            break

                if batch is not None:
                    if decay_flag:
                        self.apply_decay()

                    if step_number % self.target_update_frequency == 0:
                        logger.debug("Updating networks step {}".format(step_number))
                        self.update_target_networks(tau=self.tau)
                    decay_flag = True

                self.iteration += 1
        self.iteration -= 1
        logger.info("Total steps: {}".format(self.iteration))
        if hasattr(self, "validation_env"):
            logger.info("Running last validation process")
            for (
                val_episode_number,
                val_step_number,
                val_loss_actor,
                val_loss_critic,
                val_step_state,
            ) in self.validation_process(
                max_steps=validation_steps_per_episode,
                tb_writer_tag="End" + validation_tb_writer_tag,
                merge_graphs=merge_graphs,
            ):
                pass

        if self.tb_writer is not None:
            comp_percent = step_state["state"]["terminated"].sum().item() / len(
                step_state["state"]["terminated"]
            )
            val_comp_percent = val_step_state["state"]["terminated"].sum().item() / len(
                val_step_state["state"]["terminated"]
            )
            self.extras_hparams.update(
                {
                    "lr_actor": self.learning_rate_actor,
                    "lr_critic": self.learning_rate_critic,
                    "bsize": self.batch_size,
                    "gamma": self.gamma,
                    "carsm": self.carsm,
                    "tau": self.tau,
                    "target_update_frequency": self.target_update_frequency,
                    "entropy_coef": self.entropy_coef,
                    "entropy_coef_decay": self.entropy_coef_decay,
                    "n_steps": self.n_steps,
                }
            )
            self.tb_writer.add_hparams(
                self.extras_hparams,
                {
                    "hparam/train_loss_actor": loss_actor,
                    "hparam/train_loss_critic": loss_critic,
                    "hparam/validation_loss_actor": val_loss_actor,
                    "hparam/validation_loss_critic": val_loss_critic,
                    "hparam/train_completion": comp_percent,
                    "hparam/validation_completion": val_comp_percent,
                },
            )
            self.tb_writer.flush()
            self.tb_writer.close()

        if self.save_path is not None and loss_actor is not None:
            logger.info("Saving model in {}".format(self.save_path))
            self.save_model(self.save_path)

    def checkpoint(
        self,
        epoch_number=None,
        episode_number=None,
        step_number=None,
        loss_actor=None,
        loss_critic=None,
    ):
        logger.debug("Saving checkpoint for iteration {}".format(self.iteration))
        checkpoint = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer_actor": self.actor_optimizer.state_dict(),
            "optimizer_critic": self.critic_optimizer.state_dict(),
            "lr_schedules": {
                "actor": self.actor_scheduler.state_dict(),
                "critic": self.critic_scheduler.state_dict(),
            },
            "iteration": self.iteration,
            "hyperparameters": {
                "batch_size": self.batch_size,
                "gamma": self.gamma,
                "carsm": self.carsm,
                "tau": self.tau,
                "target_update_frequency": self.target_update_frequency,
                "entropy_coef": self.entropy_coef,
                "entropy_coef_decay": self.entropy_coef_decay,
                "n_steps": self.n_steps,
            },
        }
        checkpoint["hyperparameters"].update(self.extras_hparams)
        return checkpoint

    def save_checkpoint(self, path):
        torch.save(self.checkpoint(), path)

    def load_checkpoint(self, checkpoint):
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])

    def save_model(self, path):
        if path.startswith("gs://"):
            from google.cloud import storage

            bucket_name = path.split("gs://")[-1]
            prefix = bucket_name.split("/")
            bucket_name = prefix[0]
            prefix = "/".join(prefix[1:])

            client = storage.Client(os.environ["ML_PROJECT_ID"])
            bucket = client.get_bucket(bucket_name)

            blob = bucket.blob(os.path.join(prefix, "actor.ptc"))
            with blob.open("wb", ignore_flush=True) as f:
                torch.save(self.actor.state_dict(), f)

            blob = bucket.blob(os.path.join(prefix, "critic.ptc"))
            with blob.open("wb", ignore_flush=True) as f:
                torch.save(self.critic.state_dict(), f)
        else:
            os.makedirs(path, exist_ok=True)
            torch.save(self.actor.state_dict(), os.path.join(path, "actor.ptc"))
            torch.save(self.critic.state_dict(), os.path.join(path, "critic.ptc"))

    def load_model(self, path):
        self.actor.load_state_dict(
            torch.load(os.path.join(path, "actor.ptc"), map_location=self.device)
        )
        self.critic.load_state_dict(
            torch.load(os.path.join(path, "critic.ptc"), map_location=self.device)
        )
