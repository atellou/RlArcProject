import copy

import torch
import torch.optim as optim
from tensordict import TensorDict

from rlarcworld.utils.device import enable_cuda
from rlarcworld.utils.distributions import categorical_projection


class AmpScaler:
    def __init__(self, amp_enabled: bool, device: str) -> None:
        if amp_enabled:
            self.scaler = torch.amp.GradScaler(device=device)
            self._cm = torch.amp.autocast(device_type="cuda")
        else:
            self.scaler = None
            self._cm = None

    def __enter__(self):
        if self._cm is not None:
            return self._cm.__enter__()
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._cm is not None:
            return self._cm.__exit__(exc_type, exc_val, exc_tb)
        return False

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        if self.scaler is not None:
            return self.scaler.scale(loss)
        return loss

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        if self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()

    def unscale(self, optimizer: torch.optim.Optimizer) -> None:
        """Unscale gradients for a given optimizer when AMP is enabled.

        Call this before gradient clipping to ensure clipping happens on unscaled grads.
        """
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)


class D4PGCore:
    """
    Minimal, environment-agnostic core for D4PG.

    Responsibilities (algorithm-only):
    - Hold actor/critic and their target copies
    - Compute target distribution (Bellman projection onto atoms)
    - Compute critic and actor losses (including optional CARSM and entropy regularization)
    - Apply gradients (with optional AMP and gradient clipping)
    - Soft-update target networks

    Excludes (non-core):
    - Env stepping, dataloaders, replay buffers, logging/traceability, checkpointing
    """

    def __init__(
        self,
        actor: torch.nn.Module,
        critic: torch.nn.Module,
        *,
        policy_lr: float = 3e-4,
        critic_lr: float = 3e-3,
        lr_scheduler_kwargs: dict | None = None,
        gamma: float = 0.99,
        tau: float = 0.005,
        n_steps: int = 1,
        carsm: bool = False,
        entropy_coef: float = 0.0,
        entropy_coef_decay: float = 1.0,
        config: dict | None = None,
        max_grad_norm: float = 0.0,
        amp_enabled: bool = True,
    ) -> None:
        """
        Initialize the D4PGCore.

        Args:
            actor (torch.nn.Module): Actor network.
            critic (torch.nn.Module): Critic network.
            policy_lr (float, optional): Learning rate for actor network. Defaults to 3e-4.
            critic_lr (float, optional): Learning rate for critic network. Defaults to 3e-3.
            lr_scheduler_kwargs (Optional[Dict], optional): Keyword arguments for learning rate scheduler StepLR. Defaults to None.
            gamma (float, optional): Discount factor. Defaults to 0.99.
            tau (float, optional): Soft update parameter. Defaults to 0.005.
            n_steps (int, optional): Number of steps for n-step returns. Defaults to 1.
            carsm (bool, optional): Whether to use CARSM. Defaults to False.
            entropy_coef (float, optional): Entropy coefficient. Defaults to 0.0.
            entropy_coef_decay (float, optional): Entropy coefficient decay. Defaults to 1.0.
            config (Optional[Dict], optional): Configuration dictionary. Defaults to None.
            max_grad_norm (float, optional): Maximum gradient norm. Defaults to 0.0.
            amp_enabled (bool, optional): Whether to use automatic mixed precision. Defaults to True.
        """
        assert entropy_coef >= 0, "Entropy coefficient must be non-negative"
        assert entropy_coef_decay >= 0, "Entropy coefficient decay must be non-negative"

        # Config/device/AMP
        self.config = enable_cuda(amp_enabled=amp_enabled) if config is None else config
        self.device = self.config["device"]
        self.amp_scaler = AmpScaler(amp_enabled=amp_enabled, device=self.device)

        # Hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.n_steps = n_steps
        self.carsm = carsm
        self.entropy_coef = entropy_coef
        self.entropy_coef_decay = entropy_coef_decay
        self.max_grad_norm = max_grad_norm

        # Networks
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)

        # Target networks
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_target.to(self.device).eval()
        self.critic_target.to(self.device).eval()

        # Optimizers and schedulers
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=policy_lr)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=critic_lr)
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

        # Loss
        self.critic_criterion = torch.nn.KLDivLoss(reduction="batchmean")

    # ---------- Core computations ----------
    def compute_target_distribution(
        self,
        reward: TensorDict,
        next_state: TensorDict,
        terminated: torch.Tensor,
    ) -> TensorDict:
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
            with self.amp_scaler:
                action_probs = self.actor_target(next_state)
                next_q_dist = self.critic_target(next_state, action_probs)

        assert (
            next_q_dist.keys() == reward.keys()
        ), f"Mismatched keys next_q_dist={list(next_q_dist.keys())}, reward={list(reward.keys())}"

        target_dist = TensorDict(
            {
                key: categorical_projection(
                    next_q_dist[key],
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
                for key in next_q_dist.keys()
            }
        ).to(self.device)
        return target_dist

    def compute_critic_loss(
        self,
        state: TensorDict,
        action: TensorDict,
        target_dist: TensorDict,
        compute_td_error: bool = False,
    ) -> tuple[TensorDict, TensorDict, TensorDict]:
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
        # Q distribution from critic
        with self.amp_scaler:
            q_dist = self.critic(state, action)

        loss = TensorDict({}).to(self.device)
        td_error = TensorDict({}).to(self.device)
        for key in q_dist.keys():
            if compute_td_error:
                td_error[key] = torch.abs(
                    torch.sum(q_dist[key] * self.critic.z_atoms[key], dim=-1)
                    - torch.sum(target_dist[key] * self.critic.z_atoms[key], dim=-1)
                )
            loss[key] = self.critic_criterion(q_dist[key], target_dist[key])
        return loss, td_error, q_dist

    def _categorize_actions(
        self, actions: TensorDict, as_dict: bool = False
    ) -> TensorDict:
        """
        Categorizes actions into a single tensor or a dictionary of tensors.

        Args:
            actions (TensorDict): TensorDict of action probabilities.
            as_dict (bool, optional): Whether to return a dictionary of tensors. Defaults to False.

        Returns:
            TensorDict: Categorized actions.
        """
        if as_dict:
            return TensorDict(
                {k: torch.argmax(v, dim=-1) for k, v in actions.items()}
            ).to(self.device)
        return torch.cat(
            [torch.argmax(v, dim=-1).unsqueeze(-1) for v in actions.values()], dim=-1
        ).to(self.device)

    def compute_actor_loss(
        self,
        state: TensorDict,
        *,
        target_q: TensorDict | None = None,
        action_probs: TensorDict | None = None,
        critic_probs: TensorDict | None = None,
    ) -> TensorDict:
        """
        Compute the loss for the actor network.

        Args:
            state (TensorDict): TensorDict of current states.
            target_q (TensorDict, optional): Target Q-values. Required when using CARSM loss.
            action_probs (TensorDict, optional): Pre-computed action probabilities. If None, will be computed using actor network.
            critic_probs (TensorDict, optional): Pre-computed critic probabilities. If None, will be computed using critic network.

        Returns:
            TensorDict: Actor loss.
        """
        assert (
            not self.carsm or target_q is not None
        ), "target_q must be provided when using CARSM loss."

        # Action probabilities
        if action_probs is None:
            with self.amp_scaler:
                action_probs = self.actor(state)

        # Critic distribution for those actions
        if critic_probs is None:
            with self.amp_scaler:
                critic_probs = self.critic(state, action_probs)

        # Optional entropy and CARSM log_probs
        if self.entropy_coef != 0 or self.carsm:
            entropy = 0
            log_probs = []
            action = self._categorize_actions(action_probs, as_dict=True)
            for key in action_probs.keys():
                if self.carsm:
                    dist = torch.distributions.Categorical(probs=action_probs[key])
                    log_probs.append(dist.log_prob(action[key]))
                if self.entropy_coef != 0:
                    entropy += -torch.sum(
                        action_probs[key] * torch.log(action_probs[key] + 1e-10), dim=-1
                    ).mean()
            entropy /= max(1, len(action_probs))
            if self.carsm:
                log_probs = torch.sum(torch.stack(log_probs, dim=-1), dim=-1).unsqueeze(
                    -1
                )
                if len(action_probs.keys()) > 0:
                    log_probs = log_probs / len(action_probs.keys())

        # Expected Q
        loss = TensorDict({})
        for key in critic_probs.keys():
            q_exp = torch.sum(
                critic_probs[key]
                * self.critic.z_atoms[key].to(critic_probs[key].device),
                dim=-1,
                keepdim=True,
            )
            if self.carsm:
                q_exp = (target_q[key] - q_exp) * log_probs
            loss[key] = -torch.mean(q_exp)
            if self.entropy_coef != 0:
                loss[key] = loss[key] - self.entropy_coef * entropy
        return loss

    def step_schedulers_and_decay(self) -> None:
        """
        Steps the schedulers and decays the entropy coefficient.
        """
        if self.actor_scheduler is not None:
            self.actor_scheduler.step()
        if self.critic_scheduler is not None:
            self.critic_scheduler.step()
        self.entropy_coef *= self.entropy_coef_decay

    def update_target_networks(self) -> None:
        """
        Updates the target networks using the Polyak averaging method.
        """
        for tgt, src in zip(self.actor_target.parameters(), self.actor.parameters()):
            tgt.data.copy_(self.tau * src.data + (1 - self.tau) * tgt.data)
        for tgt, src in zip(self.critic_target.parameters(), self.critic.parameters()):
            tgt.data.copy_(self.tau * src.data + (1 - self.tau) * tgt.data)

    def compute_loss(
        self,
        batch: TensorDict,
        *,
        compute_td_error: bool = False,
        training: bool = True,
        update_actor: bool = True,
        update_critic: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes actor and critic losses for a batch of transitions.

        Args:
            batch (TensorDict): Batch of transitions containing:
                - state (TensorDict): Current states
                - actions (TensorDict): Actions taken
                - reward (TensorDict): Rewards received
                - next_state (TensorDict): Next states
                - terminated (torch.Tensor): Terminal flags
            compute_td_error (bool, optional): Whether to compute TD error. Defaults to False.
            training (bool, optional): Whether to compute gradients and update networks. Defaults to True.
            update_actor (bool, optional): Whether to update the actor. Defaults to True.
            update_critic (bool, optional): Whether to update the critic. Defaults to True.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Actor loss and critic loss
        """
        with getattr(torch, "enable_grad" if training else "no_grad")():
            target_probs = self.compute_target_distribution(
                batch["reward"], batch["next_state"], batch["terminated"]
            )
            # Critic
            loss_critic_td, td_err, q_dist = self.compute_critic_loss(
                batch["state"],
                batch["actions"],
                target_probs,
                compute_td_error,
            )
            loss_critic = sum(loss_critic_td.values()) / max(1, len(loss_critic_td))

            if training and update_critic:
                self.critic_optimizer.zero_grad()
                self.amp_scaler.scale(loss_critic).backward()
                if self.max_grad_norm and self.max_grad_norm > 0:
                    self.amp_scaler.unscale(self.critic_optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.critic.parameters(), self.max_grad_norm, norm_type=2.0
                    )
                self.amp_scaler.step(self.critic_optimizer)
                # Actor uses fresh target_probs
                loss_actor_td = self.compute_actor_loss(
                    batch["state"], target_q=target_probs
                )
            else:
                loss_actor_td = self.compute_actor_loss(
                    batch["state"],
                    target_q=target_probs,
                    action_probs=batch["actions"],
                    critic_probs=q_dist,
                )

            loss_actor = sum(loss_actor_td.values()) / max(1, len(loss_actor_td))

            if training and update_actor:
                self.actor_optimizer.zero_grad()
                self.amp_scaler.scale(loss_actor).backward()
                if self.max_grad_norm and self.max_grad_norm > 0:
                    self.amp_scaler.unscale(self.actor_optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.actor.parameters(), self.max_grad_norm, norm_type=2.0
                    )
                self.amp_scaler.step(self.actor_optimizer)
        return loss_actor, loss_critic
