"""Core components for D4PG training (algorithm-only logic)."""

import copy
import logging
from pathlib import Path
from typing import cast

import torch
import torch.optim as optim
from tensordict import TensorDict

from rlarcworld.algorithms.distributions import categorical_projection
from rlarcworld.runners.core import OnStepContext
from rlarcworld.utils.device import AmpScaler, CudaConfig, enable_cuda

logger = logging.getLogger(__name__)


class D4PGCore:
    """Minimal, environment-agnostic core for D4PG.

    Responsibilities (algorithm-only):
    - Hold actor/critic and their target copies
    - Compute target distribution (Bellman projection onto atoms)
    - Compute critic and actor losses (including optional CARSM and entropy
      regularization)
    - Apply gradients (with optional AMP and gradient clipping)
    - Soft-update target networks

    Excludes (non-core): env stepping, dataloaders, replay buffers,
    logging/traceability, checkpointing.
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
        compute_td_error: bool = False,
        carsm: bool = False,
        entropy_coef: float = 0.0,
        entropy_coef_decay: float = 1.0,
        target_update_frequency: int = 10,
        config: CudaConfig | None = None,
        max_grad_norm: float = 0.0,
        amp_enabled: bool = True,
    ) -> None:
        """Initialize the D4PGCore.

        Args:
            actor (torch.nn.Module): Actor network.
            critic (torch.nn.Module): Critic network.
            policy_lr (float, optional): Learning rate for actor network.
                Defaults to 3e-4.
            critic_lr (float, optional): Learning rate for critic network.
                Defaults to 3e-3.
            lr_scheduler_kwargs (Optional[Dict], optional): Keyword arguments
                for learning rate scheduler StepLR. Defaults to None.
            gamma (float, optional): Discount factor. Defaults to 0.99.
            tau (float, optional): Soft update parameter. Defaults to 0.005.
            n_steps (int, optional): Number of steps for n-step returns.
                Defaults to 1.
            carsm (bool, optional): Whether to use CARSM. Defaults to False.
            entropy_coef (float, optional): Entropy coefficient. Defaults to
                0.0.
            entropy_coef_decay (float, optional): Entropy coefficient decay.
                Defaults to 1.0.
            config (Optional[CudaConfig], optional): Configuration dictionary.
                Defaults to None.
            max_grad_norm (float, optional): Maximum gradient norm. Defaults to
                0.0.
            amp_enabled (bool, optional): Whether to use automatic mixed
                precision. Defaults to True.
            target_update_frequency (int, optional): Frequency of target
                network updates. Defaults to 10.
            compute_td_error (bool, optional): Whether to compute TD error.
                Defaults to False.
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
        self.compute_td_error = compute_td_error
        self.carsm = carsm
        self.entropy_coef = entropy_coef
        self.entropy_coef_decay = entropy_coef_decay
        self.max_grad_norm = max_grad_norm
        self.target_update_frequency = target_update_frequency

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
        """Computes the target distribution for the critic network.

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

        assert next_q_dist.keys() == reward.keys(), (
            f"Mismatched keys next_q_dist={list(next_q_dist.keys())}, "
            f"reward={list(reward.keys())}"
        )

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
        """Compute the loss for the critic network.

        Args:
            state (TensorDict): TensorDict of current states.
            action (TensorDict): TensorDict of actions.
            target_dist (TensorDict): Target distribution from the target
                critic.
            compute_td_error (bool, optional): Whether to compute the TD error.
                Defaults to False.

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
        """Categorize actions into a tensor or dictionary of tensors.

        Args:
            actions (TensorDict): TensorDict of action probabilities.
            as_dict (bool, optional): Whether to return a dictionary of
                tensors. Defaults to False.

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
        """Compute the loss for the actor network.

        Args:
            state (TensorDict): TensorDict of current states.
            target_q (TensorDict, optional): Target Q-values. Required when
                using CARSM loss.
            action_probs (TensorDict, optional): Pre-computed action
                probabilities. If None, will be computed using actor network.
            critic_probs (TensorDict, optional): Pre-computed critic
                probabilities. If None, will be computed using critic network.

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
            entropy: torch.Tensor = torch.tensor(0.0, device=self.device)
            log_probs: list[torch.Tensor] = []
            action = self._categorize_actions(action_probs, as_dict=True)
            for key in action_probs.keys():
                if self.carsm:
                    dist = torch.distributions.Categorical(probs=action_probs[key])
                    log_probs.append(dist.log_prob(action[key]))
                if self.entropy_coef != 0:
                    entropy += -torch.sum(
                        action_probs[key] * torch.log(action_probs[key] + 1e-10), dim=-1
                    ).mean()
            entropy = entropy / max(1, len(action_probs))
            if self.carsm:
                log_probs_tensor = torch.sum(
                    torch.stack(log_probs, dim=-1), dim=-1
                ).unsqueeze(-1)
                if len(action_probs.keys()) > 0:
                    log_probs_tensor = log_probs_tensor / len(action_probs.keys())

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
                # Narrow type for mypy: target_q must be provided in CARSM branch
                assert target_q is not None
                tq = cast(TensorDict, target_q)
                q_exp = (tq[key] - q_exp) * log_probs_tensor
            loss[key] = -torch.mean(q_exp)
            if self.entropy_coef != 0:
                loss[key] = loss[key] - self.entropy_coef * entropy
        return loss

    def step_schedulers_and_decay(self) -> None:
        """Steps the schedulers and decays the entropy coefficient."""
        if self.actor_scheduler is not None:
            self.actor_scheduler.step()
        if self.critic_scheduler is not None:
            self.critic_scheduler.step()
        self.entropy_coef *= self.entropy_coef_decay

    def update_target_networks(self) -> None:
        """Updates the target networks using the Polyak averaging method."""
        for tgt, src in zip(self.actor_target.parameters(), self.actor.parameters()):
            tgt.data.copy_(self.tau * src.data + (1 - self.tau) * tgt.data)
        for tgt, src in zip(self.critic_target.parameters(), self.critic.parameters()):
            tgt.data.copy_(self.tau * src.data + (1 - self.tau) * tgt.data)

    def compute_loss(
        self,
        batch: TensorDict,
        *,
        training: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute actor and critic losses for a batch of transitions.

        Args:
            batch (TensorDict): Batch containing state, actions, reward,
                next_state, and terminated.
            training (bool, optional): Whether to compute gradients and update
                networks. Defaults to True.

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
                self.compute_td_error,
            )
            loss_critic = sum(loss_critic_td.values()) / max(1, len(loss_critic_td))

            if training:
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

            if training:
                self.actor_optimizer.zero_grad()
                self.amp_scaler.scale(loss_actor).backward()
                if self.max_grad_norm and self.max_grad_norm > 0:
                    self.amp_scaler.unscale(self.actor_optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.actor.parameters(), self.max_grad_norm, norm_type=2.0
                    )
                self.amp_scaler.step(self.actor_optimizer)
        return loss_actor, loss_critic

    def on_step(self, context: OnStepContext):
        """Callback function to handle all step-related operations.

        Args:
            context: Step context containing all relevant information
        """
        # Update target networks periodically
        if context.batch is not None:
            return

        self.step_schedulers_and_decay()
        if context.iteration % self.target_update_frequency == 0:
            self.update_target_networks()

    def save_checkpoint(self, path: str | Path) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        try:
            path = Path(path)
            # If path looks like a directory (no suffix) ensure it exists
            if path.suffix == "":
                path.mkdir(parents=True, exist_ok=True)
                checkpoint_path = path / (self.__class__.__name__ + ".ptc")
                torch.save(self.checkpoint(), checkpoint_path)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                checkpoint_path = path
                torch.save(self.checkpoint(), checkpoint_path)
        except AttributeError as e:
            logger.error(
                "Error saving environment, reset should have been called by now: %s",
                e,
            )
            raise
        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def checkpoint(self):
        """Generates a dictionary with checkpointing."""
        return {
            "class": self.__class__,
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "actor_scheduler": self.actor_scheduler.state_dict()
            if self.actor_scheduler is not None
            else None,
            "critic_scheduler": self.critic_scheduler.state_dict()
            if self.critic_scheduler is not None
            else None,
            "hyperparameters": self.hyperparameters,
        }

    def load(
        self,
        checkpoint: dict | Path | str,
        *,
        strict: bool = True,
        load_optimizers: bool = True,
        load_schedulers: bool = True,
        restore_hyperparameters: bool = True,
        weights_only: bool = False,
        device: str | None = None,
    ) -> None:
        """Load state from a checkpoint dictionary produced by ``checkpoint()``.

        Args:
            checkpoint: Dictionary with keys from ``checkpoint()``.
            strict: Whether to strictly enforce that the keys in
                ``state_dict`` match.
            load_optimizers: Load optimizer state if present.
            load_schedulers: Load scheduler state if present and schedulers
                exist locally.
            restore_hyperparameters: Restore selected hyperparameters (incl.
                learning rates).
            weights_only: Whether to load only the weights.
            device: Device to load the environment on.
        """
        if isinstance(checkpoint, (Path, str)):
            cp_path = Path(checkpoint)
            if cp_path.is_dir():
                cp_path = cp_path / (self.__class__.__name__ + ".ptc")
            checkpoint = torch.load(
                cp_path,
                weights_only=weights_only,
                map_location=device,
            )
        if not isinstance(checkpoint, dict):
            raise ValueError("Checkpoint must be a dictionary or a Path.")

        if checkpoint["class"] != self.__class__:
            raise ValueError("Checkpoint must be from the same class.")

        # Load networks
        if "actor" in checkpoint:
            self.actor.load_state_dict(checkpoint["actor"], strict=strict)
        if "critic" in checkpoint:
            self.critic.load_state_dict(checkpoint["critic"], strict=strict)

        # Refresh targets to match online networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Load optimizers
        if load_optimizers:
            actor_opt_sd = checkpoint.get("actor_optimizer")
            if actor_opt_sd is not None:
                self.actor_optimizer.load_state_dict(actor_opt_sd)
            critic_opt_sd = checkpoint.get("critic_optimizer")
            if critic_opt_sd is not None:
                self.critic_optimizer.load_state_dict(critic_opt_sd)

        # Load schedulers
        if load_schedulers:
            actor_sch_sd = checkpoint.get("actor_scheduler")
            if actor_sch_sd is not None and self.actor_scheduler is not None:
                self.actor_scheduler.load_state_dict(actor_sch_sd)
            critic_sch_sd = checkpoint.get("critic_scheduler")
            if critic_sch_sd is not None and self.critic_scheduler is not None:
                self.critic_scheduler.load_state_dict(critic_sch_sd)

        # Restore hyperparameters
        if restore_hyperparameters:
            hp = checkpoint.get("hyperparameters") or {}
            # Learning rates
            if "actor_lr" in hp:
                for g in self.actor_optimizer.param_groups:
                    g["lr"] = float(hp["actor_lr"])
            if "critic_lr" in hp:
                for g in self.critic_optimizer.param_groups:
                    g["lr"] = float(hp["critic_lr"])
            # Other runtime params
            self.gamma = hp.get("gamma", self.gamma)
            self.entropy_coef = hp.get("entropy_coef", self.entropy_coef)
            self.entropy_coef_decay = hp.get(
                "entropy_coef_decay", self.entropy_coef_decay
            )
            self.target_update_frequency = hp.get(
                "target_update_frequency", self.target_update_frequency
            )
            self.tau = hp.get("tau", self.tau)
            self.max_grad_norm = hp.get("max_grad_norm", self.max_grad_norm)
            self.carsm = hp.get("carsm", self.carsm)

    @property
    def hyperparameters(self):
        """Returns the hyperparameters of the algorithm."""
        return {
            "actor_lr": self.actor_optimizer.param_groups[0]["lr"],
            "critic_lr": self.critic_optimizer.param_groups[0]["lr"],
            "actor_scheduler": self.actor_scheduler,
            "critic_scheduler": self.critic_scheduler,
            "gamma": self.gamma,
            "entropy_coef": self.entropy_coef,
            "entropy_coef_decay": self.entropy_coef_decay,
            "target_update_frequency": self.target_update_frequency,
            "tau": self.tau,
            "max_grad_norm": self.max_grad_norm,
            "carsm": self.carsm,
        }.copy()
