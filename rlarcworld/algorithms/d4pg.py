"""Core components for D4PG training (algorithm-only logic)."""

import copy
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as F
import torch.optim as optim
from tensordict import TensorDict
from torch.nn import Parameter
from torch.utils.tensorboard import SummaryWriter

from rlarcworld.algorithms.distributions import categorical_projection
from rlarcworld.common.enums import ActionKeys, RewardKeys
from rlarcworld.models.nn_modules import CriticResponseManager
from rlarcworld.trainers.core import OnStepContext
from rlarcworld.utils.checkpointing import (
    add_schema_and_class,
    assert_expected_class,
    resolve_load_path,
    resolve_save_path,
)
from rlarcworld.utils.device import AmpScaler, CudaConfig, debug_only, enable_cuda

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
        shared_params: Iterable[Parameter] | None = None,
        lr_scheduler_kwargs: dict | None = None,
        gamma: float = 0.99,
        tau: float = 0.005,
        n_steps: int = 1,
        compute_td_error: bool = True,
        carsm: bool = False,
        entropy_coef: float = 0.0,
        entropy_coef_decay: float = 1.0,
        entropy_floor_coef: float = 0.0,
        min_entropy: float = 0.0,
        submit_entropy_scale: float = 1.0,
        color_entropy_scale: float = 1.0,
        target_update_frequency: int = 10,
        model_update_frequency: int = 1,
        cache_enabled: bool = False,
        device_config: CudaConfig | None = None,
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
            shared_params (Iterable[Parameter]): List of parameters shared by
                actor and critic networks.
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
            entropy_floor_coef (float, optional): Coefficient for the minimum
                entropy floor penalty. Defaults to 0.0.
            min_entropy (float, optional): Minimum entropy floor to prevent
                complete entropy collapse. Defaults to 0.0.
            submit_entropy_scale (float, optional): Scale factor applied only to
                the submit head contribution when computing the entropy
                regularization term. Defaults to 1.0.
            color_entropy_scale (float, optional): Scale factor applied only to
                the color_values head contribution when computing the entropy
                regularization term. Defaults to 1.0.
            device_config (Optional[CudaConfig], optional): Device configuration
                dictionary. Defaults to None.
            max_grad_norm (float, optional): Maximum gradient norm. Defaults to
                0.0.
            amp_enabled (bool, optional): Whether to use automatic mixed
                precision. Defaults to True.
            target_update_frequency (int, optional): Frequency of target
                network updates. Defaults to 10.
            model_update_frequency (int, optional): Frequency of model weight
                updates (backward passes). Defaults to 1 (update every step).
            cache_enabled (bool, optional): Enable caching for imagined rollouts.
            compute_td_error (bool, optional): Whether to compute TD error.
                Defaults to True.
        """
        assert entropy_coef >= 0, "Entropy coefficient must be non-negative"
        assert entropy_coef_decay >= 0, "Entropy coefficient decay must be non-negative"
        assert entropy_floor_coef >= 0, "Entropy floor coefficient must be non-negative"
        assert min_entropy >= 0, "Minimum entropy must be non-negative"
        assert submit_entropy_scale >= 0, "Submit entropy scale must be non-negative"
        assert color_entropy_scale >= 0, "Color entropy scale must be non-negative"

        # Config/device/AMP
        self.device_config = (
            enable_cuda(amp_enabled=amp_enabled)
            if device_config is None
            else device_config
        )
        self.device = self.device_config["device"]
        self.amp_scaler = AmpScaler(amp_enabled=amp_enabled, device=self.device)

        # Hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.n_steps = n_steps
        self.compute_td_error = compute_td_error
        self.carsm = carsm
        self.entropy_coef = entropy_coef
        self.entropy_coef_decay = entropy_coef_decay
        self.entropy_floor_coef = entropy_floor_coef
        self.min_entropy = min_entropy
        self.submit_entropy_scale = submit_entropy_scale
        self.color_entropy_scale = color_entropy_scale
        self.max_grad_norm = max_grad_norm
        self.target_update_frequency = target_update_frequency
        self.model_update_frequency = model_update_frequency
        self._update_counter = 0
        self._accumulated_steps = 0
        self.cache_enabled = cache_enabled

        self._loss_actor_window_sum = 0.0
        self._loss_critic_window_sum = 0.0
        self.loss_actor_window_avg: float | None = None
        self.loss_critic_window_avg: float | None = None

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
        # IMPORTANT: Materialize once; avoid iterator exhaustion bugs when filtering.
        self.shared_params = list(shared_params) if shared_params is not None else None
        self._shared_param_ids = (
            {id(p) for p in self.shared_params}
            if self.shared_params is not None
            else set()
        )
        self.actor_optimizer = self._build_actor_optimizer(policy_lr)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=critic_lr)

        self._t0_actor_lr = policy_lr
        self._t0_critic_lr = critic_lr
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
        self.tb_writer_frequency = 1

    def _build_actor_optimizer(self, lr: float) -> optim.AdamW:
        if self.shared_params is None:
            return optim.AdamW(self.actor.parameters(), lr=lr)
        actor_head = [
            p for p in self.actor.parameters() if id(p) not in self._shared_param_ids
        ]
        return optim.AdamW(actor_head, lr=lr)

    # ----------- Diagnosis methods -----------

    def _log_nonfinite_params(self, module: torch.nn.Module, *, name: str) -> bool:
        """Log non-finite parameter stats. Returns True if any were found."""
        found = False
        for p_name, p in module.named_parameters():
            if not torch.isfinite(p).all():
                found = True
                safe_p = torch.nan_to_num(
                    p.detach().float(), nan=0.0, posinf=0.0, neginf=0.0
                )
                logger.error(
                    "Non-finite parameter detected after optimizer step. "
                    "module=%s param=%s shape=%s min=%s max=%s",
                    name,
                    p_name,
                    tuple(p.shape),
                    float(safe_p.min().item()),
                    float(safe_p.max().item()),
                )
        return found

    def _log_nonfinite_grads(self, module: torch.nn.Module, *, name: str) -> bool:
        """Log non-finite gradient stats. Returns True if any were found."""
        found = False
        for p_name, p in module.named_parameters():
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all():
                found = True
                g = p.grad.detach().float()
                safe_g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
                logger.error(
                    "Non-finite gradient detected before optimizer step. "
                    "module=%s param=%s shape=%s min=%s max=%s",
                    name,
                    p_name,
                    tuple(p.grad.shape),
                    float(safe_g.min().item()),
                    float(safe_g.max().item()),
                )
        return found

    def _assert_finite_grads_before_step(self) -> None:
        nonfinite_actor = self._log_nonfinite_grads(self.actor, name="actor")
        nonfinite_critic = self._log_nonfinite_grads(self.critic, name="critic")
        if nonfinite_actor or nonfinite_critic:
            raise RuntimeError(
                "Non-finite gradients detected before optimizer step "
                "(see logs for details)."
            )

    def _assert_finite_params_after_step(self) -> None:
        nonfinite_actor = self._log_nonfinite_params(self.actor, name="actor")
        nonfinite_critic = self._log_nonfinite_params(self.critic, name="critic")
        if nonfinite_actor or nonfinite_critic:
            raise RuntimeError(
                "Non-finite parameters detected after optimizer step "
                "(see logs for details)."
            )

    @debug_only
    def _assert_finite_critic_heads(
        self, target_dist: torch.Tensor, log_q: torch.Tensor, key: str
    ):
        if not torch.isfinite(target_dist).all():
            safe_t = torch.nan_to_num(
                target_dist.detach().float(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            logger.error(
                "Non-finite target_dist in critic loss. head=%s min=%s max=%s",
                key,
                float(safe_t.min().item()),
                float(safe_t.max().item()),
            )
            raise RuntimeError(
                f"Non-finite target_dist detected for critic head '{key}'."
            )
        if not torch.isfinite(log_q).all():
            safe_lq = torch.nan_to_num(
                log_q.detach().float(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            logger.error(
                "Non-finite log_q in critic loss. head=%s min=%s max=%s",
                key,
                float(safe_lq.min().item()),
                float(safe_lq.max().item()),
            )
            raise RuntimeError(f"Non-finite log_q detected for critic head '{key}'.")

    @debug_only
    def _assert_critic_finit_loss(self, loss: torch.Tensor, key: str):
        if not torch.isfinite(loss).all():
            logger.error(
                "Non-finite critic loss scalar. head=%s value=%s",
                key,
                float(torch.nan_to_num(loss.detach().float(), nan=0.0).item()),
            )
            raise RuntimeError(f"Non-finite critic loss detected for head '{key}'.")

    @debug_only
    def _assert_finite_action_probs(self, action_probs: TensorDict):
        for _k, _v in action_probs.items():
            if not torch.isfinite(_v).all():
                try:
                    finite_mask = torch.isfinite(_v)
                    bad_rows = (~finite_mask).any(dim=-1)
                    bad_idx = bad_rows.nonzero(as_tuple=False).squeeze(-1)
                    row_sum = _v.float().sum(dim=-1)
                    safe_v = torch.nan_to_num(
                        _v.float(), nan=0.0, posinf=0.0, neginf=0.0
                    )
                    safe_row_sum = torch.nan_to_num(
                        row_sum, nan=0.0, posinf=0.0, neginf=0.0
                    )
                    logger.error(
                        "Non-finite action probabilities detected. "
                        "head=%s bad_rows=%d/%d "
                        "prob_min=%s prob_max=%s "
                        "row_sum_min=%s row_sum_max=%s "
                        "bad_idx(sample)=%s",
                        _k,
                        int(bad_rows.sum().item()),
                        int(_v.shape[0]),
                        float(safe_v.min().item()),
                        float(safe_v.max().item()),
                        float(safe_row_sum.min().item()),
                        float(safe_row_sum.max().item()),
                        bad_idx[:10].detach().cpu().tolist(),
                    )
                except Exception:
                    logger.exception(
                        "Failed to compute diagnostics for non-finite action "
                        "probabilities (head=%s)",
                        _k,
                    )
                raise RuntimeError(
                    f"Non-finite action probabilities detected for head '{_k}'."
                )

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
                next_q_response: CriticResponseManager = self.critic_target(
                    next_state, action_probs
                )

        assert set(next_q_response.keys()) == set(reward.keys()), (
            f"Mismatched keys next_q_dist={list(next_q_response.keys())}, "
            f"reward={list(reward.keys())}"
        )

        target_dist = TensorDict(
            {
                key: categorical_projection(
                    next_q_response.q_dist[key],
                    reward[key],
                    terminated,
                    self.gamma,
                    self.critic_target.z_atoms[key],
                    self.critic_target.v_min[key],
                    self.critic_target.v_max[key],
                    self.critic_target.num_atoms[key],
                    normalize_projection=True,
                    n_steps=self.n_steps if key in RewardKeys.get_n_step_keys() else 1,
                )
                for key in next_q_response.keys()
            }
        ).to(self.device)
        return target_dist

    def set_tb_writer(self, tb_writer: SummaryWriter, frequency: int) -> None:
        """Set the TensorBoard writer for logging."""
        self.tb_writer = tb_writer
        self.tb_writer_frequency = frequency

    def compute_critic_loss(
        self,
        state: TensorDict,
        action: TensorDict,
        target_dist: TensorDict,
        compute_td_error: bool = False,
    ) -> tuple[TensorDict, TensorDict, CriticResponseManager]:
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
            q_response: CriticResponseManager = self.critic(state, action)

        loss = TensorDict({}).to(self.device)
        td_error = TensorDict({}).to(self.device)
        log_q = q_response.q_log_dist
        for key in log_q.keys():
            self._assert_finite_critic_heads(target_dist[key], log_q[key], key)
            if compute_td_error:
                td_error[key] = (
                    F.kl_div(log_q[key], target_dist[key], reduction="none")
                    .sum(dim=-1)
                    .detach()
                )
            loss[key] = self.critic_criterion(log_q[key], target_dist[key])
            self._assert_critic_finit_loss(loss[key], key)
        return loss, td_error, q_response

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
                q-distribution (probabilities over atoms). If None, will be
                computed using critic network.

        Returns:
            TensorDict: Actor loss.
        """
        assert not self.carsm or target_q is not None, (
            "target_q must be provided when using CARSM loss."
        )

        # Action probabilities
        if action_probs is None:
            if self.amp_scaler.scaler is not None:
                with torch.amp.autocast(device_type="cuda", enabled=False):
                    action_probs = self.actor(state)
            else:
                action_probs = self.actor(state)

        self._assert_finite_action_probs(action_probs)

        # Critic distribution for those actions
        if critic_probs is None:
            if self.amp_scaler.scaler is not None:
                with torch.amp.autocast(device_type="cuda", enabled=False):
                    critic_probs = self.critic(state, action_probs).q_dist
            else:
                critic_probs = self.critic(state, action_probs).q_dist

        for _k, _v in critic_probs.items():
            if not torch.isfinite(_v).all():
                raise RuntimeError(
                    f"Non-finite critic distribution detected for head '{_k}'."
                )

        # Optional entropy and CARSM log_probs
        if self.entropy_coef != 0 or self.entropy_floor_coef != 0 or self.carsm:
            entropy: torch.Tensor = torch.tensor(0.0, device=self.device)
            log_probs: list[torch.Tensor] = []
            action = self._categorize_actions(action_probs, as_dict=True)
            entropy_weight_sum: torch.Tensor = torch.tensor(0.0, device=self.device)
            for key in action_probs.keys():
                if self.carsm:
                    dist = torch.distributions.Categorical(probs=action_probs[key])
                    log_probs.append(dist.log_prob(action[key]))
                if self.entropy_coef != 0 or self.entropy_floor_coef != 0:
                    head_entropy = -torch.sum(
                        action_probs[key] * torch.log(action_probs[key] + 1e-10), dim=-1
                    ).mean()
                    head_weight = 1.0
                    if key == ActionKeys.SUBMIT.value:
                        head_weight = float(self.submit_entropy_scale)
                    elif key == ActionKeys.COLOR_VALUES.value:
                        head_weight = float(self.color_entropy_scale)
                    entropy = entropy + head_entropy * head_weight
                    entropy_weight_sum = entropy_weight_sum + head_weight

            if self.entropy_coef != 0 or self.entropy_floor_coef != 0:
                entropy = entropy / torch.clamp(entropy_weight_sum, min=1.0)
            if self.carsm:
                log_probs_tensor = torch.sum(
                    torch.stack(log_probs, dim=-1), dim=-1
                ).unsqueeze(-1)
                if len(action_probs.keys()) > 0:
                    log_probs_tensor = log_probs_tensor / len(action_probs.keys())

        # Expected Q
        loss = TensorDict({})
        for key in critic_probs.keys():
            z_atoms = self.critic.z_atoms[key].to(critic_probs[key].device).float()
            q_exp = torch.sum(
                critic_probs[key].float() * z_atoms,
                dim=-1,
                keepdim=True,
            )
            if not torch.isfinite(q_exp).all():
                raise RuntimeError(f"Non-finite expected Q detected for head '{key}'.")
            if self.carsm:
                # Narrow type for mypy: target_q must be provided in CARSM branch
                assert target_q is not None
                tq = cast(TensorDict, target_q)
                tq_exp = torch.sum(
                    tq[key].to(critic_probs[key].device).float() * z_atoms,
                    dim=-1,
                    keepdim=True,
                )
                if not torch.isfinite(tq_exp).all():
                    raise RuntimeError(
                        f"Non-finite target expected Q detected for head '{key}'."
                    )
                q_exp = q_exp + (tq_exp - q_exp).detach() * log_probs_tensor
            loss[key] = -torch.mean(q_exp)
            if not torch.isfinite(loss[key]).all():
                raise RuntimeError(f"Non-finite actor loss detected for head '{key}'.")
            if self.entropy_coef != 0:
                if not torch.isfinite(entropy).all():
                    raise RuntimeError("Non-finite entropy detected in actor loss.")
                loss[key] = loss[key] - self.entropy_coef * entropy
            if self.entropy_floor_coef != 0:
                if not torch.isfinite(entropy).all():
                    raise RuntimeError("Non-finite entropy detected in actor loss.")
                entropy_floor_penalty = torch.relu(self.min_entropy - entropy)
                if not torch.isfinite(entropy_floor_penalty).all():
                    raise RuntimeError(
                        "Non-finite entropy floor penalty detected in actor loss."
                    )
                loss[key] = loss[key] + self.entropy_floor_coef * entropy_floor_penalty
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
        force_update: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, TensorDict]:
        """Compute actor and critic losses for a batch of transitions.

        Args:
            batch (TensorDict): Batch containing state, actions, reward,
                next_state, and terminated.
            training (bool, optional): Whether to compute gradients and update
                networks. Defaults to True.
            force_update (bool, optional): Force weight update regardless of
                model_update_frequency. Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, TensorDict]: Actor loss, critic loss,
                and TD error.
        """
        if training:
            self.actor.train()
            self.critic.train()
        else:
            self.actor.eval()
            self.critic.eval()

        with getattr(torch, "enable_grad" if training else "no_grad")():
            should_step = force_update or (
                (self._update_counter + 1) % self.model_update_frequency == 0
            )

            target_probs = self.compute_target_distribution(
                batch["reward"], batch["next_state"], batch["terminated"]
            )
            # Critic
            loss_critic_td, td_err, q_response = self.compute_critic_loss(
                batch["state"],
                batch["actions"],
                target_probs,
                self.compute_td_error,
            )
            loss_critic = sum(loss_critic_td.values()) / max(1, len(loss_critic_td))

            if training:
                # Actor uses fresh target_probs
                loss_actor_td = self.compute_actor_loss(
                    batch["state"], target_q=target_probs
                )

                # Zero gradients only when starting a new accumulation cycle
                if self._update_counter % self.model_update_frequency == 0:
                    self.critic_optimizer.zero_grad()
                    self.actor_optimizer.zero_grad()
                    self._accumulated_steps = 0
                    self._loss_actor_window_sum = 0.0
                    self._loss_critic_window_sum = 0.0
                    self.loss_actor_window_avg = None
                    self.loss_critic_window_avg = None

                loss_actor = sum(loss_actor_td.values()) / max(1, len(loss_actor_td))

                # Scale losses for gradient accumulation (AMP-safe, CPU/MPS-safe)
                loss_critic_scaled = loss_critic / self.model_update_frequency
                loss_actor_scaled = loss_actor / self.model_update_frequency

                # Accumulate gradients
                self.amp_scaler.scale(loss_critic_scaled).backward()
                self.amp_scaler.scale(loss_actor_scaled).backward()
                self._accumulated_steps += 1

                # Step optimizer only after accumulating N gradients
                if should_step:
                    # Clip gradients if needed
                    self.amp_scaler.unscale(self.critic_optimizer)
                    self.amp_scaler.unscale(self.actor_optimizer)
                    if self.max_grad_norm and self.max_grad_norm > 0:
                        grad_norm_critic = torch.nn.utils.clip_grad_norm_(
                            self.critic.parameters(), self.max_grad_norm, norm_type=2.0
                        )
                        grad_norm_actor = torch.nn.utils.clip_grad_norm_(
                            self.actor.parameters(), self.max_grad_norm, norm_type=2.0
                        )
                        self._assert_finite_grads_before_step()
                    else:
                        # Compute norms without clipping for logging
                        grad_norm_critic = torch.nn.utils.clip_grad_norm_(
                            self.critic.parameters(),
                            max_norm=float("inf"),
                            norm_type=2.0,
                        )
                        grad_norm_actor = torch.nn.utils.clip_grad_norm_(
                            self.actor.parameters(),
                            max_norm=float("inf"),
                            norm_type=2.0,
                        )

                    self._assert_finite_grads_before_step()

                    # Log gradient norms to TensorBoard
                    if (
                        hasattr(self, "tb_writer")
                        and self.tb_writer is not None
                        and self._update_counter % self.tb_writer_frequency == 0
                    ):
                        self.tb_writer.add_scalar(
                            "grad_l2_norm/actor", grad_norm_actor, self._update_counter
                        )
                        self.tb_writer.add_scalar(
                            "grad_l2_norm/critic",
                            grad_norm_critic,
                            self._update_counter,
                        )

                    self.amp_scaler.step(self.critic_optimizer)
                    self.amp_scaler.step(self.actor_optimizer)
                    self.amp_scaler.update()

                    self._assert_finite_params_after_step()

                    if self.cache_enabled:
                        logger.debug(
                            f"Model stepped (counter={self._update_counter}, "
                            f"accumulated={self._accumulated_steps}), clearing cache"
                        )
                        # Window averages for logging
                        self._loss_actor_window_sum += float(loss_actor.detach())
                        self._loss_critic_window_sum += float(loss_critic.detach())
                        if self._accumulated_steps > 1:
                            self.loss_actor_window_avg = (
                                self._loss_actor_window_sum / self._accumulated_steps
                            )
                            self.loss_critic_window_avg = (
                                self._loss_critic_window_sum / self._accumulated_steps
                            )
                        self.actor.clear_example_cache()
                        self.critic.clear_example_cache()
                        self.actor.enable_example_cache()
                        self.critic.enable_example_cache()
                else:
                    # Keep cache enabled during accumulation to allow cache hits
                    if self.cache_enabled:
                        self.actor.enable_example_cache()
                        self.critic.enable_example_cache()
            else:
                loss_actor_td = self.compute_actor_loss(
                    batch["state"],
                    target_q=target_probs,
                    action_probs=batch["actions"],
                    critic_probs=q_response.q_dist,
                )
                loss_actor = sum(loss_actor_td.values()) / max(1, len(loss_actor_td))

            if training:
                self._update_counter += 1
        return loss_actor, loss_critic, td_err

    def on_step(self, ctx: OnStepContext):
        """Callback function to handle all step-related operations.

        Args:
            ctx: Step ctx containing all relevant information
        """
        # Update target networks periodically
        if ctx.batch is None or ctx.warmup:
            return

        self.step_schedulers_and_decay()
        if ctx.iteration % self.target_update_frequency == 0:
            self.update_target_networks()

    def save_checkpoint(self, path: str | Path) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        try:
            checkpoint_path = resolve_save_path(path, self.__class__.__name__)
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
        payload = {
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
            "update_counter": self._update_counter,
            "accumulated_steps": self._accumulated_steps,
        }
        return add_schema_and_class(payload, self.__class__.__name__)

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
            cp_path = resolve_load_path(checkpoint, self.__class__.__name__)
            checkpoint = torch.load(
                cp_path, weights_only=weights_only, map_location=device
            )
        if not isinstance(checkpoint, dict):
            raise ValueError("Checkpoint must be a dictionary or a Path.")

        # If a class marker is present, validate it; otherwise accept
        if "class" in checkpoint:
            assert_expected_class(checkpoint["class"], self.__class__.__name__)

        # If only weights are requested or present, disable other loads
        if weights_only:
            load_optimizers = False
            load_schedulers = False
            restore_hyperparameters = False

        # Load networks
        if "actor" in checkpoint:
            try:
                self.actor.load_state_dict(checkpoint["actor"], strict=strict)
            except Exception as e:
                logger.warning("Failed to load actor state_dict: %s", e)
        if "critic" in checkpoint:
            try:
                self.critic.load_state_dict(checkpoint["critic"], strict=strict)
            except Exception as e:
                logger.warning("Failed to load critic state_dict: %s", e)

        # Refresh targets to match online networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_target.eval()
        self.critic_target.eval()

        # Load optimizers
        if load_optimizers:
            try:
                actor_opt_sd = checkpoint.get("actor_optimizer")
                if actor_opt_sd is not None:
                    self.actor_optimizer.load_state_dict(actor_opt_sd)
            except Exception as e:
                logger.warning("Failed to load actor optimizer state: %s", e)
            try:
                critic_opt_sd = checkpoint.get("critic_optimizer")
                if critic_opt_sd is not None:
                    self.critic_optimizer.load_state_dict(critic_opt_sd)
            except Exception as e:
                logger.warning("Failed to load critic optimizer state: %s", e)

        # Load schedulers
        if load_schedulers:
            try:
                actor_sch_sd = checkpoint.get("actor_scheduler")
                if actor_sch_sd is not None and self.actor_scheduler is not None:
                    self.actor_scheduler.load_state_dict(actor_sch_sd)
            except Exception as e:
                logger.warning("Failed to load actor scheduler state: %s", e)
            try:
                critic_sch_sd = checkpoint.get("critic_scheduler")
                if critic_sch_sd is not None and self.critic_scheduler is not None:
                    self.critic_scheduler.load_state_dict(critic_sch_sd)
            except Exception as e:
                logger.warning("Failed to load critic scheduler state: %s", e)

        # Restore hyperparameters
        if restore_hyperparameters:
            try:
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
                self.entropy_floor_coef = hp.get(
                    "entropy_floor_coef", self.entropy_floor_coef
                )
                self.min_entropy = hp.get("min_entropy", self.min_entropy)
                self.target_update_frequency = hp.get(
                    "target_update_frequency", self.target_update_frequency
                )
                self.tau = hp.get("tau", self.tau)
                self.max_grad_norm = hp.get("max_grad_norm", self.max_grad_norm)
                self.carsm = hp.get("carsm", self.carsm)
            except Exception as e:
                logger.warning("Failed to restore hyperparameters: %s", e)

        # Restore gradient accumulation counters
        self._update_counter = checkpoint.get("update_counter", 0)
        self._accumulated_steps = checkpoint.get("accumulated_steps", 0)
        logger.info(
            f"Restored counters: update_counter={self._update_counter}, "
            f"accumulated_steps={self._accumulated_steps}"
        )

    @property
    def hyperparameters(self):
        """Returns the hyperparameters of the algorithm."""
        return {
            "t0_actor_lr": self._t0_actor_lr,
            "t0_critic_lr": self._t0_critic_lr,
            "actor_lr": self.actor_optimizer.param_groups[0]["lr"],
            "critic_lr": self.critic_optimizer.param_groups[0]["lr"],
            "actor_scheduler": None
            if self.actor_scheduler is None
            else {
                "type": self.actor_scheduler.__class__.__name__,
                "gamma": self.actor_scheduler.gamma,
                "step_size": self.actor_scheduler.step_size,
            },
            "critic_scheduler": None
            if self.critic_scheduler is None
            else {
                "type": self.critic_scheduler.__class__.__name__,
                "gamma": self.critic_scheduler.gamma,
                "step_size": self.critic_scheduler.step_size,
            },
            "gamma": self.gamma,
            "entropy_coef": self.entropy_coef,
            "entropy_coef_decay": self.entropy_coef_decay,
            "entropy_floor_coef": self.entropy_floor_coef,
            "min_entropy": self.min_entropy,
            "target_update_frequency": self.target_update_frequency,
            "tau": self.tau,
            "max_grad_norm": self.max_grad_norm,
            "carsm": self.carsm,
        }.copy()
