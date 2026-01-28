"""TensorBoard logging utilities with decoupled callbacks.

This module provides a minimal, decoupled TensorBoard integration that plugs
into the Trainer via callbacks. It avoids coupling to the core or trainer
internals beyond public attributes exposed on the OnStepContext.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Optional

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from tensordict import TensorDict
from torch.utils.tensorboard import SummaryWriter

from rlarcworld.common.enums import RewardKeys
from rlarcworld.trainers.core import (
    ActorCriticCoreProto,
    OnStepContext,
    ReplayBufferProto,
)

logger = logging.getLogger(__name__)


class TensorBoardWriterManager:
    """Owns a SummaryWriter and exposes train/validation callbacks.

    - Fully decoupled from Trainer/Core; only consumes OnStepContext
    - Minimal default metrics; safe if certain fields are unavailable
    """

    def __init__(
        self,
        experiment_path: str | Path,
        *,
        experiment_name: str | None = None,
        experiment_description: str | None = None,
        logdir_name: str = "tensorboard",
        replay_buffer: ReplayBufferProto | None = None,
        warmup_buffer_ratio: float = 0.2,
        medium_frequency: int = 100,
        minimal_frequency: int = 10,
    ) -> None:
        """Create a manager and underlying SummaryWriter.

        Args:
            experiment_path: Root path for this run's artifacts.
            experiment_name: Name of the experiment.
            experiment_description: Description of the experiment.
            logdir_name: Directory name under experiment root for TB logs.
            replay_buffer: Replay buffer to log stats from.
            warmup_buffer_ratio: Ratio used to display buffer warmup threshold.
            medium_frequency: Frequency at which to log medium-tier metrics.
            minimal_frequency: Frequency at which to log minimal metrics.
        """
        self.minimal_frequency = minimal_frequency
        self.root = Path(experiment_path)
        self.root.mkdir(parents=True, exist_ok=True)
        self.logdir = self.root / logdir_name
        self.logdir.mkdir(parents=True, exist_ok=True)
        if SummaryWriter is None:
            raise RuntimeError(
                "torch.utils.tensorboard.SummaryWriter is not available."
            )
        self.writer = SummaryWriter(log_dir=str(self.logdir))

        if experiment_name is not None or experiment_description is not None:
            title = (experiment_name or "").strip()
            title = title.replace("_", " ").replace("-", " ").strip().title()
            desc = (experiment_description or "").strip()
            text = "# {title}\n\n{desc}".format(
                title=title or "Experiment",
                desc=desc or "(no description)",
            )
            # Use step=1 so it shows up when TensorBoard defaults to "latest step".
            self.writer.add_text("Experiment", text, global_step=1)
            self.writer.flush()
        self.replay_buffer = replay_buffer
        self.warmup_buffer_ratio = float(warmup_buffer_ratio)
        self.medium_frequency = int(max(1, medium_frequency))
        self.curriculum_actual_usage = 0
        self.total_samples = 0

    # -------------------- Public API --------------------
    def close(self) -> None:
        """Flush and close the summary writer."""
        try:
            self.writer.flush()
            self.writer.close()
        except Exception:
            pass

    def on_train_step(self, ctx: OnStepContext) -> None:
        """Callback to log training-step metrics."""
        self._log_minimal_metrics(ctx, phase="train")
        if ctx.warmup:
            return
        self._maybe_log_medium_metrics(ctx, phase="train")

    def on_validation_step(self, ctx: OnStepContext) -> None:
        """Callback to log validation-step metrics."""
        if ctx.warmup:
            return
        self._log_minimal_metrics(ctx, phase="val")
        self._maybe_log_medium_metrics(ctx, phase="val")

    def log_hparams(
        self,
        hparams: dict[str, Any],
        metrics: Optional[dict[str, float]] = None,
    ) -> None:
        """Log hyperparameters once at the beginning of a run.

        Args:
            hparams: Hyperparameters to log.
            metrics: Metrics to log.
        """
        metrics = metrics or {}
        try:
            self.writer.add_hparams(hparams, metrics)
        except Exception:
            logger.warning("Failed to log hyperparameters")
            pass

    # -------------------- External helpers --------------------

    def curriculum_stats(
        self,
        iteration: int,
        curriculum_action_stats: dict[str, float],
    ) -> None:
        """Callback to log curriculum metrics."""

        def to_space_case(s: str) -> str:
            parts = [p for p in s.replace("-", "_").split("_") if p]
            return " ".join(p[:1].upper() + p[1:] for p in parts)

        # Also log totals for quick sanity checks.
        totals = {
            "curriculum_generated": float(curriculum_action_stats["samples_total"]),
            "training_usage": float(self.curriculum_actual_usage),
            "actor_generated_usage": float(
                self.total_samples - self.curriculum_actual_usage
            ),
            "total_samples": float(self.total_samples),
        }
        denom_curriculum = max(1.0, totals["curriculum_generated"])
        denom_samples = max(1.0, totals["total_samples"])
        counts = {
            k: float(v) / denom_curriculum
            for k, v in curriculum_action_stats.items()
            if k
            not in {
                "oracle_calls",
                "samples_total",
            }
        }
        used_for_training = totals["training_usage"] / denom_samples
        if counts:
            keys = sorted(counts.keys())
            vals = [counts[k] for k in keys]
            fig, ax = plt.subplots(figsize=(max(6.0, 0.6 * len(keys)), 4.0), dpi=150)
            ax.bar([to_space_case(k) for k in keys], vals)
            ax.bar_label(ax.containers[0], fmt="%.4f")
            ax.set_title(
                "Action strategy counts, %.2f%% used for training" % used_for_training
            )
            ax.set_ylabel("Usage [%]")
            ax.tick_params(axis="x", labelrotation=75)
            ax.margins(y=0.1)
            fig.tight_layout()
            self.writer.add_figure(
                "curriculum/action_strategy_counts",
                fig,
                global_step=int(iteration),
            )
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(5.0, 3.5), dpi=150)
        total_keys = list(totals.keys())
        total_vals = [totals[k] for k in total_keys]
        ax.bar([to_space_case(k) for k in total_keys], total_vals)
        ax.bar_label(ax.containers[0], fmt="%.0f")
        ax.set_title("Sample totals")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", labelrotation=75)
        ax.margins(y=0.1)

        if any(v <= 0 for v in totals.values()):
            ax.set_yscale("symlog", linthresh=1.0)
        else:
            ax.set_yscale("log")
        ax.yaxis.grid(True, which="both", linestyle="-", linewidth=0.5, alpha=0.35)
        ax.set_axisbelow(True)
        fig.tight_layout()
        self.writer.add_figure(
            "curriculum/action_strategy_totals",
            fig,
            global_step=int(iteration),
        )
        plt.close(fig)
        self.writer.flush()

    # -------------------- Internal helpers --------------------

    def _add_submission_metrics(
        self, submit: torch.Tensor, reward_td: TensorDict, iteration: int, phase: str
    ) -> None:
        """Add submission metrics scalars."""
        batch_size = int(submit.shape[0])
        submit_count = int(submit.sum().detach().cpu().item())

        scalars = {
            "rate": float(submit_count) / float(max(1, batch_size)),
        }

        # Success/failure: use binary reward if available
        binary_reward = reward_td.get("binary", None)

        if isinstance(binary_reward, torch.Tensor):
            binary_reward = binary_reward.reshape(-1)
            submitted_mask = torch.isclose(submit, torch.ones_like(submit))
            n_submitted = int(submitted_mask.sum().detach().cpu().item())
            if n_submitted > 0:
                submitted_reward = binary_reward[submitted_mask].long()
                success_count = int((submitted_reward > 0).sum().detach().cpu().item())
                failure_count = int((submitted_reward <= 0).sum().detach().cpu().item())

                scalars["success_rate"] = float(success_count) / float(n_submitted)
                scalars["failure_rate"] = float(failure_count) / float(n_submitted)

        if scalars:
            self.writer.add_scalars(
                main_tag="{phase}/submission".format(phase=phase),
                tag_scalar_dict=scalars,
                global_step=iteration,
            )

    def _add_env_info_metrics(
        self, info: TensorDict, *, phase: str, iteration: int
    ) -> None:
        """Add environment info metrics."""
        try:
            scalars = {}
            if "grid_perfect" in info.keys():
                grid_perfect = info["grid_perfect"].detach().float().reshape(-1)
                perfect_rate = float(grid_perfect.mean().cpu().item())
                scalars["grid_perfect_rate"] = perfect_rate
            if "improvement" in info.keys():
                improvement = info["improvement"].detach().reshape(-1)
                improvement_rate = float((improvement > 0).float().mean().cpu().item())
                scalars["improvement_rate"] = improvement_rate

            if scalars:
                self.writer.add_scalars(
                    main_tag="{phase}/env/base_reward".format(phase=phase),
                    tag_scalar_dict=scalars,
                    global_step=iteration,
                )

            scalars = {}
            key_aliases = {"shaping", "submit_term", "not_submit_term", "no_op"}
            for canonical in key_aliases:
                if canonical in info.keys():
                    v = info[canonical].detach().float().reshape(-1)
                    if v.numel() == 0:
                        continue
                    scalars[f"{canonical}_mean"] = float(v.mean().cpu().item())
            if scalars:
                self.writer.add_scalars(
                    main_tag="{phase}/env/pixel_reward".format(phase=phase),
                    tag_scalar_dict=scalars,
                    global_step=iteration,
                )
        except Exception:
            pass

    def _add_loss_metrics(
        self,
        ctx: OnStepContext,
        iteration: int,
        phase: str,
    ) -> None:
        """Add loss metrics."""
        actor_losses: dict[str, float] = {}
        critic_losses: dict[str, float] = {}

        if isinstance(ctx.loss_actor, torch.Tensor):
            actor_losses["loss"] = float(ctx.loss_actor.detach())
        if isinstance(ctx.loss_critic, torch.Tensor):
            critic_losses["loss"] = float(ctx.loss_critic.detach())

        core = ctx.core
        if (
            hasattr(core, "loss_actor_window_avg")
            and core.loss_actor_window_avg is not None
        ):
            actor_losses["window_avg"] = float(core.loss_actor_window_avg)
        if (
            hasattr(core, "loss_critic_window_avg")
            and core.loss_critic_window_avg is not None
        ):
            critic_losses["window_avg"] = float(core.loss_critic_window_avg)

        if actor_losses:
            self.writer.add_scalars(
                main_tag="{phase}/losses/actor".format(phase=phase),
                tag_scalar_dict=actor_losses,
                global_step=iteration,
            )
        if critic_losses:
            self.writer.add_scalars(
                main_tag="{phase}/losses/critic".format(phase=phase),
                tag_scalar_dict=critic_losses,
                global_step=iteration,
            )

    def _add_reward_metrics(
        self, reward_td: TensorDict, submit: torch.Tensor, iteration: int, phase: str
    ) -> None:
        """Add reward metrics."""
        # Handle dict-like rewards
        pixel_rewards = {}
        binary_rewards = {}
        pixel_rewards_sub = {}
        binary_rewards_sub = {}
        pixel_rewards_not_sub = {}
        binary_rewards_not_sub = {}
        pixel_keys = [k.value for k in RewardKeys.get_pixel_wise_keys()]
        binary_keys = [k.value for k in RewardKeys.get_binary_keys()]
        submitted_mask = torch.isclose(submit, torch.ones_like(submit))
        not_mask = ~submitted_mask
        for key, val in reward_td.items():
            # Compute mean across all dimensions
            mean_val = torch.mean(val).detach().cpu().item()
            submitted_mean = torch.mean(val[submitted_mask]).detach().cpu().item()
            not_submitted_mean = torch.mean(val[not_mask]).detach().cpu().item()
            if key in pixel_keys:
                pixel_rewards[key] = mean_val
                pixel_rewards_sub[key] = submitted_mean
                pixel_rewards_not_sub[key] = not_submitted_mean
            elif key in binary_keys:
                binary_rewards[key] = mean_val
                binary_rewards_sub[key] = submitted_mean
                binary_rewards_not_sub[key] = not_submitted_mean

        if pixel_rewards:
            self.writer.add_scalars(
                main_tag="{phase}/rewards/pixel_wise".format(phase=phase),
                tag_scalar_dict=pixel_rewards,
                global_step=iteration,
            )
            self.writer.add_scalars(
                main_tag="{phase}/rewards/pixel_wise/submitted".format(phase=phase),
                tag_scalar_dict=pixel_rewards_sub,
                global_step=iteration,
            )
            self.writer.add_scalars(
                main_tag="{phase}/rewards/pixel_wise/not_submitted".format(phase=phase),
                tag_scalar_dict=pixel_rewards_not_sub,
                global_step=iteration,
            )

        if binary_rewards:
            self.writer.add_scalars(
                main_tag="{phase}/rewards/binary".format(phase=phase),
                tag_scalar_dict=binary_rewards,
                global_step=iteration,
            )
            self.writer.add_scalars(
                main_tag="{phase}/rewards/binary/submitted".format(phase=phase),
                tag_scalar_dict=binary_rewards_sub,
                global_step=iteration,
            )
            self.writer.add_scalars(
                main_tag="{phase}/rewards/binary/not_submitted".format(phase=phase),
                tag_scalar_dict=binary_rewards_not_sub,
                global_step=iteration,
            )

    def _add_q_expected_metrics(
        self,
        critic_probs: TensorDict,
        z_atoms: TensorDict,
        iteration: int,
        phase: str,
    ) -> None:
        """Add critic expected Q metrics."""
        q_expected_mean_base: dict[str, float] = {}
        q_expected_mean_n_step: dict[str, float] = {}
        q_expected_std: dict[str, float] = {}
        base_keys = [k.value for k in RewardKeys.get_base_keys()]
        n_step_keys = [k.value for k in RewardKeys.get_n_step_keys()]

        for key, probs in critic_probs.items():
            z = z_atoms.get(key)
            if z is None:
                continue
            z = z.to(probs.device)
            q = torch.sum(probs * z, dim=-1)
            q_mean = float(q.mean().detach().cpu().item())
            q_std = float(q.std(unbiased=False).detach().cpu().item())
            if key in base_keys:
                q_expected_mean_base[key] = q_mean
            elif key in n_step_keys:
                q_expected_mean_n_step[key] = q_mean
            q_expected_std[key] = q_std

        if q_expected_mean_base:
            self.writer.add_scalars(
                main_tag="{phase}/critic/q_expected_mean/base_rewards".format(
                    phase=phase
                ),
                tag_scalar_dict=q_expected_mean_base,
                global_step=iteration,
            )
        if q_expected_mean_n_step:
            self.writer.add_scalars(
                main_tag="{phase}/critic/q_expected_mean/n_step_rewards".format(
                    phase=phase
                ),
                tag_scalar_dict=q_expected_mean_n_step,
                global_step=iteration,
            )
        self.writer.add_scalars(
            main_tag="{phase}/critic/q_expected_std".format(phase=phase),
            tag_scalar_dict=q_expected_std,
            global_step=iteration,
        )

    def _add_td_error_metrics(
        self,
        td_error: TensorDict,
        iteration: int,
        phase: str,
    ) -> None:
        """Custom-scalar layouts for TD-error stats."""
        td_error_mean = {}
        td_error_std = {}
        td_error_max = {}
        for key, val in td_error.items():
            try:
                v = val.detach().float().reshape(-1)
                v = v[torch.isfinite(v)]
                if v.numel() == 0:
                    continue
                td_error_mean[key] = float(v.mean().cpu())
                td_error_std[key] = float(v.std(unbiased=False).cpu())
                td_error_max[key] = float(v.max().cpu())

            except Exception:
                continue
        if td_error_mean:
            self.writer.add_scalars(
                main_tag="{phase}/td_error/mean".format(phase=phase),
                tag_scalar_dict=td_error_mean,
                global_step=iteration,
            )
        if td_error_std:
            self.writer.add_scalars(
                main_tag="{phase}/td_error/std".format(phase=phase),
                tag_scalar_dict=td_error_std,
                global_step=iteration,
            )
        if td_error_max:
            self.writer.add_scalars(
                main_tag="{phase}/td_error/max".format(phase=phase),
                tag_scalar_dict=td_error_max,
                global_step=iteration,
            )

    def _add_step_hyperparams(self, core: ActorCriticCoreProto, iteration: int) -> None:
        # Optimizer learning rates
        lr_scalars = {}
        if hasattr(core, "actor_scheduler") and isinstance(
            core.actor_scheduler, optim.lr_scheduler.StepLR
        ):
            actor_lr = core.actor_scheduler.get_last_lr()[0]
            lr_scalars["actor"] = actor_lr
        elif hasattr(core, "actor_optimizer") and core.actor_optimizer is not None:
            try:
                actor_lr = core.actor_optimizer.param_groups[0]["lr"]
                lr_scalars["actor"] = actor_lr
            except Exception:
                pass
        if hasattr(core, "critic_scheduler") and isinstance(
            core.critic_scheduler, optim.lr_scheduler.StepLR
        ):
            critic_lr = core.critic_scheduler.get_last_lr()[0]
            lr_scalars["critic"] = critic_lr
        elif hasattr(core, "critic_optimizer") and core.critic_optimizer is not None:
            try:
                critic_lr = core.critic_optimizer.param_groups[0]["lr"]
                lr_scalars["critic"] = critic_lr
            except Exception:
                pass
        if lr_scalars:
            self.writer.add_scalars(
                main_tag="hyperparams/lr",
                tag_scalar_dict=lr_scalars,
                global_step=iteration,
            )

        # Algo hyperparams (plateau diagnostics)
        if hasattr(core, "entropy_coef"):
            self.writer.add_scalar(
                "hyperparams/entropy_coef", float(core.entropy_coef), iteration
            )
        # Sampler hyperparams if present
        sampler = None
        if self.replay_buffer is not None:
            sampler = getattr(self.replay_buffer, "sampler", None)
        if sampler is not None:
            replay_scalars = {}
            if hasattr(sampler, "alpha") and sampler.alpha is not None:
                replay_scalars["alpha"] = float(sampler.alpha)
            if hasattr(sampler, "beta") and sampler.beta is not None:
                replay_scalars["beta"] = float(sampler.beta)
            if replay_scalars:
                self.writer.add_scalars(
                    main_tag="hyperparams/replay_buffer",
                    tag_scalar_dict=replay_scalars,
                    global_step=iteration,
                )

    # -------------------- InternalLogger Callbacks --------------------
    def _log_minimal_metrics(self, ctx: OnStepContext, *, phase: str) -> None:
        """Log cheap, robust metrics that rely only on ctx/core public API.

        Args:
            ctx: Training step ctx.
            phase: Phase of the training step.
        """
        # Nicer visualizations
        iteration = ctx.iteration

        if (iteration % self.minimal_frequency != 0 and not ctx.run_end) or ctx.warmup:
            return

        # Epoch
        epoch = ctx.epoch
        if epoch is not None and phase == "train":
            self.writer.add_scalars(
                "steps_data",
                tag_scalar_dict={
                    "epoch": epoch,
                    "epoch_step_idx": ctx.epoch_step_idx,
                    "episode_step_idx": ctx.episode_step_idx,
                },
                global_step=iteration,
            )
            self._add_step_hyperparams(ctx.core, iteration)

        self._add_loss_metrics(ctx, phase=phase, iteration=iteration)

        batch: TensorDict = ctx.batch
        self.total_samples += batch.shape[0]
        curriculum_used = batch["info"].get("curriculum_used", None)
        actor_mask = torch.ones(batch.shape[0], dtype=torch.bool)
        used_rate = 0.0
        if isinstance(curriculum_used, torch.Tensor):
            self.curriculum_actual_usage += int(
                curriculum_used.sum().detach().cpu().item()
            )
            used_rate = float(curriculum_used.float().mean().detach().cpu().item())
            actor_mask = (curriculum_used == 0).reshape(-1).nonzero(as_tuple=True)[0]

        self.writer.add_scalar(f"curriculum/{phase}/usage_rate", used_rate, iteration)

        if used_rate < 1.0:
            batch = batch[actor_mask]
            # Reward heads: log per-batch average for each reward key
            # Submission stats (rate, success, failure)
            submit_raw: torch.Tensor = batch["actions"]["submit"]
            if submit_raw.ndim >= 2:
                # action probabilities -> discrete submit decision
                submit = torch.argmax(submit_raw, dim=-1)
            else:
                # already discrete
                submit = submit_raw
            submit = submit.reshape(-1).to(dtype=torch.long)

            reward_td = batch["reward"]
            self._add_reward_metrics(
                reward_td, submit, phase=phase, iteration=iteration
            )

            info = batch.get("info", None)
            if info is not None and hasattr(info, "keys"):
                self._add_env_info_metrics(info, phase=phase, iteration=iteration)

            self._add_submission_metrics(submit, reward_td, iteration, phase)

        # Ensure periodic flush for visibility without too much overhead
        try:
            if iteration % self.medium_frequency == 0:
                self.writer.flush()
        except Exception:
            logger.warning("Failed to flush TensorBoard writer")
            pass

    def _maybe_log_medium_metrics(self, ctx: OnStepContext, *, phase: str) -> None:
        """Medium-frequency metrics (policy entropy, Q estimates, norms).

        Computed safely with no_grad, gated by `medium_frequency` to keep
        overhead small.
        """
        iteration = ctx.iteration
        if (iteration % self.medium_frequency != 0 and not ctx.run_end) or ctx.warmup:
            return

        logger.debug(
            "Logging medium metrics for iteration %d [phase: %s]", iteration, phase
        )

        batch: TensorDict = ctx.batch
        state = batch["state"]
        core = ctx.core

        with torch.no_grad():
            action_probs = core.actor(state)

        with torch.no_grad():
            critic_probs: TensorDict = core.critic(state, action_probs).q_dist

        # Critic expected Q per head (mean)
        self._add_q_expected_metrics(
            critic_probs, core.critic.z_atoms, iteration, phase
        )

        # Policy entropy across action heads
        entropy = 0.0
        n_heads = 0
        per_head_entropy: dict[str, float] = {}
        per_head_max_prob: dict[str, float] = {}
        for probs in action_probs.values():
            p = probs.clamp(min=1e-10)
            entropy_t = -torch.sum(p * torch.log(p + 1e-10), dim=-1).mean()
            entropy += float(entropy_t.detach().cpu())
            n_heads += 1
        for head_name, probs in action_probs.items():
            p = probs.clamp(min=1e-10)
            entropy_t = -torch.sum(p * torch.log(p + 1e-10), dim=-1).mean()
            per_head_entropy[head_name] = float(entropy_t.detach().cpu())
            per_head_max_prob[head_name] = float(
                probs.max(dim=-1).values.mean().detach().cpu()
            )
        if n_heads > 0:
            entropy /= n_heads
            self.writer.add_scalar(f"{phase}/policy/actor_entropy", entropy, iteration)
        if per_head_entropy:
            self.writer.add_scalars(
                main_tag=f"{phase}/policy/entropy",
                tag_scalar_dict=per_head_entropy,
                global_step=iteration,
            )
        if per_head_max_prob:
            self.writer.add_scalars(
                main_tag=f"{phase}/policy/max_prob",
                tag_scalar_dict=per_head_max_prob,
                global_step=iteration,
            )

        # TD-error metrics
        td_error = getattr(ctx, "td_error", None)
        if td_error is not None and hasattr(td_error, "items"):
            self._add_td_error_metrics(td_error, iteration, phase)

        # Weight norms (L2) for actor and critic
        if phase == "train":

            def l2_norm(module) -> float:
                total = 0.0
                for p in module.parameters():
                    if p is None:
                        continue
                    total += float(p.detach().norm(2).item())
                return total

            actor_norm = l2_norm(core.actor)
            critic_norm = l2_norm(core.critic)
            self.writer.add_scalars(
                main_tag="weight_l2_norm",
                tag_scalar_dict={"actor": actor_norm, "critic": critic_norm},
                global_step=iteration,
            )


def make_tensorboard_callbacks(
    experiment_path: str | Path,
    *,
    experiment_name: str | None = None,
    experiment_description: str | None = None,
    replay_buffer: Any | None = None,
    warmup_buffer_ratio: float = 0.2,
    medium_frequency: int = 100,
    minimal_frequency: int = 1,
) -> tuple[
    TensorBoardWriterManager,
    Callable[[OnStepContext], None],
    Callable[[OnStepContext], None],
]:
    """Factory to create a manager and its train/val callbacks.

    The caller should retain the manager
    reference to log hparams and close it if desired.

    Args:
        experiment_path: Root path for this run's artifacts.
        experiment_name: Name of the experiment.
        experiment_description: Description of the experiment.
        replay_buffer: Replay buffer to attach for logging stats.
        warmup_buffer_ratio: Ratio used to display buffer warmup threshold.
        medium_frequency: Frequency at which to log medium-tier metrics.
        minimal_frequency: Frequency at which to log minimal-tier metrics.

    Returns:
        A tuple of (manager, on_train_step_callback, on_validation_step_callback).
    """
    mgr = TensorBoardWriterManager(
        experiment_path,
        experiment_name=experiment_name,
        experiment_description=experiment_description,
        replay_buffer=replay_buffer,
        warmup_buffer_ratio=warmup_buffer_ratio,
        medium_frequency=medium_frequency,
        minimal_frequency=minimal_frequency,
    )
    return mgr, mgr.on_train_step, mgr.on_validation_step
