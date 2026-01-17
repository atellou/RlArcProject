"""TensorBoard logging utilities with decoupled callbacks.

This module provides a minimal, decoupled TensorBoard integration that plugs
into the Trainer via callbacks. It avoids coupling to the core or trainer
internals beyond public attributes exposed on the OnStepContext.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.optim as optim
from tensordict import TensorDict
from torch.utils.tensorboard import SummaryWriter

from rlarcworld.trainers.core import OnStepContext, ReplayBufferProto

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
        logdir_name: str = "tensorboard",
        replay_buffer: ReplayBufferProto | None = None,
        warmup_buffer_ratio: float = 0.2,
        medium_frequency: int = 100,
        minimal_frequency: int = 10,
    ) -> None:
        """Create a manager and underlying SummaryWriter.

        Args:
            experiment_path: Root path for this run's artifacts.
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
        self.replay_buffer = replay_buffer
        self.warmup_buffer_ratio = float(warmup_buffer_ratio)
        self.medium_frequency = int(max(1, medium_frequency))

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

    # -------------------- Internal helpers --------------------
    def _log_minimal_metrics(self, ctx: OnStepContext, *, phase: str) -> None:
        """Log cheap, robust metrics that rely only on ctx/core public API.

        Args:
            ctx: Training step ctx.
            phase: Phase of the training step.
        """
        # Nicer visualizations
        iteration = ctx.iteration

        if iteration % self.minimal_frequency != 0 or ctx.warmup:
            return

        # Losses
        if ctx.loss_actor is not None:
            self.writer.add_scalar(
                f"{phase}/loss/actor", float(ctx.loss_actor.detach()), iteration
            )
        if ctx.loss_critic is not None:
            self.writer.add_scalar(
                f"{phase}/loss/critic", float(ctx.loss_critic.detach()), iteration
            )

        core = ctx.core
        if (
            hasattr(core, "loss_actor_window_avg")
            and core.loss_actor_window_avg is not None
        ):
            self.writer.add_scalar(
                f"{phase}/loss/actor_window_avg",
                float(core.loss_actor_window_avg),
                iteration,
            )
        if (
            hasattr(core, "loss_critic_window_avg")
            and core.loss_critic_window_avg is not None
        ):
            self.writer.add_scalar(
                f"{phase}/loss/critic_window_avg",
                float(core.loss_critic_window_avg),
                iteration,
            )

        # Optimizer learning rates
        if hasattr(core, "actor_scheduler") and isinstance(
            core.actor_scheduler, optim.lr_scheduler.StepLR
        ):
            actor_lr = core.actor_scheduler.get_last_lr()[0]
            self.writer.add_scalar(f"{phase}/optim/actor_lr", actor_lr, iteration)
        if hasattr(core, "critic_scheduler") and isinstance(
            core.critic_scheduler, optim.lr_scheduler.StepLR
        ):
            critic_lr = core.critic_scheduler.get_last_lr()[0]
            self.writer.add_scalar(f"{phase}/optim/critic_lr", critic_lr, iteration)

        # Algo hyperparams that change over time (e.g., entropy coef)
        if (
            hasattr(core, "entropy_coef")
            and hasattr(core, "entropy_coef_decay")
            and core.entropy_coef_decay != 1.0
        ):
            self.writer.add_scalar(
                f"{phase}/algo/entropy_coef", float(core.entropy_coef), iteration
            )

        # Reward heads: log per-batch average for each reward key
        batch: TensorDict = ctx.batch

        reward_td = batch["reward"]
        # Handle dict-like rewards
        for key, val in reward_td.items():
            # Compute mean across all dimensions
            mean_val = float(val.float().mean().detach().cpu().item())
            self.writer.add_scalar(f"{phase}/reward/{key}", mean_val, iteration)

        # Submission stats (rate, success, failure)
        submit_raw: torch.Tensor = batch["actions"]["submit"]
        if submit_raw.ndim >= 2:
            # action probabilities -> discrete submit decision
            submit = torch.argmax(submit_raw, dim=-1)
        else:
            # already discrete
            submit = submit_raw

        submit = submit.reshape(-1).to(dtype=torch.long)
        batch_size = int(submit.shape[0])
        submit_count = int(submit.sum().detach().cpu().item())

        self.writer.add_scalar(
            f"{phase}/submission/rate",
            float(submit_count) / float(max(1, batch_size)),
            iteration,
        )
        self.writer.add_scalar(
            f"{phase}/submission/count",
            float(submit_count),
            iteration,
        )

        # Success/failure: use binary reward if available
        binary_reward = reward_td["binary"]

        if isinstance(binary_reward, torch.Tensor):
            binary_reward = binary_reward.reshape(-1)
            submitted_mask = submit == 1
            n_submitted = int(submitted_mask.sum().detach().cpu().item())
            if n_submitted > 0:
                submitted_reward = binary_reward[submitted_mask]
                success_count = int((submitted_reward > 0).sum().detach().cpu().item())
                failure_count = int((submitted_reward < 0).sum().detach().cpu().item())

                self.writer.add_scalar(
                    f"{phase}/submission/success_rate",
                    float(success_count) / float(n_submitted),
                    iteration,
                )
                self.writer.add_scalar(
                    f"{phase}/submission/failure_rate",
                    float(failure_count) / float(n_submitted),
                    iteration,
                )
                self.writer.add_scalar(
                    f"{phase}/submission/success_count",
                    float(success_count),
                    iteration,
                )
                self.writer.add_scalar(
                    f"{phase}/submission/failure_count",
                    float(failure_count),
                    iteration,
                )

        # Sampler hyperparams if present
        sampler = None
        if self.replay_buffer is not None:
            sampler = getattr(self.replay_buffer, "sampler", None)
        if sampler is not None:
            if hasattr(sampler, "alpha") and sampler.alpha is not None:
                self.writer.add_scalar(
                    f"{phase}/replay/alpha", float(sampler.alpha), iteration
                )
            if hasattr(sampler, "beta") and sampler.beta is not None:
                self.writer.add_scalar(
                    f"{phase}/replay/beta", float(sampler.beta), iteration
                )

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
        if iteration % self.medium_frequency != 0 or ctx.warmup:
            return

        logger.debug(
            "Logging medium metrics for iteration %d [phase: %s]", iteration, phase
        )

        batch: TensorDict = ctx.batch
        state = batch["state"]
        core = ctx.core

        # actor -> action_probs
        with torch.no_grad():
            action_probs = core.actor(state)

        # Policy entropy across action heads
        entropy = 0.0
        n_heads = 0
        for probs in action_probs.values():
            p = probs.clamp(min=1e-10)
            entropy_t = -torch.sum(p * torch.log(p + 1e-10), dim=-1).mean()
            entropy += float(entropy_t.detach().cpu())
            n_heads += 1
        if n_heads > 0:
            entropy /= n_heads
            self.writer.add_scalar(f"{phase}/policy/entropy", entropy, iteration)

        # Critic expected Q per head (mean)
        if hasattr(core, "critic") and hasattr(core.critic, "z_atoms"):
            with torch.no_grad():
                critic_probs = core.critic(state, action_probs)
            for key, probs in critic_probs.items():
                z = core.critic.z_atoms.get(key)
                if z is None:
                    continue
                z = z.to(probs.device)
                q = torch.sum(probs * z, dim=-1)
                q_mean = float(q.mean().detach().cpu())
                self.writer.add_scalar(
                    f"{phase}/critic/q_expected_mean/{key}",
                    q_mean,
                    iteration,
                )

        # Weight norms (L2) for actor and critic
        def l2_norm(module) -> float:
            total = 0.0
            for p in module.parameters():
                if p is None:
                    continue
                total += float(p.detach().norm(2).item())
            return total

        actor_norm = l2_norm(core.actor)
        critic_norm = l2_norm(core.critic)
        self.writer.add_scalar(f"{phase}/actor/weight_l2_norm", actor_norm, iteration)
        self.writer.add_scalar(f"{phase}/critic/weight_l2_norm", critic_norm, iteration)


def make_tensorboard_callbacks(
    experiment_path: str | Path,
    *,
    replay_buffer: Any | None = None,
    warmup_buffer_ratio: float = 0.2,
    medium_frequency: int = 100,
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
        replay_buffer: Replay buffer to attach for logging stats.
        warmup_buffer_ratio: Ratio used to display buffer warmup threshold.
        medium_frequency: Frequency at which to log medium-tier metrics.

    Returns:
        A tuple of (manager, on_train_step_callback, on_validation_step_callback).
    """
    mgr = TensorBoardWriterManager(
        experiment_path,
        replay_buffer=replay_buffer,
        warmup_buffer_ratio=warmup_buffer_ratio,
        medium_frequency=medium_frequency,
    )
    return mgr, mgr.on_train_step, mgr.on_validation_step
