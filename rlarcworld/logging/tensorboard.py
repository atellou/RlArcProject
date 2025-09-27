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
from torch.utils.tensorboard import SummaryWriter

from rlarcworld.runners.core import OnStepContext, ReplayBufferProto

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
    ) -> None:
        """Create a manager and underlying SummaryWriter.

        Args:
            experiment_path: Root path for this run's artifacts.
            logdir_name: Directory name under experiment root for TB logs.
            replay_buffer: Replay buffer to log stats from.
            warmup_buffer_ratio: Ratio used to display buffer warmup threshold.
            medium_frequency: Frequency at which to log medium-tier metrics.
        """
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
        self._maybe_log_medium_metrics(ctx, phase="train")

    def on_validation_step(self, ctx: OnStepContext) -> None:
        """Callback to log validation-step metrics."""
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
        """Log cheap, robust metrics that rely only on context/core public API.

        Args:
            ctx: Training step context.
            phase: Phase of the training step.
        """
        step = int(getattr(ctx, "iteration", 0))
        if step <= 0:
            # Avoid writing at step 0 for nicer charts
            step = 1

        # Losses (if available)
        try:
            if ctx.loss_actor is not None:
                self.writer.add_scalar(
                    f"{phase}/loss/actor", float(ctx.loss_actor), step
                )
            if ctx.loss_critic is not None:
                self.writer.add_scalar(
                    f"{phase}/loss/critic", float(ctx.loss_critic), step
                )
        except Exception:
            logger.error("Failed to log loss metrics")
            pass

        # Optimizer learning rates
        try:
            core = ctx.core
            actor_lr = None
            critic_lr = None
            if hasattr(core, "actor_optimizer"):
                for g in core.actor_optimizer.param_groups:
                    actor_lr = float(g.get("lr", None))
                    break
            if hasattr(core, "critic_optimizer"):
                for g in core.critic_optimizer.param_groups:
                    critic_lr = float(g.get("lr", None))
                    break
            if actor_lr is not None:
                self.writer.add_scalar(f"{phase}/optim/actor_lr", actor_lr, step)
            if critic_lr is not None:
                self.writer.add_scalar(f"{phase}/optim/critic_lr", critic_lr, step)
        except Exception:
            logger.warning("Failed to log optimizer learning rates")
            pass

        # Algo hyperparams that change over time (e.g., entropy coef)
        try:
            if hasattr(ctx.core, "entropy_coef"):
                self.writer.add_scalar(
                    f"{phase}/algo/entropy_coef", float(ctx.core.entropy_coef), step
                )
        except Exception:
            logger.warning("Failed to log algorithm hyperparams")
            pass

        # Replay buffer stats (size and warmup progress)
        try:
            if self.replay_buffer is not None and hasattr(
                self.replay_buffer, "storage"
            ):
                size = len(self.replay_buffer.storage)
                cap = getattr(self.replay_buffer.storage, "max_size", None)
                if cap is None:
                    cap = getattr(self.replay_buffer.storage, "_max_size", None)
                if isinstance(cap, (int, float)) and cap:
                    warmup_progress = float(size) / float(cap)
                else:
                    warmup_progress = 0.0
                self.writer.add_scalar(f"{phase}/replay/size", int(size), step)
                self.writer.add_scalar(f"{phase}/replay/capacity", int(cap or 0), step)
                self.writer.add_scalar(
                    f"{phase}/replay/warmup_progress",
                    warmup_progress,
                    step,
                )
                self.writer.add_scalar(
                    f"{phase}/replay/warmup_threshold",
                    float(self.warmup_buffer_ratio),
                    step,
                )
            # Sampler hyperparams if present
            sampler = (
                getattr(self.replay_buffer, "sampler", None)
                if self.replay_buffer is not None
                else None
            )
            if sampler is not None:
                if hasattr(sampler, "alpha") and sampler.alpha is not None:
                    self.writer.add_scalar(
                        f"{phase}/replay/alpha", float(sampler.alpha), step
                    )
                if hasattr(sampler, "beta") and sampler.beta is not None:
                    self.writer.add_scalar(
                        f"{phase}/replay/beta", float(sampler.beta), step
                    )
        except Exception:
            logger.warning("Failed to log replay buffer stats")
            pass

        # Ensure periodic flush for visibility without too much overhead
        try:
            if step % 100 == 0:
                self.writer.flush()
        except Exception:
            logger.warning("Failed to flush TensorBoard writer")
            pass

    def _maybe_log_medium_metrics(self, ctx: OnStepContext, *, phase: str) -> None:
        """Medium-frequency metrics (policy entropy, Q estimates, norms).

        Computed safely with no_grad, gated by `medium_frequency` to keep
        overhead small.
        """
        try:
            step = int(getattr(ctx, "iteration", 0))
            if step <= 0 or step % self.medium_frequency != 0:
                return
            if ctx.batch is None:
                return
            state = ctx.batch.get("state") if hasattr(ctx.batch, "get") else None
            if state is None:
                return
            core = ctx.core
            # actor -> action_probs
            with torch.no_grad():
                action_probs = core.actor(state)
            # Policy entropy across action heads
            try:
                entropy = 0.0
                n_heads = 0
                for probs in action_probs.values():
                    # clamp for numerical stability
                    p = probs.clamp(min=1e-10)
                    entropy_t = -torch.sum(p * torch.log(p + 1e-10), dim=-1).mean()
                    entropy += float(entropy_t.detach().cpu())
                    n_heads += 1
                if n_heads > 0:
                    entropy /= n_heads
                    self.writer.add_scalar(f"{phase}/policy/entropy", entropy, step)
            except Exception:
                logger.warning("Failed to log policy entropy")
                pass
            # Critic expected Q per head (mean)
            try:
                if hasattr(core, "critic") and hasattr(core.critic, "z_atoms"):
                    with torch.no_grad():
                        critic_probs = core.critic(state, action_probs)
                    for key, probs in critic_probs.items():
                        z = core.critic.z_atoms.get(key)
                        if z is None:
                            continue
                        # move z to probs device for multiply
                        z = z.to(probs.device)
                        q = torch.sum(probs * z, dim=-1)
                        q_mean = float(q.mean().detach().cpu())
                        self.writer.add_scalar(
                            f"{phase}/critic/q_expected_mean/{key}",
                            q_mean,
                            step,
                        )
            except Exception:
                logger.warning("Failed to log critic expected Q")
                pass
            # Weight norms (L2) for actor and critic
            try:

                def l2_norm(module) -> float:
                    total = 0.0
                    for p in module.parameters():
                        if p is None:
                            continue
                        total += float(p.detach().norm(2).item())
                    return total

                actor_norm = l2_norm(core.actor)
                critic_norm = l2_norm(core.critic)
                self.writer.add_scalar(
                    f"{phase}/actor/weight_l2_norm", actor_norm, step
                )
                self.writer.add_scalar(
                    f"{phase}/critic/weight_l2_norm", critic_norm, step
                )
            except Exception:
                logger.warning("Failed to log weight norms")
                pass
        except Exception:
            logger.warning("Failed to log medium metrics")
            pass


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
