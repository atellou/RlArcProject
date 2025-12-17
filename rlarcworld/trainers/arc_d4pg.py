"""ArcD4PG class to run an experiment with the D4PG algorithm and ARC environment."""

try:
    # NumPy reconstruct function used by some pickled arrays
    from numpy.core.multiarray import _reconstruct as _np_reconstruct
except Exception:  # pragma: no cover
    try:
        from numpy._core.multiarray import _reconstruct as _np_reconstruct
    except Exception:
        _np_reconstruct = None  # fallback; will skip adding if unavailable
import logging
import os
import random
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
from tensordict import TensorDict
from tensordict._reductions import _make_td
from torch.serialization import add_safe_globals
from torchrl.data.replay_buffers import (
    LazyMemmapStorage,
    PrioritizedSampler,
    TensorDictReplayBuffer,
)

from rlarcworld.algorithms.d4pg import D4PGCore

# Re-export config classes for convenient imports
from rlarcworld.common.configs import ArcD4PGConfig, RlArcWorldViTARCConfig
from rlarcworld.common.enums import RewardKeys
from rlarcworld.environments.steppers.arc_stepper import PixelWiseArcStepper
from rlarcworld.environments.steppers.reward_world_model import ArcWorldModelStepper
from rlarcworld.logging.tensorboard import make_tensorboard_callbacks
from rlarcworld.models.actor import ViTARCActor
from rlarcworld.models.critic import VitARCCritic
from rlarcworld.models.reward_world_model import RewardWorldModel
from rlarcworld.runners.core import OnStepContext, ReplayBufferProto, Trainer
from rlarcworld.utils.checkpointing import (
    add_schema_and_class,
    resolve_load_path,
    resolve_save_path,
)
from rlarcworld.utils.device import enable_cuda
from rlarcworld.utils.torch_queue import TorchQueue

logger = logging.getLogger(__name__)


class ArcD4PG:
    """Complete D4PG experiment orchestrator for ARC tasks.

    This class brings together all the decoupled components:
    - D4PGCore: Algorithm implementation
    - ViTARCActor & VitARCCritic: Neural networks
    - ArcBatchGridEnv: Environment wrapper
    - Trainer: Training loop orchestration
    """

    def __init__(
        self,
        config: ArcD4PGConfig,
        **kwargs,
    ):
        """Initialize the ArcD4PG experiment.

        Args:
            config: Complete experiment configuration
            **kwargs: Additional arguments passed to D4PGCore
        """
        # Store full config for reference
        config = config.get_validated_config()
        self.exp_config = config

        # Set random seed
        torch.manual_seed(config.system.seed)

        # Configure device and logging
        if config.system.device is None:
            self.cuda_config = enable_cuda(amp_enabled=config.system.amp_enabled)
        else:
            self.cuda_config = {
                "device": config.system.device,
                "amp_enabled": config.system.amp_enabled,
                "use_checkpointing": False,
            }
        self.device = self.cuda_config["device"]

        # Store commonly accessed values for convenience
        self.grid_size = config.env.grid_size
        self.color_values = config.env.color_values
        self.batch_size = config.env.batch_size
        self.max_steps_per_epoch = config.training.max_steps_per_epoch
        self.validation_steps_per_epoch = config.training.validation_steps_per_epoch
        self.min_train_batch_size = config.training.min_train_batch_size

        # Validation policies
        self.validation_epochs = config.training.validation_epochs
        self.validation_steps_frequency = config.training.validation_steps_frequency

        # Checkpoint policies
        self.ckpt_save_frequency = config.checkpoint.save_frequency
        if config.checkpoint.checkpoint_dir is None:
            self.checkpoint_dir = Path(
                "./experiments/{}-{}/".format(
                    self.__class__.__name__,
                    datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"),
                )
            )
        else:
            self.checkpoint_dir = Path(config.checkpoint.checkpoint_dir)

        logger.info("Checkpoints will be saved to: %s", self.checkpoint_dir.resolve())

        # Initialize networks
        logger.info("Initializing actor and critic networks...")
        logger.info("Creating actor network...")

        # Use config's rl_vit_arc_config or create one with best settings
        rl_vit_arc_config = config.network or RlArcWorldViTARCConfig.best_config(
            config.env.grid_size
        )

        self.actor = ViTARCActor(
            color_values=config.env.color_values,
            rl_vit_arc_config=rl_vit_arc_config,
            device_config=self.cuda_config,
        )
        logger.info("Actor network created successfully")

        logger.info("Creating critic network...")
        self.critic = VitARCCritic(
            color_values=config.env.color_values,
            rl_vit_arc_config=rl_vit_arc_config,
            num_atoms=config.distributional.num_atoms,
            v_min=config.distributional.v_min,
            v_max=config.distributional.v_max,
            device_config=self.cuda_config,
        )
        logger.info("Critic network created successfully")

        # Create training stepper
        logger.info("Creating training stepper...")
        self.training_stepper = PixelWiseArcStepper(
            max_steps_per_epoch=self.max_steps_per_epoch,
            dataset_path=config.dataset_path,
            grid_size=config.env.grid_size,
            color_values=config.env.color_values,
            actor=self.actor,
            config=self.cuda_config,
            max_penalty=config.env.max_penalty,
            n_step_config=config.d4pg.n_step_config,
        )
        logger.info("Training stepper created successfully")

        # Create validation stepper
        logger.info("Creating validation stepper...")
        self.validation_stepper = PixelWiseArcStepper(
            max_steps_per_epoch=self.validation_steps_per_epoch,
            dataset_path=config.dataset_path,
            grid_size=config.env.grid_size,
            color_values=config.env.color_values,
            actor=self.actor,
            config=self.cuda_config,
            max_penalty=config.env.max_penalty,
        )
        logger.info("Validation stepper created successfully")

        # Create world model stepper (optional Dyna-style imagination)
        self.use_world_model = config.world_model.enabled
        self.world_model_stepper: Optional[ArcWorldModelStepper] = None
        if config.world_model.enabled:
            logger.info("Creating world model stepper...")
            reward_model = RewardWorldModel(
                color_values=config.env.color_values,
                rl_vit_arc_config=rl_vit_arc_config,
                config=self.cuda_config,
            )
            self.world_model_stepper = ArcWorldModelStepper(
                grid_size=config.env.grid_size,
                color_values=config.env.color_values,
                actor=self.actor,
                config=self.cuda_config,
                reward_model=reward_model,
                reward_model_lr=config.world_model.learning_rate,
                lr_scheduler_kwargs=config.world_model.lr_scheduler_kwargs,
                step_horizon=config.world_model.imagination_horizon,
                n_step_config=config.d4pg.n_step_config,
                gamma=config.d4pg.gamma,
                model_update_frequency=config.world_model.model_update_frequency,
                imagination_ratio=config.world_model.imagination_ratio,
                imagination_frequency=config.world_model.imagination_frequency,
            )
            logger.info("World model stepper created successfully")

        # Initialize D4PG core
        logger.info("Initializing D4PG algorithm core...")
        self.d4pg_core = D4PGCore(
            actor=self.actor,
            critic=self.critic,
            policy_lr=config.d4pg.policy_lr,
            critic_lr=config.d4pg.critic_lr,
            gamma=config.d4pg.gamma,
            tau=config.d4pg.tau,
            lr_scheduler_kwargs=config.d4pg.lr_scheduler_kwargs,
            n_steps=config.d4pg.n_step_config.n_steps,
            config=self.cuda_config,
            amp_enabled=config.system.amp_enabled,
            carsm=config.d4pg.carsm,
            entropy_coef=config.d4pg.entropy_coef,
            entropy_coef_decay=config.d4pg.entropy_coef_decay,
            target_update_frequency=config.d4pg.target_update_frequency,
            max_grad_norm=config.d4pg.max_grad_norm,
            compute_td_error=config.d4pg.compute_td_error,
            **kwargs,
        )

        # Initialize trainer with callbacks
        self._setup_trainer(
            replay_buffer_size=config.replay_buffer.size,
            alpha=config.replay_buffer.alpha,
            beta=config.replay_buffer.beta,
            warmup_buffer_ratio=config.replay_buffer.warmup_ratio,
        )

    def _get_hparams(self):
        hparams = {
            "grid_size": self.grid_size,
            "color_values": self.color_values,
            "batch_size": self.batch_size,
            "max_steps_per_epoch": self.max_steps_per_epoch,
            "validation_steps_per_epoch": self.validation_steps_per_epoch,
            "device": self.device,
            "use_world_model": self.use_world_model,
        }
        # trainer hyperparameters dict is nested; flatten selected

        hp = self.trainer.hyperparameters
        if isinstance(hp, dict):
            # core hp is a property on trainer, not nested
            core_hp = hp.get("core", {}) if isinstance(hp.get("core"), dict) else {}
            hparams.update(
                {
                    "t0_actor_lr": core_hp.get("t0_actor_lr", None),
                    "t0_critic_lr": core_hp.get("t0_critic_lr", None),
                    "actor_lr": core_hp.get("actor_lr", None),
                    "critic_lr": core_hp.get("critic_lr", None),
                    "gamma": core_hp.get("gamma", None),
                    "tau": core_hp.get("tau", None),
                    "entropy_coef": core_hp.get("entropy_coef", None),
                }
            )

        # Add ViTARC encoder hyperparameters
        vcfg = self.actor.rl_vit_arc_config
        hparams.update(
            {
                "vitarc_d_model": getattr(vcfg, "d_model", None),
                "vitarc_d_ff": getattr(vcfg, "d_ff", None),
                "vitarc_num_layers": getattr(vcfg, "num_layers", None),
                "vitarc_num_heads": getattr(vcfg, "num_heads", None),
                "vitarc_use_ope": getattr(vcfg, "use_ope", None),
                "vitarc_use_index_grid": getattr(vcfg, "use_index_grid", None),
                "vitarc_use_temporal": getattr(vcfg, "use_temporal", None),
                "vitarc_use_examples": getattr(vcfg, "use_examples", None),
                "vitarc_ape_type": getattr(vcfg, "ape_type", None),
                "vitarc_rpe_type": getattr(vcfg, "rpe_type", None),
                "vitarc_ape_mixer": getattr(vcfg, "ape_mixer", None),
                "vitarc_trainable_ape": getattr(vcfg, "trainable_ape", None),
            }
        )

        # Add world model hparams if enabled
        if isinstance(self.world_model_stepper, ArcWorldModelStepper):
            hparams["imagination_count"] = self.world_model_stepper.imagination_count

        return hparams

    def _setup_trainer(
        self,
        replay_buffer_size: int,
        alpha: float,
        beta: float,
        warmup_buffer_ratio: float,
    ) -> None:
        """Setup the trainer with appropriate callbacks."""
        # Setup step callbacks
        on_step_callbacks: OrderedDict[str, Callable[[OnStepContext], None]] = (
            self._on_train_callbacks()
        )

        # Set up replay buffer
        self.replay_buffer_priority_key = "td_error"
        sampler = PrioritizedSampler(
            replay_buffer_size,
            alpha=alpha,
            beta=beta,
        )
        storage = LazyMemmapStorage(replay_buffer_size)
        self.replay_buffer = TensorDictReplayBuffer(
            storage=storage,
            sampler=sampler,
            priority_key=self.replay_buffer_priority_key,
        )

        self.trainer = Trainer(
            core=self.d4pg_core,
            batch_size=self.batch_size,
            train_step_stream_factory=self.training_stepper.create_step_stream_from_samples,
            validation_step_stream_factory=self.validation_stepper.create_step_stream_from_samples,
            replay_buffer=self.replay_buffer,
            replay_buffer_update_priority=self._replay_buffer_update_priority,
            warmup_buffer_ratio=warmup_buffer_ratio,
            on_train_step=on_step_callbacks,
            on_validation_step=self._on_validation_callbacks(),
            device=self.device,
            min_train_batch_size=self.min_train_batch_size,
        )

        # Attach TensorBoard callbacks (train and validation)
        try:
            if not self.exp_config.logging.tensorboard_enabled:
                logger.info("TensorBoard disabled in config, skipping setup")
                return
            self.tb_manager, tb_train_cb, tb_val_cb = make_tensorboard_callbacks(
                self.checkpoint_dir,
                replay_buffer=self.replay_buffer,
                warmup_buffer_ratio=warmup_buffer_ratio,
            )
            # Extend training callbacks
            self.trainer.on_train_step = self.trainer.on_train_step or OrderedDict()
            self.trainer.on_train_step["tensorboard"] = tb_train_cb
            # Set validation callbacks
            self.trainer.on_validation_step = (
                self.trainer.on_validation_step or OrderedDict()
            )
            self.trainer.on_validation_step["tensorboard"] = tb_val_cb
        except Exception as e:
            logger.warning("TensorBoard not attached: %s", e)

    def _replay_buffer_update_priority(
        self,
        ctx: OnStepContext,
        info: dict[str, torch.Tensor],
        replay_buffer: ReplayBufferProto,
    ) -> None:
        """Update replay buffer priority.

        Args:
            ctx: Training ctx with batch and TD-error
            info: Metadata from replay buffer sample containing indices
            replay_buffer: The replay buffer to update
        """
        # These checks should not be needed since caller validates, but kept for safety
        if ctx.td_error is None or info is None:
            raise ValueError("Skipping priority update: missing td_error or info")

        td_error = ctx.td_error

        # Check if info has index metadata (required for update_tensordict_priority)
        if "index" not in info.keys():
            raise ValueError(
                "Info does not have index metadata, cannot update priorities. "
                "Available keys: %s",
                list(info.keys()),
            )

        if td_error.get(RewardKeys.PIXEL_WISE_N_REWARD.value) is not None:
            replay_buffer.update_priority(
                info["index"], td_error.get(RewardKeys.PIXEL_WISE_N_REWARD.value)
            )
        else:
            replay_buffer.update_priority(
                info["index"], td_error.get(RewardKeys.PIXEL_WISE.value)
            )

    def _on_validation_callbacks(
        self,
    ) -> OrderedDict[str, Callable[[OnStepContext], Any]]:
        """Build validation step callbacks.

        Returns:
            Dictionary of callbacks for validation steps
        """
        if self.exp_config.checkpoint.improvement_mode == "validation":
            logger.info("Improvement checkpoint saving enabled for validation")
            return OrderedDict(
                [
                    ("improvement_ckpt", self._improvement_checkpoint),
                ]
            )
        return OrderedDict()

    def _on_train_callbacks(
        self,
    ) -> OrderedDict[str, Callable[[OnStepContext], Any]]:
        """Build training step callbacks.

        Returns:
            Dictionary of callbacks for training steps
        """
        # Track best (lower is better) actor/critic losses for improvement checkpoints
        self.best_loss_actor: torch.Tensor = torch.tensor(float("inf"))
        self.best_loss_critic: torch.Tensor = torch.tensor(float("inf"))

        callbacks: OrderedDict[str, Callable[[OnStepContext], Any]] = OrderedDict(
            {"core_step": self.d4pg_core.on_step, "logging": self._log_step}
        )

        if self.ckpt_save_frequency > 0:
            logger.info(
                f"Periodic checkpoint enabled every {self.ckpt_save_frequency} steps"
            )
            callbacks["periodic_ckpt"] = self._periodic_checkpoint

        if self.exp_config.checkpoint.improvement_mode == "train":
            logger.info("Improvement checkpoint saving enabled for training")
            callbacks["improvement_ckpt"] = self._improvement_checkpoint

        # Add world model callback if enabled
        if isinstance(self.world_model_stepper, ArcWorldModelStepper):
            callbacks["world_model"] = self.world_model_stepper.training_callback

        return callbacks

    def _log_step(self, ctx: OnStepContext):
        """Log training step information.

        Args:
            ctx: Step ctx containing training information
        """
        if ctx.iteration % self.exp_config.logging.log_frequency == 0:
            msg = f"Epoch {ctx.epoch}, Step {ctx.step_idx}, " + (
                f"Buffer Warmup: {ctx.buffer_warmup:.1f}%, "
                if ctx.buffer_warmup is not None
                else ""
            )
            if ctx.warmup:
                msg += " (Warmup), "
            if ctx.loss_actor is not None:
                msg += (
                    f"Iteration {ctx.iteration}: "
                    f"Actor Loss: {ctx.loss_actor:.4f}, "
                    f"Critic Loss: {ctx.loss_critic:.4f}"
                )
                # Add world model metrics if enabled
                if isinstance(self.world_model_stepper, ArcWorldModelStepper):
                    wm_losses = self.world_model_stepper.get_average_model_losses()
                    if wm_losses:
                        wm_str = ", ".join(
                            f"{k}: {v:.4f}" for k, v in wm_losses.items()
                        )
                        msg += f", RWM Loss: [{wm_str}]"
                    msg += (
                        f", Imaginations: {self.world_model_stepper.imagination_count}"
                    )
            logger.info(msg)

    def _periodic_checkpoint(self, ctx: OnStepContext):
        """Periodic lightweight checkpoint: core + trainer + config + provenance.

        This runs frequently (every self.save_frequency steps) and avoids
        saving large components such as environment, stepper and replay buffer.

        Args:
            ctx: Step ctx containing training information
        """
        if (
            ctx.warmup
            or ctx.iteration <= 0
            or ctx.iteration % self.ckpt_save_frequency != 0
        ):
            return
        fname = os.path.join(
            self.checkpoint_dir,
            "checkpoints/periodic/d4pg_cp-per_i_{:06d}.pth".format(ctx.iteration),
        )
        # Save minimal components for fast, small checkpoint files
        self.save_checkpoint(
            fname,
            include_core=True,
            include_trainer=True,
            include_env=False,
            include_stepper=False,
            include_rng=True,
            include_config=True,
            include_provenance=True,
            save_replay_buffer=False,
        )

    def _improvement_checkpoint(self, ctx: OnStepContext):
        """Save core-only checkpoint when losses improve.

        Lower losses are better. We checkpoint the core networks and optimizers
        only, to capture the best model so far with minimal size.

        Args:
            ctx: Step ctx containing training information
        """
        if ctx.loss_actor is None or ctx.loss_critic is None:
            return
        improved = False
        if ctx.loss_actor.item() < self.best_loss_actor:
            self.best_loss_actor = ctx.loss_actor.item()
            improved = True
        if ctx.loss_critic.item() < self.best_loss_critic:
            self.best_loss_critic = ctx.loss_critic.item()
            improved = True
        if not improved:
            return
        if self.exp_config.checkpoint.improvement_mode is None:
            raise ValueError("Improvement mode not configured")
        fname = os.path.join(
            self.checkpoint_dir,
            "checkpoints/best/{}/d4pg_cp-best_i_{:06d}-la_{:.4f}-lc_{:.4f}.pth".format(
                self.exp_config.checkpoint.improvement_mode,
                int(ctx.iteration),
                ctx.loss_actor.item(),
                ctx.loss_critic.item(),
            ),
        )
        # Core-only save for best improvements
        try:
            self.d4pg_core.save_checkpoint(path=fname)
            logger.info("Saved improvement core checkpoint: %s", fname)
        except Exception as e:
            logger.warning("Failed to save improvement checkpoint %s: %s", fname, e)

    def train(self, epochs: int = 10) -> None:
        """Run the complete training loop.

        Args:
            epochs: Number of epochs to train
        """
        logger.info(f"Starting D4PG training for {epochs} epochs")
        failed = False
        try:
            assert self.trainer is not None
            self.trainer.train(
                epochs=epochs,
                max_steps_per_epoch=self.max_steps_per_epoch,
                validation_epochs=self.validation_epochs,
                validation_steps_per_epoch=self.validation_steps_per_epoch,
                validation_steps_frequency=self.validation_steps_frequency,
            )
            logger.info("Training completed successfully")
            if hasattr(self, "tb_manager") and self.tb_manager is not None:
                self.tb_manager.log_hparams(self._get_hparams())
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            failed = True
            raise
        finally:
            _buffer_warmup, warmup = self.trainer.warmup_info()
            if warmup:
                logger.warning("Skipping final checkpoint during warmup period")
                return
            logger.info("Saving final checkpoint...")
            self.save_checkpoint(
                os.path.join(
                    self.checkpoint_dir,
                    "checkpoints/{}/d4pg_cp-failed-{}.pth".format(
                        "failure" if failed else "success",
                        datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"),
                    ),
                ),
                save_replay_buffer=failed,
            )

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        save_replay_buffer: bool = False,
        include_core: bool = True,
        include_env: bool = True,
        include_trainer: bool = True,
        include_stepper: bool = True,
        include_world_model: bool = True,
        include_rng: bool = True,
        include_config: bool = True,
        include_provenance: bool = True,
    ) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint
            save_replay_buffer: Whether to save replay buffer
            include_core: Whether to include core checkpoint
            include_env: Whether to include environment checkpoint
            include_trainer: Whether to include trainer checkpoint
            include_stepper: Whether to include stepper checkpoint
            include_world_model: Whether to include world model checkpoint
            include_rng: Whether to include RNG checkpoint
            include_config: Whether to include config checkpoint
            include_provenance: Whether to include provenance checkpoint
        """
        # Resolve target path (file or directory semantics)
        checkpoint_path = resolve_save_path(path, self.__class__.__name__)
        if not (
            hasattr(self, "trainer")
            and hasattr(self, "training_stepper")
            and hasattr(self.training_stepper, "environment")
            and hasattr(self.training_stepper, "core")
        ):
            raise ValueError(
                "Experiment must have environment, trainer, and stepper to checkpoint"
            )

        # Core, env, trainer, stepper, world_model (conditionally included)
        core_ckpt = self.d4pg_core.checkpoint() if include_core else None
        env_ckpt = (
            self.training_stepper.environment.checkpoint() if include_env else None
        )
        trainer_ckpt = self.trainer.checkpoint() if include_trainer else None
        stepper_ckpt = (
            self.training_stepper.core.checkpoint() if include_stepper else None
        )
        world_model_ckpt = (
            self.world_model_stepper.checkpoint()
            if include_world_model and self.world_model_stepper is not None
            else None
        )

        # RNG states
        rng_state = (
            {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.random.get_rng_state(),
                "torch_cuda": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                ),
            }
            if include_rng
            else None
        )

        # Config snapshot and provenance
        config = (
            {
                "grid_size": self.grid_size,
                "color_values": self.color_values,
                "batch_size": self.batch_size,
                "n_steps": getattr(self, "n_steps", None),
                "gamma": getattr(self, "gamma", None),
                "tau": getattr(self, "tau", None),
                "policy_lr": getattr(self, "policy_lr", None),
                "critic_lr": getattr(self, "critic_lr", None),
                "device": self.device,
                "amp_enabled": getattr(self, "amp_enabled", True),
                "seed": getattr(self, "seed", None),
                "dataset_path": getattr(self, "dataset_path", None),
                "use_world_model": self.use_world_model,
            }
            if include_config
            else None
        )

        provenance = (
            {
                "created_at": datetime.now(timezone.utc).isoformat() + "Z",
                "schema": 1,
                "env": {
                    "torch_version": torch.__version__,
                    "numpy_version": np.__version__,
                    "python_version": sys.version,
                },
            }
            if include_provenance
            else None
        )

        # Optional replay buffer metadata (buffer saved separately for size reasons)
        rb_meta: dict[str, Any] | None = None
        if (
            save_replay_buffer
            and hasattr(self.trainer, "replay_buffer")
            and self.trainer.replay_buffer is not None
        ):
            try:
                ckpt_dir = Path(checkpoint_path).parent
                rb_file = ckpt_dir / "replay_buffer.ptc"
                # Persist underlying buffer via its API if available
                if hasattr(self.trainer.replay_buffer, "save"):
                    self.trainer.replay_buffer.save(str(rb_file))
                    meta: dict[str, Any] = {"file": str(rb_file.name)}
                elif hasattr(self.trainer.replay_buffer, "state_dict"):
                    torch.save(self.trainer.replay_buffer.state_dict(), rb_file)
                    meta = {"file": str(rb_file.name)}
                else:
                    logger.warning(
                        "Replay buffer does not have a save or state_dict method"
                    )
                # Attach sampler metadata if we saved a buffer file
                if "meta" in locals():
                    sampler = getattr(self.trainer.replay_buffer, "sampler", None)
                    if sampler is not None:
                        meta["sampler"] = {
                            "alpha": getattr(sampler, "alpha", None),
                            "beta": getattr(sampler, "beta", None),
                        }
                    rb_meta = meta
            except Exception as e:
                logger.warning(f"Failed to save replay buffer: {e}")
                rb_meta = None

        payload = {
            "core": core_ckpt,
            "env": env_ckpt,
            "trainer": trainer_ckpt,
            "stepper": stepper_ckpt,
            "world_model": world_model_ckpt,
            "rng": rng_state,
            "config": config,
            "provenance": provenance,
            "replay_buffer": rb_meta,
        }
        checkpoint = add_schema_and_class(payload, self.__class__.__name__)

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        load_replay_buffer: bool = False,
        weights_only: bool = False,
    ) -> None:
        """Load model checkpoint.

        Args:
            path: Path to load checkpoint from
            load_replay_buffer: Whether to load replay buffer
            weights_only: Whether to load weights only
        """
        cp_path = resolve_load_path(path, self.__class__.__name__)
        # Allowlist project-specific classes for safe unpickling with weights_only
        try:
            safe_list = [TorchQueue, _make_td, TensorDict]
            if _np_reconstruct is not None:
                safe_list.append(_np_reconstruct)
            add_safe_globals(safe_list)
        except Exception:
            pass
        checkpoint = torch.load(
            cp_path, map_location=self.device, weights_only=weights_only
        )
        restored_core = False
        restored_env = False
        restored_trainer = False
        restored_stepper = False
        restored_world_model = False
        restored_rng = False
        restored_replay = False

        # Restore core
        core_state = checkpoint.get("core")
        if isinstance(core_state, dict):
            self.d4pg_core.load(core_state, device=self.device)
            restored_core = True

        # Restore environment/wrapper
        env_state = checkpoint.get("env")
        if env_state is not None and hasattr(self, "training_stepper"):
            self.training_stepper.environment.load(env_state, device=self.device)
            restored_env = True

        # Restore trainer progress
        trainer_state = checkpoint.get("trainer")
        if trainer_state is not None and hasattr(self, "trainer"):
            self.trainer.load(trainer_state)
            restored_trainer = True

        # Restore stepper progress (does not restore sampler)
        stepper_state = checkpoint.get("stepper")
        if stepper_state is not None and hasattr(self, "training_stepper"):
            self.training_stepper.core.load(stepper_state)
            restored_stepper = True

        # Restore world model stepper
        world_model_state = checkpoint.get("world_model")
        if world_model_state is not None and self.world_model_stepper is not None:
            self.world_model_stepper.load(world_model_state)
            restored_world_model = True

        # Restore RNGs
        rng = checkpoint.get("rng")
        if not weights_only and isinstance(rng, dict):
            try:
                if rng.get("python") is not None:
                    random.setstate(rng["python"])
                if rng.get("numpy") is not None:
                    np.random.set_state(rng["numpy"])
                if rng.get("torch_cpu") is not None:
                    cpu_state = rng["torch_cpu"]
                    if not isinstance(cpu_state, torch.ByteTensor):
                        cpu_state = torch.as_tensor(cpu_state, dtype=torch.uint8).cpu()
                    torch.random.set_rng_state(cpu_state)
                if rng.get("torch_cuda") is not None and torch.cuda.is_available():
                    cuda_states = rng["torch_cuda"]
                    if isinstance(cuda_states, (list, tuple)):
                        coerced = []
                        for s in cuda_states:
                            if not isinstance(s, torch.ByteTensor):
                                s = torch.as_tensor(s, dtype=torch.uint8).cpu()
                            coerced.append(s)
                        torch.cuda.set_rng_state_all(coerced)
                    else:
                        # Single-device state fallback
                        s = cuda_states
                        if not isinstance(s, torch.ByteTensor):
                            s = torch.as_tensor(s, dtype=torch.uint8).cpu()
                        torch.cuda.set_rng_state_all([s])
            except Exception as e:
                logger.warning(f"Failed to restore RNG states: {e}")
            else:
                restored_rng = True

        # Restore critic LR scheduler state if present in core checkpoint
        if isinstance(core_state, dict):
            try:
                sched_state = core_state.get("critic_scheduler")
                if (
                    sched_state is not None
                    and self.d4pg_core.critic_scheduler is not None
                ):
                    self.d4pg_core.critic_scheduler.load_state_dict(sched_state)
            except Exception as e:
                logger.warning(f"Failed to restore critic scheduler: {e}")

        # Restore replay buffer if requested and available
        if (
            load_replay_buffer
            and hasattr(self.trainer, "replay_buffer")
            and self.trainer.replay_buffer is not None
        ):
            rb_meta = checkpoint.get("replay_buffer")
            if rb_meta and isinstance(rb_meta, dict):
                rb_file = rb_meta.get("file")
                if rb_file:
                    rb_path = Path(cp_path).parent / rb_file
                    try:
                        if hasattr(self.trainer.replay_buffer, "load"):
                            self.trainer.replay_buffer.load(str(rb_path))
                        elif hasattr(self.trainer.replay_buffer, "load_state_dict"):
                            state = torch.load(rb_path, map_location=self.device)
                            self.trainer.replay_buffer.load_state_dict(state)
                        sampler = getattr(self.trainer.replay_buffer, "sampler", None)
                        if sampler is not None and isinstance(
                            rb_meta.get("sampler"), dict
                        ):
                            s = rb_meta["sampler"]
                            if s.get("alpha") is not None:
                                sampler.alpha = s["alpha"]
                            if s.get("beta") is not None:
                                sampler.beta = s["beta"]
                        restored_replay = True
                    except Exception as e:
                        logger.warning(f"Failed to restore replay buffer: {e}")

        logger.info(
            "Loaded checkpoint components — core=%s, env=%s, trainer=%s, "
            "stepper=%s, world_model=%s, rng=%s, replay=%s from %s",
            restored_core,
            restored_env,
            restored_trainer,
            restored_stepper,
            restored_world_model,
            restored_rng,
            restored_replay,
            path,
        )
