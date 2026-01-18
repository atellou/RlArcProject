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
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.distributed as dist
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
from rlarcworld.common.configs import (
    ArcD4PGConfig,
    ReproducibilityConfig,
    RlArcWorldViTARCConfig,
)
from rlarcworld.common.enums import RewardKeys
from rlarcworld.environments.steppers.arc_stepper import PixelWiseArcStepper
from rlarcworld.environments.steppers.reward_world_model import ArcWorldModelStepper
from rlarcworld.logging.tensorboard import make_tensorboard_callbacks
from rlarcworld.models.actor import ViTARCActor
from rlarcworld.models.critic import VitARCCritic
from rlarcworld.models.nn_modules import RlArcWorldViTARC
from rlarcworld.models.reward_world_model import RewardWorldModel
from rlarcworld.trainers.core import OnStepContext, ReplayBufferProto, Trainer
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

        self.global_seed: int | None = None
        if config.reproducibility.global_seed is not None:
            self._setup_reproducibility(config=config.reproducibility)

        # Setup in logical order
        self._setup_system(config)
        self._setup_networks(config)
        self._setup_steppers(config)
        self._setup_core(config, **kwargs)
        self._setup_trainer(
            replay_buffer_size=config.replay_buffer.size,
            alpha=config.replay_buffer.alpha,
            beta=config.replay_buffer.beta,
            warmup_buffer_ratio=config.replay_buffer.warmup_ratio,
        )

    def _dist_rank(self) -> int:
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
        return 0

    def _is_rank0(self) -> bool:
        return self._dist_rank() == 0

    def _setup_reproducibility(self, config: ReproducibilityConfig) -> None:
        """Reproducibility: set global seeds and deterministic flags."""
        if not isinstance(config.global_seed, int):
            raise ValueError("Global seed must be an integer for reproducibility")
        self.global_seed = config.global_seed

        msg = "Setting: Global seed: {}".format(self.global_seed)

        # Optional deterministic mode
        # NOTE: CuBLAS determinism requires this env var to be set *before* any
        # CUDA context is initialized.
        if config.torch_deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

        # Optional deterministic mode
        # NOTE: CuBLAS determinism requires this env var to be set *before* any
        # CUDA context is initialized.
        if config.torch_deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

        # Python / NumPy / Torch seeds
        random.seed(self.global_seed)
        np.random.seed(self.global_seed)
        torch.manual_seed(self.global_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.global_seed)

        if config.torch_deterministic:
            msg += ", torch_deterministic: True"
            torch.use_deterministic_algorithms(True, warn_only=False)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # TF32 can introduce nondeterminism; disable in deterministic mode.
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        logger.info(msg)

    def _setup_system(self, config: ArcD4PGConfig) -> None:
        """Configure system settings: seed, device, and paths."""
        # Configure device
        if config.system.device is None:
            local_rank_env = os.environ.get("LOCAL_RANK")
            device_index = int(local_rank_env) if local_rank_env is not None else None
            self.cuda_config = enable_cuda(
                amp_enabled=config.system.amp_enabled,
                use_checkpointing=config.system.use_checkpointing,
                device_index=device_index,
            )
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
        self.max_steps_per_episode = config.training.max_steps_per_episode
        self.validation_steps_per_epoch = config.training.validation_steps_per_epoch
        self.validation_steps_per_episode = config.training.validation_steps_per_episode
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

    def _setup_networks(self, config: ArcD4PGConfig) -> None:
        """Initialize actor and critic networks.

        Args:
            config: Experiment configuration
        """
        logger.info("Initializing actor and critic networks...")

        # Use config's rl_vit_arc_config or create one with best settings
        self.rl_vit_arc_config = config.network or RlArcWorldViTARCConfig.best_config(
            config.env.grid_size
        )

        if config.d4pg.share_network:
            self.rl_vit_arc = RlArcWorldViTARC(
                self.rl_vit_arc_config,
                verbose=True,
                checkpoint=self.cuda_config.get("use_checkpointing", False)
                if self.cuda_config
                else False,
            )

        logger.info("Creating actor network...")
        self.actor = ViTARCActor(
            color_values=config.env.color_values,
            rl_vit_arc=self.rl_vit_arc
            if config.d4pg.share_network
            else self.rl_vit_arc_config,
            device_config=self.cuda_config,
        )
        logger.info("Actor network created successfully")

        # Ensure distributional config includes n-step reward keys when required
        if config.d4pg.n_step_config.n_steps > 1:
            step_keys = RewardKeys.get_n_step_keys()
            for n_key in step_keys:
                if n_key.value not in config.distributional.num_atoms:
                    raise ValueError(
                        f"Distributional config missing num_atoms for {n_key.value}"
                    )

        logger.info("Creating critic network...")
        self.critic = VitARCCritic(
            color_values=config.env.color_values,
            rl_vit_arc=self.rl_vit_arc_config
            if not config.d4pg.share_network
            else self.rl_vit_arc,
            num_atoms=config.distributional.num_atoms,
            v_min=config.distributional.v_min,
            v_max=config.distributional.v_max,
            device_config=self.cuda_config,
        )
        logger.info("Critic network created successfully")

    def _setup_steppers(self, config: ArcD4PGConfig) -> None:
        """Create training, validation, and world model steppers.

        Args:
            config: Experiment configuration
        """
        # Prepare reproducibility generators / seeds
        seed_cfg = config.reproducibility
        ds_shuffle_seed = seed_cfg.dataset_seed or self.global_seed
        aug_gen = None
        if seed_cfg.augmentation_seed is not None:
            aug_gen = torch.Generator(device="cpu").manual_seed(
                seed_cfg.augmentation_seed
            )
        elif self.global_seed is not None:
            aug_gen = torch.Generator(device="cpu").manual_seed(self.global_seed)

        # Create training stepper
        logger.info("Creating training stepper...")
        self.training_stepper = PixelWiseArcStepper(
            max_steps_per_episode=self.max_steps_per_episode,
            dataset=config.dataset,
            grid_size=config.env.grid_size,
            color_values=config.env.color_values,
            actor=self.actor,
            device_config=self.cuda_config,
            batch_size=self.batch_size,
            augmentation_enabled=config.training.augmentation_enabled,
            max_examples=10,
            dataset_shuffle_seed=ds_shuffle_seed,
            augmentation_generator=aug_gen,
            max_penalty=config.env.max_penalty,
            n_step_config=config.d4pg.n_step_config,
        )
        logger.info("Training stepper created successfully")

        # Create validation stepper (use same seeds)
        logger.info("Creating validation stepper...")
        self.validation_stepper = PixelWiseArcStepper(
            max_steps_per_episode=self.validation_steps_per_episode,
            dataset=config.dataset,
            grid_size=config.env.grid_size,
            color_values=config.env.color_values,
            actor=self.actor,
            device_config=self.cuda_config,
            batch_size=self.batch_size,
            augmentation_enabled=False,
            max_examples=10,
            dataset_shuffle_seed=ds_shuffle_seed,
            augmentation_generator=None,
            max_penalty=config.env.max_penalty,
            n_step_config=config.d4pg.n_step_config,
        )
        logger.info("Validation stepper created successfully")

        # Create world model stepper (optional Dyna-style imagination)
        self.world_model_stepper: Optional[ArcWorldModelStepper] = None
        if config.world_model.enabled:
            logger.info("Creating world model stepper...")
            reward_model = RewardWorldModel(
                color_values=config.env.color_values,
                rl_vit_arc_config=self.rl_vit_arc_config,
                device_config=self.cuda_config,
            )
            self.world_model_stepper = ArcWorldModelStepper(
                grid_size=config.env.grid_size,
                color_values=config.env.color_values,
                actor=self.actor,
                device_config=self.cuda_config,
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

    def _setup_core(self, config: ArcD4PGConfig, **kwargs) -> None:
        """Initialize the D4PG algorithm core.

        Args:
            config: Experiment configuration
            **kwargs: Additional arguments passed to D4PGCore
        """
        logger.info("Initializing D4PG algorithm core...")
        self.d4pg_core = D4PGCore(
            actor=self.actor,
            critic=self.critic,
            policy_lr=config.d4pg.policy_lr,
            critic_lr=config.d4pg.critic_lr,
            shared_params=self.rl_vit_arc.parameters()
            if config.d4pg.share_network
            else None,
            gamma=config.d4pg.gamma,
            tau=config.d4pg.tau,
            lr_scheduler_kwargs=config.d4pg.lr_scheduler_kwargs,
            n_steps=config.d4pg.n_step_config.n_steps,
            device_config=self.cuda_config,
            amp_enabled=config.system.amp_enabled,
            carsm=config.d4pg.carsm,
            entropy_coef=config.d4pg.entropy_coef,
            entropy_coef_decay=config.d4pg.entropy_coef_decay,
            target_update_frequency=config.d4pg.target_update_frequency,
            model_update_frequency=config.d4pg.model_update_frequency,
            cache_enabled=config.d4pg.cache_enabled,
            max_grad_norm=config.d4pg.max_grad_norm,
            compute_td_error=config.d4pg.compute_td_error,
            **kwargs,
        )

    def _get_hparams(self) -> dict:
        """Get hyperparameters from stored config for logging/checkpointing.

        Returns:
            Dictionary of hyperparameters sourced from exp_config.
        """
        cfg = self.exp_config
        d4pg = cfg.d4pg
        net = cfg.network
        wm = cfg.world_model

        hparams = {
            # Environment
            "grid_size": cfg.env.grid_size,
            "color_values": cfg.env.color_values,
            "max_penalty": cfg.env.max_penalty,
            "batch_size": cfg.env.batch_size,
            # Training
            "max_steps_per_epoch": cfg.training.max_steps_per_epoch,
            "validation_steps_per_epoch": cfg.training.validation_steps_per_epoch,
            # D4PG
            "policy_lr": d4pg.policy_lr,
            "critic_lr": d4pg.critic_lr,
            "gamma": d4pg.gamma,
            "tau": d4pg.tau,
            "n_steps": d4pg.n_step_config.n_steps,
            "entropy_coef": d4pg.entropy_coef,
            "entropy_coef_decay": d4pg.entropy_coef_decay,
            "carsm": d4pg.carsm,
            "target_update_frequency": d4pg.target_update_frequency,
            "max_grad_norm": d4pg.max_grad_norm,
            # Distributional
            "num_atoms": cfg.distributional.num_atoms,
            "v_min_pixel_wise": cfg.distributional.v_min.get("pixel_wise"),
            "v_max_pixel_wise": cfg.distributional.v_max.get("pixel_wise"),
            # Network
            "vitarc_d_model": net.d_model,
            "vitarc_d_ff": net.d_ff,
            "vitarc_num_layers": net.num_layers,
            "vitarc_num_heads": net.num_heads,
            "vitarc_use_ope": net.use_ope,
            "vitarc_use_index_grid": net.use_index_grid,
            "vitarc_use_temporal": net.use_temporal,
            "vitarc_use_examples": net.use_examples,
            "vitarc_ape_type": net.ape_type,
            "vitarc_rpe_type": net.rpe_type,
            # World model
            "world_model_enabled": wm.enabled,
            "world_model_lr": wm.learning_rate if wm.enabled else None,
            "imagination_horizon": wm.imagination_horizon if wm.enabled else None,
            "imagination_ratio": wm.imagination_ratio if wm.enabled else None,
            # System
            "device": str(self.device),
            "amp_enabled": cfg.system.amp_enabled,
            "seed": cfg.reproducibility.global_seed,
        }

        # Add runtime world model stats if available
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
            if not self._is_rank0():
                logger.info("Non-rank0 process: skipping TensorBoard setup")
                return
            if not self.exp_config.logging.tensorboard_enabled:
                logger.info("TensorBoard disabled in config, skipping setup")
                return
            self.tb_manager, tb_train_cb, tb_val_cb = make_tensorboard_callbacks(
                self.checkpoint_dir,
                replay_buffer=self.replay_buffer,
                warmup_buffer_ratio=warmup_buffer_ratio,
                minimal_frequency=self.exp_config.logging.tb_minimal_frequency,
                medium_frequency=self.exp_config.logging.tb_medium_frequency,
            )
            self.d4pg_core.set_tb_writer(
                self.tb_manager.writer, self.tb_manager.minimal_frequency
            )
            # Extend training callbacks
            self.trainer.on_train_step = self.trainer.on_train_step or OrderedDict()
            self.trainer.on_train_step["tensorboard"] = tb_train_cb
            # Set validation callbacks
            self.trainer.on_validation_step = (
                self.trainer.on_validation_step or OrderedDict()
            )
            self.trainer.on_validation_step["tensorboard"] = tb_val_cb
            logger.info("TensorBoard attached!!!")
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
        callbacks: OrderedDict[str, Callable[[OnStepContext], Any]] = OrderedDict()

        self._validation_loss_actor_accumulator: list[float] = []
        self._validation_loss_critic_accumulator: list[float] = []
        self._validation_rewards_accumulator: dict[str, list[float]] = {}

        # Keep basic progress visibility during validation (which may be slow)
        self._validation_run_start_t: float | None = None
        callbacks["val_progress_log"] = self._validation_progress_log
        callbacks["val_end_log"] = self._validation_end_log

        if self.exp_config.checkpoint.improvement_checkpoint_enabled:
            logger.info("Validation improvement checkpoint enabled")
            # Initialize tracking for validation-based improvement
            self._best_validation_reward: float = float("-inf")
            self._validation_reward_accumulator: list[float] = []
            callbacks["improvement_ckpt"] = self._validation_improvement_checkpoint

        return callbacks

    def _validation_progress_log(self, ctx: OnStepContext) -> None:
        if ctx.run_end:
            self._validation_run_start_t = None
            return
        if not self._is_rank0():
            return

        # Initialize timer on first step
        if ctx.epoch_step_idx == 0 or self._validation_run_start_t is None:
            self._validation_run_start_t = time.monotonic()

        # Log every 10 validation steps by default (cheap visibility)
        if ctx.epoch_step_idx % 10 != 0:
            return

        elapsed_s = time.monotonic() - self._validation_run_start_t
        # We usually run validation for a single epoch; estimate remaining steps
        remaining = max(
            int(self.validation_steps_per_epoch) - int(ctx.epoch_step_idx) - 1, 0
        )
        per_step = elapsed_s / max(int(ctx.epoch_step_idx) + 1, 1)
        eta_s = remaining * per_step

        logger.info(
            "Validation progress: step %d/%s (elapsed %.1fs, eta %.1fs)",
            int(ctx.epoch_step_idx),
            str(self.validation_steps_per_epoch),
            float(elapsed_s),
            float(eta_s),
        )

    def _validation_end_log(self, ctx: OnStepContext) -> None:
        if ctx.loss_actor is not None:
            try:
                self._validation_loss_actor_accumulator.append(
                    float(ctx.loss_actor.item())
                )
            except Exception:
                pass
        if ctx.loss_critic is not None:
            try:
                self._validation_loss_critic_accumulator.append(
                    float(ctx.loss_critic.item())
                )
            except Exception:
                pass

        if ctx.batch is not None and ctx.batch.get("reward") is not None:
            for key in [rk.value for rk in RewardKeys]:
                reward_tensor = ctx.batch["reward"].get(key)
                if reward_tensor is None:
                    continue
                try:
                    reward_val = float(reward_tensor.mean().item())
                except Exception:
                    continue
                if key not in self._validation_rewards_accumulator:
                    self._validation_rewards_accumulator[key] = []
                self._validation_rewards_accumulator[key].append(reward_val)

        if not ctx.run_end:
            return
        if not self._is_rank0():
            self._validation_loss_actor_accumulator.clear()
            self._validation_loss_critic_accumulator.clear()
            self._validation_rewards_accumulator.clear()
            return

        mean_loss_actor = (
            sum(self._validation_loss_actor_accumulator)
            / len(self._validation_loss_actor_accumulator)
            if self._validation_loss_actor_accumulator
            else None
        )
        mean_loss_critic = (
            sum(self._validation_loss_critic_accumulator)
            / len(self._validation_loss_critic_accumulator)
            if self._validation_loss_critic_accumulator
            else None
        )
        rewards_str = ""
        if self._validation_rewards_accumulator:
            rewards_str = ", ".join(
                f"{k}: {sum(v) / len(v):.6f}"
                for k, v in self._validation_rewards_accumulator.items()
                if v
            )

        msg = (
            f"Validation summary @ train_iter={self.trainer.train_iteration}: "
            + (
                f"loss_actor={mean_loss_actor:.7f}, "
                if mean_loss_actor is not None
                else ""
            )
            + (
                f"loss_critic={mean_loss_critic:.7f}"
                if mean_loss_critic is not None
                else ""
            )
        )
        if rewards_str:
            msg += f", rewards=[{rewards_str}]"
        logger.info(msg)

        self._validation_loss_actor_accumulator.clear()
        self._validation_loss_critic_accumulator.clear()
        self._validation_rewards_accumulator.clear()

    def _on_train_callbacks(
        self,
    ) -> OrderedDict[str, Callable[[OnStepContext], Any]]:
        """Build training step callbacks.

        Returns:
            Dictionary of callbacks for training steps
        """
        callbacks: OrderedDict[str, Callable[[OnStepContext], Any]] = OrderedDict(
            {"core_step": self.d4pg_core.on_step, "logging": self._log_step}
        )

        if self.ckpt_save_frequency > 0:
            logger.info(
                f"Periodic checkpoint enabled every {self.ckpt_save_frequency} steps"
            )
            callbacks["periodic_ckpt"] = self._periodic_checkpoint

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
            msg = (
                f"Epoch {ctx.epoch}, "
                + f"Step [Episode/Epoch] {ctx.episode_step_idx}/{ctx.epoch_step_idx}"
            )
            if ctx.warmup:
                msg += (
                    f", Buffer Warmup: {ctx.buffer_warmup:.1f}%"
                    if ctx.buffer_warmup is not None
                    else ""
                )
            if ctx.loss_actor is not None:
                msg += (
                    f". Iteration {ctx.iteration}: "
                    f"Actor Loss: {ctx.loss_actor:.7f}, "
                    f"Critic Loss: {ctx.loss_critic:.7f}"
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
                batch: TensorDict = ctx.batch
                reward_td: TensorDict = batch["reward"]
                items_iter = reward_td.items() if hasattr(reward_td, "items") else []
                reward_pairs: list[str] = []
                for key, val in items_iter:
                    try:
                        mean_val = float(val.float().mean().detach().cpu().item())
                    except Exception:
                        mean_val = float(val)
                    reward_pairs.append(f"{key}={mean_val:.4f}")
                if reward_pairs:
                    reward_pairs.sort()
                    msg += ", Rewards: [" + ",".join(reward_pairs) + "]"
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

    def _validation_improvement_checkpoint(self, ctx: OnStepContext):
        """Save checkpoint when validation reward improves.

        Accumulates rewards during validation run and saves checkpoint at run_end
        if mean reward exceeds previous best. Uses actual policy performance
        rather than loss values.

        Args:
            ctx: Step ctx containing validation information
        """
        # Accumulate rewards during validation
        if ctx.batch is not None and ctx.batch.get("reward") is not None:
            # Extract reward from batch - prefer n-step reward if available
            reward_td: dict = ctx.batch["reward"]
            if RewardKeys.PIXEL_WISE_N_REWARD.value in reward_td.keys():
                reward = float(
                    reward_td[RewardKeys.PIXEL_WISE_N_REWARD.value].mean().item()
                )
            elif RewardKeys.PIXEL_WISE.value in reward_td.keys():
                reward = float(reward_td[RewardKeys.PIXEL_WISE.value].mean().item())
            else:
                logger.warning(
                    "No expected reward keys found in validation batch. "
                    "Expected rewards are {}. Got {}.".format(
                        ["pixel_wise_n_reward", "pixel_wise"], ctx.batch["reward"]
                    )
                )
                return
            self._validation_reward_accumulator.append(reward)

        # Only checkpoint at end of validation run
        if not ctx.run_end:
            return

        if not self._validation_reward_accumulator:
            logger.info(
                "No validation rewards accumulated; skipping improvement checkpoint. "
                "(Expected rewards under ctx.batch['reward'].)"
            )
            return

        mean_reward = sum(self._validation_reward_accumulator) / len(
            self._validation_reward_accumulator
        )
        self._validation_reward_accumulator.clear()  # Reset for next validation

        if mean_reward <= self._best_validation_reward:
            logger.debug(
                f"Validation reward {mean_reward:.4f} did not improve "
                f"over best {self._best_validation_reward:.4f}"
            )
            return

        self._best_validation_reward = mean_reward
        fname = os.path.join(
            self.checkpoint_dir,
            f"checkpoints/best/validation/d4pg_cp-best_e_{self.trainer.train_epoch:04d}_i_{self.trainer.train_iteration:06d}"
            f"-reward_{mean_reward:.4f}.pth",
        )
        try:
            self.d4pg_core.save_checkpoint(path=fname)
            logger.info(
                f"Saved validation improvement checkpoint: {fname} "
                f"(reward: {mean_reward:.4f})"
            )
        except Exception as e:
            logger.warning("Failed to save improvement checkpoint %s: %s", fname, e)

    def train(self, epochs: int = 10) -> None:
        """Run the complete training loop.

        This method is cloud-friendly: the `epochs` parameter represents the
        **total desired epochs**, not additional epochs to run. When resuming
        from a checkpoint, training continues from the saved epoch until the
        total epoch count is reached.

        Example:
            - Fresh start with epochs=100: runs epochs 0-99
            - Resume at epoch 50 with epochs=100: runs epochs 50-99
            - Resume at epoch 50 with epochs=50: exits immediately (already done)

        Args:
            epochs: Total number of epochs to complete. If resuming from a
                checkpoint where current_epoch >= epochs, training exits early.
        """
        logger.info(f"Starting D4PG training for {epochs} epochs")

        # Ensure networks are in train mode
        self.actor.train()
        self.critic.train()

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
            self.save_final_checkpoint(failed=failed)

    def save_final_checkpoint(self, *, failed: bool, reason: str | None = None) -> None:
        """Save final checkpoint."""
        if hasattr(self, "tb_manager") and self.tb_manager is not None:
            try:
                logger.info("Flushing TensorBoard writer...")
                self.tb_manager.close()
            except Exception as e:
                logger.warning(f"Failed to close TensorBoard writer: {e}")

        if not hasattr(self, "trainer") or self.trainer is None:
            logger.warning("Skipping final checkpoint: trainer not initialized")
            return

        _buffer_warmup, warmup = self.trainer.warmup_info()
        if warmup:
            logger.warning("Skipping final checkpoint during warmup period")
            return

        now_str = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        if reason == "spot":
            rel = f"checkpoints/spot/d4pg_cp-spot-{now_str}.pth"
        else:
            rel = "checkpoints/{}/d4pg_cp-failed-{}.pth".format(
                "failure" if failed else "success",
                now_str,
            )
        logger.info("Saving final checkpoint...")
        self.save_checkpoint(os.path.join(self.checkpoint_dir, rel))

    def save_checkpoint(
        self,
        checkpoint_path: str | Path,
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
            checkpoint_path: Path to save checkpoint
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
        checkpoint_path = resolve_save_path(checkpoint_path, self.__class__.__name__)
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

        # Config snapshot from exp_config (source of truth)
        config = self._get_hparams() if include_config else None

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

        if self._is_rank0():
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
        map_location: str | torch.device = self.device
        if dist.is_available() and dist.is_initialized():
            map_location = "cpu"
        checkpoint = torch.load(
            cp_path, map_location=map_location, weights_only=weights_only
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
