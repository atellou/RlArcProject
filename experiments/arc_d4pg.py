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

from rlarcworld.agent.actor import ArcActorNetwork
from rlarcworld.agent.critic import ArcCriticNetwork
from rlarcworld.algorithms.d4pg import D4PGCore
from rlarcworld.arc_dataset import ArcDataset, ArcSampleTransformer
from rlarcworld.common.enums import RewardKeys
from rlarcworld.environments.arc_batch_grid_env import (
    ArcBatchGridEnv,
    ArcEnvRunningMode,
)
from rlarcworld.environments.reward_processor import RewardProcessor
from rlarcworld.environments.stepper import EnvStepper
from rlarcworld.environments.wrappers.pixel_aware_reward import PixelAwareRewardWrapper
from rlarcworld.logging.tensorboard import make_tensorboard_callbacks
from rlarcworld.runners.core import OnStepContext, ReplayBufferProto, Trainer
from rlarcworld.utils.checkpointing import (
    add_schema_and_class,
    resolve_load_path,
    resolve_save_path,
)
from rlarcworld.utils.device import CudaConfig, enable_cuda
from rlarcworld.utils.torch_queue import TorchQueue

logger = logging.getLogger(__name__)


class ArcStepper:
    """ArcStepper class to handle environment stepping and episode simulation."""

    def __init__(
        self,
        max_steps_per_epoch: int,
        dataset_path: str,
        grid_size: int,
        color_values: int,
        env_mode: ArcEnvRunningMode,
        device: str,
        actor: ArcActorNetwork,
        config: CudaConfig,
    ) -> None:
        """Creates a stepper for the environment and dataset.

        Args:
            max_steps_per_epoch (int): Maximum steps per epoch.
            dataset_path (str): Path to the dataset.
            grid_size (int): Size of the ARC grid.
            color_values (int): Number of possible colors.
            env_mode (ArcEnvRunningMode): Environment running mode.
            device (str): Device to use.
            actor (ArcActorNetwork): Actor network.
            config (dict): Configuration dictionary.
        """
        # Initialize dataset and environment
        logger.info("Setting up dataset and environment...")
        self.env_mode = env_mode

        transform = ArcSampleTransformer(
            output_size=(grid_size, grid_size),
            examples_stack_dim=10,
            padding_constant_value=-1,
            zero_based_correction=1,
        )
        self.dataset: ArcDataset = ArcDataset(
            dataset_path, keep_in_memory=False, transform=transform
        )

        base_env = ArcBatchGridEnv(
            grid_size=grid_size,
            color_values=color_values,
            device=device,
        )
        # Wrap with pixel-aware rewards to expose get_source_env_reward and n-step API
        self.environment = PixelAwareRewardWrapper(
            base_env,
            max_penalty=-1.0,
            n_steps=base_env.n_steps,
            gamma=base_env.gamma,
            apply_clamp=False,
        )

        # Ensure rewards are structured for the critic heads via on_step processor
        reward_on_step = RewardProcessor(
            n_steps=self.environment.n_steps, device=device
        ).get_on_step_rewards()

        self.stepper = EnvStepper(
            actor=actor,
            device=device,
            env=self.environment,
            sampler=iter(self.dataset),
            max_steps=max_steps_per_epoch,
            env_sample_reset=self.reset_env,
            amp_enabled=config.get("amp_enabled", True),
            on_step=reward_on_step,
        )

    def reset_env(self, env, sample, **kwargs):
        """Reset the environment with a sample from the dataset.

        Args:
            env: The environment to reset.
            sample: The sample from the dataset.
            **kwargs: Additional keyword arguments.
        """
        options: dict = kwargs.setdefault("options", {"mode": self.env_mode.value})
        running_mode = options.setdefault("mode", self.env_mode.value)
        if running_mode != self.env_mode.value:
            raise ValueError(f"Unsupported running mode: {running_mode}")

        # Accept single-sample outputs from dataset and add batch dim if needed
        task = sample["task"]
        examples = sample["examples"]
        # Ensure 'input' and 'output' have shape (B, H, W)
        inp = task["input"]
        if inp.ndim == 2:
            inp = inp.unsqueeze(0)
        batched_task = {"input": inp}

        if running_mode == ArcEnvRunningMode.TRAIN.value:
            out = task["output"]
            if out.ndim == 2:
                out = out.unsqueeze(0)
            batched_task["output"] = out

        # Ensure examples have shape (B, E, 2, H, W)
        if examples.ndim == 4:
            examples = examples.unsqueeze(0)
        args = {"batch": batched_task, "examples": examples}
        kwargs["options"].update(args)
        env.reset(**kwargs)

    def create_step_stream_from_samples(self, **kwargs):
        """Create a step stream from samples."""
        return self.stepper.create_step_stream_from_samples(**kwargs)


class ArcD4PG:
    """Complete D4PG experiment orchestrator for ARC tasks.

    This class brings together all the decoupled components:
    - D4PGCore: Algorithm implementation
    - ArcActorNetwork & ArcCriticNetwork: Neural networks
    - ArcBatchGridEnv: Environment wrapper
    - Trainer: Training loop orchestration
    """

    def __init__(
        self,
        dataset_path: str,
        *,
        # Environment config
        grid_size: int = 30,
        color_values: int = 10,
        batch_size: int = 32,
        # Network config
        embedding_size: int = 128,
        # D4PG config
        policy_lr: float = 3e-4,
        critic_lr: float = 3e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        n_steps: int = 1,
        # Distributional RL config
        num_atoms: Optional[dict[str, int]] = None,
        v_min: Optional[dict[str, int]] = None,
        v_max: Optional[dict[str, int]] = None,
        # Replay buffer config
        replay_buffer_size: int = 100000,
        alpha: float = 0.6,
        beta: float = 0.4,
        warmup_buffer_ratio: float = 0.2,
        # Training config
        max_steps_per_epoch: int = 1000,
        validation_epochs: int = 1,
        validation_steps_per_epoch: int = 100,
        validation_epochs_frequency: int = 1,
        validation_steps_frequency: int = 100,
        # Checkpoint policies
        periodic_ckpt_interval: int = 1000,
        experiment_path: str | Path | None = None,
        # System config
        amp_enabled: bool = True,
        device: Optional[str] = None,
        seed: int = 42,
        # Trainer guards
        min_train_batch_size: int = 2,
        # Data config
        **kwargs,
    ):
        """Initialize the ArcD4PG experiment.

        Args:
            grid_size: Size of the ARC grid
            color_values: Number of possible colors
            batch_size: Training batch size
            embedding_size: Neural network embedding dimension
            policy_lr: Actor learning rate
            critic_lr: Critic learning rate
            gamma: Discount factor
            tau: Soft update coefficient
            n_steps: N-step returns
            num_atoms: Number of atoms for distributional critic
            v_min: Minimum values for distributional critic
            v_max: Maximum values for distributional critic
            replay_buffer_size: Replay buffer size
            alpha: Prioritized replay buffer alpha
            beta: Prioritized replay buffer beta
            warmup_buffer_ratio: Replay buffer warmup ratio
            max_steps_per_epoch: Maximum steps per training epoch
            validation_epochs: Number of epochs to validate for
            validation_steps_per_epoch: Number of steps per validation epoch
            validation_epochs_frequency: Frequency of validation epochs
            validation_steps_frequency: Frequency of validation steps
            periodic_ckpt_interval: Periodic checkpoint interval
            experiment_path: Path to experiment directory
            amp_enabled: Enable automatic mixed precision
            device: Device to use (auto-detected if None)
            seed: Random seed
            dataset_path: Path to ARC dataset
            min_train_batch_size: Minimum batch size for training
            **kwargs: Additional arguments passed to components
        """
        # Set random seed
        torch.manual_seed(seed)

        # Configure device and logging
        if device is None:
            self.config = enable_cuda(amp_enabled=amp_enabled)
        else:
            self.config = {
                "device": device,
                "amp_enabled": amp_enabled,
                "use_checkpointing": False,
            }
        self.device = self.config["device"]

        # Store configuration
        self.grid_size = grid_size
        self.color_values = color_values
        self.batch_size = batch_size
        self.max_steps_per_epoch = max_steps_per_epoch
        self.validation_steps_per_epoch = validation_steps_per_epoch
        self.min_train_batch_size = int(min_train_batch_size)

        # Validation policies
        self.validation_epochs = validation_epochs
        self.validation_steps_per_epoch = validation_steps_per_epoch
        self.validation_epochs_frequency = validation_epochs_frequency
        self.validation_steps_frequency = validation_steps_frequency

        # Checkpoint policies
        self.periodic_ckpt_interval = periodic_ckpt_interval
        if experiment_path is None:
            self.experiment_path = Path(
                "./experiments/{}-{}/".format(
                    self.__class__.__name__,
                    datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"),
                )
            )
        else:
            self.experiment_path = Path(experiment_path)

        logger.info(
            "Experiment outputs will be written to: %s", self.experiment_path.resolve()
        )

        # Default distributional RL parameters
        if num_atoms is None:
            num_atoms = {"pixel_wise": 51, "binary": 51}
        if v_min is None:
            v_min = {"pixel_wise": -10, "binary": -1}
        if v_max is None:
            v_max = {"pixel_wise": 10, "binary": 1}

        # Initialize networks
        logger.info("Initializing actor and critic networks...")
        logger.info("Creating actor network...")
        self.actor = ArcActorNetwork(
            grid_size=grid_size,
            color_values=color_values,
            embedding_size=embedding_size,
            config=self.config,
        )
        logger.info("Actor network created successfully")

        logger.info("Creating critic network...")
        self.critic = ArcCriticNetwork(
            grid_size=grid_size,
            color_values=color_values,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max,
            embedding_size=embedding_size,
            config=self.config,
        )
        logger.info("Critic network created successfully")

        # Create training stepper
        logger.info("Creating training stepper...")
        self.training_stepper = ArcStepper(
            max_steps_per_epoch=self.max_steps_per_epoch,
            dataset_path=dataset_path,
            grid_size=grid_size,
            color_values=color_values,
            env_mode=ArcEnvRunningMode.TRAIN,
            device=self.device,
            actor=self.actor,
            config=self.config,
        )
        logger.info("Training stepper created successfully")

        # Create validation stepper
        logger.info("Creating validation stepper...")
        self.validation_stepper = ArcStepper(
            max_steps_per_epoch=self.validation_steps_per_epoch,
            dataset_path=dataset_path,
            grid_size=grid_size,
            color_values=color_values,
            env_mode=ArcEnvRunningMode.TRAIN,
            device=self.device,
            actor=self.actor,
            config=self.config,
        )
        logger.info("Validation stepper created successfully")

        # Initialize D4PG core
        logger.info("Initializing D4PG algorithm core...")
        self.d4pg_core = D4PGCore(
            actor=self.actor,
            critic=self.critic,
            policy_lr=policy_lr,
            critic_lr=critic_lr,
            gamma=gamma,
            tau=tau,
            n_steps=n_steps,
            config=self.config,
            amp_enabled=amp_enabled,
            **kwargs,
        )

        # Initialize trainer with callbacks
        self._setup_trainer(
            replay_buffer_size=replay_buffer_size,
            alpha=alpha,
            beta=beta,
            warmup_buffer_ratio=warmup_buffer_ratio,
        )

    def _get_hparams(self):
        hparams = {
            "grid_size": self.grid_size,
            "color_values": self.color_values,
            "batch_size": self.batch_size,
            "max_steps_per_epoch": self.max_steps_per_epoch,
            "validation_steps_per_epoch": self.validation_steps_per_epoch,
            "device": self.device,
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
            device=self.device,
            min_train_batch_size=self.min_train_batch_size,
        )

        # Attach TensorBoard callbacks (train and validation)
        try:
            self.tb_manager, tb_train_cb, tb_val_cb = make_tensorboard_callbacks(
                self.experiment_path,
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
        context: OnStepContext,
        info: dict[str, torch.Tensor],
        replay_buffer: ReplayBufferProto,
    ) -> None:
        """Update replay buffer priority.

        Args:
            context: Training context with batch and TD-error
            info: Metadata from replay buffer sample containing indices
            replay_buffer: The replay buffer to update
        """
        # These checks should not be needed since caller validates, but kept for safety
        if context.td_error is None or info is None:
            raise ValueError("Skipping priority update: missing td_error or info")

        td_error = context.td_error

        # Check if info has index metadata (required for update_tensordict_priority)
        if "index" not in info.keys():
            raise ValueError(
                "Info does not have index metadata, cannot update priorities. "
                "Available keys: %s",
                list(info.keys()),
            )

        if td_error.get(RewardKeys.N_REWARD.value) is not None:
            replay_buffer.update_priority(
                info["index"], td_error.get(RewardKeys.N_REWARD.value)
            )
        else:
            replay_buffer.update_priority(
                info["index"], td_error.get(RewardKeys.PIXEL_WISE.value)
            )

    def _on_train_callbacks(self) -> OrderedDict[str, Callable[[OnStepContext], None]]:
        """Log training step information.

        Returns:
            Dictionary of callbacks for training steps
        """
        # Track best (lower is better) actor/critic losses for improvement checkpoints
        self.best_loss_actor: torch.Tensor = torch.tensor(float("inf"))
        self.best_loss_critic: torch.Tensor = torch.tensor(float("inf"))

        return OrderedDict(
            {
                "core_step": self.d4pg_core.on_step,
                "logging": self._log_step,
                "periodic_ckpt": self._periodic_checkpoint,
                "improvement_ckpt": self._improvement_checkpoint,
            }
        )

    def _log_step(self, context: OnStepContext) -> None:
        """Log training step information.

        Args:
            context: Step context containing training information
        """
        if context.iteration % 100 == 0 and context.loss_actor is not None:
            logger.info(
                f"Epoch {context.epoch}, Step {context.step_idx}, "
                f"Iteration {context.iteration}: "
                f"Actor Loss: {context.loss_actor:.4f}, "
                f"Critic Loss: {context.loss_critic:.4f}"
            )

    def _periodic_checkpoint(self, context: OnStepContext) -> None:
        """Periodic lightweight checkpoint: core + trainer + config + provenance.

        This runs frequently (every self.periodic_ckpt_interval steps) and avoids
        saving large components such as environment, stepper and replay buffer.

        Args:
            context: Step context containing training information
        """
        if self.periodic_ckpt_interval <= 0:
            return
        if (
            context.iteration <= 0
            or context.iteration % self.periodic_ckpt_interval != 0
        ):
            return
        fname = os.path.join(
            self.experiment_path,
            "checkpoints/periodic/d4pg_cp-per_i_{:06d}.pth".format(context.iteration),
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

    def _improvement_checkpoint(self, context: OnStepContext) -> None:
        """Save core-only checkpoint when losses improve.

        Lower losses are better. We checkpoint the core networks and optimizers
        only, to capture the best model so far with minimal size.

        Args:
            context: Step context containing training information
        """
        if context.loss_actor is None or context.loss_critic is None:
            return
        improved = False
        if context.loss_actor.item() < self.best_loss_actor:
            self.best_loss_actor = context.loss_actor.item()
            improved = True
        if context.loss_critic.item() < self.best_loss_critic:
            self.best_loss_critic = context.loss_critic.item()
            improved = True
        if not improved:
            return
        fname = os.path.join(
            self.experiment_path,
            "checkpoints/best/d4pg_cp-best_i_{:06d}-la_{:.4f}-lc_{:.4f}.pth".format(
                int(context.iteration),
                context.loss_actor.item(),
                context.loss_critic.item(),
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
                validation_epochs_frequency=self.validation_epochs_frequency,
                validation_steps_frequency=self.validation_steps_frequency,
            )
            logger.info("Training completed successfully")

            self.tb_manager.log_hparams(self._get_hparams())
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            failed = True
            raise
        finally:
            self.save_checkpoint(
                os.path.join(
                    self.experiment_path,
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
            and hasattr(self.training_stepper, "stepper")
        ):
            raise ValueError(
                "Experiment must have environment, trainer, and stepper to checkpoint"
            )

        # Core, env, trainer, stepper (conditionally included)
        core_ckpt = self.d4pg_core.checkpoint() if include_core else None
        env_ckpt = (
            self.training_stepper.environment.checkpoint() if include_env else None
        )
        trainer_ckpt = self.trainer.checkpoint() if include_trainer else None
        stepper_ckpt = (
            self.training_stepper.stepper.checkpoint() if include_stepper else None
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
            self.training_stepper.stepper.load(stepper_state)
            restored_stepper = True

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
            "stepper=%s, rng=%s, replay=%s from %s",
            restored_core,
            restored_env,
            restored_trainer,
            restored_stepper,
            restored_rng,
            restored_replay,
            path,
        )
