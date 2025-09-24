"""ArcD4PG class to run an experiment with the D4PG algorithm and ARC environment."""

import logging
import random
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
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
from rlarcworld.environments.arc_batch_grid_env import ArcBatchGridEnv
from rlarcworld.environments.reward_processor import RewardProcessor
from rlarcworld.environments.stepper import EnvStepper
from rlarcworld.environments.wrappers.rewards import PixelAwareRewardWrapper
from rlarcworld.runners.core import OnStepContext, Trainer
from rlarcworld.utils.checkpointing import (
    add_schema_and_class,
    resolve_load_path,
    resolve_save_path,
)
from rlarcworld.utils.device import CudaConfig, enable_cuda
from rlarcworld.utils.logging_utils import configure_logger
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
            device (str): Device to use.
            actor (ArcActorNetwork): Actor network.
            config (dict): Configuration dictionary.
        """
        # Initialize dataset and environment
        logger.info("Setting up dataset and environment...")

        def reset_env(env, sample, seed):
            # Accept single-sample outputs from dataset and add batch dim if needed
            task = sample["task"]
            examples = sample["examples"]
            # Ensure 'input' and 'output' have shape (B, H, W)
            inp = task["input"]
            out = task["output"]
            if inp.ndim == 2:
                inp = inp.unsqueeze(0)
            if out.ndim == 2:
                out = out.unsqueeze(0)
            batched_task = {"input": inp, "output": out}
            # Ensure examples have shape (B, E, 2, H, W)
            if examples.ndim == 4:
                examples = examples.unsqueeze(0)
            args = {"batch": batched_task, "examples": examples}
            env.reset(options=args, seed=seed)

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
            env_sample_reset=reset_env,
            amp_enabled=config.get("amp_enabled", True),
            on_step=reward_on_step,
        )

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
        validation_steps_per_epoch: int = 100,
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
            validation_steps_per_epoch: Number of steps per validation epoch
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
        configure_logger()

        # Store configuration
        self.grid_size = grid_size
        self.color_values = color_values
        self.batch_size = batch_size
        self.max_steps_per_epoch = max_steps_per_epoch
        self.validation_steps_per_epoch = validation_steps_per_epoch
        self.min_train_batch_size = int(min_train_batch_size)

        # Default distributional RL parameters
        if num_atoms is None:
            num_atoms = {"pixel_wise": 51, "binary": 51}
        if v_min is None:
            v_min = {"pixel_wise": -10, "binary": -1}
        if v_max is None:
            v_max = {"pixel_wise": 10, "binary": 1}

        # Initialize networks
        logger.info("Initializing actor and critic networks...")
        self.actor = ArcActorNetwork(
            grid_size=grid_size,
            color_values=color_values,
            embedding_size=embedding_size,
            config=self.config,
        )

        self.critic = ArcCriticNetwork(
            grid_size=grid_size,
            color_values=color_values,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max,
            embedding_size=embedding_size,
            config=self.config,
        )

        # Create training stepper
        self.training_stepper = ArcStepper(
            max_steps_per_epoch=self.max_steps_per_epoch,
            dataset_path=dataset_path,
            grid_size=grid_size,
            color_values=color_values,
            device=self.device,
            actor=self.actor,
            config=self.config,
        )

        # Create validation stepper
        self.validation_stepper = ArcStepper(
            max_steps_per_epoch=self.validation_steps_per_epoch,
            dataset_path=dataset_path,
            grid_size=grid_size,
            color_values=color_values,
            device=self.device,
            actor=self.actor,
            config=self.config,
        )

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

        logger.info("ArcD4PG experiment initialized successfully")

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
            OrderedDict()
        )
        on_step_callbacks["core_step"] = self.d4pg_core.on_step
        on_step_callbacks["logging"] = self._log_step

        # Setup exception callbacks
        on_exception_callbacks: OrderedDict[
            str, Callable[[Exception, OnStepContext], None]
        ] = OrderedDict()
        on_exception_callbacks["log_exception"] = self._log_exception

        # Set up replay buffer
        sampler = PrioritizedSampler(
            replay_buffer_size,
            alpha=alpha,
            beta=beta,
        )
        storage = LazyMemmapStorage(replay_buffer_size)
        self.replay_buffer = TensorDictReplayBuffer(storage=storage, sampler=sampler)

        self.trainer = Trainer(
            core=self.d4pg_core,
            batch_size=self.batch_size,
            train_step_stream_factory=self.training_stepper.create_step_stream_from_samples,
            validation_step_stream_factory=self.validation_stepper.create_step_stream_from_samples,
            replay_buffer=self.replay_buffer,
            warmup_buffer_ratio=warmup_buffer_ratio,
            on_train_step=on_step_callbacks,
            on_exception=on_exception_callbacks,
            device=self.device,
            min_train_batch_size=self.min_train_batch_size,
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

    def _log_exception(self, e: Exception, context: OnStepContext) -> None:
        """Log training exceptions.

        Args:
            e: Exception that occurred
            context: Step context when exception occurred
        """
        logger.error(
            f"Training exception {e} at epoch {context.epoch}, "
            f"step {context.step_idx}, iteration {context.iteration}"
        )

    def train(self, epochs: int = 10) -> None:
        """Run the complete training loop.

        Args:
            epochs: Number of epochs to train
        """
        logger.info(f"Starting D4PG training for {epochs} epochs")

        try:
            assert self.trainer is not None
            self.trainer.train(
                epochs=epochs, max_steps_per_epoch=self.max_steps_per_epoch
            )
            logger.info("Training completed successfully")
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise

    def save_checkpoint(
        self, path: str | Path, *, save_replay_buffer: bool = False
    ) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint
            save_replay_buffer: Whether to save replay buffer
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

        # Core, env, trainer, stepper
        core_ckpt = self.d4pg_core.checkpoint()
        env_ckpt = self.training_stepper.environment.checkpoint()
        trainer_ckpt = self.trainer.checkpoint()
        stepper_ckpt = self.training_stepper.stepper.checkpoint()

        # RNG states
        rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.random.get_rng_state(),
            "torch_cuda": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        }

        # Config snapshot and provenance
        config = {
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

        provenance = {
            "created_at": datetime.utcnow().isoformat() + "Z",
            "schema": 1,
            "env": {
                "torch_version": torch.__version__,
                "numpy_version": np.__version__,
                "python_version": sys.version,
            },
        }

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
        self, path: str | Path, *, load_replay_buffer: bool = False
    ) -> None:
        """Load model checkpoint.

        Args:
            path: Path to load checkpoint from
            load_replay_buffer: Whether to load replay buffer
        """
        cp_path = resolve_load_path(path, self.__class__.__name__)
        # Allowlist project-specific classes for safe unpickling with weights_only
        try:
            add_safe_globals([TorchQueue, _make_td])
        except Exception:
            pass
        checkpoint = torch.load(cp_path, map_location=self.device)

        # Restore core
        core_state = checkpoint.get("core")
        if core_state is not None:
            self.d4pg_core.load(core_state, device=self.device)

        # Restore environment/wrapper
        env_state = checkpoint.get("env")
        if env_state is not None and hasattr(self, "training_stepper"):
            self.training_stepper.environment.load(env_state, device=self.device)

        # Restore trainer progress
        trainer_state = checkpoint.get("trainer")
        if trainer_state is not None and hasattr(self, "trainer"):
            self.trainer.load(trainer_state)

        # Restore stepper progress (does not restore sampler)
        stepper_state = checkpoint.get("stepper")
        if stepper_state is not None and hasattr(self, "training_stepper"):
            self.training_stepper.stepper.load(stepper_state)

        # Restore RNGs
        rng = checkpoint.get("rng", {})
        try:
            if rng.get("python") is not None:
                random.setstate(rng["python"])
            if rng.get("numpy") is not None:
                np.random.set_state(rng["numpy"])
            if rng.get("torch_cpu") is not None:
                torch.random.set_rng_state(rng["torch_cpu"])
            if rng.get("torch_cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["torch_cuda"])
        except Exception as e:
            logger.warning(f"Failed to restore RNG states: {e}")
        if (
            core_state["critic_scheduler"] is not None
            and self.d4pg_core.critic_scheduler is not None
        ):
            self.d4pg_core.critic_scheduler.load_state_dict(
                core_state["critic_scheduler"]
            )

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
                    except Exception as e:
                        logger.warning(f"Failed to restore replay buffer: {e}")

        logger.info(f"Checkpoint loaded from {path}")
