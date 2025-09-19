"""ArcD4PG class to run an experiment with the D4PG algorithm and ARC environment."""

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Optional

import torch

from rlarcworld.agent.actor import ArcActorNetwork
from rlarcworld.agent.critic import ArcCriticNetwork
from rlarcworld.algorithms.d4pg import D4PGCore
from rlarcworld.arc_dataset import ArcDataset, ArcSampleTransformer
from rlarcworld.environments.arc_batch_grid_env import ArcBatchGridEnv
from rlarcworld.environments.stepper import EnvStepper
from rlarcworld.runners.core import OnStepContext, Trainer
from rlarcworld.utils.device import CudaConfig, enable_cuda
from rlarcworld.utils.logging_utils import configure_logger

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
            args = {"batch": sample["task"], "examples": sample["examples"]}
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

        self.environment = ArcBatchGridEnv(
            grid_size=grid_size,
            color_values=color_values,
            device=device,
        )

        self.stepper = EnvStepper(
            actor=actor,
            device=device,
            env=self.environment,
            sampler=iter(self.dataset),
            max_steps=max_steps_per_epoch,
            env_sample_reset=reset_env,
            amp_enabled=config.get("amp_enabled", True),
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
        # Training config
        max_steps_per_epoch: int = 1000,
        validation_steps_per_epoch: int = 100,
        warmup_buffer_ratio: float = 0.2,
        # System config
        amp_enabled: bool = True,
        device: Optional[str] = None,
        seed: int = 42,
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
            max_steps_per_epoch: Maximum steps per training epoch
            validation_steps_per_epoch: Number of steps per validation epoch
            warmup_buffer_ratio: Replay buffer warmup ratio
            amp_enabled: Enable automatic mixed precision
            device: Device to use (auto-detected if None)
            seed: Random seed
            dataset_path: Path to ARC dataset
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
            config=dict(self.config),
            amp_enabled=amp_enabled,
            **kwargs,
        )

        # Initialize trainer with callbacks
        self._setup_trainer(warmup_buffer_ratio)

        logger.info("ArcD4PG experiment initialized successfully")

    def _setup_trainer(self, warmup_buffer_ratio: float) -> None:
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

        self.trainer: Optional[Trainer] = Trainer(
            core=self.d4pg_core,
            batch_size=self.batch_size,
            train_step_stream_factory=self.training_stepper.create_step_stream_from_samples,
            validation_step_stream_factory=self.validation_stepper.create_step_stream_from_samples,
            replay_buffer=None,  # Using direct sampling for now
            warmup_buffer_ratio=warmup_buffer_ratio,
            on_train_step=on_step_callbacks,
            on_exception=on_exception_callbacks,
            device=self.device,
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

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "core": self.d4pg_core.checkpoint(),
            "config": {
                "grid_size": self.grid_size,
                "color_values": self.color_values,
                "batch_size": self.batch_size,
            },
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint.

        Args:
            path: Path to load checkpoint from
        """
        checkpoint = torch.load(path, map_location=self.device)

        # Load core state
        core_state = checkpoint["core"]
        self.actor.load_state_dict(core_state["actor"])
        self.critic.load_state_dict(core_state["critic"])
        self.d4pg_core.actor_optimizer.load_state_dict(core_state["actor_optimizer"])
        self.d4pg_core.critic_optimizer.load_state_dict(core_state["critic_optimizer"])

        if (
            core_state["actor_scheduler"] is not None
            and self.d4pg_core.actor_scheduler is not None
        ):
            self.d4pg_core.actor_scheduler.load_state_dict(
                core_state["actor_scheduler"]
            )
        if (
            core_state["critic_scheduler"] is not None
            and self.d4pg_core.critic_scheduler is not None
        ):
            self.d4pg_core.critic_scheduler.load_state_dict(
                core_state["critic_scheduler"]
            )

        logger.info(f"Checkpoint loaded from {path}")
