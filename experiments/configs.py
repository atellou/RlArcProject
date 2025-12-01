"""Configuration dataclasses for ArcD4PG experiments."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rlarcworld.common.configs import NStepMinVMaxConfig


@dataclass
class EnvConfig:
    """Environment configuration.

    Attributes:
        grid_size (int): Size of the grid.
        color_values (int): Number of color values.
        batch_size (int): Batch size.
    """

    grid_size: int = 30
    color_values: int = 10
    batch_size: int = 32


@dataclass
class NetworkConfig:
    """Neural network configuration.

    Attributes:
        embedding_size (int): Size of the embedding.
    """

    embedding_size: int = 128


@dataclass
class D4PGConfig:
    """D4PG algorithm configuration.

    Attributes:
        policy_lr (float): Learning rate for the policy.
        critic_lr (float): Learning rate for the critic.
        gamma (float): Discount factor.
        tau (float): Soft update parameter.
        n_steps (int): Number of steps to look ahead.
    """

    policy_lr: float = 3e-4
    critic_lr: float = 3e-3
    gamma: float = 0.99
    tau: float = 0.005
    n_step_config: NStepMinVMaxConfig = field(
        default_factory=lambda: NStepMinVMaxConfig(n_steps=1)
    )


@dataclass
class DistributionalConfig:
    """Distributional RL configuration.

    Attributes:
        num_atoms (dict[str, int]): Number of atoms for each distribution.
        v_min (dict[str, int]): Minimum value for each distribution.
        v_max (dict[str, int]): Maximum value for each distribution.
    """

    num_atoms: dict[str, int] = field(
        default_factory=lambda: {"pixel_wise": 51, "binary": 51}
    )
    v_min: dict[str, int] = field(
        default_factory=lambda: {"pixel_wise": -10, "binary": -1}
    )
    v_max: dict[str, int] = field(
        default_factory=lambda: {"pixel_wise": 10, "binary": 1}
    )


@dataclass
class ReplayBufferConfig:
    """Replay buffer configuration.

    Attributes:
        size (int): Size of the replay buffer.
        alpha (float): Alpha parameter for prioritized replay.
        beta (float): Beta parameter for prioritized replay.
        warmup_ratio (float): Ratio of warmup steps.
    """

    size: int = 100000
    alpha: float = 0.6
    beta: float = 0.4
    warmup_ratio: float = 0.2


@dataclass
class TrainingConfig:
    """Training loop configuration.

    Attributes:
        max_steps_per_epoch (int): Maximum steps per epoch.
        validation_epochs (int): Number of validation epochs.
        validation_steps_per_epoch (int): Number of validation steps per epoch.
        validation_epochs_frequency (int): Frequency of validation epochs.
    """

    max_steps_per_epoch: int = 1000
    validation_epochs: int = 1
    validation_steps_per_epoch: int = 100
    validation_epochs_frequency: int = 1
    validation_steps_frequency: int = 100
    min_train_batch_size: int = 2


@dataclass
class WorldModelConfig:
    """World model (Dyna-style imagination) configuration.

    Attributes:
        enabled (bool): Whether to enable the world model.
        learning_rate (float): Learning rate for the world model.
        imagination_horizon (int): Imagination horizon.
        imagination_ratio (int): Imagination ratio.
        imagination_frequency (int): Imagination frequency.
        model_update_frequency (int): Model update frequency.
    """

    enabled: bool = False
    learning_rate: float = 1e-4
    imagination_horizon: int = 5
    imagination_ratio: int = 5
    imagination_frequency: int = 10
    model_update_frequency: int = 10


@dataclass
class CheckpointConfig:
    """Checkpointing configuration.

    Attributes:
        periodic_interval (int): Periodic interval for checkpointing.
        experiment_path (Optional[str | Path]): Path to the experiment directory.
    """

    periodic_interval: int = 1000
    experiment_path: Optional[str | Path] = None


@dataclass
class SystemConfig:
    """System configuration.

    Attributes:
        amp_enabled (bool): Whether to enable automatic mixed precision.
        device (Optional[str]): Device to use for training.
        seed (int): Random seed.
    """

    amp_enabled: bool = True
    device: Optional[str] = None
    seed: int = 42


@dataclass
class ArcD4PGConfig:
    """Complete configuration for ArcD4PG experiment.

    Groups all sub-configurations into a single config object.

    Attributes:
        dataset_path (str): Path to the dataset.
        env (EnvConfig): Environment configuration.
        network (NetworkConfig): Network configuration.
        d4pg (D4PGConfig): D4PG configuration.
        distributional (DistributionalConfig): Distributional configuration.
        replay_buffer (ReplayBufferConfig): Replay buffer configuration.
        training (TrainingConfig): Training configuration.
        world_model (WorldModelConfig): World model configuration.
        checkpoint (CheckpointConfig): Checkpoint configuration.
        system (SystemConfig): System configuration.

    Example:
        ```python
        config = ArcD4PGConfig(
            dataset_path="path/to/dataset",
            env=EnvConfig(grid_size=6, color_values=5),
            training=TrainingConfig(max_steps_per_epoch=100),
            world_model=WorldModelConfig(enabled=True),
        )
        experiment = ArcD4PG(config)
        ```
    """

    dataset_path: str
    env: EnvConfig = field(default_factory=EnvConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    d4pg: D4PGConfig = field(default_factory=D4PGConfig)
    distributional: DistributionalConfig = field(default_factory=DistributionalConfig)
    replay_buffer: ReplayBufferConfig = field(default_factory=ReplayBufferConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
