"""Common configurations."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from transformers.modeling_outputs import BaseModelOutput

from rlarcworld.common.enums import StateKeys


@dataclass
class NStepMinVMaxConfig:
    """Neural network configuration.

    Attributes:
        n_steps (int): Number of steps for n-step returns.
        pixel_wise_min_v (Optional[float]): Size of the embedding.
        pixel_wise_max_v (Optional[float]): Size of the embedding.
        binary_min_v (Optional[float]): Size of the embedding.
        binary_max_v (Optional[float]): Size of the embedding.
    """

    n_steps: int = 1
    pixel_wise_min_v: Optional[float] = None
    pixel_wise_max_v: Optional[float] = None
    binary_min_v: Optional[float] = None
    binary_max_v: Optional[float] = None

    def get_pixel_wise_minmax(self) -> dict[str, float] | None:
        """Get the pixel-wise minimum value and maximum value."""
        if self.pixel_wise_min_v is None or self.pixel_wise_max_v is None:
            return None
        return {"min": self.pixel_wise_min_v, "max": self.pixel_wise_max_v}

    def get_binary_minmax(self) -> dict[str, float] | None:
        """Get the binary minimum value and maximum value."""
        if self.binary_min_v is None or self.binary_max_v is None:
            return None
        return {"min": self.binary_min_v, "max": self.binary_max_v}


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
class RlArcWorldViTARCConfig:
    """Configuration for the RlArcWorld ViTARC nn module/encoder.

    A configuration class for the ViTARC used in RlArcWorld.
    This configuration defines the hyperparameters for the ViTARC-based neural network.

    Attributes:
        grid_size: Size of the input grid (height = width).
        d_model: Model dimension for embeddings and hidden layers.
        d_ff: Feed-forward dimension.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        dropout: Dropout rate.
        ape_type: Type of absolute position encoding ('none', 'learnable', 'sinusoidal')
        rpe_type: Type of relative position encoding ('none', 'learnable', 'sinusoidal')
        use_ope: Whether to use object position encoding.
        ape_mixer: Mixer type for absolute position encoding.
        trainable_ape: Whether positional embeddings are trainable.
        grids_to_encode: List of grid keys to encode.
        color_values: Number of color values in the dataset.
    """

    grid_size: int
    d_model: int = 128
    d_ff: int = 512
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    ape_type: str | None = None
    rpe_type: str | None = None
    use_ope: bool | None = None
    ape_mixer: str | None = None
    trainable_ape: bool = False
    grids_to_encode: list[str] | None = None
    use_index_grid: bool = True
    color_values: int | None = None

    @classmethod
    def best_config(
        cls,
        grid_size: int,
        use_temporal: bool = False,
        use_examples: bool = False,
        use_index_grid: bool = True,
    ) -> "RlArcWorldViTARCConfig":
        """Create a config with the best settings from the ViTARC paper.

        Best configuration from ViTARC paper ablation study (arXiv:2410.06405):
        - APE: SinusoidalAPE2D (2D sinusoidal absolute position encoding)
        - RPE: Two-slope-Alibi (largest performance gain)
        - PEmixer: weighted_sum_no_norm_vec (second largest gain)
        - OPE: True (third largest gain, object-based positional encoding)

        Args:
            grid_size: Size of the input grid (height = width).
            use_temporal: Whether to use temporal grids for encoding.
            use_examples: Whether to use example grids for encoding.
            use_index_grid: Whether to use index grid for encoding.

        Returns:
            RlArcWorldViTARCConfig with best settings for training.
        """
        grid_to_encode = [StateKeys.GRID.value]
        if use_temporal:
            grid_to_encode.extend([StateKeys.LAST_GRID.value, StateKeys.INITIAL.value])
        if use_examples:
            grid_to_encode.append(StateKeys.EXAMPLES.value)
        return cls(
            grid_size=grid_size,
            d_model=128,
            d_ff=512,
            num_layers=4,
            num_heads=8,
            dropout=0.1,
            ape_type="SinusoidalAPE2D",
            rpe_type="Two-slope-Alibi",
            use_ope=True,
            ape_mixer="weighted_sum_no_norm_vec",
            trainable_ape=False,
            grids_to_encode=grid_to_encode,
            use_index_grid=use_index_grid,
        )


@dataclass
class ViTARCEncoderOutput:
    """Structured output for ViTARC encoder modules.

    Exposes both the conditioned hidden states and the intermediate artifacts
    necessary for downstream consumers to reuse embeddings or perform custom
    pooling/attention.

    Returns:
        hidden_states: The conditioned hidden states from the encoder.
        attention_mask: The attention mask used during encoding.
        encoder_outputs: The full encoder outputs for potential reuse.
        input_ids: The input token IDs.
        temporal_ids: Optional temporal position IDs.
        object_idx: Optional object indices for object-aware processing.
        example_hidden: Optional example-level hidden states.
        example_mask: Optional example-level attention mask.
        index_ids: Optional index/timestamp IDs aligned with input_ids.
    """

    hidden_states: torch.Tensor
    attention_mask: torch.Tensor | None
    encoder_outputs: BaseModelOutput
    input_ids: torch.Tensor
    temporal_ids: torch.Tensor | None
    object_idx: torch.Tensor | None
    example_hidden: torch.Tensor | None
    example_mask: torch.Tensor | None
    index_ids: torch.Tensor | None


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
    network: RlArcWorldViTARCConfig = field(
        default_factory=lambda: RlArcWorldViTARCConfig.best_config(
            grid_size=30, use_temporal=True, use_examples=True, use_index_grid=True
        )
    )
    d4pg: D4PGConfig = field(default_factory=D4PGConfig)
    distributional: DistributionalConfig = field(default_factory=DistributionalConfig)
    replay_buffer: ReplayBufferConfig = field(default_factory=ReplayBufferConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
