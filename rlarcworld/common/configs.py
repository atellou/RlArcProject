"""Common configurations."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from transformers.modeling_outputs import BaseModelOutput

from rlarcworld.common.enums import RewardKeys

logger = logging.getLogger(__name__)


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
        max_penalty (float): Maximum penalty for the environment reward.
        batch_size (int): Batch size.
    """

    grid_size: int = 30
    color_values: int = 10
    max_penalty: float = 1.0
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
        use_index_grid: Whether to use index grid.
        use_temporal: Whether to use temporal encoding.
        use_examples: Whether to use examples.
        example_compression_tokens: Number of tokens to compress each example to.
        grid_compression_tokens: Number of tokens to compress grid(None=no compression).
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
    use_index_grid: bool = True
    use_temporal: bool = True
    use_examples: bool = True
    example_compression_tokens: int = 64
    grid_compression_tokens: int | None = 128
    compression_method: str = "learned"  # "pooling" or "learned"
    max_learnable_examples: int = 1
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
            use_index_grid=use_index_grid,
            use_temporal=use_temporal,
            use_examples=use_examples,
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
        share_network (bool): Whether to share the network between actor and critic.
        policy_lr (float): Learning rate for the policy.
        critic_lr (float): Learning rate for the critic.
        gamma (float): Discount factor.
        tau (float): Soft update parameter.
        lr_scheduler (dict | None): Learning rate scheduler configuration.
        n_step_config (NStepMinVMaxConfig): N-step return configuration.
        compute_td_error (bool): Whether to compute TD error.
        carsm (bool): Whether to use CARSM regularization.
        entropy_coef (float): Entropy regularization coefficient.
        entropy_coef_decay (float): Entropy coefficient decay rate.
        target_update_frequency (int): Frequency of target network updates.
        max_grad_norm (float): Maximum gradient norm for clipping.
    """

    share_network: bool = True
    policy_lr: float = 3e-4
    critic_lr: float = 3e-3
    gamma: float = 0.99
    tau: float = 0.005
    lr_scheduler_kwargs: Optional[dict] = None
    n_step_config: NStepMinVMaxConfig = field(
        default_factory=lambda: NStepMinVMaxConfig(n_steps=1)
    )
    compute_td_error: bool = True
    carsm: bool = False
    entropy_coef: float = 0.0
    entropy_coef_decay: float = 1.0
    target_update_frequency: int = 10
    model_update_frequency: int = 1
    cache_enabled: bool = False
    max_grad_norm: float = 0.0


@dataclass
class DistributionalConfig:
    """Distributional RL configuration.

    Attributes:
        num_atoms (dict[str, int]): Number of atoms for each distribution.
        v_min (dict[str, int]): Minimum value for each distribution.
        v_max (dict[str, int]): Maximum value for each distribution.
    """

    num_atoms: dict[str, int] = field(
        default_factory=lambda: {"pixel_wise": 21, "binary": 21}
    )
    v_min: dict[str, int] = field(
        default_factory=lambda: {"pixel_wise": -1, "binary": -1}
    )
    v_max: dict[str, int] = field(
        default_factory=lambda: {"pixel_wise": 1, "binary": 1}
    )

    def scale_by_penalty(self, penalty: float) -> "DistributionalConfig":
        """Scale v_min, v_max, and num_atoms by the given penalty factor.

        Args:
            penalty (float): Penalty scaling factor (typically max_penalty value).
                When max_penalty > 0, scales the support range and atom count
                accordingly.

        Returns:
            DistributionalConfig: New config with scaled bounds and atoms.
        """
        if penalty <= 0:
            return self

        # Scale support bounds by penalty for every key present
        scaled_v_min = {k: int(v * penalty) for k, v in self.v_min.items()}
        scaled_v_max = {k: int(self.v_max[k] * penalty) for k in self.v_max}

        # Scale atom count to maintain ~0.1 unit resolution per atom
        # Preserve minimum of 11 atoms per key
        scaled_num_atoms = {
            k: max(11, int(self.num_atoms[k] * penalty)) for k in self.num_atoms
        }

        return DistributionalConfig(
            num_atoms=scaled_num_atoms,
            v_min=scaled_v_min,
            v_max=scaled_v_max,
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
    warmup_ratio: float = 0.02


@dataclass
class TrainingConfig:
    """Training loop configuration.

    Attributes:
        max_steps_per_epoch (int): Maximum steps per epoch.
        max_steps_per_episode (int): Maximum steps per episode.
        validation_epochs (int): Number of validation epochs.
        validation_steps_per_epoch (int): Number of validation steps per epoch.
        validation_steps_per_episode (int): Number of validation steps per episode.
        validation_steps_frequency (int): Frequency of validation steps.
        min_train_batch_size (int): Minimum batch size for training.
        augmentation_enabled (bool): Whether to enable data augmentation.
    """

    max_steps_per_epoch: int = 10000
    max_steps_per_episode: int = 900
    validation_epochs: int = 1
    validation_steps_per_epoch: int = 100
    validation_steps_per_episode: int = 100
    validation_steps_frequency: int = 100
    min_train_batch_size: int = 2
    augmentation_enabled: bool = False


@dataclass
class WorldModelConfig:
    """World model (Dyna-style imagination) configuration.

    Attributes:
        enabled (bool): Whether to enable the world model.
        learning_rate (float): Learning rate for the world model.
        lr_scheduler_kwargs (dict | None): Learning rate scheduler configuration.
        imagination_horizon (int): Imagination horizon.
        imagination_ratio (int): Imagination ratio.
        imagination_frequency (int): Imagination frequency.
        model_update_frequency (int): Model update frequency.
    """

    enabled: bool = False
    learning_rate: float = 1e-4
    lr_scheduler_kwargs: dict | None = None
    imagination_horizon: int = 5
    imagination_ratio: int = 5
    imagination_frequency: int = 10
    model_update_frequency: int = 10


@dataclass
class CheckpointConfig:
    """Checkpointing configuration.

    Attributes:
        save_frequency (int): Frequency for saving checkpoints.
        checkpoint_dir (Optional[str | Path]): Directory to save checkpoints.
        improvement_checkpoint_enabled (bool): Whether to save checkpoints when
            validation reward improves. Uses actual policy performance metrics.
    """

    save_frequency: int = -1
    checkpoint_dir: Optional[str | Path] = None
    improvement_checkpoint_enabled: bool = False


@dataclass
class LoggingConfig:
    """Logging configuration."""

    tensorboard_enabled: bool = False
    log_frequency: int = 100


@dataclass
class SystemConfig:
    """System configuration.

    Attributes:
        amp_enabled (bool): Whether to enable automatic mixed precision.
        device (Optional[str]): Device to use for training.
    """

    amp_enabled: bool = True
    use_checkpointing: bool = False
    fsdp_enabled: bool = False
    device: Optional[str] = None


@dataclass
class ReproducibilityConfig:
    """Control global RNG seeds and determinism flags for full reproducibility."""

    global_seed: int | None = None
    dataset_seed: int | None = None
    augmentation_seed: int | None = None
    torch_deterministic: bool = True


@dataclass
class ArcD4PGConfig:
    """Complete configuration for ArcD4PG experiment.

    Groups all sub-configurations into a single config object.

    Attributes:
        dataset (str | list[str]): Path to the dataset.
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
            dataset="path/to/dataset",
            env=EnvConfig(grid_size=6, color_values=5),
            training=TrainingConfig(max_steps_per_epoch=100),
            world_model=WorldModelConfig(enabled=True),
        )
        experiment = ArcD4PG(config)
        ```
    """

    dataset: str | list[str]
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
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    reproducibility: ReproducibilityConfig = field(
        default_factory=ReproducibilityConfig
    )
    system: SystemConfig = field(default_factory=SystemConfig)

    def get_validated_config(self) -> "ArcD4PGConfig":
        """Get a validated version of the configuration."""
        self.network.grid_size = self.env.grid_size
        self.distributional = self.distributional.scale_by_penalty(self.env.max_penalty)

        if self.d4pg.model_update_frequency == 0:
            self.d4pg.model_update_frequency = 1

        if self.d4pg.model_update_frequency > 1:
            logger.warning(
                "Model update frequency is greater than 1. Enabling example cache."
            )
            self.d4pg.cache_enabled = True

        if self.d4pg.n_step_config.n_steps > 1:
            step_keys = RewardKeys.get_n_step_keys()
            # Validate distributional contains n-step parameters
            if not set(step_keys).issubset(self.distributional.num_atoms.keys()):
                raise ValueError(
                    "Distributional config missing num_atoms for n-step rewards. Provided: {}".format(  # noqa: E501
                        self.distributional.num_atoms.keys()
                    )
                )

            # Auto-fill NStepMinVMaxConfig min/max from distributional when missing
            ns_cfg = self.d4pg.n_step_config
            # Mapping helper
            _map = {
                RewardKeys.PIXEL_WISE_N_REWARD: (
                    "pixel_wise_min_v",
                    "pixel_wise_max_v",
                ),
                RewardKeys.BINARY_N_REWARD: ("binary_min_v", "binary_max_v"),
            }
            for rk, (mn_attr, mx_attr) in _map.items():
                if getattr(ns_cfg, mn_attr) is None:
                    setattr(ns_cfg, mn_attr, self.distributional.v_min[rk.value])
                if getattr(ns_cfg, mx_attr) is None:
                    setattr(ns_cfg, mx_attr, self.distributional.v_max[rk.value])

        return self
