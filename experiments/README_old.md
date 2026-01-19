# ArcD4PG Experiment Configuration Guide

This guide documents all available hyperparameters for the ArcD4PG experiment orchestrator. Use these parameters to define and run different training experiments.

## Configuration Structure

The `ArcD4PGConfig` is the main configuration object that groups all sub-configurations. It is composed of the following configuration classes:

```python
config = ArcD4PGConfig(
    dataset="path/to/dataset",
    env=EnvConfig(...),
    network=RlArcWorldViTARCConfig(...),
    d4pg=D4PGConfig(...),
    distributional=DistributionalConfig(...),
    replay_buffer=ReplayBufferConfig(...),
    training=TrainingConfig(...),
    world_model=WorldModelConfig(...),
    checkpoint=CheckpointConfig(...),
    system=SystemConfig(...),
)

experiment = ArcD4PG(config)
experiment.train(epochs=10)
```

---

## Environment Configuration (`EnvConfig`)

Controls the ARC environment parameters.

| Parameter | Type | Default | Impact |
|-----------|------|---------|--------|
| `grid_size` | int | 30 | Size of the input/output grid (height = width). Larger grids increase computational complexity and memory usage. |
| `color_values` | int | 10 | Number of distinct color values in the ARC dataset (0-9). Must match the dataset. Affects network output dimensions. |
| `max_penalty` | float | 1.0 | Maximum penalty for incorrect submissions. Scales the reward range and distributional RL support bounds. Higher penalties expand the reward support to [-penalty, penalty]. |
| `batch_size` | int | 32 | Number of samples processed per training step. Larger batches improve gradient stability but increase memory usage. Smaller batches add more noise but may improve generalization. |

---

## Network Configuration (`RlArcWorldViTARCConfig`)

Configures the ViTARC encoder used by both actor and critic networks.

| Parameter | Type | Default | Impact |
|-----------|------|---------|--------|
| `grid_size` | int | Required | Must match `EnvConfig.grid_size`. Size of input grids. |
| `d_model` | int | 128 | Embedding dimension for transformer layers. Larger values increase model capacity and memory usage. Controls the width of hidden representations. |
| `d_ff` | int | 512 | Feed-forward network dimension in transformer blocks. Typically 4x `d_model`. Larger values increase model capacity. |
| `num_layers` | int | 4 | Number of transformer encoder layers. More layers increase model depth and capacity but slow down training. Affects ability to learn complex spatial relationships. |
| `num_heads` | int | 8 | Number of attention heads in multi-head attention. Must divide `d_model` evenly. More heads allow parallel attention patterns but increase computation. |
| `dropout` | float | 0.1 | Dropout rate for regularization (0.0-1.0). Higher values increase regularization, preventing overfitting but may reduce learning speed. |
| `ape_type` | str | "SinusoidalAPE2D" | Absolute position encoding type. Options: 'none', 'learnable', 'sinusoidal'. Affects how spatial positions are encoded. |
| `rpe_type` | str | "Two-slope-Alibi" | Relative position encoding type. Options: 'none', 'learnable', 'sinusoidal'. Improves attention mechanism's position awareness. |
| `use_ope` | bool | True | Enable object position encoding. Helps model understand object-level spatial relationships in the grid. |
| `ape_mixer` | str | "weighted_sum_no_norm_vec" | Method for mixing absolute position encodings. Affects how positional information is integrated. |
| `trainable_ape` | bool | False | Whether absolute position embeddings are trainable. Set to True for task-specific position learning. |
| `use_index_grid` | bool | True | Include index/timestamp grid in encoding. Helps model track temporal progression. |
| `use_temporal` | bool | True | Include temporal grids (current, last, initial). Enables sequence modeling. |
| `use_examples` | bool | True | Include example grids in the input. Enables few-shot learning capabilities. |
| `color_values` | int | 10 | Number of color values. Must match `EnvConfig.color_values`. |

### Quick Configuration

Use the `best_config()` class method for pre-tuned settings from the ViTARC paper:

```python
network = RlArcWorldViTARCConfig.best_config(
    grid_size=30,
    use_temporal=True,      # Include temporal grids
    use_examples=True,      # Include example grids
    use_index_grid=True,    # Include index grid
)
```

---

## D4PG Algorithm Configuration (`D4PGConfig`)

Controls the D4PG (Distributed Distributional Deterministic Policy Gradient) algorithm parameters.

| Parameter | Type | Default | Impact |
|-----------|------|---------|--------|
| `policy_lr` | float | 3e-4 | Learning rate for the actor (policy) network. Higher values speed up learning but may cause instability. Lower values provide stability but slower convergence. |
| `critic_lr` | float | 3e-3 | Learning rate for the critic (value) network. Typically 10x higher than policy_lr. Affects how quickly value estimates improve. |
| `gamma` | float | 0.99 | Discount factor (0.0-1.0). Controls how much future rewards are valued. Higher values (0.99) emphasize long-term rewards; lower values (0.9) emphasize immediate rewards. |
| `tau` | float | 0.005 | Soft update parameter for target networks. Controls the rate of target network updates. Lower values (0.005) provide stability; higher values (0.01) allow faster adaptation. |
| `lr_scheduler_kwargs` | dict \| None | None | Learning rate scheduler configuration. Pass dict with `step_size` and `gamma` keys for StepLR scheduler. Example: `{"step_size": 1000, "gamma": 0.5}`. None disables scheduling. |
| `n_step_config` | NStepMinVMaxConfig | n_steps=1 | N-step return configuration (see below). |

### Advanced D4PG Parameters

These parameters are passed to the D4PGCore algorithm and control advanced training dynamics:

| Parameter | Type | Default | Impact |
|-----------|------|---------|--------|
| `compute_td_error` | bool | True | Whether to compute TD error for prioritized replay. Disabling saves computation but loses priority information. |
| `carsm` | bool | False | Clipped Action Regularization for Soft-Modulus. Experimental regularization technique for policy stability. |
| `entropy_coef` | float | 0.0 | Entropy regularization coefficient. Higher values (0.01-0.1) encourage exploration; 0.0 disables entropy regularization. |
| `entropy_coef_decay` | float | 1.0 | Decay rate for entropy coefficient per step. Values < 1.0 gradually reduce exploration over time. |
| `target_update_frequency` | int | 10 | Frequency of target network updates (every N steps). Higher values (20-50) provide stability; lower values (5) allow faster adaptation. |
| `max_grad_norm` | float | 0.0 | Maximum gradient norm for clipping. 0.0 disables clipping; typical values: 1.0-10.0 for stability. |

### N-Step Configuration (`NStepMinVMaxConfig`)

| Parameter | Type | Default | Impact |
|-----------|------|---------|--------|
| `n_steps` | int | 1 | Number of steps for n-step returns. Higher values (3-5) reduce variance but increase bias. n_steps=1 is standard TD learning. |
| `pixel_wise_min_v` | float | None | Minimum value for pixel-wise reward distribution. Used for distributional RL value clipping. |
| `pixel_wise_max_v` | float | None | Maximum value for pixel-wise reward distribution. Used for distributional RL value clipping. |
| `binary_min_v` | float | None | Minimum value for binary reward distribution. |
| `binary_max_v` | float | None | Maximum value for binary reward distribution. |

---

## Distributional RL Configuration (`DistributionalConfig`)

Configures the distributional critic for learning value distributions instead of point estimates.

| Parameter | Type | Default | Impact |
|-----------|------|---------|--------|
| `num_atoms` | dict[str, int] | {"pixel_wise": 21, "binary": 21} | Number of atoms (support points) for each reward type. More atoms provide finer value distribution resolution but increase computation. Both use 21 atoms for symmetric [-1, 1] support. |
| `v_min` | dict[str, int] | {"pixel_wise": -1, "binary": -1} | Minimum value in the support for each reward type. Both reward types have worst case of -1 (no improvement or incorrect submission). |
| `v_max` | dict[str, int] | {"pixel_wise": 1, "binary": 1} | Maximum value in the support for each reward type. Both reward types have best case of 1 (correct submission). |

**Reward Structure Analysis**:

The ARC environment produces two reward signals:

1. **Pixel-wise reward** [-1, 0]:
   - Computed as: `(improvement - 1) * (grid_diff != 0) + (grid_diff / grid_size²) * submission * (-max_penalty) + (grid_diff == 0) * submission`
   - With default `max_penalty=0.0`: worst case is -1 (no improvement on wrong grid)
   - Best case is 1 (correct submission) or 0 (perfect match without submission)
   - Actual range: [-1, 1] for most episodes
   - **Why v_min=-1, v_max=1**: Tight bounds match actual reward range without clipping

2. **Binary reward** [-1, 1]:
   - 1 if submission is correct
   - -1 if submission is incorrect
   - 0 for intermediate steps
   - **Why v_min=-1, v_max=1**: Tight bounds match actual reward range

3. **Atom counts**:
   - **Both reward types (21 atoms)**: Support range of 2 units ([-1, 1]) with 21 atoms = ~0.1 units per atom. Sufficient resolution for learning the discrete reward structure without over-parameterization.

**Impact**: Distributional RL learns the full distribution of returns, not just the mean. This provides:

- Better uncertainty estimates
- More stable learning
- Better handling of multimodal value distributions
- Proper support coverage prevents reward clipping and distribution collapse

---

## Replay Buffer Configuration (`ReplayBufferConfig`)

Configures the prioritized experience replay buffer.

| Parameter | Type | Default | Impact |
|-----------|------|---------|--------|
| `size` | int | 100000 | Maximum number of transitions stored. Larger buffers provide more diverse samples but increase memory usage. Smaller buffers reduce memory but may cause overfitting to recent data. |
| `alpha` | float | 0.2 | Prioritization exponent (0.0-1.0). Controls how much prioritization is used. Higher values (0.6) prioritize high TD-error samples more; lower values (0.0) use random sampling. |
| `beta` | float | 0.2 | Importance sampling exponent (0.0-1.0). Controls bias correction. Higher values correct more for prioritization bias. |
| `warmup_ratio` | float | 0.2 | Ratio of buffer to fill before training starts. Ensures diverse initial data. Typical range: 0.1-0.3. |

**Impact**: Prioritized replay focuses learning on important transitions (high TD-error), improving sample efficiency and convergence speed.

---

## Training Configuration (`TrainingConfig`)

Controls the training loop parameters.

| Parameter | Type | Default | Impact |
|-----------|------|---------|--------|
| `max_steps_per_epoch` | int | 1000 | Maximum environment steps per training epoch. Larger values provide more data per epoch but increase epoch duration. |
| `validation_epochs` | int | 1 | Number of validation epochs to run. Each validation epoch evaluates on validation data. |
| `validation_steps_per_epoch` | int | 100 | Maximum steps per validation epoch. Controls validation thoroughness vs. speed. |
| `validation_steps_frequency` | int | 100 | Run quick validation every N training steps. Provides frequent feedback without full validation. |
| `min_train_batch_size` | int | 2 | Minimum batch size for training. Ensures sufficient gradient information. Must be ≤ `batch_size`. |

**Impact**: These parameters control the validation strategy:

- Frequent validation (frequency=1) provides early stopping opportunities but slows training
- Infrequent validation (frequency=10) speeds training but may miss optimal checkpoints

---

## World Model Configuration (`WorldModelConfig`)

Configures optional Dyna-style imagination for model-based RL augmentation.

| Parameter | Type | Default | Impact |
|-----------|------|---------|--------|
| `enabled` | bool | False | Enable world model training and imagination. Disabled by default; set to True for model-based augmentation. |
| `learning_rate` | float | 1e-4 | Learning rate for the reward world model. Controls how quickly the model learns to predict rewards. |
| `lr_scheduler_kwargs` | dict \| None | None | Learning rate scheduler configuration for world model. Pass dict with `step_size` and `gamma` keys for StepLR scheduler. None disables scheduling. |
| `imagination_horizon` | int | 5 | Number of steps to imagine ahead. Longer horizons provide more lookahead but may accumulate errors. |
| `imagination_ratio` | int | 5 | Ratio of imagined to real transitions. Higher values use more imagined data, reducing real data dependency. |
| `imagination_frequency` | int | 10 | Frequency of imagination updates (every N steps). Lower values update more frequently. |
| `model_update_frequency` | int | 10 | Frequency of world model training (every N steps). Controls how often the model is updated. |

**Impact**: World model enables:

- Learning from imagined rollouts (Dyna-style)
- Improved sample efficiency
- Better generalization through synthetic data
- Increased computational cost

---

## Checkpoint Configuration (`CheckpointConfig`)

Controls checkpointing behavior.

| Parameter | Type | Default | Impact |
|-----------|------|---------|--------|
| `save_frequency` | int | 1000 | Save checkpoint every N training steps. Smaller intervals (100) provide more recovery points but use more disk space. Set to 0 to disable. |
| `experiment_path` | str \| Path | None | Directory to save checkpoints and logs. If None, creates timestamped directory. Useful for organizing multiple experiments. |

**Impact**: Checkpointing enables:

- Training resumption after interruption
- Tracking best models
- Experiment reproducibility

---

## System Configuration (`SystemConfig`)

Controls system-level parameters.

| Parameter | Type | Default | Impact |
|-----------|------|---------|--------|
| `amp_enabled` | bool | True | Enable automatic mixed precision (AMP). Reduces memory usage and speeds up training on compatible GPUs. May slightly reduce numerical precision. |
| `device` | str | None | Device to use ('cuda', 'cpu', 'cuda:0', etc.). If None, automatically selects best available device. |
| `seed` | int | 42 | Random seed for reproducibility. Same seed produces identical results (with deterministic algorithms). |

---

## Example Configurations

### Baseline Configuration

```python
from rlarcworld.common.configs import ArcD4PGConfig

config = ArcD4PGConfig(
    dataset="/path/to/arc/dataset",
)

experiment = ArcD4PG(config)
experiment.train(epochs=10)
```

### Larger Model Configuration

```python
config = ArcD4PGConfig(
    dataset="/path/to/arc/dataset",
    env=EnvConfig(grid_size=30, color_values=10, batch_size=64),
    network=RlArcWorldViTARCConfig.best_config(
        grid_size=30,
        use_temporal=True,
        use_examples=True,
    ),
    d4pg=D4PGConfig(
        policy_lr=1e-4,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.005,
    ),
    training=TrainingConfig(
        max_steps_per_epoch=2000,
    ),
)

experiment = ArcD4PG(config)
experiment.train(epochs=20)
```

### Sample Efficiency Configuration (with World Model)

```python
config = ArcD4PGConfig(
    dataset="/path/to/arc/dataset",
    env=EnvConfig(batch_size=32),
    d4pg=D4PGConfig(
        policy_lr=3e-4,
        critic_lr=3e-3,
        gamma=0.99,
        tau=0.005,
    ),
    replay_buffer=ReplayBufferConfig(
        size=200000,
        alpha=0.6,
        beta=0.4,
        warmup_ratio=0.1,
    ),
    world_model=WorldModelConfig(
        enabled=True,
        learning_rate=1e-4,
        imagination_horizon=5,
        imagination_ratio=5,
    ),
    training=TrainingConfig(
        max_steps_per_epoch=1000,
        validation_steps_frequency=50,
    ),
)

experiment = ArcD4PG(config)
experiment.train(epochs=50)
```

### Fast Experimentation Configuration

```python
config = ArcD4PGConfig(
    dataset="/path/to/arc/dataset",
    env=EnvConfig(grid_size=20, batch_size=16),
    network=RlArcWorldViTARCConfig.best_config(
        grid_size=20,
        use_temporal=False,  # Reduce model size
        use_examples=False,
    ),
    training=TrainingConfig(
        max_steps_per_epoch=500,
        validation_steps_frequency=200,
    ),
    checkpoint=CheckpointConfig(
        periodic_interval=500,
    ),
)

experiment = ArcD4PG(config)
experiment.train(epochs=5)
```

## Important Notes

1. **Dataset Path**: Must be provided and point to a valid ARC dataset directory.

2. **Grid Size Consistency**: `network.grid_size` must match `env.grid_size`.

3. **Color Values**: Must match the actual dataset color range (typically 0-9).

4. **Learning Rates**: Policy LR is typically 10x smaller than critic LR for stability.

5. **Gamma**: Values closer to 1.0 (0.99) work better for long-horizon tasks; lower values (0.95) for short-horizon.

6. **Batch Size**: Must be ≤ `replay_buffer.size` and ≥ `min_train_batch_size`.

7. **Validation Frequency**: More frequent validation helps catch overfitting early but slows training.

8. **Checkpoint Path**: If not specified, creates a timestamped directory in `./experiments/`.

9. **Reproducibility**: Set `system.seed` and use `system.device` for consistent results.

10. **Memory Management**: Reduce `batch_size`, `replay_buffer.size`, or `d_model` if running out of memory.
