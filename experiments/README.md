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
    logging=LoggingConfig(...),
    reproducibility=ReproducibilityConfig(...),
    system=SystemConfig(...),
)

experiment = ArcD4PG(config)
experiment.train(epochs=10)
```

---

## Environment Configuration (`EnvConfig`)

Controls the ARC environment parameters.

| Parameter | Type | Default | Impact |
| --------- | ---- | ------- | ------ |
| `grid_size` | int | 30 | Size of the input/output grid (height = width). Larger grids increase computational complexity and memory usage. |
| `color_values` | int | 11 | Number of distinct color values in the environment (upper bound). Must match the dataset encoding. Affects network output dimensions. |
| `wrong_submit_penalty` | float | 1.0 | Penalty applied when submitting an incorrect grid. Also used to scale distributional support via `DistributionalConfig.scale_by_penalty()`. |
| `step_cost` | float | 0.0 | Optional per-step cost to encourage shorter solutions. |
| `perfect_no_submit_penalty_coef` | float | 1.0 | Penalty factor applied when the grid is already perfect but the agent does not submit. |
| `zero_improvement_penalty` | float | 0.0 | Optional penalty applied when an action does not improve the grid. |
| `shaping_clamp_min` | float | 0.0 | Lower clamp for shaping reward (useful to avoid overly negative shaping early). |
| `shaping_clamp_min_tau_steps` | int | 0 | Optional schedule length (in steps) for the shaping clamp behavior. |
| `batch_size` | int | 32 | Number of samples processed per environment step (batched env/stepper). Larger batches improve gradient stability but increase memory usage. |

---

## Network Configuration (`RlArcWorldViTARCConfig`)

Configures the ViTARC encoder used by both actor and critic networks.

| Parameter | Type | Default | Impact |
| --------- | ---- | ------- | ------ |
| `grid_size` | int | Required | Must match `EnvConfig.grid_size`. Size of input grids. |
| `d_model` | int | 128 | Embedding dimension for transformer layers. Larger values increase model capacity and memory usage. Controls the width of hidden representations. |
| `d_ff` | int | 512 | Feed-forward network dimension in transformer blocks. Typically 4x `d_model`. Larger values increase model capacity. |
| `num_layers` | int | 4 | Number of transformer encoder layers. More layers increase model depth and capacity but slow down training. Affects ability to learn complex spatial relationships. |
| `num_heads` | int | 8 | Number of attention heads in multi-head attention. Must divide `d_model` evenly. More heads allow parallel attention patterns but increase computation. |
| `dropout` | float | 0.1 | Dropout rate for regularization (0.0-1.0). Higher values increase regularization, preventing overfitting but may reduce learning speed. |
| `ape_type` | str \| None | None | Absolute position encoding type. Common choice: `"SinusoidalAPE2D"`. |
| `rpe_type` | str \| None | None | Relative position encoding type. Common choice: `"Two-slope-Alibi"`. |
| `use_ope` | bool \| None | None | Enable object position encoding. Common choice: `true`. |
| `ape_mixer` | str \| None | None | Method for mixing absolute position encodings. Common choice: `"weighted_sum_no_norm_vec"`. |
| `trainable_ape` | bool | False | Whether absolute position embeddings are trainable. Set to True for task-specific position learning. |
| `use_index_grid` | bool | True | Include index/timestamp grid in encoding. Helps model track temporal progression. |
| `use_temporal` | bool | True | Include temporal grids (current, last, initial). Enables sequence modeling. |
| `use_examples` | bool | True | Include example grids in the input. Enables few-shot learning capabilities. |
| `example_compression_tokens` | int | 64 | Tokens used to compress each example (when examples are enabled). |
| `grid_compression_tokens` | int \| None | 128 | Tokens used to compress the main grid (None disables compression). |
| `compression_method` | str | "learned" | Compression method: `"learned"` or `"pooling"`. |
| `max_learnable_examples` | int | 1 | Max number of examples handled by the learned example compressor. |
| `color_values` | int \| None | None | If set, must match `EnvConfig.color_values`. Often inferred from env/config. |

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
| --------- | ---- | ------- | ------ |
| `share_network` | bool | True | Whether actor and critic share a common encoder/trunk. |
| `policy_lr` | float | 3e-4 | Learning rate for the actor (policy) network. Higher values speed up learning but may cause instability. Lower values provide stability but slower convergence. |
| `critic_lr` | float | 3e-3 | Learning rate for the critic (value) network. Typically 10x higher than policy_lr. Affects how quickly value estimates improve. |
| `gamma` | float | 0.99 | Discount factor (0.0-1.0). Controls how much future rewards are valued. Higher values (0.99) emphasize long-term rewards; lower values (0.9) emphasize immediate rewards. |
| `tau` | float | 0.005 | Soft update parameter for target networks. Controls the rate of target network updates. Lower values (0.005) provide stability; higher values (0.01) allow faster adaptation. |
| `lr_scheduler_kwargs` | dict \| None | None | Learning rate scheduler configuration. Pass dict with `step_size` and `gamma` keys for StepLR scheduler. Example: `{"step_size": 1000, "gamma": 0.5}`. None disables scheduling. |
| `n_step_config` | NStepMinVMaxConfig | n_steps=1 | N-step return configuration (see below). |

### Advanced D4PG Parameters

These parameters are passed to the D4PGCore algorithm and control advanced training dynamics:

| Parameter | Type | Default | Impact |
| --------- | ---- | ------- | ------ |
| `compute_td_error` | bool | True | Whether to compute TD error for prioritized replay. Disabling saves computation but loses priority information. |
| `carsm` | bool | False | Clipped Action Regularization for Soft-Modulus. Experimental regularization technique for policy stability. |
| `entropy_coef` | float | 0.0 | Entropy regularization coefficient. Higher values (0.01-0.1) encourage exploration; 0.0 disables entropy regularization. |
| `entropy_coef_decay` | float | 1.0 | Decay rate for entropy coefficient per step. Values < 1.0 gradually reduce exploration over time. |
| `entropy_floor_coef` | float | 0.0 | Coefficient for a penalty when entropy drops below `min_entropy` (helps prevent entropy collapse). |
| `min_entropy` | float | 0.01 | Minimum entropy target used by the entropy-floor penalty. |
| `submit_entropy_scale` | float | 1.0 | Scale factor applied to the submit head contribution in entropy regularization. |
| `color_entropy_scale` | float | 1.0 | Scale factor applied to the color head contribution in entropy regularization. |
| `target_update_frequency` | int | 10 | Frequency of target network updates (every N steps). Higher values (20-50) provide stability; lower values (5) allow faster adaptation. |
| `model_update_frequency` | int | 1 | Update frequency for model weights (e.g. accumulate steps and update every N steps). Values > 1 can reduce compute cost. |
| `cache_enabled` | bool | False | Enables example caching, typically used when `model_update_frequency > 1`. |
| `max_grad_norm` | float | 0.0 | Maximum gradient norm for clipping. 0.0 disables clipping; typical values: 1.0-10.0 for stability. |

### N-Step Configuration (`NStepMinVMaxConfig`)

| Parameter | Type | Default | Impact |
| --------- | ---- | ------- | ------ |
| `n_steps` | int | 1 | Number of steps for n-step returns. Higher values (3-5) reduce variance but increase bias. n_steps=1 is standard TD learning. |
| `pixel_wise_min_v` | float | None | Minimum value for pixel-wise reward distribution. Used for distributional RL value clipping. |
| `pixel_wise_max_v` | float | None | Maximum value for pixel-wise reward distribution. Used for distributional RL value clipping. |
| `binary_min_v` | float | None | Minimum value for binary reward distribution. |
| `binary_max_v` | float | None | Maximum value for binary reward distribution. |

---

## Distributional RL Configuration (`DistributionalConfig`)

Configures the distributional critic for learning value distributions instead of point estimates.

| Parameter | Type | Default | Impact |
| --------- | ---- | ------- | ------ |
| `num_atoms` | dict[str, int] | {"pixel_wise": 21, "binary": 21} | Number of atoms (support points) for each reward type. More atoms provide finer value distribution resolution but increase computation. Both use 21 atoms for symmetric [-1, 1] support. |
| `v_min` | dict[str, int] | {"pixel_wise": -1, "binary": -1} | Minimum value in the support for each reward type. Both reward types have worst case of -1 (no improvement or incorrect submission). |
| `v_max` | dict[str, int] | {"pixel_wise": 1, "binary": 1} | Maximum value in the support for each reward type. Both reward types have best case of 1 (correct submission). |

**Reward Structure Analysis**:

The ARC environment produces two reward signals:

1. **Pixel-wise reward** [-1, 0]:
   - Computed as: `(improvement - 1) * (grid_diff != 0) + (grid_diff / grid_size²) * submission * (-wrong_submit_penalty) + (grid_diff == 0) * submission`
   - With default `wrong_submit_penalty=0.0`: worst case is -1 (no improvement on wrong grid)
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
| --------- | ---- | ------- | ------ |
| `size` | int | 100000 | Maximum number of transitions stored. Larger buffers provide more diverse samples but increase memory usage. Smaller buffers reduce memory but may cause overfitting to recent data. |
| `alpha` | float | 0.6 | Prioritization exponent (0.0-1.0). Controls how much prioritization is used. Higher values prioritize high TD-error samples more; lower values use random sampling. |
| `beta` | float | 0.4 | Importance sampling exponent (0.0-1.0). Controls bias correction. Higher values correct more for prioritization bias. |
| `warmup_ratio` | float | 0.02 | Ratio of buffer to fill before training starts. Ensures diverse initial data. |

**Impact**: Prioritized replay focuses learning on important transitions (high TD-error), improving sample efficiency and convergence speed.

---

## Training Configuration (`TrainingConfig`)

Controls the training loop parameters.

| Parameter | Type | Default | Impact |
| --------- | ---- | ------- | ------ |
| `max_steps_per_epoch` | int | 10000 | Maximum environment steps per training epoch. Larger values provide more data per epoch but increase epoch duration. |
| `max_steps_per_episode` | int | 900 | Maximum steps per episode before forcing termination/reset. |
| `validation_epochs` | int | 1 | Number of validation epochs to run when validation is triggered. |
| `validation_steps_per_epoch` | int | 100 | Steps to run for each validation epoch. |
| `validation_steps_per_episode` | int | 100 | Maximum episode length during validation rollouts. |
| `validation_steps_frequency` | int | 100 | Run validation every N training steps. |
| `min_train_batch_size` | int | 2 | Minimum batch size for training updates. Must be ≤ `env.batch_size`. |
| `augmentation_enabled` | bool | False | Enable dataset augmentation in the training stepper. |
| `enable_curriculum` | bool | False | Enable curriculum wrapper around the environment stepper. |
| `curriculum_replace_probabilities` | object | see below | Replace probabilities used by the curriculum across phases (warmup/episode start/train). |
| `curriculum_pixel_location_strategy` | str | "random" | Strategy used by curriculum when selecting which pixel to act on (`random`, `top_left`, `bottom_right`). |

### Curriculum replace probabilities (`CurriculumReplaceProbabilitiesConfig`)

| Parameter | Type | Default | Impact |
| --------- | ---- | ------- | ------ |
| `on_warmup` | float | 1.0 | Replace probability while the replay buffer is warming up. |
| `on_episode_start` | float | 1.0 | Replace probability during the initial steps of each episode. |
| `on_train` | float | 0.0 | Replace probability during normal training (after warmup and episode start). |
| `episode_start_steps` | int | 100 | Number of initial episode steps considered “episode start”. |

**Impact**: These parameters control the validation strategy:

- Frequent validation (frequency=1) provides early stopping opportunities but slows training
- Infrequent validation (frequency=10) speeds training but may miss optimal checkpoints

---

## World Model Configuration (`WorldModelConfig`)

Configures optional Dyna-style imagination for model-based RL augmentation.

| Parameter | Type | Default | Impact |
| --------- | ---- | ------- | ------ |
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
| --------- | ---- | ------- | ------ |
| `save_frequency` | int | -1 | Save checkpoint every N training steps. Set to `-1` to disable periodic saves. |
| `checkpoint_dir` | str \| Path \| None | None | Directory to save checkpoints and logs. If None, experiment code creates a timestamped directory. |
| `improvement_checkpoint_enabled` | bool | False | If enabled, save checkpoints when validation improves (policy-performance based metrics). |

---

## Logging Configuration (`LoggingConfig`)

Controls console/TensorBoard logging behavior.

| Parameter | Type | Default | Impact |
| --------- | ---- | ------- | ------ |
| `tensorboard_enabled` | bool | False | Enable TensorBoard logging/callbacks. |
| `log_frequency` | int | 100 | Console/logging frequency in training steps. |
| `tb_minimal_frequency` | int | 1 | TensorBoard minimal logging frequency. |
| `tb_medium_frequency` | int | 100 | TensorBoard medium logging frequency. |

**Impact**: Checkpointing enables:

- Training resumption after interruption
- Tracking best models
- Experiment reproducibility

---

## System Configuration (`SystemConfig`)

Controls system-level parameters.

| Parameter | Type | Default | Impact |
| --------- | ---- | ------- | ------ |
| `amp_enabled` | bool | True | Enable automatic mixed precision (AMP). Reduces memory usage and speeds up training on compatible GPUs. May slightly reduce numerical precision. |
| `use_checkpointing` | bool | False | Enable activation checkpointing (trades compute for memory in large models). |
| `device` | str \| None | None | Device to use ('cuda', 'cpu', 'cuda:0', etc.). If None, automatically selects best available device. |

---

## Reproducibility Configuration (`ReproducibilityConfig`)

Controls RNG seeds and deterministic flags.

| Parameter | Type | Default | Impact |
| --------- | ---- | ------- | ------ |
| `global_seed` | int \| None | None | Global seed applied to Python/NumPy/Torch when set. |
| `dataset_seed` | int \| None | None | Seed used for dataset shuffling (can be set separately from `global_seed`). |
| `augmentation_seed` | int \| None | None | Seed for deterministic data augmentation. |
| `torch_deterministic` | bool | True | Enables deterministic Torch algorithms (may reduce performance). |

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
        save_frequency=500,
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

9. **Reproducibility**: Set `reproducibility.global_seed` (and optionally `dataset_seed` / `augmentation_seed`) for consistent results.

10. **Memory Management**: Reduce `batch_size`, `replay_buffer.size`, or `d_model` if running out of memory.
