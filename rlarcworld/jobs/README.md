# Training Jobs

This directory contains the main training scripts and configurations for the RL-ARC project.

## Files

### `train.py`

The main training script that implements the D4PG algorithm for the ARC environment.

#### Usage

```bash
python train.py \
    --config_file rlarcworld/jobs/config.yaml \
    --config_key default \
    --log_level INFO \
    --log_convention json \
    --storage_uri /path/to/storage \
    --training_data_uri /path/to/training/data \
    --validation_data_uri /path/to/validation/data \
    --models_dir /path/to/save/models \
    --checkpoint_dir /path/to/save/checkpoints \
    --tensorboard_log_dir /path/for/tensorboard/logs \
    --checkpoint_frequency 10000
```

#### Arguments

- `--config_file`: Path to the YAML configuration file (default: 'rlarcworld/jobs/config.yaml')
- `--config_key`: Key for the configuration in the YAML file (default: 'test')
- `--log_level`: Logging level (e.g., 'INFO', 'WARNING', 'ERROR')
- `--log_convention`: Logging format convention. Supported values:
  - `json`: Output logs in JSON format (default)
  - `text`: Output logs in plain text format
- `--storage_uri`: Base storage URI (used as fallback for other paths)
- `--training_data_uri`: Path to the training data directory
- `--validation_data_uri`: Path to the validation data directory
- `--models_dir`: Directory to save trained models (default: '${storage_uri}/models')
- `--checkpoint_dir`: Directory to save training checkpoints (default: '${storage_uri}/checkpoints')
- `--tensorboard_log_dir`: Directory for TensorBoard logs (default: '${storage_uri}/tensorboard_logs')
- `--checkpoint_frequency`: How often to save checkpoints (in steps, default: 10000)

#### Environment Variables

All arguments can also be set via environment variables:

- `STORAGE_URI` → `--storage_uri`
- `AIP_TRAINING_DATA_URI` → `--training_data_uri`
- `AIP_VALIDATION_DATA_URI` → `--validation_data_uri`
- `AIP_MODEL_DIR` → `--models_dir`
- `AIP_CHECKPOINT_DIR` → `--checkpoint_dir`
- `AIP_TENSORBOARD_LOG_DIR` → `--tensorboard_log_dir`

#### Example: Running with Environment Variables

```bash
export STORAGE_URI=/path/to/storage
export AIP_TRAINING_DATA_URI=${STORAGE_URI}/training_data
export AIP_VALIDATION_DATA_URI=${STORAGE_URI}/validation_data
export AIP_MODEL_DIR=${STORAGE_URI}/models
export AIP_CHECKPOINT_DIR=${STORAGE_URI}/checkpoints
export AIP_TENSORBOARD_LOG_DIR=${STORAGE_URI}/tensorboard_logs

python train.py \
    --config_key default \
    --log_level INFO \
    --checkpoint_frequency 5000
```

### `config.yaml`

Configuration file containing hyperparameters and training settings. Multiple configurations can be defined and selected using the `--config_key` argument.

#### File Structure

The configuration file uses YAML format with a hierarchical structure:

- Each top-level key represents a named configuration
- Configurations can inherit from others using YAML anchors (`&default` and `<<: *default`)
- The `--config_key` parameter selects which configuration to use

#### Example Configuration

```yaml
default: &default
  # Environment
  batch_size: 128
  grid_size: 30
  color_values: 11
  
  # Training
  max_epochs: 1
  max_steps: 1000000
  validation_steps_frequency: 1000
  
  # D4PG Hyperparameters
  policy_lr: 0.0001
  critic_lr: 0.001
  gamma: 0.99
  
  # Replay Buffer
  replay_buffer_size: 1000000
  warmup_buffer_ratio: 0.1

# Additional configurations can inherit from default using YAML anchors
test: 
  <<: *default
  batch_size: 5
  max_steps: 20
  validation_steps_frequency: 5
```
