# DevOps Configuration

This directory contains the infrastructure and deployment configurations for the RL-ARC project.

## Files

### `Dockerfile`

Defines the container image for running the training jobs. Based on `pytorch-gpu.2-3.py310` for GPU support.

### `dev.docker-compose.yml`

Docker Compose configuration for local development and testing.

#### Usage

```bash
# Build the container
docker-compose -f dev.docker-compose.yml build

# Start the container
docker-compose -f dev.docker-compose.yml up -d

# Access the container
docker exec -it rl_train_job bash
```

#### Enabling GPU Support

To enable GPU support, modify `dev.docker-compose.yml` to include:

```yaml
services:
  rl_train_job:
    # ... existing configuration ...
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### `.env`

Environment variables for the application. Copy `.env.example` to `.env` and update the values as needed.

### `main.py`

Entry point script for the containerized application when running on Google Cloud's Vertex AI. This script handles:

- **Vertex AI Initialization**: Configures the Vertex AI environment with project settings and authentication
- **Experiment Tracking**: Sets up experiment tracking with a name based on the provided config key
- **Environment Setup**: Manages environment variables and service account credentials
- **Configuration Loading**: Processes command-line arguments for config file and key

**Command Line Arguments**:

- `--config_file`: Path to the YAML configuration file
- `--config_key`: Key specifying which configuration to use from the YAML file
- `--log_level`: Sets the logging verbosity level (e.g., 'INFO', 'WARNING')

**Required Environment Variables**:

- `GCP_TRAIN_IMAGE`: Container image URI for the training job
- `SERVICE_ACCOUNT`: Service account to run the job as
- `PROJECT_ID`: GCP project ID
- `STORAGE_URI`: Cloud Storage bucket for staging and outputs
- `REGION`: GCP region (defaults to 'us-central1')

This script is automatically used as the entry point when the container starts in the Google Cloud environment.

## Local Development Setup

1. Ensure Docker and Docker Compose are installed
2. Install NVIDIA Container Toolkit for GPU support
3. Copy `.env.example` to `.env` and configure:

   ```bash
   # Example .env
   TRAINING_DATA_URI=/path/to/training/data
   VALIDATION_DATA_URI=/path/to/validation/data
   MODELS_DIR=/path/to/save/models
   CHECKPOINT_DIR=/path/to/save/checkpoints
   TENSORBOARD_LOG_DIR=/path/for/tensorboard/logs
   ```

4. Build and start the container:

   ```bash
   docker-compose -f dev.docker-compose.yml build
   docker-compose -f dev.docker-compose.yml up -d
   ```

5. Monitor training:
   - View logs: `docker logs -f rl_train_job`
   - Access TensorBoard at `http://localhost:6006`

## Troubleshooting

- **GPU not detected**: Ensure NVIDIA drivers and the NVIDIA Container Toolkit are properly installed
- **Permission issues**: Make sure the current user has access to the Docker daemon
- **Port conflicts**: Check if port 6006 is already in use for TensorBoard
