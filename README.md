# Reinforcement Learning for ARC (Abstraction and Reasoning Corpus)

This project implements a Deep Reinforcement Learning (RL) approach to solve tasks from the Abstraction and Reasoning Corpus (ARC) challenge. It uses the Distributional Deep Deterministic Policy Gradient (D4PG) algorithm with specialized neural network architectures for the ARC environment.

## Table of Contents

- [Infrastructure](#infrastructure)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Training](#training)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Dataset](#dataset)
- [Cloud Integration](#cloud-integration)


## Infrastructure

The project is designed to run in containerized environments with support for GPU acceleration:

- **Containerization**: Docker support with GPU capabilities
- **Cloud Integration**: Compatible with GCP Vertex AI and Cloud Storage
- **Environment**: Python 3.x with PyTorch and TorchRL
- **Monitoring**: TensorBoard integration for experiment tracking

Key components:

- `devops/`: Contains Docker configuration and deployment scripts
  - `Dockerfile`: Base image with all dependencies
  - `dev.docker-compose.yml`: Local development setup
  - `.env`: Environment configuration

## Installation

### Prerequisites

- Docker and Docker Compose
- NVIDIA Container Toolkit (for GPU support)
- Python 3.8+

### Setup

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd RlArcProject
   ```

2. Build the Docker image:

   ```bash
   docker-compose -f devops/dev.docker-compose.yml build
   ```

3. Start the development container:

   ```bash
   docker-compose -f devops/dev.docker-compose.yml up -d
   ```

## Project Structure

```text
RlArcProject/
├── rlarcworld/              # Main package
│   ├── agent/               # Agent implementations
│   ├── algorithms/          # RL algorithms (D4PG)
│   ├── dataset/             # Data loading and processing
│   ├── environments/        # ARC environment and wrappers
│   └── jobs/                # Training and evaluation scripts
│       ├── config.yaml      # Configuration templates
│       └── train.py         # Main training script
├── tests/                   # Unit and integration tests
├── devops/                  # Deployment and infrastructure
└── runs/                    # Training outputs and checkpoints
```

## Configuration

The project uses YAML configuration files to manage hyperparameters and experiment settings. Key configurations include:

- Environment parameters (grid size, color values)
- D4PG hyperparameters (learning rates, gamma, etc.)
- Training settings (batch sizes, buffer sizes)
- Logging and validation frequencies

Example configuration can be found in `rlarcworld/jobs/config.yaml`.

## Training

To train the model:

```bash
python -m rlarcworld.jobs.train \
    --config-key default \
    --training-data-uri /path/to/training/data \
    --validation-data-uri /path/to/validation/data \
    --models-dir /path/to/save/models \
    --checkpoint-dir /path/to/save/checkpoints \
    --tensorboard-log-dir /path/for/tensorboard/logs
```

## Evaluation

The model can be evaluated using the validation set during training. The evaluation metrics are logged to TensorBoard for monitoring.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The ARC Challenge by François Chollet
- PyTorch and TorchRL communities
- Google Cloud Platform

## Dataset

For development purposes, the test set consists of 80 files (20%) of the evaluation set (320 files) and the training set is composed of 400 files. The IDs of the files used are in [dataset_ids.txt](dataset_ids.txt).

## Cloud Integration

- [GCP Cloud Storage Convention](https://cloud.google.com/vertex-ai/docs/training/code-requirements#fuse)
- [Environment variables for special Cloud Storage directories](https://cloud.google.com/vertex-ai/docs/training/code-requirements#environment-variables)