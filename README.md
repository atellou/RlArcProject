# Reinforcement Learning for ARC (Abstraction and Reasoning Corpus)

This project implements a Deep Reinforcement Learning (RL) approach to solve tasks from the Abstraction and Reasoning Corpus (ARC) challenge. It uses the Distributional Deep Deterministic Policy Gradient (D4PG) algorithm with specialized neural network architectures for the ARC environment.

## Quickstart

### Local

1. Install (Python 3.12)

   ```bash
   python -m pip install -e .
   ```

2. List available experiments

   ```bash
   python experiments/run_experiment.py --list
   ```

3. Run an experiment

   ```bash
   python experiments/run_experiment.py \
     --experiment stable_baseline \
     --epochs 10 \
     --config experiments/configs/experiments.yaml \
     --base experiments/configs/base.yaml
   ```

4. Override dataset path (optional)

   ```bash
   python experiments/run_experiment.py \
     --experiment stable_baseline \
     --epochs 10 \
     --dataset /path/to/arc/dataset
   ```

5. Healthcheck (quick checkpoint save/load test)

   ```bash
   python experiments/run_experiment.py --healthcheck
   ```

### SageMaker

1. Install SageMaker dependencies locally

   ```bash
   python -m pip install -e ".[aws-sagemaker]"
   ```

2. Upload a dataset tarball to S3

   SageMaker expects a single tarball in the train channel. You can create one from a text file listing JSON samples:

   - `dataset_ids.txt` in the repo is an example of a sample list file.

3. Launch a job

   ```bash
   python aws_sagemaker/launch.py \
     --role-arn arn:aws:iam::<acct>:role/SageMakerExecRole \
     --s3-train-tar s3://<bucket>/datasets/arc_train.tar.gz \
     --experiment lr_3e4_nstep_1 \
     --instance-type ml.g5.12xlarge \
     --epochs 10
   ```

   - The container entrypoint is `aws_sagemaker/train.py`.
   - SageMaker checkpoints are synced via `/opt/ml/checkpoints`.
   - Logs stream to CloudWatch under `/aws/sagemaker/TrainingJobs`.

For more details, see `aws_sagemaker/README.md`.

## Table of Contents

- [Quickstart](#quickstart)
  - [Local](#local)
  - [SageMaker](#sagemaker)
- [Infrastructure](#infrastructure)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Setup](#setup)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Training](#training)
- [Monitoring with TensorBoard](#monitoring-with-tensorboard)
  - [Starting TensorBoard](#starting-tensorboard)
  - [Key Metrics Tracked](#key-metrics-tracked)
  - [Viewing Specific Runs](#viewing-specific-runs)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Dataset](#dataset)
- [Cloud Integration](#cloud-integration)

## Infrastructure

TheProject is designed to run in containerized environments with support for GPU acceleration:

- **Containerization**: Docker support with GPU capabilities
- **Cloud Integration**: Compatible with GCP Vertex AI and Cloud Storage
- **Environment**: Python 3.x with PyTorch and TorchRL
- **Monitoring**: TensorBoard integration for experiment tracking

Key components:

- `aws_sagemaker/`: SageMaker launcher, entrypoint, and container image
- `experiments/`: Experiment runner and YAML configs
- `rlarcworld/`: Core library (env, models, algorithms, trainers)

## Installation

### Prerequisites

- Python 3.12
- (Optional) NVIDIA GPU + CUDA for faster training
- (Optional) Docker for building the SageMaker training image (`aws_sagemaker/Dockerfile`)

### Setup

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd RlArcProject
   ```

2. Install the package (editable mode):

   ```bash
   python -m pip install -e .
   ```

3. (Optional) Install SageMaker extras for launching jobs:

   ```bash
   python -m pip install -e ".[aws-sagemaker]"
   ```

## Project Structure

```text
RlArcProject/
├── rlarcworld/              # Core library (env/models/algorithms/trainers/logging)
├── experiments/             # Local experiment runner + YAML configs
│   ├── run_experiment.py
│   └── configs/
│       ├── base.yaml
│       └── experiments.yaml
├── aws_sagemaker/            # SageMaker launcher + training entrypoint + Dockerfile
│   ├── launch.py
│   ├── train.py
│   └── lr_experiment/        # Example SageMaker experiment configs
├── checkpoints/              # Default local checkpoint root (runs create subfolders)
├── logs/                     # Default local logs
└── tests/                    # Unit/integration tests
```

## Configuration

The project uses YAML configuration files to manage hyperparameter and experiment settings. Key configurations include:

- Environment parameter (grid size, color values)
- D4PG hyperparameter (learning rates, gamma, etc.)
- Training setting (batch sizes, buffer sizes)
- Logging and validation frequency

Example configurations:

- `experiments/configs/base.yaml`
- `experiments/configs/experiments.yaml`
- `aws_sagemaker/lr_experiment/cloud_base.yaml`
- `aws_sagemaker/lr_experiment/experiments.yaml`

## Training

To train the model:

```bash
python experiments/run_experiment.py \
  --experiment stable_baseline \
  --epochs 10
```

## Monitoring with TensorBoard

This project uses TensorBoard for tracking and visualizing training progress, including metrics, model graphs, and other useful information.

### Starting TensorBoard

1. **Local Installation**:

   If you have TensorBoard installed locally, you can start it with:

   ```bash
   tensorboard --logdir=/path/to/tensorboard/logs
   ```

   Then open your browser to `http://localhost:6006/`

2. **Docker Container**:

   If you're using the provided Docker setup, you can access TensorBoard by exposing the port in your `docker-compose.yml`:

   ```yaml
   ports:
     - "6006:6006"
   ```

   Then start TensorBoard inside the container:

   ```bash
   tensorboard --logdir=/path/to/tensorboard/logs --host=0.0.0.0
   ```

   Access it at `http://localhost:6006/` from your host machine.

### Key Metrics Tracked

- **Training Loss**: Actor and critic losses over time
- **Reward**: Training and validation rewards per episode
- **Hyperparameter**: Learning rates and other training parameters
- **Gradient**: Histograms of model gradients
- **Model Graph**: Computation graphs of your models

### Viewing Specific Runs

If you have multiple runs, you can compare them by selecting which runs to display in the TensorBoard interface. The runs are automatically organized by their timestamp and configuration.

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
