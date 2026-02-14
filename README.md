# RlArcProject

This repository is a **modularized implementation for applying Reinforcement Learning (RL) algorithms to the Abstraction and Reasoning Corpus (ARC)**. It provides **Gymnasium-based environments** with a **pixel-level action space** (choose an `(x, y)` location, a `color`, and optionally `submit`) so agents can iteratively edit an output grid. Because the environments follow Gymnasium conventions and the stepping logic is separated into composable components, the system is designed to be **extended through wrappers** (reward shaping, curricula, alternative termination rules, etc.) and reused with additional RL algorithms beyond what is currently implemented.

## Why this repo

ARC is often approached as supervised program induction, but it can also be framed as **sequential decision-making**:

- **State**: the current grid(s), task examples, and step context.
- **Action**: a pixel-level edit (location + color) and a `submit` decision.
- **Reward**: shaped signals (pixel-wise improvement, penalties, submit reward) to guide exploration.

This repo focuses on making those pieces explicit and swappable.

## Key design

- **Environment is Gymnasium-first**
  - `rlarcworld.environments.arc_batch_grid_env.ArcBatchGridEnv` is a batched `gym.Env` for grid editing.
  - `rlarcworld.environments.arc_batch_grid_env.ArcActionSpace` defines a `gym.spaces.Dict` action space with:
    - `x_location`: discrete index in `[0, grid_size)`
    - `y_location`: discrete index in `[0, grid_size)`
    - `color_values`: discrete index in `[0, color_values)`
    - `submit`: discrete `{0, 1}`

- **Steppers separate “how to roll out episodes” from the algorithm**
  - `rlarcworld.environments.steppers.arc_stepper.PixelWiseArcStepper` ties together:
    - ARC dataset loading (`rlarcworld.arc_dataset`)
    - environment stepping (`ArcBatchGridEnv`)
    - optional augmentation and curriculum
  - This keeps the algorithm core environment-agnostic.

- **Wrappers are the primary extension mechanism**
  - `rlarcworld.environments.wrappers.*` contains pluggable wrappers (e.g. curriculum and reward-related wrappers).
  - Because the underlying env is Gymnasium-compatible, adding wrappers follows the standard RL pattern.

## Trainers and algorithms

This repo deliberately separates:

- **Algorithm-only logic ("Core")**
- **Training-loop orchestration ("Trainer")**
- **ARC-specific experiment wiring ("Experiment")**

### 1) Generic trainer (`Trainer`)

`rlarcworld.trainers.core.Trainer` is a lightweight orchestration layer that:

- pulls transitions from a `train_step_stream_factory`
- optionally stores/samples from a replay buffer
- calls the algorithm core to compute updates and sync targets
- triggers pluggable callbacks (logging, validation, checkpointing)

It is designed to be reusable across multiple off-policy actor-critic methods.

### 2) D4PG implementation (`D4PGCore`)

`rlarcworld.algorithms.d4pg.D4PGCore` is the **environment-agnostic** algorithm core for D4PG:

- actor/critic + target networks
- distributional critic updates (categorical projection)
- optional entropy regularization / stability helpers
- AMP support, gradient clipping, target soft-updates

### 3) ARC + D4PG experiment (`ArcD4PG`)

`rlarcworld.trainers.arc_d4pg.ArcD4PG` is the ARC-specific experiment orchestrator that wires:

- models (`rlarcworld.models.*`)
- steppers (`PixelWiseArcStepper` for training/validation)
- replay buffers (TorchRL replay buffer components)
- the generic `Trainer`
- checkpointing and TensorBoard callbacks

Most practical usage starts here.

## Setup [uv](https://docs.astral.sh/uv/)

This project uses `uv` for environment and dependency management.

```bash
uv venv
uv sync
uv pip install -e .
```

Dependencies that are **not included by default**:

- **Optional extra: `aws-sagemaker`**
  - Install when you need SageMaker tooling.
  - Example:

    ```bash
    uv sync --extra aws-sagemaker
    ```

- **Dependency group: `dev`**
  - Includes tooling like `pytest`, `pre-commit`, `mypy`, notebooks, etc.
  - Example:

    ```bash
    uv sync --group dev
    ```

## Running experiments

The main entry point is:

```bash
uv run python experiments/run_experiment.py --list
uv run python experiments/run_experiment.py --experiment <name> --epochs 10
```

Experiment configuration is YAML-driven:

- `experiments/configs/base.yaml` contains default values
- `experiments/configs/*.yaml` override subsets for specific sweeps/ablations

Important config groups:

- `env`: `grid_size`, `color_values`, penalties, batch size
- `d4pg`: learning rates, entropy knobs, target update frequencies, n-step settings
- `distributional`: atoms / support ranges
- `replay_buffer`, `training`, `checkpoint`, `logging`, `system`, `reproducibility`

## Installation

- Python: **3.12** (see `pyproject.toml`)
- Package: `rlarcworld`

Install with your preferred workflow (e.g. `pip install -e .` in a virtual environment). Dependencies are declared in `pyproject.toml`.

## Repository layout

- `rlarcworld/`
  - `environments/`
    - `arc_batch_grid_env.py`: Gymnasium env + pixel-level `Dict` action space
    - `steppers/`: rollouts/episode simulation
    - `wrappers/`: extensibility points (curriculum, reward shaping, etc.)
  - `algorithms/`: algorithm cores (e.g. `d4pg.py`)
  - `trainers/`: generic trainer + ARC-specific experiment runner(s)
  - `models/`: ViT-based actor/critic and supporting modules
  - `common/`: configs/enums/protocols
  - `utils/`: checkpointing, device helpers, config loader, queues
- `experiments/`
  - `run_experiment.py`: CLI to run YAML-configured experiments
  - `configs/`: experiment YAMLs
- `tests/`: unit + integration tests

## Extending to other RL algorithms

The easiest way to add a new algorithm is to follow the same three-layer split:

1. **Implement a new `*Core`** in `rlarcworld/algorithms/`
   - Keep it environment-agnostic.
   - Consume batches in the same transition format produced by steppers (`TensorDict` with keys like state/action/reward/next_state/terminated).
   - Expose the minimal API expected by `Trainer` (see protocols in `rlarcworld.common.protocols`).

2. **Reuse `Trainer`** in `rlarcworld/trainers/core.py`
   - Plug your core in.
   - Optionally plug in a replay buffer implementation and priority update function.

3. **Create an ARC experiment wrapper** (like `ArcD4PG`)
   - Wire networks, steppers, logging, checkpointing.
   - Add a config dataclass (see `rlarcworld.common.configs.ArcD4PGConfig`) and a YAML schema under `experiments/configs/`.

Because the ARC environment uses Gymnasium and the pixel action space is explicit, algorithms that operate over **discrete multi-head actions** (or that can be adapted to them) can be integrated without rewriting the environment.
