# RLArcProject TODOs: Utils Refactor Follow-ups

This document tracks follow-ups and improvements for the recent `rlarcworld/utils/` refactor.

## High Priority

We will not maintain backward compatibility. Update code and tests to the new layout and merge when green.

### rlarcworld

- [x] Check D4PG trainer implementation and update to be able to run multiple algorithms.
- [x] Implement inference for core algorithms to use on testing or validation environment.
- [ ] Review checkpoints and create tests.

### Experiments

- [ ] Implement TensorBoard logging for experiment callbacks.
- [ ] Update jobs to a separate structure that allows the resources in rlarcworld to be used in the jobs. Jobs is in reality a space to define the experiments to run.
  - [x] Use Trainer to train the algorithm.
    - [ ] Use priority replay buffer.
  - [ ] Use checkpointing to save the algorithm state.
    - [ ] Define on Step checkpointing.
    - [ ] Define on Train checkpointing.
    - [ ] Define on Exception checkpointing.
  - [x] Use logging to log the algorithm state.
  - [x] Use ArcDataset to load the training dataset.
    - [x] Use ArcSampleTransformer to transform the dataset.
  - [x] Use ArcBatchGridEnv to create the environment.
    - [x] Use EnvStepper to create the stepper.
  - [x] Use ArcActorNetwork and ArcCriticNetwork to create the networks.
  - [x] Use D4PGCore to create the core.
  - [x] Setup Validation procedures.
    - [x] Use ArcDataset to load the validation dataset.
    - [x] Use ArcBatchGridEnv to create the validation environment.
    - [x] Use EnvStepper to create the validation stepper.

### devops

- [ ] Update divide structure defined to allow the jobs to be run in on cloud resources.

## Medium Priority

- [x] BetaScheduler robustness (`rlarcworld/utils/schedulers.py`)

  - [x] Support end < start
  - [x] Use `internal_step / steps` instead of `/ end` in `step()`
  - [x] Clamp progress: `min(1.0, self.internal_step / max(self.steps, 1))` to avoid tiny overshoots

- [x] AMP autocast usage (`rlarcworld/environments/stepper.py`)

  - [x] Do not pass `None` to `torch.amp.autocast`; use conditional or `nullcontext`
  - [x] Consider consolidating with a shared AMP utility/context if needed
  - [x] Provide a small testing utility to generate device-aware synthetic steps to reduce duplication in AMP tests

## Test Coverage

- [x] Add/extend unit tests under `tests/` for utils

  - [x] `schedulers.BetaScheduler`: monotonic increase, saturation at `end`, state_dict/load roundtrip
  - [x] `device.enable_cuda`: mock `torch.cuda.is_available()` and device capability to validate fields
  - [x] `distributions.categorical_projection`: shape checks, probability mass preservation (sum≈1), atoms alignment
  - [x] `dict_utils.get_nested_ref`: nested creation, default callable behavior
  - [x] `logging_utils.configure_logger`: single handler after multiple calls

## Nice-to-Have

- [x] Add type annotations and docstrings consistency across all modules
  - [x] Utils
  - [x] Environments
  - [x] Agent
  - [x] Algorithms
- [x] Centralize constants or enums (e.g., logger formats) if reused
  - [x] Utils
  - [x] Environments
  - [x] Agent
  - [x] Algorithms

## Pytest Migration Plan

- [x] Restructure tests into focused pytest modules (no monolithic test_algorithms)

  - [x] `tests/test_d4pg_core_compute.py` – math-only tests for `D4PGCore`
  - [x] `tests/test_d4pg_trainer.py` – trainer loop with `D4PGTrainer` (+ PER warmup, AMP, grad clipping, target updates)
  - [ ] `tests/test_checkpointing.py` – re-scoped checkpoint/state dict validation
  - [x] `tests/test_env_stepper.py` – step dict keys/shapes; AMP vs non-AMP
  - [x] `tests/test_schedulers.py` – `BetaScheduler` behavior
  - [x] `tests/test_logging_utils.py` – logger handler dedupe
  - [x] `tests/test_torch_queue.py` – keep/convert existing tests to pytest
  - [x] `tests/test_distributions.py` – `categorical_projection` (shapes, normalization, n-steps)
  - [x] `tests/test_arc_dataset.py` – Evaluates the dataset loading and transformations
  - [x] `tests/test_arc_environment.py` – Evaluates the env progress
  - [x] `tests/test_networks.py` – Evaluates the networks

- [x] Introduce pytest fixtures in `tests/conftest.py`

  - [x] `device`, `grid_size`, `color_values`, `batch_size`
  - [x] `env` and wrapped env, parametrized over `n_steps`
  - [x] `dataset`, `train_loader`, `val_loader`
  - [x] `actor`, `critic` with consistent `v_min`/`v_max`/`num_atoms`
  - [x] `d4pg_core` (parametrize `amp_enabled`)
  - [x] `trainer`

  - [x] Add `make_synthetic_step_on_device(device)` helper to avoid CPU/CUDA mismatches in specialized tests (e.g., AMP)

- [ ] Parametrized tests

  - [x] `categorical_projection` (shapes, normalization)
  - [x] `BetaScheduler` (increasing, clamping)
  - [x] AMP/no-AMP code paths where relevant
- [x] Pre-commit hooks
- [x] Linters for import validity and formatting

---
