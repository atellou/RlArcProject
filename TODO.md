# RLArcProject TODOs: Utils Refactor Follow-ups

This document tracks follow-ups and improvements for the recent `rlarcworld/utils/` refactor.

## High Priority

We will not maintain backward compatibility. Update code and tests to the new layout and merge when green.

- [ ] Update all imports to new utils structure

  - [ ] Use `from rlarcworld.utils.distributions import categorical_projection`
  - [ ] Use `from rlarcworld.utils.dict_utils import get_nested_ref`
  - [ ] Use `from rlarcworld.utils.schedulers import BetaScheduler`
  - [ ] Use `from rlarcworld.utils.device import enable_cuda`
  - [ ] Use `from rlarcworld.utils.torch_queue import TorchQueue`
  - [ ] Use `from rlarcworld.utils.logging_utils import configure_logger`

- [ ] Logger configuration hardening (`rlarcworld/utils/logging_utils.py`)

  - [ ] Avoid duplicate `StreamHandler` additions when `configure_logger()` is called multiple times (tests often reconfigure logging)
  - [ ] Keep JSON and plain-text format options stable

## Medium Priority

- [ ] BetaScheduler robustness (`rlarcworld/utils/schedulers.py`)

  - [x] Use `internal_step / steps` instead of `/ end` in `step()`
  - [ ] Clamp progress: `progress = min(internal_step, steps)/max(steps,1)` to avoid tiny overshoots

- [ ] AMP autocast usage (`rlarcworld/utils/env_stepper.py`)

  - [x] Do not pass `None` to `torch.amp.autocast`; use conditional or `nullcontext`
  - [ ] Consider consolidating with a shared AMP utility/context if needed
  - [ ] Provide a small testing utility to generate device-aware synthetic steps to reduce duplication in AMP tests

- [ ] EnvStepper episode/reset behavior (`rlarcworld/utils/env_stepper.py`)

  - [ ] Prefer a pattern where the trainer provides per-episode batches for `reset()` rather than internally consuming `DataLoader` with `next(iter(samples))`
  - [ ] Ensure step streams align with trainer’s episode orchestration

- [ ] Reward processing ergonomics (`rlarcworld/utils/reward_utils.py`)

  - [ ] Consider optional `critic` argument to auto-extract `v_min/v_max` for `n_step_reward` when available to reduce plumbing at call sites

## Test Coverage

- [ ] Add/extend unit tests under `tests/` for utils

  - [ ] `schedulers.BetaScheduler`: monotonic increase, saturation at `end`, state_dict/load roundtrip
  - [ ] `device.enable_cuda`: mock `torch.cuda.is_available()` and device capability to validate fields
  - [ ] `distributions.categorical_projection`: shape checks, probability mass preservation (sum≈1), atoms alignment
  - [ ] `dict_utils.get_nested_ref`: nested creation, default callable behavior
  - [ ] `logging_utils.configure_logger`: single handler after multiple calls

## Nice-to-Have

- [ ] Add type annotations and docstrings consistency across all utils modules
- [ ] Centralize constants or enums (e.g., logger formats) if reused

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
  - [ ] AMP/no-AMP code paths where relevant
- [ ] Pre-commit hooks
- [x] Linters for import validity and formatting

---

Context: utils modules currently present under `rlarcworld/utils/`:

- `device.py`
- `dict_utils.py`
- `distributions.py`
- `env_stepper.py`
- `logging_utils.py`
- `reward_utils.py`
- `schedulers.py`
- `torch_queue.py`
