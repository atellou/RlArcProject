# Experiment run order + commands

This file documents a **budget-friendly order** for running ARC RL experiments and provides **copy/paste commands** using `experiments/run_experiments.sh`.

## Notes on sample count and replay buffer sizing

Start with a smaller sample list (e.g. **100 tasks**) for early sweeps because it:

- reduces iteration time (more configs tested per hour)
- makes regressions and instability easier to reproduce/debug
- keeps the replay buffer from being dominated by a very wide task distribution before the policy is functional

Once defaults are stable, scale to larger lists to validate generalization:

- **100 tasks**: fast dynamics debugging and hyperparameter triage
- **300-500 tasks**: more representative stability check before committing to full runs
- **1400 tasks**: full dataset validation

Replay buffer sizing should scale with dataset breadth to avoid excessive recency bias:

- **100 tasks**: `replay_buffer.size=10_000` is a good default
- **300-500 tasks**: consider `replay_buffer.size=50_000`
- **1400 tasks**: consider `replay_buffer.size=100_000+` (if memory/time allow)

## 0) One-time setup

- Make sure your environment variables are loaded (if you use a `.env`):

```bash
set -o allexport; source .env; set +o allexport
```

- Define these shell variables once per session:

```bash
# Adjust to your paths
BASE_CONFIG="./experiments/configs/base.yaml"
SAMPLES_PATH_LIST="./experiments/arc_sample.txt"

# Suggested defaults
SEED=42
EPOCHS=2
```

## 1) Healthcheck (verify everything runs end-to-end)

```bash
./experiments/run_experiments.sh \
  --config ./experiments/configs/healthcheck.yaml \
  --base "${BASE_CONFIG}" \
  --samples-path-list "${SAMPLES_PATH_LIST}" \
  --seed "${SEED}" \
  --epochs "${EPOCHS}" \
  --experiments healthcheck
```

## 2) Reward shaping sweep (pick the best reward dynamics first)

Run these first because they directly change whether submitting is net-negative and whether the agent gets stuck in `improvement == 0`.

```bash
./experiments/run_experiments.sh \
  --config ./experiments/configs/reward_exp.yaml \
  --base "${BASE_CONFIG}" \
  --samples-path-list "${SAMPLES_PATH_LIST}" \
  --seed "${SEED}" \
  --epochs "${EPOCHS}" \
  --experiments no_submit_penalty,added_submit_penalty,added_improvement_penalty,all_penalties,all_penalties_low_improvement_penalty,all_penalties_medium_improvement_penalty,all_penalties_low_perfect_no_submit_penalty,all_penalties_low_both
```

Decision after this stage:

- Choose the best reward config (or best 2) based on:
  - `train/reward/*` trend (less cyclic, less negative drift)
  - submission success/failure rates
  - `grid_perfect_rate` and `improvement_rate`

Decision after this stage:

- Promote the best reward configs to curriculum tests.

## 3) Curriculum experiments (reward configs under curriculum)

Run curriculum with the selected reward settings and compare warm-start-only vs warm-start + ongoing assist.

```bash
./experiments/run_experiments.sh \
  --config ./experiments/configs/curriculum_exp.yaml \
  --base "${BASE_CONFIG}" \
  --samples-path-list "${SAMPLES_PATH_LIST}" \
  --seed "${SEED}" \
  --epochs "${EPOCHS}" \
  --experiments low_both_curriculum_ongoing,low_both_curriculum_episode_start_ongoing,low_improvement_penalty_curriculum_ongoing,low_improvement_penalty_curriculum_episode_start_ongoing
```

Decision after this stage:

- Pick **one** entropy regime that:
  - avoids entropy collapse
  - avoids extreme oscillation
  - yields the best reward/submission metrics under the best reward shaping

## 4) Entropy regularization sweep (pick stable exploration)

Run on top of the selected reward setting(s) from stage 2.

```bash
./experiments/run_experiments.sh \
  --config ./experiments/configs/entropy_exp.yaml \
  --base "${BASE_CONFIG}" \
  --samples-path-list "${SAMPLES_PATH_LIST}" \
  --seed "${SEED}" \
  --epochs "${EPOCHS}" \
  --experiments no_entropy_regularization,small_entropy_regularization,strong_entropy_regularization,perpetual_entropy_regularization
```

Decision after this stage:

- Pick **one** entropy regime that:
  - avoids entropy collapse
  - avoids extreme oscillation
  - yields the best reward/submission metrics under the best reward shaping

## 5) Submit-head entropy scale sweep

Run only after stage 3 (otherwise it’s too many degrees of freedom).

Recommended first sweep values:

- `submit_entropy_scale`: `0.0`, `0.25`, `0.5` (optionally include `1.0` as baseline)

```bash
./experiments/run_experiments.sh \
  --config ./experiments/configs/entropy_exp.yaml \
  --base "${BASE_CONFIG}" \
  --samples-path-list "${SAMPLES_PATH_LIST}" \
  --seed "${SEED}" \
  --epochs "${EPOCHS}" \
  --experiments submit_entropy_scale_0_0,submit_entropy_scale_0_25,submit_entropy_scale_0_5
```

## 6) Learning-rate sweep (run before CARSM)

Recommendation: run LR sweeps **before** CARSM. CARSM changes the actor objective and typically changes gradient scale, so doing LR after CARSM confounds attribution.

Exception: if you see obvious instability (NaNs, exploding TD-error, rapidly growing weight norms), do a **tiny** LR sanity sweep earlier (2-3 runs) just to get a stable baseline.

Example LR sweep command:

```bash
./experiments/run_experiments.sh \
  --config ./experiments/configs/lr_exp.yaml \
  --base "${BASE_CONFIG}" \
  --samples-path-list "${SAMPLES_PATH_LIST}" \
  --seed "${SEED}" \
  --epochs "${EPOCHS}" \
  --experiments lr_1e3_nstep_1,lr_3e4_nstep_1,lr_1e4_nstep_1
```

## Optional: CARSM

Recommendation: run CARSM **after** you have selected:

- one reward shaping config (stage 2)
- one entropy regime (stage 3)
- one `submit_entropy_scale` (stage 4)

Then do a small A/B (CARSM off vs on) rather than adding it to the initial grid.
