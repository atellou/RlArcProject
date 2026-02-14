# Experiment run order + commands

This file documents a **budget-friendly order** for running ARC RL experiments and provides **copy/paste commands** using `experiments/run_experiments.sh`.

## Notes on sample count and replay buffer sizing

Smaller sample lists (e.g. **100 tasks**) from `experiments/experiemnts_arc_sample.txt` for early sweeps, the sample was generated randomly, using `aws_sagemaker/lr_experiment/lr_experiments.ipynb` and should be modified to pint to the ARC dataset folders in the local filesystem.

## 0) One-time setup

- Make sure your environment variables are loaded (if you use a `.env`):

```bash
set -o allexport; source .env; set +o allexport
```

- Define these shell variables once per session:

```bash
# Adjust to your paths
BASE_CONFIG="./experiments/configs/base.yaml"
SAMPLES_PATH_LIST="./experiments/experiemnts_arc_sample.txt"

# Example values for budget-friendly runs
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
  --experiments re1_no_penalty,re2_wrong_submit_only,re3_no_op_only,re4_step_cost_only,re5_wrong_submit_plus_no_op,re6_wrong_submit_plus_step_cost,re7_perfect_no_submit_only,re8_wrong_submit_plus_no_op_plus_step_cost,re9_wrong_submit_plus_no_op_plus_perfect_no_submit,re10_all_penalties_plus_negative_shaping,re11_all_penalties_plus_positive_shaping,re12_all_penalties_plus_positive_shaping,re13_all_penalties_plus_positive_shaping
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
  --experiments cu1_base_no_curriculum,cu2_curriculum_mild,cu3_curriculum_medium,cu4_curriculum_aggressive,cu6_curriculum_only_start_up,cu7_curriculum_medium_50_50,cu8_curriculum_only_improve
```

## 4) Entropy regularization sweep (pick stable exploration)

Run on top of the selected reward setting(s) from stage 2.

```bash
./experiments/run_experiments.sh \
  --config ./experiments/configs/entropy_exp.yaml \
  --base "${BASE_CONFIG}" \
  --samples-path-list "${SAMPLES_PATH_LIST}" \
  --seed "${SEED}" \
  --epochs "${EPOCHS}" \
  --experiments en1_no_entropy_regularization,en2_entropy_floor_weak,en3_entropy_floor_base,en4_entropy_floor_strong,en5_entropy_floor_base_plus_small_entropy_coef,en6_entropy_floor_base_heads,en7_curriculum_mild_entropy_floor_with_heads_entropy
```

Decision after this stage:

- Pick **one** entropy regime that:
  - avoids entropy collapse
  - avoids extreme oscillation
  - yields the best reward/submission metrics under the best reward shaping

## 6) [CARSM](https://arxiv.org/abs/2002.03534v2)

```bash
./experiments/run_experiments.sh \
  --config ./experiments/configs/entropy_exp.yaml \
  --base "${BASE_CONFIG}" \
  --samples-path-list "${SAMPLES_PATH_LIST}" \
  --seed "${SEED}" \
  --epochs "${EPOCHS}" \
  --experiments ca1_add_carsm,ca2_add_carsm_no_curriculum
```

## 7) TD / n-step sweep (`td_exp.yaml`)

Once you have a stable reward + exploration setup, sweep `n_steps` to test credit assignment horizon.

```bash
./experiments/run_experiments.sh \
  --config ./experiments/configs/td_exp.yaml \
  --base "${BASE_CONFIG}" \
  --samples-path-list "${SAMPLES_PATH_LIST}" \
  --seed "${SEED}" \
  --epochs "${EPOCHS}" \
  --experiments ns_base_nstep_5,ns_base_nstep_15,ns_base_nstep_30,ns_base_nstep_50
```

## 8) World model sweep (`world_model_experiment.yaml`)

If you want model-based augmentation (Dyna-style imagination), test enabling the world model on top of the selected baseline.

```bash
./experiments/run_experiments.sh \
  --config ./experiments/configs/world_model_experiment.yaml \
  --base "${BASE_CONFIG}" \
  --samples-path-list "${SAMPLES_PATH_LIST}" \
  --seed "${SEED}" \
  --epochs "${EPOCHS}" \
  --experiments base_wm,wm_ns_base_nstep_5,wm_ns_base_nstep_15,wm_ns_base_nstep_30,wm_ns_base_nstep_50
