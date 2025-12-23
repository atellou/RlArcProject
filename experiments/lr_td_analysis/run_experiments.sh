#!/bin/bash

set -euo pipefail

# Run experiments for learning rate and TD coefficient analysis
#
# Usage:
#   SAMPLE_PATH_LIST=./arc_sample.txt ./experiments/lr_td_analysis/run_experiments.sh
#
# Notes:
# - All experiments use the same seed.
# - Epochs are controlled via --epochs.
# - max_steps_per_epoch is controlled in the base config.

ROOT_DIR="$(pwd)"

CONFIG_PATH="${ROOT_DIR}/experiments/lr_td_analysis/experiments.yaml"
BASE_PATH="${ROOT_DIR}/experiments/configs/base.yaml"
SAMPLE_PATH_LIST="${ROOT_DIR}/experiments/lr_td_analysis/arc_sample.txt"

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
SEED="123"
EPOCHS="10"

experiments=(
  lr_1e3_nstep_1
  lr_1e3_nstep_5
  lr_1e3_nstep_15
  lr_1e3_nstep_30
  lr_1e3_nstep_50
  lr_3e4_nstep_1
  lr_3e4_nstep_5
  lr_3e4_nstep_15
  lr_3e4_nstep_30
  lr_3e4_nstep_50
  lr_1e4_nstep_1
  lr_1e4_nstep_5
  lr_1e4_nstep_15
  lr_1e4_nstep_30
  lr_1e4_nstep_50
  lr_3e5_nstep_1
  lr_3e5_nstep_5
  lr_3e5_nstep_15
  lr_3e5_nstep_30
  lr_3e5_nstep_50
)

if [[ ! -f "${SAMPLE_PATH_LIST}" ]]; then
  echo "ERROR: SAMPLE_PATH_LIST not found: ${SAMPLE_PATH_LIST}" >&2
  exit 1
fi

echo "Running ${#experiments[@]} experiments"
echo "  CONFIG_PATH=${CONFIG_PATH}"
echo "  BASE_PATH=${BASE_PATH}"
echo "  SAMPLE_PATH_LIST=${SAMPLE_PATH_LIST}"
echo "  SEED=${SEED}"
echo "  EPOCHS=${EPOCHS}"

for exp_name in "${experiments[@]}"; do
  echo "------------------------------------------------------------"
  echo "Starting experiment: ${exp_name}"
  echo "------------------------------------------------------------"

  uv run python experiments/run_experiment.py \
    --experiment "${exp_name}" \
    --config "${CONFIG_PATH}" \
    --base "${BASE_PATH}" \
    --samples-path-list "${SAMPLE_PATH_LIST}" \
    --epochs "${EPOCHS}" \
    --seed "${SEED}"
done

echo "All experiments completed."
