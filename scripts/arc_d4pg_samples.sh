#!/usr/bin/env bash
set -euo pipefail

# ARC D4PG CLI samples
# Usage: source this file or run directly. You can override vars below when running, e.g.:
#   GRID_SIZE=6 COLOR_VALUES=5 ./scripts/arc_d4pg_samples.sh train_quick

# Defaults (override by exporting before running)
: "${GRID_SIZE:=6}"
: "${COLOR_VALUES:=5}"
: "${BATCH_SIZE:=4}"
: "${EPOCHS:=1}"
: "${MAX_STEPS:=5}"
: "${CHECKPOINT_DIR:=./tmp_ckpt}"
: "${CHECKPOINT_FILE:=d4pg_final.pt}"
: "${DATASET_PATH:=}"
: "${LOG_LEVEL:=INFO}"
: "${MIN_TRAIN_BS:=2}"
: "${SAVE_RB:=0}"
: "${LOAD_RB:=0}"

# Helper to maybe add dataset flag
maybe_dataset_flag() {
  if [[ -n "${DATASET_PATH}" ]]; then
    printf -- " --dataset-path %q" "${DATASET_PATH}"
  else
    printf -- ""
  fi
}

# Helpers to toggle replay buffer flags
maybe_save_rb_flag() {
  if [[ "${SAVE_RB}" == "1" || "${SAVE_RB}" == "true" ]]; then
    printf -- " --save-replay-buffer"
  else
    printf -- ""
  fi
}

maybe_load_rb_flag() {
  if [[ "${LOAD_RB}" == "1" || "${LOAD_RB}" == "true" ]]; then
    printf -- " --load-replay-buffer"
  else
    printf -- ""
  fi
}

# 1) Quick train (small epoch, saves checkpoint)
train_quick() {
  uv run python -m examples.run_d4pg_experiment \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --min-train-batch-size "${MIN_TRAIN_BS}" \
    --max-steps-per-epoch "${MAX_STEPS}" \
    --grid-size "${GRID_SIZE}" \
    --color-values "${COLOR_VALUES}" \
    --no-amp \
    --save-checkpoint \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --log-level "${LOG_LEVEL}" \
    $(maybe_dataset_flag) \
    $(maybe_save_rb_flag)
}

# 2) Load-only verification (no training)
load_only() {
  uv run python -m examples.run_d4pg_experiment \
    --epochs 0 \
    --batch-size "${BATCH_SIZE}" \
    --min-train-batch-size "${MIN_TRAIN_BS}" \
    --grid-size "${GRID_SIZE}" \
    --color-values "${COLOR_VALUES}" \
    --no-amp \
    --load-checkpoint "${CHECKPOINT_DIR}/${CHECKPOINT_FILE}" \
    --log-level "${LOG_LEVEL}" \
    $(maybe_dataset_flag) \
    $(maybe_load_rb_flag)
}

# 3) Evaluate-only run (from checkpoint)
#    Note: set EVAL_EPISODES to override (default 10 as per script)
eval_only() {
  uv run python -m examples.run_d4pg_experiment \
    --epochs 0 \
    --batch-size "${BATCH_SIZE}" \
    --min-train-batch-size "${MIN_TRAIN_BS}" \
    --grid-size "${GRID_SIZE}" \
    --color-values "${COLOR_VALUES}" \
    --no-amp \
    --load-checkpoint "${CHECKPOINT_DIR}/${CHECKPOINT_FILE}" \
    --evaluate \
    --log-level "${LOG_LEVEL}" \
    $(maybe_dataset_flag) \
    $(maybe_load_rb_flag)
}

# 4) Medium train (more steps per epoch)
train_medium() {
  local steps=${1:-50}
  uv run python -m examples.run_d4pg_experiment \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --min-train-batch-size "${MIN_TRAIN_BS}" \
    --max-steps-per-epoch "${steps}" \
    --grid-size "${GRID_SIZE}" \
    --color-values "${COLOR_VALUES}" \
    --no-amp \
    --save-checkpoint \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --log-level "${LOG_LEVEL}" \
    $(maybe_dataset_flag)
}

# Print help
help_msg() {
  cat <<EOF
ARC D4PG sample commands

Environment overrides:
  GRID_SIZE (default: ${GRID_SIZE})
  COLOR_VALUES (default: ${COLOR_VALUES})
  BATCH_SIZE (default: ${BATCH_SIZE})
  EPOCHS (default: ${EPOCHS})
  MAX_STEPS (default: ${MAX_STEPS})
  CHECKPOINT_DIR (default: ${CHECKPOINT_DIR})
  CHECKPOINT_FILE (default: ${CHECKPOINT_FILE})
  DATASET_PATH (default: empty uses test data fallback)
  LOG_LEVEL (default: ${LOG_LEVEL})

Commands:
  train_quick   - Train for a short epoch and save checkpoint
  load_only     - Load checkpoint and do no training
  eval_only     - Load checkpoint and run evaluation only
  train_medium  - Train with more steps per epoch (arg1=steps, default 50)
  help          - Show this message

Examples:
  ./scripts/arc_d4pg_samples.sh train_quick
  ./scripts/arc_d4pg_samples.sh load_only
  ./scripts/arc_d4pg_samples.sh eval_only
  MAX_STEPS=100 ./scripts/arc_d4pg_samples.sh train_medium 100
EOF
}

cmd=${1:-help}
case "$cmd" in
  train_quick) train_quick ;;
  load_only)   load_only ;;
  eval_only)   eval_only ;;
  train_medium) shift || true; train_medium "$@" ;;
  help|*)      help_msg ;;
 esac
