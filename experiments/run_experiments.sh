#!/bin/bash

set -euo pipefail

# Run an experiment list

ROOT_DIR="$(pwd)"

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTORCH_ALLOC_CONF="expandable_segments:True"

print_usage() {
  echo "Usage: $0 [--config PATH] [--base PATH] [--samples-path-list PATH] [--seed INT] [--epochs INT] [--experiment-folder-location PATH] [--experiments NAME1,NAME2,...] [--experiment NAME]... [NAME ...]" >&2
}

experiments=()
experiments_arg=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --base)
      BASE_PATH="$2"
      shift 2
      ;;
    --samples-path-list)
      SAMPLE_PATH_LIST="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --experiment-folder-location)
      EXPERIMENT_FOLDER_LOCATION="$2"
      shift 2
      ;;
    --experiments)
      experiments_arg="$2"
      shift 2
      ;;
    --experiment|-e)
      experiments+=("$2")
      shift 2
      ;;
    --help|-h)
      print_usage
      exit 0
      ;;
    --*)
      echo "ERROR: Unknown flag: $1" >&2
      print_usage
      exit 1
      ;;
    *)
      experiments+=("$1")
      shift
      ;;
  esac
done

: "${CONFIG_PATH:?Missing required --config PATH}"
: "${BASE_PATH:?Missing required --base PATH}"
: "${SAMPLE_PATH_LIST:?Missing required --samples-path-list PATH}"
: "${SEED:?Missing required --seed INT}"
: "${EPOCHS:?Missing required --epochs INT}"

if [[ -n "${experiments_arg}" ]]; then
  IFS="," read -r -a _exp_arr <<< "${experiments_arg}"
  for _e in "${_exp_arr[@]}"; do
    if [[ -n "${_e}" ]]; then
      experiments+=("${_e}")
    fi
  done
fi

if [[ ${#experiments[@]} -eq 0 ]]; then
  echo "ERROR: No experiments provided." >&2
  print_usage
  exit 1
fi

NPROC_PER_NODE="$(uv run python -c 'import torch; print(torch.cuda.device_count())')"
if [[ -z "${NPROC_PER_NODE}" || "${NPROC_PER_NODE}" -lt 1 ]]; then
  echo "ERROR: No CUDA devices found (torch.cuda.device_count()=${NPROC_PER_NODE})" >&2
  exit 1
fi
echo "Using torchrun with nproc_per_node=${NPROC_PER_NODE}"

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
if [[ -n "${EXPERIMENT_FOLDER_LOCATION:-}" ]]; then
  echo "  EXPERIMENT_FOLDER_LOCATION=${EXPERIMENT_FOLDER_LOCATION}"
fi
echo "  experiments=${experiments[@]}"

for exp_name in "${experiments[@]}"; do
  echo "------------------------------------------------------------"
  echo "Starting experiment: ${exp_name}"
  echo "------------------------------------------------------------"

  extra_args=()
  if [[ -n "${EXPERIMENT_FOLDER_LOCATION:-}" ]]; then
    extra_args+=("--experiment-folder-location" "${EXPERIMENT_FOLDER_LOCATION}")
  fi

  uv run torchrun --standalone --nproc_per_node "${NPROC_PER_NODE}" experiments/run_experiment.py \
    --experiment "${exp_name}" \
    --config "${CONFIG_PATH}" \
    --base "${BASE_PATH}" \
    --samples-path-list "${SAMPLE_PATH_LIST}" \
    --epochs "${EPOCHS}" \
    --seed "${SEED}" \
    "${extra_args[@]}"
done

echo "All experiments completed."
