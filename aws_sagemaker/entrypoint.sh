#!/usr/bin/env bash
set -euo pipefail

# SageMaker passes hyperparameters as CLI args; forward them to train.py.
# Using exec ensures SIGTERM reaches torchrun directly (best-effort spot handling).

NNODES="${SM_NUM_NODES:-1}"
NPROC_PER_NODE="${SM_NUM_GPUS:-1}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# When SageMaker runs SourceCode.command, it may not append hyperparameters as
# CLI args. In that case, fetch them from the standard hyperparameters.json.
HP_JSON="/opt/ml/input/config/hyperparameters.json"
HP_ARGS=()
if [[ -f "${HP_JSON}" ]]; then
  while IFS= read -r -d '' tok; do
    HP_ARGS+=("$tok")
  done < <(
    python - <<'PY'
import json
import sys

path = "/opt/ml/input/config/hyperparameters.json"
try:
    with open(path, "r", encoding="utf-8") as f:
        hp = json.load(f)
except Exception:
    hp = {}

for k, v in hp.items():
    if v is None:
        continue
    # SageMaker uses strings for hyperparameters; keep as-is.
    sys.stdout.write(f"--{k}\0")
    sys.stdout.write(f"{v}\0")
PY
  )
fi

exec torchrun --nnodes="${NNODES}" --nproc_per_node="${NPROC_PER_NODE}" "${SCRIPT_DIR}/train.py" "${HP_ARGS[@]}" "$@"
