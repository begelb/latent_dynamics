#!/usr/bin/env bash

set -euo pipefail

readonly CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly PYTHON_BIN="${CODE_ROOT}/.venv/bin/python"
readonly SWEEP_TAG="patrick_paper_3x5_v1"
readonly SWEEP_ROOT="${CODE_ROOT}/output/leslie3d_example2_seedsweep_${SWEEP_TAG}"
readonly STATUS_FILE="${SWEEP_ROOT}/run_status.txt"

readonly -a COMMON_ARGS=(
  --example leslie3d_example2
  --trajectory-length 20
  --total-initial-conditions 10000
  --cmgdb-subdiv 25,28,29
  --box-map-backend adaptive_precomputed
  --bounds-data-role train_pairs
  --adaptive-precompute-subdiv init
  --tag "${SWEEP_TAG}"
  --figures morse
)
readonly -a FULL_GRID_ARGS=(
  --ic-seeds 1,2,3,4,5
  --model-seeds 0,1,2
)

mkdir -p "${SWEEP_ROOT}"
cd "${CODE_ROOT}"

export PYTHONPATH="${CODE_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
# The CSR cache has no size ceiling; this only pre-allocates the edge buffer
# once instead of growing it geometrically. It is a sizing hint, not a limit.
export CMGDB_MAPGRAPH_RESERVE_EDGES=1200000000

CURRENT_PHASE="initializing"

record_status() {
  printf '%s\t%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" | tee "${STATUS_FILE}"
}

finish() {
  local exit_code=$?
  if [[ ${exit_code} -eq 0 ]]; then
    record_status "completed"
  else
    record_status "failed exit_code=${exit_code} phase=${CURRENT_PHASE}"
  fi
}
trap finish EXIT

run_sweep() {
  "${PYTHON_BIN}" -u scripts/retrain_seed_sweep.py "${COMMON_ARGS[@]}" "$@"
}

CURRENT_PHASE="preflight"
record_status "running phase=${CURRENT_PHASE}"
"${PYTHON_BIN}" -c 'import CMGDB, torch; assert hasattr(CMGDB, "ComputeConleyMorseGraphOnly"), "CMGDB graph-only entry point is unavailable"; assert hasattr(CMGDB.Model, "set_batch_map"), "CMGDB batch-map entry point is unavailable"; assert torch.backends.mps.is_available(), "MPS is unavailable"'

CURRENT_PHASE="plan"
record_status "running phase=${CURRENT_PHASE}"
run_sweep \
  "${FULL_GRID_ARGS[@]}" \
  --stages data,scale,train,diagnose,morse,render \
  --device mps \
  --dry-run > "${SWEEP_ROOT}/run_plan.json"

CURRENT_PHASE="data_scale"
record_status "running phase=${CURRENT_PHASE}"
run_sweep \
  "${FULL_GRID_ARGS[@]}" \
  --stages data,scale \
  --device cpu \
  --skip-completed

CURRENT_PHASE="train_diagnose"
record_status "running phase=${CURRENT_PHASE}"
run_sweep \
  "${FULL_GRID_ARGS[@]}" \
  --stages train,diagnose \
  --device mps \
  --skip-completed

CURRENT_PHASE="morse"
for dataset_seed in 1 2 3 4 5; do
  for model_seed in 0 1 2; do
    CURRENT_PHASE="morse dataset=${dataset_seed} model=${model_seed}"
    record_status "running phase=${CURRENT_PHASE}"
    run_sweep \
      --ic-seeds "${dataset_seed}" \
      --model-seeds "${model_seed}" \
      --stages morse \
      --device mps \
      --skip-completed
  done
done

CURRENT_PHASE="render_summary"
record_status "running phase=${CURRENT_PHASE}"
run_sweep \
  "${FULL_GRID_ARGS[@]}" \
  --stages data,scale,train,diagnose,morse,render \
  --device mps \
  --skip-completed

CURRENT_PHASE="verify_summary"
record_status "running phase=${CURRENT_PHASE}"
SWEEP_SUMMARY_PATH="${SWEEP_ROOT}/sweep_summary.json" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

summary_path = Path(os.environ["SWEEP_SUMMARY_PATH"])
summary = json.loads(summary_path.read_text())
outcome = summary["outcome"]
expected_counts = {
    "planned_cells": 15,
    "training_complete": 15,
    "morse_graphs_complete": 15,
    "morse_sets_complete": 15,
    "classified_cells": 15,
}
observed_counts = {key: outcome.get(key) for key in expected_counts}
if observed_counts != expected_counts:
    raise SystemExit(
        f"incomplete final summary: expected {expected_counts}, observed {observed_counts}"
    )
cells = summary.get("cells", [])
required_artifacts = ("checkpoint", "training_summary", "morse_graph", "morse_sets")
if len(cells) != 15 or any(
    not all(cell.get("artifacts", {}).get(name) for name in required_artifacts)
    for cell in cells
):
    raise SystemExit("one or more final cell artifacts are missing")
print(f"verified complete 15-cell summary: {summary_path}")
PY
