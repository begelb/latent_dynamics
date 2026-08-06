#!/usr/bin/env bash

set -euo pipefail

readonly CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly PYTHON_BIN="${CODE_ROOT}/.venv/bin/python"
readonly TRAIN_DEVICE="${TRAIN_DEVICE:-mps}"
readonly SWEEP_TAG="3x5_v1"
readonly SWEEP_STEM="leslie3d_groundbox_curriculum_wide_seedsweep_${SWEEP_TAG}"
readonly SWEEP_ROOT="${CODE_ROOT}/output/${SWEEP_STEM}"
readonly STATUS_FILE="${SWEEP_ROOT}/run_status.txt"

readonly -a COMMON_ARGS=(
  --example leslie3d_groundbox_curriculum_wide
  --trajectory-length 20
  --cmgdb-subdiv 25,28,29
  --box-map-backend adaptive_precomputed
  --bounds-data-role train_pairs
  --adaptive-precompute-subdiv init
  --tag "${SWEEP_TAG}"
  --figures morse
  --full-batch
)
readonly -a FULL_GRID_ARGS=(
  --ic-seeds 2158,4792,3174,688,5727
  --model-seeds 0,1,2
)

mkdir -p "${SWEEP_ROOT}"
cd "${CODE_ROOT}"

export PYTHONPATH="${CODE_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
# This is a CSR allocation hint, not a graph-size ceiling.
export CMGDB_MAPGRAPH_RESERVE_EDGES="${CMGDB_MAPGRAPH_RESERVE_EDGES:-1200000000}"

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
record_status "running phase=${CURRENT_PHASE} train_device=${TRAIN_DEVICE}"
TRAIN_DEVICE_NAME="${TRAIN_DEVICE}" "${PYTHON_BIN}" - <<'PY'
import os

import CMGDB
import torch

from latentdynamics.config import load_config

cfg = load_config("leslie3d_groundbox_curriculum_wide")
assert cfg.system.params["lower_bounds"] == [0.0, 0.0, 0.0]
assert cfg.system.params["upper_bounds"] == [110.0, 77.0, 54.0]
assert (cfg.data.n_samples_train, cfg.data.n_samples_val) == (1000, 200)
assert (cfg.data.n_iterations, cfg.data.skip) == (20, 0)
assert cfg.training.batch_size == 20000
assert cfg.training.curriculum is not None
assert cfg.training.curriculum_optimizer is not None
assert cfg.training.curriculum_optimizer.model_dump(mode="json") == {
    "name": "adamw",
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "weight_decay": 0.0,
    "amsgrad": False,
    "foreach": False,
    "fused": False,
}
assert [stage.epochs for stage in cfg.training.curriculum] == [4000, 4000, 4000]
assert [stage.loss_weights for stage in cfg.training.curriculum] == [
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [1.0, 1.0, 1.0],
]
assert cfg.training.curriculum_polish is not None
assert cfg.training.curriculum_polish.model_dump(mode="json") == {
    "name": "lbfgs",
    "device": "cpu",
    "dtype": "float64",
    "outer_steps": 12,
    "learning_rate": 0.25,
    "max_iter": 10,
    "max_eval": 25,
    "history_size": 50,
    "tolerance_grad": 1e-9,
    "tolerance_change": 1e-12,
    "line_search_fn": "strong_wolfe",
    "loss_weights": [1.0, 1.0, 1.0],
    "trainable_components": ["encoder", "latent_map", "decoder"],
}
assert hasattr(CMGDB, "ComputeConleyMorseGraphOnly"), "CMGDB graph-only entry point is unavailable"
assert hasattr(CMGDB.Model, "set_batch_map"), "CMGDB batch-map entry point is unavailable"

device = os.environ["TRAIN_DEVICE_NAME"]
if device == "mps":
    assert torch.backends.mps.is_available(), "TRAIN_DEVICE=mps, but MPS is unavailable"
elif device.startswith("cuda"):
    assert torch.cuda.is_available(), "TRAIN_DEVICE requests CUDA, but CUDA is unavailable"
elif device != "cpu":
    torch.device(device)
PY

CURRENT_PHASE="plan"
record_status "running phase=${CURRENT_PHASE}"
run_sweep \
  "${FULL_GRID_ARGS[@]}" \
  --stages data,scale,train,diagnose,morse,render,metrics \
  --device "${TRAIN_DEVICE}" \
  --dry-run > "${SWEEP_ROOT}/run_plan.json"

CURRENT_PHASE="data_scale"
record_status "running phase=${CURRENT_PHASE}"
run_sweep \
  "${FULL_GRID_ARGS[@]}" \
  --stages data,scale \
  --device cpu \
  --skip-completed

CURRENT_PHASE="train_diagnose"
record_status "running phase=${CURRENT_PHASE} adamw_device=${TRAIN_DEVICE} lbfgs_device=cpu"
run_sweep \
  "${FULL_GRID_ARGS[@]}" \
  --stages train,diagnose \
  --device "${TRAIN_DEVICE}" \
  --skip-completed

for dataset_seed in 2158 4792 3174 688 5727; do
  for model_seed in 0 1 2; do
    CURRENT_PHASE="morse dataset=${dataset_seed} model=${model_seed}"
    record_status "running phase=${CURRENT_PHASE}"
    run_sweep \
      --ic-seeds "${dataset_seed}" \
      --model-seeds "${model_seed}" \
      --stages morse \
      --device "${TRAIN_DEVICE}" \
      --skip-completed
  done
done

CURRENT_PHASE="render_summary"
record_status "running phase=${CURRENT_PHASE}"
# Re-enter every stage with skip-completed so the generic sweep summary sees all
# 15 cells, while only derived render/metric artifacts still missing are made.
run_sweep \
  "${FULL_GRID_ARGS[@]}" \
  --stages data,scale,train,diagnose,morse,render,metrics \
  --device "${TRAIN_DEVICE}" \
  --skip-completed

CURRENT_PHASE="verify_generic_summary"
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
        f"incomplete generic summary: expected {expected_counts}, observed {observed_counts}"
    )

expected_cells = {
    (dataset_seed, model_seed)
    for dataset_seed in (2158, 4792, 3174, 688, 5727)
    for model_seed in (0, 1, 2)
}
cells = summary.get("cells", [])
observed_cells = {(cell.get("ic_seed"), cell.get("model_seed")) for cell in cells}
if len(cells) != 15 or observed_cells != expected_cells:
    raise SystemExit(
        f"generic summary cell grid mismatch: expected {sorted(expected_cells)}, "
        f"observed {sorted(observed_cells)}"
    )
required_artifacts = ("checkpoint", "training_summary", "morse_graph", "morse_sets")
if any(
    not all(cell.get("artifacts", {}).get(name) for name in required_artifacts)
    for cell in cells
):
    raise SystemExit("one or more generic-summary cell artifacts are missing")
print(f"verified complete 15-cell generic summary: {summary_path}")
PY

CURRENT_PHASE="dedicated_summary"
record_status "running phase=${CURRENT_PHASE}"
"${PYTHON_BIN}" -u scripts/summarize_leslie3d_groundbox_curriculum_3x5.py \
  --sweep-root "${SWEEP_ROOT}"

CURRENT_PHASE="verify_dedicated_summary"
record_status "running phase=${CURRENT_PHASE}"
SUMMARY_ROOT="${SWEEP_ROOT}/summary" "${PYTHON_BIN}" - <<'PY'
import csv
import json
import os
from pathlib import Path

summary_root = Path(os.environ["SUMMARY_ROOT"])
cells_path = summary_root / "cells.csv"
aggregate_path = summary_root / "aggregate_summary.json"
markdown_path = summary_root / "SUMMARY.md"

aggregate = json.loads(aggregate_path.read_text())
if not isinstance(aggregate, dict):
    raise SystemExit(f"dedicated aggregate is not a JSON object: {aggregate_path}")
inventory = aggregate.get("inventory", {})
if aggregate.get("provisional") is not False or {
    "n_complete_cells": inventory.get("n_complete_cells"),
    "n_invalid_cells": inventory.get("n_invalid_cells"),
    "n_missing_cells": inventory.get("n_missing_cells"),
} != {
    "n_complete_cells": 15,
    "n_invalid_cells": 0,
    "n_missing_cells": 0,
}:
    raise SystemExit(f"dedicated aggregate is not a complete 15-cell report: {aggregate_path}")
with cells_path.open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
if len(rows) != 15:
    raise SystemExit(f"dedicated summary has {len(rows)} cells, expected 15")
if any(
    row.get("cell_status") != "complete"
    or row.get("training_contract_valid") != "True"
    for row in rows
):
    raise SystemExit("one or more dedicated-summary rows failed the frozen contract")
if not markdown_path.is_file() or markdown_path.stat().st_size == 0:
    raise SystemExit(f"dedicated Markdown summary is missing or empty: {markdown_path}")
print(f"verified strict 15-cell dedicated summary: {summary_root}")
PY

# Deliberately no margin-fine-tuning phase. That intervention is designed only
# after this baseline 3x5 topology and loss summary has been reviewed.
