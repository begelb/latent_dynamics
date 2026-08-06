#!/usr/bin/env bash

set -euo pipefail

readonly CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly PYTHON_BIN="${CODE_ROOT}/.venv/bin/python"
readonly TRAIN_DEVICE="${TRAIN_DEVICE:-mps}"
readonly SWEEP_TAG="3x5_v1"
readonly SWEEP_STEM="ives_myvatn_seedsweep_${SWEEP_TAG}"
readonly DATA_ROOT="${CODE_ROOT}/data/${SWEEP_STEM}"
readonly SWEEP_ROOT="${CODE_ROOT}/output/${SWEEP_STEM}"
readonly PLAN_FILE="${SWEEP_ROOT}/run_plan.json"
readonly STATUS_FILE="${SWEEP_ROOT}/run_status.txt"
readonly RUN_LOG="${SWEEP_ROOT}/run.log"
readonly PID_FILE="${SWEEP_ROOT}/controller.pid"
readonly SESSION_FILE="${SWEEP_ROOT}/session.txt"
readonly GENERIC_PLAN_TMP="${SWEEP_ROOT}/.generic_dry_plan.$$.json"
readonly SESSION_NAME="${RUN_SESSION_NAME:-${TMUX:-foreground}}"

readonly -a DATA_SEEDS=(2158 4792 3174 688 5727)
readonly -a MODEL_SEEDS=(0 1 2)
readonly -a COMMON_ARGS=(
  --example ives_myvatn
  --trajectory-length 70
  --cmgdb-subdiv 18,22,30
  --box-map-backend adaptive_precomputed
  --bounds-data-role system_grid
  --adaptive-precompute-subdiv init
  --tag "${SWEEP_TAG}"
  --figures morse
)
readonly -a FULL_GRID_ARGS=(
  --ic-seeds 2158,4792,3174,688,5727
  --model-seeds 0,1,2
)

if [[ ! -x "${PYTHON_BIN}" ]]; then
  printf 'missing workspace Python: %s\n' "${PYTHON_BIN}" >&2
  exit 2
fi

mkdir -p "${SWEEP_ROOT}"
cd "${CODE_ROOT}"
export PYTHONPATH="${CODE_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
# This is a CSR allocation hint, not a graph-size ceiling.
export CMGDB_MAPGRAPH_RESERVE_EDGES="${CMGDB_MAPGRAPH_RESERVE_EDGES:-1200000000}"

# A strictly verified completed sweep is immutable.  In particular, do not
# append to its log or replace its controller metadata on an accidental rerun.
if [[ -s "${SWEEP_ROOT}/summary/aggregate_summary.json" ]] && \
  "${PYTHON_BIN}" scripts/summarize_ives_myvatn_3x5.py \
    --sweep-root "${SWEEP_ROOT}" --verify >/dev/null 2>&1; then
  printf 'Ives sweep is already strictly complete; leaving it untouched: %s\n' "${SWEEP_ROOT}"
  exit 0
fi

if [[ -s "${PID_FILE}" ]]; then
  existing_pid="$(tr -dc '0-9' < "${PID_FILE}")"
  existing_command=""
  if [[ -n "${existing_pid}" ]] && kill -0 "${existing_pid}" 2>/dev/null; then
    existing_command="$(ps -p "${existing_pid}" -o command= 2>/dev/null || true)"
  fi
  if [[ "${existing_command}" == *"run_ives_myvatn_3x5.sh"* ]]; then
    printf 'an Ives controller is already live (PID %s); refusing a second launch\n' \
      "${existing_pid}" >&2
    exit 2
  fi
fi

exec > >(tee -a "${RUN_LOG}") 2>&1
printf '%s\n' "$$" > "${PID_FILE}"
{
  printf 'started_utc=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  printf 'controller_pid=%s\n' "$$"
  printf 'session=%s\n' "${SESSION_NAME}"
  printf 'host=%s\n' "$(hostname)"
  printf 'invocation=%q' "${BASH_SOURCE[0]}"
  printf ' %q' "$@"
  printf '\n'
} >> "${SESSION_FILE}"

CURRENT_PHASE="initializing"
FINAL_DETAIL=""

record_status() {
  local temporary="${STATUS_FILE}.tmp.$$"
  printf '%s\t%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" | tee "${temporary}"
  mv "${temporary}" "${STATUS_FILE}"
}

finish() {
  local exit_code=$?
  if [[ ${exit_code} -eq 0 ]]; then
    record_status "completed phase=completed${FINAL_DETAIL:+ ${FINAL_DETAIL}}"
  else
    record_status "failed exit_code=${exit_code} phase=${CURRENT_PHASE}"
  fi
}
trap finish EXIT

run_sweep() {
  "${PYTHON_BIN}" -u scripts/retrain_seed_sweep.py "${COMMON_ARGS[@]}" "$@"
}

CURRENT_PHASE="preflight"
record_status "running phase=${CURRENT_PHASE} train_device=${TRAIN_DEVICE} topology_native=cpu"
TRAIN_DEVICE_NAME="${TRAIN_DEVICE}" "${PYTHON_BIN}" - <<'PY'
import os

import CMGDB
import numpy as np
import torch

from latentdynamics.config import load_config
from latentdynamics.models import build_autoencoder
from latentdynamics.systems import build_system
from latentdynamics.training import ReconstructionLoss

cfg = load_config("ives_myvatn")
assert cfg.experiment_name == "ives_myvatn"
assert cfg.system.name == "ives"
assert cfg.system.params == {
    "coordinate_mode": "log",
    "r1": 3.873,
    "r2": 11.746,
    "c": 0.000000367282300498085,
    "d": 0.5517,
    "p": 0.06659,
    "q": 0.9026,
    "lower_bounds": [-3.0, -7.5, -3.0],
    "upper_bounds": [1.5, 1.5, 1.5],
}
assert (cfg.arch.high_dims, cfg.arch.low_dims) == (3, 2)
assert cfg.arch.component("encoder").hidden_shapes == (32,)
assert cfg.arch.component("latent_map").hidden_shapes == (64, 64, 64, 64, 64)
assert cfg.arch.component("decoder").hidden_shapes == (32,)
assert cfg.arch.component("encoder").out_activation == "tanh"
assert cfg.arch.component("latent_map").out_activation == "tanh"
assert cfg.arch.component("decoder").out_activation == "sigmoid"
assert cfg.training.learning_rate == 0.001
assert cfg.training.batch_size == 1024
assert (cfg.training.epochs, cfg.training.patience, cfg.training.lr_patience) == (500, 300, 20)
assert cfg.training.loss_weights == [1.0, 1.0, 1.0]
assert cfg.training.gradient_clip_norm is None
assert (cfg.training.scheduler_factor, cfg.training.scheduler_threshold) == (0.5, 0.0001)
assert cfg.training.scheduler_min_lr == 0.000001
assert cfg.training.curriculum is None and cfg.training.warm_start_checkpoint_dir is None
assert cfg.data.sampling_method == "uniform"
assert cfg.data.scaling == "fixed_bounds" and cfg.data.scaling_epsilon == 0.000001
assert (cfg.data.n_samples_train, cfg.data.n_samples_val) == (1000, 200)
assert (cfg.data.n_iterations, cfg.data.skip, cfg.data.val_seed) == (70, 50, 9999)
assert cfg.data.n_samples_train * (cfg.data.n_iterations - cfg.data.skip) == 20000
assert cfg.data.n_samples_val * (cfg.data.n_iterations - cfg.data.skip) == 4000
assert (cfg.cmgdb.subdiv_init, cfg.cmgdb.subdiv_min, cfg.cmgdb.subdiv_max) == (18, 22, 30)
assert cfg.cmgdb.subdiv_limit == 100000
assert cfg.cmgdb.box_map_backend == "adaptive_precomputed"
assert cfg.cmgdb.bounds_data_role == "system_grid"
assert cfg.cmgdb.bounds_grid_resolution == 64
assert cfg.cmgdb.bounds_include_latent_image is True
assert cfg.cmgdb.bounds_epsilon_frac == 0.1
assert cfg.cmgdb.bounds_clip_lower == [-1.0, -1.0]
assert cfg.cmgdb.bounds_clip_upper == [1.0, 1.0]
assert cfg.cmgdb.padding is True
assert cfg.cmgdb.adaptive_precompute_subdiv == "init"
assert cfg.cmgdb.compute_roa is False
assert hasattr(CMGDB, "ComputeConleyMorseGraphOnly"), "CMGDB graph-only entry point is unavailable"
assert hasattr(CMGDB.Model, "set_batch_map"), "CMGDB batch-map entry point is unavailable"

system = build_system(cfg.system.name, cfg.system.params)
observed = system.step(np.zeros(3, dtype=np.float64))
expected = np.array(
    [0.3287694948708851, 0.5881807125380104, 0.15185979147616613],
    dtype=np.float64,
)
np.testing.assert_allclose(observed, expected, rtol=0.0, atol=2e-15)

device_name = os.environ["TRAIN_DEVICE_NAME"]
if device_name == "mps":
    assert torch.backends.mps.is_available(), "TRAIN_DEVICE=mps, but MPS is unavailable"
elif device_name.startswith("cuda"):
    assert torch.cuda.is_available(), "TRAIN_DEVICE requests CUDA, but CUDA is unavailable"
elif device_name != "cpu":
    torch.device(device_name)
device = torch.device(device_name)
model = build_autoencoder(cfg.arch).to(device)
x = torch.rand((16, cfg.arch.high_dims), dtype=torch.float32, device=device)
target = torch.rand((16, cfg.arch.high_dims), dtype=torch.float32, device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
optimizer.zero_grad(set_to_none=True)
result = model(x, target)
loss = ReconstructionLoss(cfg.training.loss_weights).to(device)(result).total
assert bool(torch.isfinite(loss).item())
loss.backward()
assert all(
    parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
    for parameter in model.parameters()
)
optimizer.step()
for tensor in (
    result.z_t,
    result.z_tau,
    result.z_tau_pred,
    result.z_tau_pred_cycle,
    result.x_t_hat,
    result.x_tau_hat,
):
    assert tensor.device.type == device.type
    assert bool(torch.isfinite(tensor).all().item())
print(
    f"preflight passed: native graph-only/batch-map symbols and "
    f"{device_name} model forward/backward"
)
PY

CURRENT_PHASE="plan"
record_status "running phase=${CURRENT_PHASE}"
run_sweep \
  "${FULL_GRID_ARGS[@]}" \
  --stages data,scale,train,diagnose,morse,render,metrics \
  --device "${TRAIN_DEVICE}" \
  --dry-run > "${GENERIC_PLAN_TMP}"
GENERIC_PLAN_PATH="${GENERIC_PLAN_TMP}" PLAN_PATH="${PLAN_FILE}" \
  TRAIN_DEVICE_NAME="${TRAIN_DEVICE}" "${PYTHON_BIN}" - <<'PY'
import json
import os
import sys
from pathlib import Path

from latentdynamics.cli.provenance import config_hash

code_root = Path.cwd()
sys.path.insert(0, str(code_root / "scripts"))
import retrain_seed_sweep as sweep  # noqa: E402

data_seeds = [2158, 4792, 3174, 688, 5727]
model_seeds = [0, 1, 2]
generic = json.loads(Path(os.environ["GENERIC_PLAN_PATH"]).read_text())
assert generic["examples"] == ["ives_myvatn"]
assert generic["ic_seeds"] == data_seeds
assert generic["model_seeds"] == model_seeds
assert generic["n_cells"] == 15
assert generic["figures"] == ["morse"]
assert generic["cmgdb_subdiv"] == [18, 22, 30]
assert generic["box_map_backend"] == "adaptive_precomputed"
assert generic["bounds_data_role"] == "system_grid"
assert generic["adaptive_precompute_subdiv"] == "init"
assert generic["trajectory_length"] == 70
assert generic["full_batch"] is False
size = generic["data_sizes"]["ives_myvatn"]
assert size["effective_initial_conditions_per_dataset"] == {
    "train": 1000,
    "validation": 200,
    "total": 1200,
}
assert size["trajectory"] == {
    "generated_steps": 70,
    "discarded_steps": 50,
    "retained_steps": 20,
}
assert size["transition_pairs_per_dataset"] == {
    "train": 20000,
    "validation": 4000,
    "total": 24000,
}
expected_first = code_root / "output/ives_myvatn_seedsweep_3x5_v1/dataset_2158/seed_0"
expected_last = code_root / "output/ives_myvatn_seedsweep_3x5_v1/dataset_5727/seed_2"
assert Path(generic["cells"][0]["output_dir"]) == expected_first
assert Path(generic["cells"][-1]["output_dir"]) == expected_last

resolved = {}
for data_seed in data_seeds:
    cfg = sweep._dataset_config(
        "ives_myvatn",
        ic_seed=data_seed,
        model_seeds=model_seeds,
        tag="3x5_v1",
        cmgdb_subdiv=(18, 22, 30),
        box_map_backend="adaptive_precomputed",
        bounds_data_role="system_grid",
        adaptive_precompute_subdiv="init",
        trajectory_length=70,
        total_initial_conditions=None,
        full_batch=False,
    )
    resolved[str(data_seed)] = {
        "config_hash": config_hash(cfg),
        "config": cfg.model_dump(mode="json"),
    }

plan = {
    "schema_version": 1,
    "experiment_id": "ives_myvatn",
    "run_tag": "3x5_v1",
    "design": {
        "data_seeds": data_seeds,
        "model_seeds": model_seeds,
        "validation_seed": 9999,
        "n_cells": 15,
        "first_cell": str(expected_first),
        "last_cell": str(expected_last),
    },
    "generic_dry_run": generic,
    "resolved_dataset_configs": resolved,
    "phase_devices": {
        "data_and_scaling": "cpu",
        "training_and_diagnosis": os.environ.get("TRAIN_DEVICE_NAME", "mps"),
        "topology_native_graph": "cpu",
        "topology_network_evaluation": os.environ.get("TRAIN_DEVICE_NAME", "mps"),
    },
    "topology_process_policy": "one cell per process",
    "render_groups": ["morse"],
    "excluded_products": [
        "regions_of_attraction",
        "basin_plots",
        "training_data_plots",
        "latent_evolution_snapshots_or_animations",
        "density_overlays",
        "invariant_overlays",
        "separation_training_extras",
        "unrelated_paper_figures",
    ],
    "required_cell_artifacts": [
        "models/autoencoder.pt",
        "models/autoencoder.json",
        "logs/history.json",
        "training_summary.json",
        "final_losses.txt",
        "diagnose.json",
        "MG/morse_graph",
        "MG/morse_sets",
        "mg_params_log.txt",
        "MG/morse_graph.pdf",
        "MG/morse_graph.png",
        "MG/morse_sets.pdf",
        "MG/morse_sets.png",
        "metrics.json",
        "run_manifest.json",
    ],
    "pass_criterion": {
        "graph": "directed-isomorphic to four-node, three-edge branch-then-chain with two sinks",
        "fixed_point": "uniquely assigned to one sink",
        "period_12_cycle": "at least 11 of 12 phases uniquely assigned to one common other sink and none conflicting",
        "conley_periods_1_12": "recorded diagnostic only; not a pass gate",
        "sweep_completion": "all 15 cells complete, parseable, and classified; no minimum scientific pass count",
    },
}
path = Path(os.environ["PLAN_PATH"])
serialized = json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n"
if path.exists():
    try:
        existing = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"existing run plan is invalid; refusing overwrite: {path}: {exc}")
    if existing != plan:
        raise SystemExit(f"existing run plan differs; refusing overwrite: {path}")
    print(f"verified existing resolved plan: {path}")
else:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(serialized)
    temporary.replace(path)
    print(f"saved resolved plan: {path}")
PY
rm -f -- "${GENERIC_PLAN_TMP}"

CURRENT_PHASE="recovery"
record_status "running phase=${CURRENT_PHASE}"
"${PYTHON_BIN}" -u scripts/recover_ives_myvatn_3x5.py \
  --data-root "${DATA_ROOT}" --sweep-root "${SWEEP_ROOT}"

CURRENT_PHASE="data_scale"
record_status "running phase=${CURRENT_PHASE}"
run_sweep \
  "${FULL_GRID_ARGS[@]}" \
  --stages data,scale \
  --device cpu \
  --skip-completed

CURRENT_PHASE="train_diagnose"
record_status "running phase=${CURRENT_PHASE} device=${TRAIN_DEVICE}"
run_sweep \
  "${FULL_GRID_ARGS[@]}" \
  --stages train,diagnose \
  --device "${TRAIN_DEVICE}" \
  --skip-completed

for data_seed in "${DATA_SEEDS[@]}"; do
  for model_seed in "${MODEL_SEEDS[@]}"; do
    CURRENT_PHASE="morse dataset=${data_seed} model=${model_seed}"
    record_status "running phase=${CURRENT_PHASE} native_graph=cpu network_device=${TRAIN_DEVICE}"
    run_sweep \
      --ic-seeds "${data_seed}" \
      --model-seeds "${model_seed}" \
      --stages morse \
      --device "${TRAIN_DEVICE}" \
      --skip-completed
  done
done

CURRENT_PHASE="render_metrics_manifest"
record_status "running phase=${CURRENT_PHASE} figures=morse roa=false"
# Re-enter the full grid with strict stage-completeness checks.  Completed
# expensive products are skipped; this creates only missing Morse renders,
# metrics, manifests, and the final 15-cell generic sweep summary.
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

path = Path(os.environ["SWEEP_SUMMARY_PATH"])
summary = json.loads(path.read_text())
expected_counts = {
    "planned_cells": 15,
    "training_complete": 15,
    "morse_graphs_complete": 15,
    "morse_sets_complete": 15,
    "classified_cells": 15,
}
observed_counts = {key: summary["outcome"].get(key) for key in expected_counts}
if observed_counts != expected_counts:
    raise SystemExit(
        f"incomplete generic summary: expected {expected_counts}, observed {observed_counts}"
    )
expected_cells = {
    (data_seed, model_seed)
    for data_seed in (2158, 4792, 3174, 688, 5727)
    for model_seed in (0, 1, 2)
}
cells = summary.get("cells", [])
observed_cells = {(cell.get("ic_seed"), cell.get("model_seed")) for cell in cells}
if len(cells) != 15 or observed_cells != expected_cells:
    raise SystemExit(
        f"generic summary cell grid mismatch: expected {sorted(expected_cells)}, "
        f"observed {sorted(observed_cells)}"
    )
required = ("checkpoint", "training_summary", "morse_graph", "morse_sets")
if any(not all(cell.get("artifacts", {}).get(name) for name in required) for cell in cells):
    raise SystemExit("one or more generic-summary cell artifacts are missing")
print(f"verified complete 15-cell generic summary: {path}")
PY

CURRENT_PHASE="dedicated_summary"
record_status "running phase=${CURRENT_PHASE}"
"${PYTHON_BIN}" -u scripts/summarize_ives_myvatn_3x5.py --sweep-root "${SWEEP_ROOT}"

CURRENT_PHASE="strict_final_verify"
record_status "running phase=${CURRENT_PHASE}"
"${PYTHON_BIN}" -u scripts/summarize_ives_myvatn_3x5.py \
  --sweep-root "${SWEEP_ROOT}" --verify
MACHINE_PASS_COUNT="$(SUMMARY_PATH="${SWEEP_ROOT}/summary/aggregate_summary.json" \
  "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

summary = json.loads(Path(os.environ["SUMMARY_PATH"]).read_text())
inventory = summary["inventory"]
if summary.get("provisional") is not False:
    raise SystemExit("strict aggregate was marked provisional")
if (
    inventory.get("n_expected_cells") != 15
    or inventory.get("n_complete_cells") != 15
    or inventory.get("n_verified_cells") != 15
    or inventory.get("n_incomplete_cells") != 0
    or inventory.get("n_invalid_cells") != 0
    or inventory.get("n_issues") != 0
):
    raise SystemExit("strict aggregate does not certify all 15 cells")
classification = summary["classification"]
if classification.get("n_evaluated") != 15:
    raise SystemExit("strict aggregate did not classify all 15 cells")
print(classification["n_pass"])
PY
)"
FINAL_DETAIL="verified_cells=15 machine_passes=${MACHINE_PASS_COUNT}"
CURRENT_PHASE="completed"
