#!/bin/bash
#
# AMAREL / Slurm array template for one LatentDynamics config.
#
# Usage from code/ after checking the cell count:
#   ../.venv/bin/python pipeline.py --config configs/coral_data_scaling.yaml --dry-run
#   sbatch --array=0-179 slurm/pipeline_array.sbatch
#
# Override at submission time, for example:
#   sbatch --array=0-0 --export=ALL,CONFIG=configs/chafee_infante.yaml,STAGES=train,morse slurm/pipeline_array.sbatch

#SBATCH --job-name=ldyn
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p slurm/logs

CONFIG="${CONFIG:-configs/coral_data_scaling.yaml}"
STAGES="${STAGES:-train,morse}"
PYTHON="${PYTHON:-../.venv/bin/python}"
DEVICE="${DEVICE:-cpu}"
MAX_SEEDS="${MAX_SEEDS:-}"
EXPECTED_CELLS="${EXPECTED_CELLS:-}"

args=(
  pipeline.py
  --config "$CONFIG"
  --stages "$STAGES"
  --cell-index "${SLURM_ARRAY_TASK_ID}"
  --device "$DEVICE"
  --skip-completed
)

if [[ -n "$MAX_SEEDS" ]]; then
  args+=(--max-seeds "$MAX_SEEDS")
fi

if [[ -n "$EXPECTED_CELLS" ]]; then
  args+=(--expected-cells "$EXPECTED_CELLS")
fi

"$PYTHON" "${args[@]}"
