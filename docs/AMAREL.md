# AMAREL and Long-Run Workflow

The package is organized around one staged pipeline for all examples:

| Source | Paper examples | Current config(s) |
| --- | --- | --- |
| Brittany | 3D Leslie, coral sweeps, adaptive coral | `leslie3d_spurious.yaml`, `leslie3d_success.yaml`, `coral_*.yaml` |
| Patrick using Brittany's code | 10D Leslie contraction | `leslie_contraction.yaml` |
| Marcio | Chafee-Infante spectral PDE | `chafee_infante.yaml` |

Each config expands to independent cells: `(train_file, seed, output_dir)`.
Use a dry run to inspect the cell plan without generating data, training, or
calling CMGDB:

```bash
cd code
../.venv/bin/python pipeline.py --config configs/coral_data_scaling.yaml --dry-run
```

## Stage Contract

Stages are intentionally explicit:

- `data`: creates missing CSV/metadata pairs only; existing pairs are kept.
- `scale`: writes `scaler.gz` for one train file.
- `train`: writes checkpoint, architecture sidecar, history, and final losses.
- `morse`: runs CMGDB and writes only `MG/morse_graph`, `MG/morse_sets`, and
  `mg_params_log.txt`.
- `render`: reads saved Morse artifacts and writes PDF/PNG figures.
- `metrics`: reads saved checkpoints/Morse artifacts and writes `metrics.json`.

Do not use full CMGDB as a smoke test. Lightweight checks should run config
loading, cell planning, data/scale on tiny temporary data, and short training
tests. Fine CMGDB runs belong on AMAREL or another long-run machine.

## Slurm Arrays

Prepare shared data/scalers once before launching a train/Morse array:

```bash
cd code
../.venv/bin/python pipeline.py --config configs/coral_data_scaling.yaml --dry-run
../.venv/bin/python pipeline.py --config configs/coral_data_scaling.yaml --stages data,scale --skip-completed
```

Then launch one cell per array task. **Important**: do not put comma-laden
values like `STAGES=train,morse` inside `--export` — slurm parses commas in
`--export` as separators, so `STAGES=train,morse` becomes `STAGES=train`
plus a stray `morse` variable, and the `morse` stage is silently dropped.
Export env vars first, then list them by name:

```bash
CONFIG=configs/coral_data_scaling.yaml STAGES=train,morse EXPECTED_CELLS=180 \
  sbatch --array=0-179 --export=ALL,CONFIG,STAGES,EXPECTED_CELLS \
  slurm/pipeline_array.sbatch
```

For a one-seed retrain:

```bash
../.venv/bin/python pipeline.py --config configs/chafee_infante.yaml --stages data,scale --max-seeds 1 --skip-completed
CONFIG=configs/chafee_infante.yaml STAGES=train,morse MAX_SEEDS=1 EXPECTED_CELLS=1 \
  sbatch --array=0-0 --export=ALL,CONFIG,STAGES,MAX_SEEDS,EXPECTED_CELLS \
  slurm/pipeline_array.sbatch
```

The template always passes `--skip-completed`, so rerunning the same array
resumes from saved artifacts instead of repeating completed stages. Increase
`--time`, `--mem`, or add AMAREL-specific partition/GPU directives in
`slurm/pipeline_array.sbatch` as needed.

## Local Smoke Commands

```bash
cd code
../.venv/bin/python -m pytest -m "not slow"
../.venv/bin/python scripts/import_legacy_data.py --dry-run
../.venv/bin/python pipeline.py --config configs/chafee_infante.yaml --dry-run
../.venv/bin/python pipeline.py --config configs/coral_data_scaling.yaml --dry-run --max-seeds 2
```

These commands do not run CMGDB. Use `--stages render,metrics` to test saved
artifact readers, and use `--stages all` only when retraining/recomputing is
intentional.
