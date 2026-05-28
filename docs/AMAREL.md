# AMAREL and Long-Run Workflow

The package is organized around one staged pipeline for all examples:

| Paper examples | Current config(s) | Source |
| --- | --- | --- |
| 3D Leslie, coral sweeps, adaptive coral | `leslie3d_example1.yaml`, `leslie3d_example2.yaml`, `coral_*.yaml` | read-only replay from the preserved tree |
| 10D Leslie contraction | `leslie_2gen_contraction.yaml` | fresh package retrain (seed 20, subdiv 27/29/30); code-defined system, fully reproducible |
| Chafee-Infante spectral PDE | `chafee_infante.yaml` | reference data/config matched |

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
- `morse`: runs CMGDB and writes `MG/morse_graph`, `MG/morse_sets`, and
  `mg_params_log.txt`. If `cmgdb.compute_roa: true`, it also builds
  `MG/regions_of_attraction_exact.npz` on CMGDB's returned `MapGraph`.
- `render`: reads saved Morse artifacts and writes PDF/PNG figures. When an
  exact RoA artifact is present, render writes `MG/regions_of_attraction_exact.png`;
  otherwise 2-D runs fall back to the diagnostic 128x128 RoA overlay.
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

## Test 1101 ED-Cycle Sweep

The `test_1101` configs use loss weights `[1, 1, 0, 1]`, i.e. reconstruction,
input-space prediction, no direct semiconjugacy loss, and the predicted-latent
cycle loss `||E(D(G(E(x)))) - G(E(x))||^2`. This is the experiment family for
testing whether `DE ~= I_X`, `f ~= DGE`, and `ED ~= I_Z` on `GE(x)` recover
`Ef ~= GE`.

These configs also set `cmgdb.compute_roa: true`, so the `morse` stage computes
the same-cell RoA artifact before `render` makes the RoA figure. Existing
Morse outputs created before this flag was added need a fresh `morse` run
(`--force-overwrite` if the old `MG/morse_graph` and `MG/morse_sets` are still
present).

Configs:

```bash
configs/leslie2d_to_2d_test_1101.yaml
configs/leslie_2gen_contraction_test_1101.yaml
configs/leslie3d_example1_test_1101.yaml
configs/leslie3d_example2_test_1101.yaml
configs/chafee_infante_test_1101.yaml
```

Dry-run every config before submitting:

```bash
cd code
for CONFIG in \
  configs/leslie2d_to_2d_test_1101.yaml \
  configs/leslie_2gen_contraction_test_1101.yaml \
  configs/leslie3d_example1_test_1101.yaml \
  configs/leslie3d_example2_test_1101.yaml \
  configs/chafee_infante_test_1101.yaml
do
  ../.venv/bin/python pipeline.py --config "$CONFIG" --dry-run
done
```

Prepare shared data/scalers for each config:

```bash
cd code
for CONFIG in \
  configs/leslie2d_to_2d_test_1101.yaml \
  configs/leslie_2gen_contraction_test_1101.yaml \
  configs/leslie3d_example1_test_1101.yaml \
  configs/leslie3d_example2_test_1101.yaml \
  configs/chafee_infante_test_1101.yaml
do
  ../.venv/bin/python pipeline.py --config "$CONFIG" --stages data,scale --skip-completed
done
```

Submit one Slurm array per config. Keep `STAGES` in the shell environment
rather than writing `STAGES=train,morse` directly inside `--export`.

```bash
cd code
for CONFIG in \
  configs/leslie2d_to_2d_test_1101.yaml \
  configs/leslie_2gen_contraction_test_1101.yaml \
  configs/leslie3d_example1_test_1101.yaml \
  configs/leslie3d_example2_test_1101.yaml \
  configs/chafee_infante_test_1101.yaml
do
  STAGES=train,diagnose,morse,render,metrics EXPECTED_CELLS=1 \
    sbatch --array=0-0 --export=ALL,CONFIG,STAGES,EXPECTED_CELLS \
    slurm/pipeline_array.sbatch
done
```

## Local Smoke Commands

```bash
cd code
../.venv/bin/python -m pytest -m "not slow"
../.venv/bin/python pipeline.py --config configs/chafee_infante.yaml --dry-run
../.venv/bin/python pipeline.py --config configs/coral_data_scaling.yaml --dry-run --max-seeds 2
```

These commands do not run CMGDB. Use `--stages render,metrics` to test saved
artifact readers, and use `--stages all` only when retraining/recomputing is
intentional.
