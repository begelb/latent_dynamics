# LatentDynamics

Topological validation (Conley-Morse graphs) of autoencoder-learned latent
dynamics for high-dimensional discrete maps and PDEs. Companion code for the
manuscript *Rigorously Characterizing High-dimensional Dynamics by
Combinatorial-Topological Methods on a Latent Space*.

## Layout

```
code/
├── pyproject.toml                 # pip install -e .
├── reproduce_paper.py             # one entry point per paper figure
├── README.md
├── docs/PAPER_REPRODUCTION.md     # figure -> command map
├── docs/AMAREL.md                 # cluster/array workflow
├── src/latentdynamics/            # the importable package
│   ├── systems/                   # ground-truth dynamics
│   ├── sampling/                  # initial-condition + trajectory generation
│   ├── config/                    # pydantic v2 schema + YAML loader
│   ├── models/                    # encoder / latent map / decoder
│   ├── training/                  # trainer, losses, safe state_dict checkpoints
│   ├── analysis/                  # CMGDB wrapper + tau-bar tolerance + metrics
│   ├── viz/                       # one source for palette and plots
│   └── cli/                       # entry-points used by /scripts
├── configs/                       # one YAML per experiment + _shared/defaults
├── scripts/                       # thin CLI wrappers
└── tests/                         # 120+ pytest cases
```

The archived single-script sources are preserved under `code/legacy/` and
`../archive/`. Do not delete or rewrite saved `data/` or `output/` artefacts
while validating figure parity; the new pipeline can render and compute
metrics from them directly.

## Quick start

The Python virtualenv lives at the workspace root (`../.venv`). Activate or
invoke directly:

```bash
# From the repo root (this directory):
../.venv/bin/pip install -e ".[dev]"
../.venv/bin/pytest -m "not slow"   # ~2 s
../.venv/bin/pytest                  # full suite

# Re-render one paper figure from saved Morse/checkpoint artefacts:
../.venv/bin/python reproduce_paper.py --only fig_coral_basic --max-seeds 1

# For configs marked paths.read_only=true, derived outputs go to output/replay/<config-stem>/.
../.venv/bin/python pipeline.py --config configs/coral_basic.yaml --stages render,metrics

# Opt into a retrain/recompute. The data stage preserves existing CSVs.
../.venv/bin/python reproduce_paper.py --only fig_chafee_infante --stages all --max-seeds 1
```

Detailed figure -> command mapping is in `docs/PAPER_REPRODUCTION.md`.
Cluster execution notes and the Slurm array template are in `docs/AMAREL.md`
and `slurm/pipeline_array.sbatch`.
Archived Brittany/Marcio data can be audited or restored with
`scripts/import_legacy_data.py`.

## Pipeline

Each paper experiment is a config-driven sequence:

1. `make_data.run(cfg)` — generate missing train/test trajectory CSVs and
   metadata from the chosen system. Existing CSV+metadata pairs are treated as
   source artefacts and are not overwritten. Adaptive coral datasets are
   validated as precomputed inputs.
2. `scale_data.run(cfg, train_file)` — fit either a MinMax scaler or an
   identity scaler (`data.scaling: none`) and persist it as joblib.
3. `train.run(cfg, train_file, seed)` — train the unified
   `LatentDynamicsAutoencoder` (encoder + latent map + decoder). The `arch`
   schema supports shared defaults plus per-component hidden-width lists,
   activations, and terminal activations; `training` controls optimizer,
   scheduler, clipping, loss mode, and loss weights. Training emits a single
   state_dict and an architecture sidecar.
4. `diagnose.run(cfg)` — iterate the learned latent map on a grid and write a
   cheap diagnostic (`diagnose.json` plus point-cloud/orbit plots) before
   spending time on CMGDB.
5. `morse_graph.run(cfg)` — infer latent bounds, build the CMGDB box map,
   or use fixed `cmgdb.lower_bounds` / `cmgdb.upper_bounds`, then compute the
   Conley-Morse graph.
6. `render_stage(cfg)` — re-render Morse graph/set plots and experiment extras
   from saved DOT/CSV/checkpoint artefacts only; it does not invoke CMGDB.
7. Per-experiment metric (`unique_membership` for coral, tau-bar tolerance
   for the Leslie failure case, etc.).

Preserved paper configs set `paths.read_only: true`. In that mode `data`,
`scale`, `train`, and `morse` are blocked unless `--force-overwrite` is passed.
Derived stages (`diagnose`, `render`, `metrics`, and `run_manifest.json`) read
the source artefacts but write to `output/replay/<experiment_name>/...` by
default, or to a custom `--replay-root`. This makes replay deterministic
without dirtying tracked `data/` or `output/` artefacts.

Long runs can be split into independent cells:

```bash
../.venv/bin/python pipeline.py --config configs/coral_data_scaling.yaml --dry-run
../.venv/bin/python pipeline.py --config configs/coral_data_scaling.yaml --cell-index 0 --skip-completed
```

Each cell is one `(train_file, seed)` pair, which maps directly to a Slurm
array task on AMAREL.

Saved Morse-set CSVs can be reopened as overlay-ready matplotlib figures:

```python
from latentdynamics.viz import plot_morse_sets_from_csv

plot = plot_morse_sets_from_csv("output/coral/seed_0/MG/morse_sets")
plot.ax.scatter([0.0], [plot.label_to_y[0]], color="black", zorder=10)
plot.fig.savefig("output/coral/seed_0/MG/morse_sets_overlay.png")
```

This keeps CMGDB computation, saved artefacts, and figure postprocessing as
separate steps.

## Adding a new experiment

1. Drop a YAML in `configs/<name>.yaml` (deep-merged with
   `configs/_shared/defaults.yaml`).
2. Register it in `reproduce_paper.py::EXPERIMENTS`.
3. Cover the routing/config behavior in `tests/test_experiments.py` and add
   focused tests for any new system, metric, or renderer.

## CMGDB pin

`pyproject.toml` pins `CMGDB==1.3.2`; newer versions may break the `BoxMap` /
`ComputeConleyMorseGraph` contract.
