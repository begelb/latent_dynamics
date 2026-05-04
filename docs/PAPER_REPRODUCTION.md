# Reproducing the paper figures

Every computational figure in `paper/main.tex` is driven by one YAML config
under `configs/` and the staged pipeline in `src/latentdynamics/cli/`.

## Setup

Python 3.13 is required. The pre-built `.venv/` at the project root has every
dependency pinned (Torch 2.11, CMGDB 1.3.2, pydantic 2). To install or
re-create it:

```bash
python3.13 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## Single command per figure

The `reproduce_paper.py` script at the repository root defaults to
`render,metrics`, so it reads saved DOT/CSV/checkpoint artefacts and does not
invoke training or CMGDB. Use `--stages all` only when you intend to retrain
and recompute. The `data` stage is non-destructive: existing CSV+metadata
pairs are kept as source artefacts, and adaptive coral datasets are validated
as precomputed inputs.

```bash
# Render/metrics for all configured figures from saved artefacts:
python reproduce_paper.py

# One figure:
python reproduce_paper.py --only fig_coral_basic

# Full retrain/recompute for one seed:
python reproduce_paper.py --only fig_chafee_infante --stages all --max-seeds 1

# Sweep smoke check on laptop (limit to 3 seeds):
python reproduce_paper.py --only fig_coral_data_scaling --max-seeds 3
```

For AMAREL, use `pipeline.py --dry-run` to get the cell count and submit
`slurm/pipeline_array.sbatch`; see `docs/AMAREL.md`.

| Paper reference   | Experiment id                  | Config                              | Notes                                        |
| ----------------- | ------------------------------ | ----------------------------------- | -------------------------------------------- |
| Fig. 1.83         | `fig_leslie_contraction`       | `configs/leslie_contraction.yaml`   | 2D Leslie + contracting tail; CMGDB smin=28, smax=29 |
| Fig. 1.214        | `fig_leslie3d_spurious`        | `configs/leslie3d_spurious.yaml`    | legacy checkpoint; tau-bar < max semiconjugacy |
| Sec. 1.211 success | `fig_leslie3d_success`         | `configs/leslie3d_success.yaml`     | 3 hidden layers; tau-bar > max semiconjugacy |
| Sec. 1.256 PDE    | `fig_chafee_infante`           | `configs/chafee_infante.yaml`       | additive loss, ODE time-tau map              |
| Fig. 1.376        | `fig_coral_basic`              | `configs/coral_basic.yaml`          | 1D Morse graph, 13D coral; bistable          |
| Fig. 1.469        | `fig_coral_data_scaling`       | `configs/coral_data_scaling.yaml`   | 6 sizes x 30 seeds; sobol sampling           |
| Fig. 1.528        | `fig_coral_adaptive`           | `configs/coral_adaptive.yaml`       | augmented dataset; M = 100,200,300,400,500   |

## Expected outputs

For one-shot experiments (e.g. `fig_chafee_infante`):

```
output/chafee_infante/
  seed_0/
    final_losses.txt
    logs/history.json
    models/autoencoder.pt    # state_dict (weights_only=True safe)
    models/autoencoder.json  # architecture sidecar
    MG/
      morse_graph.pdf
      morse_graph.png
      morse_sets.pdf
      morse_sets.png
      morse_sets               # CSV with [a, b, label] columns
    mg_params_log.txt
    metrics.json             # per-seed paper metric where implemented
```

For sweep experiments (`fig_coral_data_scaling`):

```
output/coral/
  train_<N>/
    seed_<k>/
      ... (as above, per seed)
```

Adaptive coral uses the same layout with `train_500_<M>_adaptive/`:

```
output/coral/
  train_500_<M>_adaptive/
    seed_<k>/
    ... (as above, per seed)
```

## Figure postprocessing overlays

The expensive CMGDB stage writes `MG/morse_sets` via `CMGDB.SaveMorseSets`.
Postprocessing should read that saved CSV instead of recomputing CMGDB:

```python
from latentdynamics.viz import plot_morse_sets_from_csv

plot = plot_morse_sets_from_csv("output/coral/seed_0/MG/morse_sets")
plot.ax.scatter([0.0], [plot.label_to_y[0]], color="black", zorder=10)
plot.fig.savefig("output/coral/seed_0/MG/morse_sets_overlay.png")
```

The base plotter only draws Morse sets. Figure-specific code can then add
points, lines, shaded regions, arrows, or annotations on the returned axis.

## Modular hyperparameters

The `arch` block has shared defaults plus per-component overrides:

```yaml
arch:
  num_layers: 2
  hidden_shape: 64
  high_dims: 64
  low_dims: 2
  activation: tanh
  encoder_out_activation: none
  latent_out_activation: none
  decoder_out_activation: none
  encoder:
    hidden_shapes: [64, 32]
  latent_map:
    hidden_shapes: [32, 32]
  decoder:
    hidden_shapes: [32, 64]
```

Use `hidden_shapes` for asymmetric networks. Use the flat
`num_layers`/`hidden_shape` fields for repeated-width networks. Training
hyperparameters live in `training` (`learning_rate`, `batch_size`, `epochs`,
`patience`, `loss_weights`, `loss_mode`, scheduler settings, and optional
`gradient_clip_norm`). Raw-coordinate experiments can set `data.scaling: none`.
CMGDB bounds can be inferred from encoded data or fixed with
`cmgdb.lower_bounds` and `cmgdb.upper_bounds`.

## Legacy data import

Archived source data can be re-imported without blind overwrites:

```bash
python scripts/import_legacy_data.py --dry-run
python scripts/import_legacy_data.py
```

The importer checks Brittany's coral/Leslie data against `code/data`, converts
Marcio's headerless `archive/marcio/scripts/train_data.csv` into the active
Chafee-Infante CSV format, backs up replaced files under
`data/_pre_import_backup/<timestamp>/`, and writes
`data/legacy_import_manifest.json`. Marcio did not save a separate test split,
so the imported `test.csv` mirrors the archived training pairs and is marked
as `test_mirror_of_train` in metadata.

## Determinism caveats

`tests/test_reproducibility.py` verifies that the same `--seed` produces
bitwise-identical state dicts on the same machine. Cross-machine determinism
is not guaranteed (BLAS / cuDNN / MPS heuristics can vary). For
publication-grade exact-match reproducibility, run on the same hardware as
the paper authors with `torch.use_deterministic_algorithms(True)`.

## CMGDB version pinning

The Morse-graph computation depends on `CMGDB==1.3.2`. Newer versions may
change the `BoxMap` or `ComputeConleyMorseGraph` signatures. The pin is in
`pyproject.toml`.

## Legacy artefacts in `code/`

The pre-restructure code lives under `code/legacy/` and `../archive/`.
Brittany's three-file pickled `nn.Module` checkpoints are loadable without
rewriting them through `latentdynamics.training.load_legacy_checkpoint`.
`scripts/migrate_legacy_checkpoints.py` remains available if you want a new
state_dict + sidecar copy for separate analysis.

## Map of figures to specific commands

```bash
# Basic sanity on a laptop (~20 minutes per experiment without sweeps):
python reproduce_paper.py --only fig_coral_basic --max-seeds 1
python reproduce_paper.py --only fig_chafee_infante --max-seeds 1

# Cluster-scale full reproduction:
sbatch slurm/coral_data_scaling.sbatch    # not yet present; M6+
python reproduce_paper.py                  # everything sequentially
```
