# Reproducing the paper figures

Every computational figure in `paper/main.tex` is represented by one YAML
config under `configs/` and the staged pipeline in `src/latentdynamics/cli/`.
The configs are meant to encode the archived historical runs as far as the
saved files allow, while sharing one higher-level workflow across Leslie,
Chafee-Infante, and Coral examples.

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
`render,metrics`, so it reads saved DOT/CSV/checkpoint artifacts and does not
invoke training or CMGDB. Some paper figures are still partial because their
archives lack non-empty checkpoints or raw CMGDB DOT/CSV files; the contracts
below state which commands are expected to work today. Use `--stages all` only
when you intend to retrain and recompute. The `data` stage is non-destructive:
existing CSV+metadata pairs are kept as source artifacts, and adaptive coral
datasets are validated as precomputed inputs.

```bash
# Render/metrics for one replay-ready figure:
python reproduce_paper.py --only fig_leslie3d_spurious

# One known-good coral sweep cell:
python pipeline.py --config configs/coral_data_scaling.yaml --stages render,metrics --cell-index 120 --expected-cells 180

# Full retrain/recompute for one seed:
python reproduce_paper.py --only fig_chafee_infante --stages all --max-seeds 1

# Sweep smoke check on laptop (limit to 3 seeds):
python reproduce_paper.py --only fig_coral_data_scaling --max-seeds 3
```

For AMAREL, use `pipeline.py --dry-run` to get the cell count and submit
`slurm/pipeline_array.sbatch`; see `docs/AMAREL.md`.

| Paper reference   | Experiment id                  | Config                              | Contract                                                 | Notes                                        |
| ----------------- | ------------------------------ | ----------------------------------- | -------------------------------------------------------- | -------------------------------------------- |
| Fig. 1.83         | `fig_leslie_contraction`       | `configs/leslie_contraction.yaml`   | [contract](figure_contracts/leslie_contraction.md)       | read-only replay from Patrick `Leslie10D`; source script/raw CSVs missing |
| Fig. 1.214        | `fig_leslie3d_spurious`        | `configs/leslie3d_spurious.yaml`    | [contract](figure_contracts/leslie3d_spurious.md)        | replay-ready (legacy 3-file checkpoint); read_only |
| Sec. 1.211 success| `fig_leslie3d_success`         | `configs/leslie3d_success.yaml`     | [contract](figure_contracts/leslie3d_success.md)         | read-only replay from Patrick `Leslie3D`; source script/raw CSVs missing |
| Sec. 1.256 PDE    | `fig_chafee_infante`           | `configs/chafee_infante.yaml`       | [contract](figure_contracts/chafee_infante.md)           | Marcio data/config matched; weights conversion and raw CMGDB DOT/CSV still missing |
| Fig. 1.376        | `fig_coral_basic`              | `configs/coral_basic.yaml`          | [contract](figure_contracts/coral_basic.md)              | read-only Brittany replay blocked by 0-byte `train_500` checkpoints |
| Fig. 1.469        | `fig_coral_data_scaling`       | `configs/coral_data_scaling.yaml`   | [contract](figure_contracts/coral_data_scaling.md)       | partial read-only Brittany replay: selected `train_100`, complete `train_2000` |
| Fig. 1.528        | `fig_coral_adaptive`           | `configs/coral_adaptive.yaml`       | [contract](figure_contracts/coral_adaptive.md)           | partial read-only Brittany replay: M = 100, 200, 300; 400, 500 blocked |

For per-figure expected outputs, hyperparameter audits with archive
line citations, status, and verification recipes, see the contracts
linked above.

## Interpretation of reproduction

There are two distinct modes:

1. **Replay** reads preserved data, scalers, checkpoints, CMGDB DOT/CSV files,
   and logs. It is the default mode for inspecting paper figures and metrics.
2. **Fresh reproduction** regenerates missing artifacts by rerunning data,
   training, and CMGDB stages. It is useful for filling archive gaps and for
   robustness checks, but it is not guaranteed to land on the same Morse graph
   for every seed, hardware backend, or optimizer trajectory.

The paper's theoretical guarantee is conditional. Once a run has the required
learned map, CMGDB structure, and verified error/tolerance bounds, the
corresponding structure is certified for the original dynamics. The pipeline
therefore treats diagnostics, CMGDB outputs, and paper-specific metrics as
validation gates rather than decorative outputs.

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
replay_sources/coral/
  train_<N>/
    seed_<k>/
      ... (as above, per seed)
```

Adaptive coral uses the same layout with `train_500_<M>_adaptive/`:

```
replay_sources/coral/
  train_500_<M>_adaptive/
    seed_<k>/
    ... (as above, per seed)
```

## Figure postprocessing overlays

The expensive CMGDB stage writes `MG/morse_sets` via `CMGDB.SaveMorseSets`.
Postprocessing should read that saved CSV instead of recomputing CMGDB:

```python
from latentdynamics.viz import plot_morse_sets_from_csv

plot = plot_morse_sets_from_csv("replay_sources/coral/seed_0/MG/morse_sets")
plot.ax.scatter([0.0], [plot.label_to_y[0]], color="black", zorder=10)
plot.fig.savefig("replay/coral_basic/seed_0/MG/morse_sets_overlay.png")
```

The base plotter only draws Morse sets. Figure-specific code can then add
points, lines, shaded regions, arrows, or annotations on the returned axis.
Use this layer for paper or slide polish: crops, axis limits, trajectory
overlays, highlighted Morse sets, and label placement should be explicit
render/postprocessing steps, not manual edits to preserved CMGDB artifacts.

## Modular hyperparameters

The `arch` block specifies hyperparameters per network (`encoder`, `latent_map`, `decoder`). Shared values at the top level act as defaults that any component may override.

Most paper configs are symmetric — all three networks share width, depth, and activation — and use the terse shortcut form:

```yaml
arch:
  num_layers: 3
  hidden_shape: 64
  high_dims: 13
  low_dims: 1
```

Asymmetric architectures, or per-network activation choices, use the per-component form. Each component block accepts `hidden_shapes`, `num_layers` + `hidden_shape`, `activation`, and `out_activation`:

```yaml
arch:
  high_dims: 64
  low_dims: 2
  activation: tanh                  # shared default
  encoder:
    hidden_shapes: [64, 32]         # asymmetric widths
    out_activation: none
  latent_map:
    hidden_shapes: [32, 32]
    activation: relu                # override shared default
    out_activation: none
  decoder:
    hidden_shapes: [32, 64]
    out_activation: none
```

Shared `arch.num_layers` / `arch.hidden_shape` are optional. If every component supplies its own `hidden_shapes`, the shared fields may be omitted entirely (see `configs/chafee_infante.yaml`). If any component lacks an explicit width specification, either provide it per-component or fall back to the shared fields.

A worked example exercising asymmetric widths, per-network activations, and an `out_activation` override lives at `configs/scratch/asymmetric_example.yaml`.

Training hyperparameters live under `training`. Data and CMGDB settings are unchanged by this refactor.

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

## Legacy artifacts in `code/`

The pre-restructure code lives under `code/legacy/` and `../archive/`.
Archived three-file pickled `nn.Module` checkpoints are loadable without
rewriting them through `latentdynamics.training.load_legacy_checkpoint`.
`scripts/migrate_legacy_checkpoints.py` remains available if you want a new
state_dict + sidecar copy for separate analysis.

## Map of figures to specific commands

```bash
# Basic replay sanity on a laptop:
python reproduce_paper.py --only fig_leslie3d_spurious
python pipeline.py --config configs/coral_data_scaling.yaml --stages render,metrics --cell-index 120 --expected-cells 180

# Cluster-scale jobs use the generic array template after checking the cell count:
python pipeline.py --config configs/coral_data_scaling.yaml --dry-run
# Then submit slurm/pipeline_array.sbatch with the reported EXPECTED_CELLS.
```
