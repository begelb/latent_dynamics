# LatentDynamics

Companion code for *Rigorously Characterizing High-dimensional Dynamics by
Combinatorial-Topological Methods on a Latent Space* (`paper/main.tex`).

For each high-dimensional discrete map `f : X -> X`, the package trains an
autoencoder `(E, D)` and a latent dynamics `G : Z -> Z`, then computes a Morse
graph for `G` with [CMGDB](https://github.com/bernardorivas/CMGDB) and runs the
paper's tolerance / success-metric checks against the result. The seven
applications in Section 5 of the paper are encoded as YAML configs and a single
`reproduce_paper.py` entry point.

## What the paper asks of the code

For data points `(x_t, x_{t+1})` with `x_{t+1} = f(x_t)`, the trainer minimizes

```
L1(x_t, x_{t+1}) = || D(E(x_t))     - x_t   ||^2     # reconstruction
L2(x_t, x_{t+1}) = || D(G(E(x_t)))  - x_{t+1} ||^2   # prediction
L3(x_t, x_{t+1}) = || G(E(x_t))     - E(x_{t+1}) ||^2 # semiconjugacy error
L4(x_t, x_{t+1}) = || E(D(G(E(x_t)))) - G(E(x_t)) ||^2 # ED cycle on predicted latent states
```

with weights `(w1, w2, w3)` or `(w1, w2, w3, w4)`. Existing three-weight
configs set `w4 = 0`. After training, CMGDB returns a Morse graph
`MG(G)` over the latent box; for each minimal node `q` the package can:

- compute the **tolerance** `tau(N_q, G)` of the corresponding attracting block
  `N_q = |pi^{-1}(q)|` (Theorem 4.4 / Theorem `thm:main_alt`); the run passes the
  hypothesis check when
  `sup_{x in E^{-1}(N_q)} || G(E(x)) - E(f(x)) || <= tau(N_q, G)`;
- compute the **success metric** `1_{x*}` (Section 5.5.1) for known reference
  points `S = {a_0, a_1, r}`, requiring that an attractor's encoded image lie in
  a unique minimal `|pi^{-1}(q)|` while the unstable point lies in a non-minimal
  one.

The conclusion `Inv(E^{-1}(N), f) != \emptyset` (Corollary `cor`) is what gets
certified for `f` once the tolerance hypothesis verifies for `G`.

## Paper-experiment map

Every figure in `paper/main.tex` Section 5 has one YAML and one figure
contract. Detailed hyperparameter audits, archive line citations, and per-figure
verification recipes live in `docs/figure_contracts/`.

| Paper       | Section / Fig.            | Experiment id              | Config                              | Reproducibility status |
| ----------- | ------------------------- | -------------------------- | ----------------------------------- | --- |
| 10D Embedded Leslie  | §5.2,  Fig. `lesliecontraction_dynamics`  | `fig_leslie_2gen_contraction`   | `configs/leslie_2gen_contraction.yaml`   | fresh retrain (seed 20); fully reproducible from the config |
| 3D Leslie spurious   | §5.3.1, Fig. `3D_Leslie_latent_dynamics`  | `fig_leslie3d_example1`    | `configs/leslie3d_example1.yaml`    | read-only replay (provided 3-file checkpoint) |
| 3D Leslie success    | §5.3.2, Fig. `3D_Leslie_latent_dynamics_success` | `fig_leslie3d_example2` | `configs/leslie3d_example2.yaml` | read-only replay (provided checkpoints) |
| Chafee-Infante PDE   | §5.4,  Fig. `ci_morse_graph_dynamics`     | `fig_chafee_infante`       | `configs/chafee_infante.yaml`       | read-only replay (provided model); fresh-retrain variant available |
| Coral basic          | §5.5,   Fig. `coral_latent_dynamics`      | `fig_coral_basic`          | `configs/coral_basic.yaml`          | read-only replay; fresh retrain available |
| Coral data-scaling   | §5.5.2, Fig. `coral_success_rates_init`   | `fig_coral_data_scaling`   | `configs/coral_data_scaling.yaml`   | read-only replay (partial seeds); fresh sweep available |
| Coral adaptive       | §5.5.3, Fig. `coral_success_rates_adaptive` | `fig_coral_adaptive`     | `configs/coral_adaptive.yaml`       | read-only replay (partial `M`); fresh sweep available |

Hyperparameters in the configs are a superset of paper Tables 3, 4, 5
(architecture, training, data), recovered either from the archived run logs or
from `paper/main.tex` itself when the archives don't pin them. See the per-figure
contracts for line-by-line provenance.

### Key parameter values from the paper

The configs encode these directly; this table is for reference.

| Experiment        | Ambient `n` | Latent | Hidden L / W | `(w_1, w_2, w_3)` | Sampling | `(N_train, N_test, T)`         | CMGDB `(init/min/max)` |
| ----------------- | ----------: | -----: | ------------ | ----------------- | -------- | ------------------------------ | ---------------------- |
| 10D Leslie        | 10          | 2      | 4 / 64       | `(100, 10, 20)`   | uniform  | `(8000, 2000, 20)`             | `27 / 29 / 30`         |
| 3D Leslie spurious| 3           | 2      | 3 / 32       | `(10, 10, 1)`     | uniform  | `(3200, 800, 30)`              | `23 / 23 / 27`         |
| 3D Leslie success | 3           | 2      | 2 / 64       | `(100, 10, 20)`   | uniform  | `(8000, 2000, 20)`             | `25 / 28 / 29`         |
| Chafee-Infante    | 64          | 2      | 2 / 64*      | `(1, 1, 0)` add.  | uniform  | `(1000, 200, 30)` (no scaling) | `10 / 14 / 28`         |
| Coral basic       | 13          | 1      | 3 / 64       | `(10, 10, 1)`     | uniform  | `([500], 10000, 20)`           | shared default         |
| Coral data-scaling| 13          | 1      | 3 / 64       | `(10, 10, 1)`     | Sobol    | `({100..5000}, 10000, 20)` × 30 seeds | shared default |
| Coral adaptive    | 13          | 1      | 3 / 64       | `(10, 10, 1)`     | adaptive | `(500 + M, 10000, 20)`, M ∈ {100..500} × 30 seeds | shared default |

`*` Chafee-Infante uses asymmetric per-component widths
(`encoder 64->64->32->2, latent 2->32->32->2, decoder 2->32->64->64`).
Every config is fully self-contained: each YAML spells out every tunable
hyperparameter (Adam settings, ReduceLROnPlateau parameters, scaling, CMGDB
subdivisions, paths, seeds) so every knob is visible at a glance.

## Layout

```
code/
├── README.md                         # this file
├── pyproject.toml                    # pip install -e .  (CMGDB from a git fork)
├── reproduce_paper.py                # one entry point per paper figure
├── pipeline.py                       # single-config staged runner
├── docs/
│   ├── PAPER_REPRODUCTION.md         # figure -> command map, in detail
│   ├── AMAREL.md                     # cluster + Slurm array workflow
│   └── figure_contracts/             # one .md per paper figure, line-cited
├── configs/                          # one fully-explicit YAML per experiment
│   └── scratch/                      # local examples or user-created writable copies
├── src/latentdynamics/
│   ├── systems/      # ground-truth f: LeslieContraction, LeslieModel3D, RedCoralModel, ChafeeInfante
│   ├── sampling/     # uniform / Sobol / adaptive trajectory generation, scaling
│   ├── models/       # Encoder + LatentMap + Decoder (per-component MLP widths)
│   ├── training/     # trainer, L1+L2+L3 losses, checkpoints (state_dict + arch sidecar)
│   ├── analysis/     # CMGDB wrapper, tau-bar tolerance, coral unique-membership metric
│   ├── viz/          # palette, Morse graph/set rendering, plot_morse_sets_from_csv
│   ├── config/       # pydantic v2 schema + YAML loader
│   └── cli/          # entry-points consumed by reproduce_paper.py / pipeline.py
├── scripts/
│   └── migrate_legacy_checkpoints.py # convert 3-file pickled modules -> state_dict + sidecar
├── slurm/pipeline_array.sbatch       # AMAREL array template
└── tests/                            # 140+ pytest cases
```

The pre-restructure single-script versions are preserved under `code/legacy/`
and `../archive/`; do not delete or rewrite saved `data/` or `output/` artifacts
while validating figure parity.

## Pipeline (per experiment)

`pipeline.run` chains the same seven stages for every config; individual stages
are exposed as CLI entry points under `latentdynamics.cli.*`.

1. `make_data` — build `D(T, N) = {(f^{k-1}(x_i), f^k(x_i))}` from
   `system.name` + `system.params`. Existing CSV+metadata pairs are **never**
   overwritten; adaptive coral datasets are validated as precomputed inputs.
2. `scale_data` — fit `MinMaxScaler(0, 1)` on `vstack(x_train, y_train)` and
   joblib it; or `data.scaling: none` for raw-coordinate runs (Chafee-Infante).
3. `train` — train the unified `LatentDynamicsAutoencoder` (encoder + latent
   map + decoder) under
   `loss_total = w1 * L1 + w2 * L2 + w3 * L3 + w4 * L4`, Adam +
   `ReduceLROnPlateau`, optional `gradient_clip_norm`. Saves a single
   `models/autoencoder.pt` (`state_dict`, `weights_only=True` safe) and a
   companion `models/autoencoder.json` architecture sidecar.
4. `diagnose` — iterate `G` on a latent grid and write `diagnose.json` plus
   point-cloud / orbit plots before paying the CMGDB cost.
5. `morse_graph` — derive the latent box (encode train+test, expand the first
   dim by `bounds_epsilon_frac`) or take fixed `cmgdb.lower_bounds` /
   `upper_bounds`, build the box map (`padding` configurable), call
   `CMGDB.ComputeConleyMorseGraph`. Writes `MG/morse_graph[.pdf,.png]`,
   `MG/morse_sets[.pdf,.png]`, `MG/morse_sets` (CSV), `mg_params_log.txt`.
6. `render_stage` — re-renders Morse-graph / Morse-set plots from the saved
   DOT/CSV/checkpoint only. Does not invoke CMGDB.
7. `metrics` — per-experiment paper metric. For coral: the
   `unique_membership` metric `1_{x*}` from §5.5.1. For 3D Leslie: the tau-bar
   tolerance check from §5.3.

Configs marked `paths.read_only: true` block `data, scale, train, morse` unless
`--force-overwrite` is passed. Derived stages (`diagnose, render, metrics, run_manifest.json`)
read the source artifacts but write to `replay/<experiment_name>/...` by
default. This makes replay deterministic without dirtying the preserved trees.

## Quick start

```bash
# Use the pre-built venv at the repo root.
../.venv/bin/pip install -e ".[dev]"
../.venv/bin/pytest -m "not slow"            # ~2 s
../.venv/bin/pytest                           # full suite

# Render one replay-ready paper figure from saved artifacts (no CMGDB, no training):
../.venv/bin/python reproduce_paper.py --only fig_leslie3d_example1

# Read-only configs route derived outputs to replay/<name>/; select usable
# coral cells until the zero-byte source artifacts are re-synced.
../.venv/bin/python pipeline.py --config configs/coral_data_scaling.yaml --stages render,metrics --cell-index 120 --expected-cells 180

# Opt into a retrain/recompute. Chafee can be rerun from the unified config;
# coral full recomputation is intentionally out of the default paper replay
# path because it is a large sweep and the preserved replay tree is read-only.
../.venv/bin/python reproduce_paper.py --only fig_chafee_infante --stages all --max-seeds 1
```

Sweep configs decompose into `(train_file, seed)` cells, one per Slurm array
task; `pipeline.py --dry-run` reports the cell count, and
`slurm/pipeline_array.sbatch` is the template.

## Reproduction stance (what "reproducing the paper" means)

There are two distinct modes:

1. **Replay** (default) reads preserved data, scalers, checkpoints, and CMGDB
   DOT/CSV files and rebuilds figures + metrics. This is the primary mode of
   inspection.
2. **Fresh reproduction** regenerates artifacts via the data, training, and
   CMGDB stages. Useful for filling archive gaps and robustness checks; not
   guaranteed to land on the same Morse graph for every seed/hardware.

The paper's guarantee is conditional: once a run has the required learned map,
CMGDB structure, and verified tolerance / semiconjugacy bounds, the
corresponding combinatorial structure is certified for `f`. The pipeline
therefore treats `diagnose`, CMGDB outputs, and paper-specific metrics as
validation gates, not decorative outputs.

`tests/test_reproducibility.py` checks that the same `--seed` yields
bitwise-identical state dicts on the same machine; cross-machine determinism is
not guaranteed (BLAS / cuDNN / MPS heuristics).

## Postprocessing overlays

CMGDB writes `MG/morse_sets` via `CMGDB.SaveMorseSets`. Figure-specific
postprocessing should consume that CSV instead of recomputing CMGDB:

```python
from latentdynamics.viz import plot_morse_sets_from_csv

plot = plot_morse_sets_from_csv("replay_sources/coral/seed_0/MG/morse_sets")
plot.ax.scatter([0.0], [plot.label_to_y[0]], color="black", zorder=10)
plot.fig.savefig("replay/coral_basic/seed_0/MG/morse_sets_overlay.png")
```

The default plotter only draws Morse sets / Morse graphs. Trajectory overlays,
crops, axis limits, and label placement (e.g. the `latent_trajectory.png`
figure of §5.3.1) belong in render hooks or small notebooks that read saved
artifacts; do not bake them into training or CMGDB.

## Adding a new experiment

1. Drop a YAML in `configs/<name>.yaml`. Copy an existing config and edit
   the values you need; every config spells out every field, so there is no
   hidden inheritance.
2. Register it in `reproduce_paper.py::EXPERIMENTS`.
3. Add a `docs/figure_contracts/<name>.md` recording archive source, known
   gaps, expected Morse graph (number of nodes / minimal nodes / Conley
   indices), and the verification recipe.
4. Cover the routing/config behaviour in `tests/test_experiments.py` and add
   focused tests for any new system, metric, or renderer.

## Pins and caveats

- `pyproject.toml` installs CMGDB from `github.com/bernardorivas/CMGDB`
  (unpinned, tracks `master`); upstream API changes to `BoxMap` /
  `ComputeConleyMorseGraph` may break the pipeline.
- Provided three-file pickled `nn.Module` checkpoints can be loaded with
  `latentdynamics.training.load_legacy_checkpoint` without rewriting them;
  `scripts/migrate_legacy_checkpoints.py` produces a `state_dict` + sidecar
  copy when needed.
- Code identifiers may keep legacy names (`leslie_2gen_contraction`); paper-facing
  docs describe that example as the **10D Embedded Leslie** map (a 2D
  Leslie/Ricker map embedded in 10D with eight contracting tail coordinates).
