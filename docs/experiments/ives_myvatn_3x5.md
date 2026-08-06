# Ives Lake Mývatn 3x5 replication

Date configured: 2026-08-06  
Launched: 2026-08-06 15:15 UTC  
Status at handoff: **live; training and diagnosis in progress**  
Persistent session: `latent-ives-myvatn-3x5` (`tmux`, protected by `caffeinate`)  
Scope: five independently sampled training datasets by three independent
network initializations, for 15 cells. The `3x5` label follows the established
launcher naming convention: it means **3 model seeds x 5 data seeds**, not
three datasets by five models.

## Objective

This experiment ports Bernardo's Lake Mývatn midge--algae--detritus example
into the current configuration, dataset, trainer, CMGDB, and seed-sweep paths.
It asks how often the learned two-dimensional dynamics recover both the saved
four-node branch-chain Morse graph and the two distinct invariant objects: a
period-12 cycle and a fixed point.

This is an updated 15-cell replication, not a byte-for-byte replay of the
single archived run. The physical map, parameter values, system box, affine
normalization, network, and CMGDB geometry come from the canonical standalone
script. Dataset isolation, a true validation holdout, generic modern training,
optimized topology evaluation, strict completion checks, and aggregate
classification use the current codebase.

## Sources and conflict resolution

The local sources of truth are:

- canonical standalone experiment:
  [`archive/bernardo/archive/legacy_systems/IvesMap3D_cmgdb.py`](../../../archive/bernardo/archive/legacy_systems/IvesMap3D_cmgdb.py);
- archived system implementation:
  [`archive/bernardo/src/systems/ives_model.py`](../../../archive/bernardo/src/systems/ives_model.py);
- older JSON config:
  [`archive/bernardo/configs/ives_model_log.json`](../../../archive/bernardo/configs/ives_model_log.json);
- saved baseline run:
  [`archive/bernardo/data/reference_runs/ives_model_log_baseline/`](../../../archive/bernardo/data/reference_runs/ives_model_log_baseline/);
- canonical invariant coordinates:
  [`archive/bernardo/data/barycenters/ives_model_log.csv`](../../../archive/bernardo/data/barycenters/ives_model_log.csv).

The archive is internally inconsistent. The standalone script and the saved
run's `model_config.json` agree on the network and transient, so they take
precedence over the older JSON where they conflict.

| field | older JSON | canonical script / saved run | frozen choice |
|---|---|---|---|
| physical box | `[-6.5,-6.5,-6.5]` to `[1.5,1.5,1.5]` | `[-3,-7.5,-3]` to `[1.5,1.5,1.5]` | canonical script |
| influx `c` | rounded `3.68e-7` | `10^-6.435` | `0.000000367282300498085` |
| exponent `q` | rounded `0.902` | `0.9026` | `0.9026` |
| autoencoder width | JSON says 64 | script and saved run say 32 | one width-32 hidden layer |
| hidden activations | JSON terminal-style fields are ambiguous | script uses ReLU hidden layers | ReLU hidden layers |
| transient | JSON starts at zero | saved run records `min_steps=50` | discard 50 steps |
| data mode | JSON says grid | script and saved run say uniform | uniform initial conditions |
| CMGDB ladder | `24/24/30` | `18/22/30` | `18/22/30` |

The paper provenance recorded in the archived implementation is Ives et al.
(2008), *Nature* 452, 84--87, Box 1. No paper PDF for this example is present
in the workspace, so the executable archive is the parameter-level source for
this port.

## Physical map and coordinates

Let `(m,a,d)` denote positive linear-scale midge, algae, and detritus abundance,
and set `R = a + p*d`. One map step is

```text
m' = r1*m*(1 + m/R)^(-q)
a' = max(c, r2*a/(1+a) - (a/R)*m' + c)
d' = max(c, retention*d + a - (p*d/R)*m' + c)
```

The learned system operates in componentwise base-10 log coordinates: input
log states are exponentiated, the map above is evaluated, and the output is
mapped back with `log10`. The frozen parameters are:

| parameter | value |
|---|---:|
| `r1` | `3.873` |
| `r2` | `11.746` |
| `c` | `0.000000367282300498085` (`10^-6.435`) |
| detritus retention | `0.5517` |
| `p` | `0.06659` |
| `q` | `0.9026` |

Initial conditions are uniform on the log-coordinate box

```text
[-3.0, -7.5, -3.0] x [1.5, 1.5, 1.5].
```

## Replication and data contract

Each of the five physical datasets is reused by three independently initialized
models. Dataset/scaler artifacts are shared only inside that dataset; model,
diagnosis, topology, render, and metric artifacts are isolated per cell.

| dataset | training-IC seed | model seeds |
|---:|---:|---|
| 1 | `2158` | `0,1,2` |
| 2 | `4792` | `0,1,2` |
| 3 | `3174` | `0,1,2` |
| 4 | `688` | `0,1,2` |
| 5 | `5727` | `0,1,2` |

Validation uses seed `9999` for every dataset. The five validation files must
therefore be byte-identical; the five training files must be distinct. The
strict summary records and checks their hashes and seeds.

For every dataset:

- training initial conditions: 1,000;
- validation initial conditions: 200;
- generated final time: `T=70`;
- discarded transient: the first 50 steps;
- retained transitions per initial condition: 20, from times 50 through 69;
- expected training pairs: **20,000**;
- expected validation pairs: **4,000**.

Thus the five saved dataset trees contain 100,000 training pairs and 20,000
validation pairs in total. Here 1,000 means 1,000 training initial conditions;
it is not a train/validation split of a 1,000-IC budget.

Scaling is the archive's explicit fixed-box affine map

```text
(x - lower_bounds) / (upper_bounds - lower_bounds + 1e-6),
```

persisted through the modern scaler interface. It is not a dataset-fitted
MinMax transformation. The same frozen system box and epsilon `1e-6` apply to
training data, validation data, and the invariant-point CSV.

## Network and training

The ambient and latent dimensions are `3 -> 2`. The exact architecture is:

| component | architecture | hidden activation | output activation |
|---|---|---|---|
| encoder `E` | `3 -> 32 -> 2` | ReLU | tanh |
| latent map `G` | `2 -> 64 -> 64 -> 64 -> 64 -> 64 -> 2` | ReLU | tanh |
| decoder `D` | `2 -> 32 -> 3` | ReLU | sigmoid |

The generic trainer minimizes, in configuration-weight order,

```text
MSE(D(E(x)), x)
+ MSE(D(G(E(x))), y)
+ MSE(G(E(x)), E(y)).
```

All weights are one. Training uses Adam with learning rate `0.001`, batch size
`1024`, at most 500 epochs, early-stopping patience 300, and no gradient
clipping. `ReduceLROnPlateau` has factor `0.5`, patience 20, threshold `1e-4`,
and minimum learning rate `1e-6`.

The modern trainer intentionally differs from the archive by evaluating a
separately generated 4,000-pair holdout and restoring the lowest holdout-loss
checkpoint. The archived loop selected its validation subset from the training
pairs and returned the endpoint reached at stopping. The archive's optional
separation penalty remains disabled; the invariant points are used only for
post-training classification, never as a training objective.

The archived standalone run targeted 100,000 training pairs (5,000 ICs after
its 50-step transient), used one global NumPy/Torch seed, and produced one
model. This replication instead freezes 1,000 train ICs plus 200 true
validation ICs and crosses five data seeds with three model seeds.

## CMGDB and requested figures

Each cell uses the archived subdivision schedule `18/22/30` for
`init/min/max`, subdivision limit `100000`, padding enabled, and the modern
`adaptive_precomputed` backend. The dense table is built only through
`subdiv_init=18`; corners introduced at later levels are evaluated in batches
on demand. Running topology one cell per process releases each lookup and
native graph before the next cell.

Latent bounds reproduce the canonical script rather than being inferred from
the sampled train or validation pairs:

1. construct a `64 x 64 x 64` Cartesian grid on the full physical system box;
2. apply the fixed-box scaler and encode all grid points;
3. include both encoded points `E(x)` and their latent images `G(E(x))`;
4. expand the coordinatewise range by 10%;
5. clip the result to `[-1,-1] x [1,1]`.

The requested render group is only `morse`, producing the Morse graph and
Morse sets. Exact regions of attraction are disabled. RoA products, basin
plots, training-data plots, latent-evolution snapshots/animations, density
overlays, invariant overlays, separation-training extras, and unrelated paper
figures are outside scope and must not be scheduled. The invariant CSV is
consumed by the summary classifier without requesting an overlay.

## Machine-checkable success criterion

The canonical invariant file copied into the package is
[`src/latentdynamics/reference_data/ives_myvatn_invariant_points.csv`](../../src/latentdynamics/reference_data/ives_myvatn_invariant_points.csv).
Rows with `vertex=0` are the 12 ordered phases of the period-12 orbit; the one
row with `vertex=1` is the fixed point.

A cell passes only when both of the following hold.

1. Its complete Morse graph is isomorphic to the archived four-node
   branch-chain. There must be exactly four nodes and exactly three directed
   edges: one root has edges to a direct sink and to a middle node, and the
   middle node has one edge to the other sink. Equivalently, after assigning
   role names, the edges are exactly `root -> direct_sink`, `root -> middle`,
   and `middle -> terminal_sink`. Extra nodes, edges, or sinks fail.
2. After fixed-box scaling and encoding the 13 canonical points, the fixed
   point is uniquely contained in one graph sink, and at least 11 of the 12
   cycle phases are each uniquely contained in one common graph sink. The
   cycle sink and fixed-point sink must be different. A multiply assigned
   phase does not count toward the 11-phase coverage threshold.

The graph test is node-ID invariant. No archived numeric node identifier is
hard-coded. Classification retains, rather than reducing to a Boolean:

- every node identifier and the full directed adjacency list;
- node in/out degrees, root/middle/sink roles, and the normalized graph
  signature;
- every full Conley-index tuple and inferred period when available;
- the two sink identifiers and minimal/sink status;
- each invariant row's original coordinate, scaled coordinate, encoded
  coordinate, containing Morse-node IDs, containing sink IDs, and unique or
  ambiguous status;
- the fixed-point sink assignment, consensus cycle-sink assignment, covered
  cycle component IDs, uncovered/ambiguous component IDs, and exact coverage
  count out of 12;
- separate graph-shape, fixed-assignment, cycle-assignment, distinct-sink, and
  combined pass flags.

Scientific success is classified per cell. Experiment completion is a separate
condition: the strict verifier must classify all 15 planned cells and confirm
all required artifacts before reporting a final pass count. There is no
predeclared minimum number of passing cells for the sweep itself.

The archived baseline evidence is a four-node graph with adjacency
`0 -> {}`, `1 -> {0}`, `2 -> {}`, and `3 -> {2,1}`. This establishes the
node-ID-invariant target shape; it does not make archived IDs `0` and `2`
semantic labels for new runs.

## Execution and artifact paths

The packaged configuration is
[`src/latentdynamics/configs/ives_myvatn.yaml`](../../src/latentdynamics/configs/ives_myvatn.yaml).
The frozen launcher is intended at
[`scripts/run_ives_myvatn_3x5.sh`](../../scripts/run_ives_myvatn_3x5.sh), and
the dedicated strict summarizer at
[`scripts/summarize_ives_myvatn_3x5.py`](../../scripts/summarize_ives_myvatn_3x5.py).

Run from `code/` with:

```bash
bash scripts/run_ives_myvatn_3x5.sh
```

The persistent controller writes to:

```text
data/ives_myvatn_seedsweep_3x5_v1/
output/ives_myvatn_seedsweep_3x5_v1/
```

Within the output root, the intended controller-level artifacts are:

- `run_plan.json`: resolved five-dataset, 15-cell dry plan;
- `controller.pid`: PID of the persistent controller;
- `session.txt`: launch/session identity and invocation metadata;
- `run_status.txt`: latest timestamped controller phase/status;
- `run.log`: complete persistent-controller log;
- `sweep_summary.json`: generic grid/artifact summary;
- `summary/cells.csv` and `summary/cells.json`: per-cell raw classifications;
- `summary/aggregate_summary.json`: strict aggregate result;
- `summary/SUMMARY.md`: readable aggregate report;
- `recovery/<UTC timestamp>/`: recoverable moves of any interrupted partial
  artifact group, with a manifest.

Expected cell directories are
`output/ives_myvatn_seedsweep_3x5_v1/dataset_<IC seed>/seed_<model seed>/`.
The first is `dataset_2158/seed_0`; the last is `dataset_5727/seed_2`.

The launcher phases are preflight, resolved-plan save, partial-artifact
recovery, five-dataset generation and fixed scaling, 15 model trainings and
diagnoses, isolated per-cell topology, Morse-only rendering, generic summary,
dedicated classification, and strict verification. `--skip-completed` is safe
only when the stage-specific completeness checks accept every required file;
a checkpoint by itself is not a complete trained cell.

The dedicated summarizer's default strict mode writes no final report unless
all 15 cells validate. `--allow-incomplete` is reserved for an explicitly
provisional progress report, and `--verify` validates existing final artifacts
without rewriting them.
