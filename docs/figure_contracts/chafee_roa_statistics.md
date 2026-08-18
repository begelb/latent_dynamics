# tab_chafee_roa_statistics

Paper tables `tab:ci_dimension_roa_statistics` and `tab:basins_attraction`
(§`sec:appendix_ci_roa_statistics` and §`sec:chafee_infante`): correct
classification rates of regions of attraction for the Chafee--Infante
example across latent dimensions, and the derived
undetermined/TP/FP/precision/recall summary.

## The 45 computations

Three latent dimensions x five training datasets x three training runs, all
scored against the same 10,000 test trajectories:

- **d=1** — local retrain (5 datasets x 3 model seeds) by
  `scripts/run_chafee_d1_matched_5x3.py` on the coauthor's five archived
  training datasets (sha-pinned in the experiment plan).
- **d=2** — the coauthor's 15 archived checkpoints
  (`run_dataset_{1..5}/ci_model_weights_{1..3}.pth`); basins recomputed
  locally by `scripts/analyze_chafee_d2_archive.py` (recorded wall time
  36.13 s for all 15 runs).
- **d=3** — local retrain by `scripts/run_chafee_d3_matched_5x3_training.py`
  plus the on-demand basin computation
  (`scripts/run_chafee_d3_ondemand_5x3_controller.py` /
  `_worker.py`).

Common settings: uniform CMGDB grid with `2^8 = 256` intervals per dimension
(so `256^d` cells), domain padding `0.1` (`bounds_epsilon_frac`), strict
singleton-all-reachable basin semantics
(`latentdynamics.analysis.basin_statistics`,
`latentdynamics.analysis.regions_of_attraction`); training per the paper
(full-batch 30,000, 4,000 epochs, lr 3e-3, loss weights (1,1,0)). Every one
of the 45 runs has exactly two attracting minimal Morse sets, as the paper
asserts.

## Truth labels

The coauthor's 10,000-trajectory integration (ICs sampled with seed 9551;
7,862 trajectories converged by t=6) completed by a recorded local
continuation of the 2,138 unresolved trajectories using both LSODA and BDF
solvers: **all 2,138 stragglers converged** (latest by t=660.9), the two
solvers agreed on all 2,138 labels, and the final split is 5,030 negative /
4,970 positive. The continuation record and completed labels ship in
`replay_sources/chafee_infante/continuation_10000/` (including
`run_continuation.py`, the results npz, and the statistics files).

## Frozen statistics

`replay_sources/chafee_infante/continuation_10000/updated_paper_statistics.csv`
(46 lines; SHA-256
`eb61bc81a92379b9c4433975cebe8b10e52bd1383d95832c786cdc3b164bd09e`)
and `updated_paper_statistics.json` (SHA-256
`fee3cabd2041bec93212c2709f2250d27db5e53042f79008d4d32a6359de430f`);
frozen copies ship under `artifacts/reference_results/chafee_infante/`.
Every per-run value, mean, and SD printed in
`tab:ci_dimension_roa_statistics` matches these files, and the
`run_directory` column maps each row to its saved run directory.

Headline published values:

| d | mean +- SD (correct classification) |
|---|----|
| 1 | 42.82 +- 17.89 |
| 2 | 56.93 +- 11.36 |
| 3 | 60.11 +- 7.77 (excluding dataset 1: 62.46 +- 1.12) |

`tab:basins_attraction` (means over the 15 runs per dimension, percent of
all 10,000 initial conditions):

| row | d=1 | d=2 | d=3 |
|---|---|---|---|
| Undetermined, converges to M(0+) | 29.35 | 23.16 | 19.57 |
| Undetermined, converges to M(0-) | 27.25 | 19.59 | 20.25 |
| TP, M(0+) | 20.65 | 26.92 | 30.68 |
| TP, M(0-) | 22.17 | 30.00 | 29.43 |
| FP, M(0+) | 0.282 | 0.104 | 0.023 |
| FP, M(0-) | 0.303 | 0.220 | 0.048 |
| Precision, M(0+) | 98.65 | 99.62 | 99.93 |
| Precision, M(0-) | 98.65 | 99.27 | 99.84 |
| Recall, M(0+) | 41.05 | 53.53 | 61.00 |
| Recall, M(0-) | 44.61 | 60.37 | 59.21 |

Per the caption, `FN(solution) = FP(other solution) + undetermined initial
conditions converging to that solution`; columns sum to 100% per dimension in
the outcome block.

## Regeneration

Rescoring is pure post-processing of saved artifacts — no training and no
CMGDB run:

```bash
python scripts/chafee_basin_table.py
```

The script derives both tables from the per-IC record
`ci_completed_10k_raw_classifications_45_runs.csv` (450,000 rows = 45 runs x
10,000 ICs; SHA-256
`51ac54913cc5dad768d8f1fc4879b5c9d2a750f937a202fc5692c65f7d8baa9d`)
together with the undetermined-per-solution split (SHA-256
`523ea857a969aa7535cb17d4a5d63431610dcfddf48747518db4ae8e6b523490`),
and is validated against every printed value. The per-IC record itself is
rebuilt from each run's `trajectory_basin_labels.npy` and the continuation
results.

## Per-run replay

Replaying an individual run (rather than rescoring) requires the fetched
45-run bundle: the saved run directories
`chafee_d1_matched_d2_archive_5x3_roa_v1/` (20 MB),
`chafee_d2_archive_5x3_roa_v1/` (6 MB),
`chafee_d3_matched_d2_archive_5x3_training_v1/` (5.7 MB, models), and
`chafee_d3_matched_d2_archive_5x3_ondemand_v2/` (52 MB), each holding
per-run `trajectory_basin_labels.npy`, `basin_statistics.json`, the uniform
grids (`MG_uniform_s8`/`s16`/`s24`), bounds, and manifests. d=1/d=3
retraining is stochastic: the printed table replays only from the saved run
directories, not from fresh retraining. Fresh d=2 basin recomputation
additionally requires the coauthor's 15 archived checkpoints (in the
`chafee_infante` bundle's `reference_inputs/`) and the five training datasets
(in the optional `chafee_training_datasets` bundle).

## Verification

```bash
python scripts/chafee_basin_table.py
# Output must equal every printed value of both tables, including the
# per-run rows of tab:ci_dimension_roa_statistics via
# updated_paper_statistics.csv.
```

## Known limitations and open items

- The mapping of CMGDB node labels to `M(0+)`/`M(0-)` signs in
  `tab:basins_attraction` is marked as provisional in the manuscript caption.
- The paper's claim that all 45 computations have exactly two attracting
  minimal Morse sets is recorded per run (`basin_statistics.json`,
  d=2 `validation_report.json`).
- The precision/recall rows are arithmetic derivations of the outcome rows
  (caption formula); `scripts/chafee_basin_table.py` recomputes them from
  the per-IC record rather than trusting the typeset values.
