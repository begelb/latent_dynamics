# Figure reproduction contracts

One file per paper figure or table family, pinning hyperparameters and
artifact checksums, recording the exact reproduction command, and stating the
expected scientific output. New experiments should add a corresponding
contract before being merged.

These contracts are the bridge between the preserved author artifacts and the
unified config pipeline. They distinguish preserved paper artifacts from
fresh reproduction attempts, record which values are known from the preserved
runs, and leave unknown values explicit instead of smoothing them over.

| Active paper reference | Contract |
|------------------------|----------|
| `fig:lesliecontraction_dynamics` (incl. direct 2D reference) | [`leslie_2gen_contraction.md`](leslie_2gen_contraction.md) |
| `fig:3D_Leslie_direct` | [`leslie3d_ground_truth.md`](leslie3d_ground_truth.md) |
| `fig:3D_Leslie_latent`, `fig:3D_Leslie_latent_coarse` | [`leslie3d_example1.md`](leslie3d_example1.md) |
| `sec:chafee_infante` figures | [`chafee_infante.md`](chafee_infante.md) |
| `tab:ci_dimension_roa_statistics`, `tab:basins_attraction` | [`chafee_roa_statistics.md`](chafee_roa_statistics.md) |
| `fig:coral_latent_dynamics`, `tab:coral_data` | [`coral_basic.md`](coral_basic.md) |
| `tab:sampled_residual_tolerance` (all 15 rows) | [`sampled_residual_tolerance.md`](sampled_residual_tolerance.md) |

## Contract template

Each figure contract has the following fixed sections (table contracts adapt
them to rows/artifacts):

1. **Paper figures** — the manuscript panels covered by the contract, by
   published file name.
2. **Source of paper run** — training and CMGDB provenance, parameter logs,
   saved data/scaler/checkpoint, and saved Morse graph/set artifacts, with
   paths under the repository's `replay_sources/` (fetched artifact bundles)
   or `artifacts/reference_results/` (frozen result files).
3. **Status** — replay-ready, fresh-reproducible, static, or open, with the
   exact scope of each.
4. **Reproduction commands** — the read-only replay command plus any
   validated fresh-run command. If fresh reproduction needs a writable copy,
   that is stated explicitly.
5. **Expected scientific output** — number of Morse sets, Hasse edges,
   Conley indices, and the metrics fields that should be present.
6. **Hyperparameter audit** — table of preserved value vs packaged YAML
   value, with the source for each preserved value.
7. **Postprocessing notes** — figure-specific render choices such as axis
   limits, overlays, alignment constraints, or no-legend variants.
8. **Verification** — concrete shell recipe (checksums, count gates, or
   index checks) to confirm reproduction.

Fresh runs should not be described as guaranteed paper reproduction until
their diagnostics, CMGDB output, and paper-specific validation metrics pass.
The theory applies after those bounds and hypotheses are verified; it does
not promise that every stochastic retraining run will produce the same latent
Morse graph.

## Decisions captured by the contracts

- `leslie3d_ground_truth` keeps the six Morse sets and verified local Conley
  indices from the saved adaptive run, but displays the transitive reduction
  of exhaustive level-33 reachability among those saved sets. This removes the
  adaptive graph's inherited `3 -> 2` relation and adds `3 -> 1`; it is not a
  claim that the full uniform level-33 SCC inventory was enumerated. The
  paper's cubical no-legend panel is regenerated (visually identically) by
  the renderer's `--no-legend` flag; the committed bytes were a one-off
  export.
- `leslie3d_example1` shows the saved fine adaptive cover in its unmerged
  panels and a recorded live rebuild in the merged and uniform-depth panels;
  the two differ by 1 and 12 cells in the two large minimal nodes (backend
  sensitivity) and match exactly on the merged nodes. The merged fiber's
  Conley index comes from a verified index pair (322 cells), not from the
  literal union of the fine covers.
- `leslie_2gen_contraction` is the config id for the 10D Embedded Leslie
  example: a 2D Leslie/Ricker map embedded in 10D with eight contracting tail
  coordinates. It is a fresh package retrain (seed 20) computed at subdivision
  27/29/30, fully reproducible from
  `src/latentdynamics/configs/leslie_2gen_contraction.yaml`; the system is
  defined in code (`src/latentdynamics/systems/leslie.py`,
  `LeslieContraction`) and a read-only replay mirror lives under
  `replay_sources/leslie_2gen_contraction/`. The direct 2D reference is the
  `(26,30,40)` on-demand run with six nodes (trivial-index origin node
  restored), minimal nodes `{0, 2}`.
- `chafee_infante` combines two provenance lines that must remain distinct.
  The two-dimensional fine and coarse panels come from the coauthor's
  production computation (fine panels are recolored author PDFs, not
  recomputable byte-exactly); the one- and three-dimensional panels come from
  the saved latent-dimension study. Verified local index pairs reconstructed
  from the exact saved level-24 boxes supply all 11 Conley-index annotations
  in the active three-dimensional graph.
- `chafee_roa_statistics` scores 45 saved computations against author truth
  labels completed by a recorded two-solver continuation; the printed tables
  are pure post-processing of the frozen per-IC record.
- `coral_basic` is a read-only replay over the preserved coral tree. The
  featured three-node paper example is specifically `train_500/seed_16`. The
  archived dataset metadata records scrambled Sobol sampling, and the
  packaged YAML now says `sobol` (an earlier copy carried a stale `uniform`
  value).
- `sampled_residual_tolerance` records that every table row is a
  finite-sample estimate produced by
  `latentdynamics.analysis.sampled_metrics` with fixed seeds; frozen
  full-precision results ship with the repository and round exactly to the
  published values.
