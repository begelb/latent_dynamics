# Figure reproduction contracts

One file per paper figure, pinning hyperparameters to archive sources by
line number, recording the exact reproduction command, and stating the
expected scientific output. New experiments should add a corresponding
contract before being merged.

These contracts are the bridge between the archived author-specific scripts and
the unified config pipeline. They should distinguish preserved paper artifacts
from fresh reproduction attempts, record which values are known from the
archive, and leave unknown values explicit instead of smoothing them over.

| Active paper reference | Contract |
|------------------------|----------|
| `fig:lesliecontraction_dynamics` | [`leslie_2gen_contraction.md`](leslie_2gen_contraction.md) |
| `fig:3D_Leslie_ground_truth` | [`leslie3d_ground_truth.md`](leslie3d_ground_truth.md) |
| `fig:3D_Leslie_latent_dynamics` | [`leslie3d_example1.md`](leslie3d_example1.md) |
| `fig:3D_Leslie_latent_dynamics_success` | [`leslie3d_example2.md`](leslie3d_example2.md) |
| `sec:chafee_infante` | [`chafee_infante.md`](chafee_infante.md) |
| `fig:coral_latent_dynamics` | [`coral_basic.md`](coral_basic.md) |
| coral data-scaling figures | [`coral_data_scaling.md`](coral_data_scaling.md) |
| coral adaptive-sampling figures | [`coral_adaptive.md`](coral_adaptive.md) |

## Contract template

Each contract has the following fixed sections:

1. **Paper figures** — paths under `paper/figures/`.
2. **Source of paper run** — the original training script, CMGDB script,
   `mg_params_log.txt`, saved data/scaler, saved checkpoint, saved
   Morse graph/set artifacts, and any source notes, with file paths under
   `archive/<who>/`.
3. **Status** — one of:
   - `replay-ready` (saved DOT/CSV/checkpoint sufficient to render +
     metric without re-running CMGDB or training);
   - `partial read-only replay` (some seeds replayable, others
     fresh-reproducible from a writable copy);
   - `fresh-reproducible` (regenerated end-to-end from a code-defined
     system via `--stages all`).
4. **Reproduction commands** — the read-only replay command plus any validated
   fresh-run command. If fresh reproduction needs a writable copy,
   say that explicitly.
5. **Expected scientific output** — number of Morse sets, Hasse edges,
   Conley indices, and the metrics.json fields that should be present.
6. **Hyperparameter audit** — table of `archive value` vs `YAML value`
   with the source line for each archive value.
7. **Postprocessing notes** — optional figure-specific render choices such as
   axis limits, overlays, trajectory annotations, or slide-only crops.
8. **Verification** — concrete shell+grep recipe to confirm reproduction.

Fresh runs should not be described as guaranteed paper reproduction until their
diagnostics, CMGDB output, and paper-specific validation metrics pass. The
theory applies after those bounds and hypotheses are verified; it does not
promise that every stochastic retraining run will produce the same latent Morse
graph.

## Decisions captured by the contracts

- `leslie3d_ground_truth` keeps the six Morse sets and verified local Conley
  indices from the saved adaptive run, but displays the transitive reduction
  of exhaustive level-33 reachability among those saved sets. This removes the
  adaptive graph's inherited `3 -> 2` relation and adds `3 -> 1`; it is not a
  claim that the full uniform level-33 SCC inventory was enumerated.
- `leslie3d_example1` has two preserved computations that must not be mixed.
  The active paper graph is the May 27 rerender of the current replay tree;
  an earlier run with a smaller latent box remains recoverable from Git. The
  two raw graphs have the same topology, but different bounds and annotations.
- `leslie_2gen_contraction` is the config id for the 10D Embedded Leslie
  example: a 2D Leslie/Ricker map embedded in 10D with eight contracting tail
  coordinates. It is a fresh package retrain (seed 20) computed at subdivision
  27/29/30, fully reproducible from `src/latentdynamics/configs/leslie_2gen_contraction.yaml`; the
  system is defined in code (`src/latentdynamics/systems/leslie.py`,
  `LeslieContraction`) and a read-only replay mirror lives under
  `replay_sources/leslie_2gen_contraction/`. The Morse graph has five nodes,
  with two attractors: an invariant circle `(x-1, x-1, 0)` and a period-six
  orbit `(x^6-1, 0, 0)`.
- `leslie3d_example2` is an artifact-only replay over the preserved Leslie 3D
  checkpoint and CMGDB output. Its generated render manifest contains an
  obsolete inference for the Leslie parameters and is not training provenance.
- `chafee_infante` combines two provenance lines that must remain distinct.
  The two-dimensional and coarse panels come from Marcio's production
  computation; the one- and three-dimensional panels come from the saved
  latent-dimension study. The original three-dimensional adaptive stage is a
  topology-only fallback, but verified local index pairs reconstructed from
  its exact saved level-24 boxes now supply all 11 Conley-index annotations in
  the active graph.
- `coral_basic`, `coral_data_scaling`, `coral_adaptive` are read-only replay
  configs over the preserved coral tree. The non-adaptive replay tree now has
  model triplets, raw graphs, raw sets, and parameter logs for all 30 seeds at
  all six sample sizes. Some cells lack rendered PDFs, which can be regenerated
  from the saved raw artifacts. The featured three-node paper example is
  specifically `train_500/seed_16`. Its archived dataset metadata records
  scrambled Sobol sampling; the replay YAML's `uniform` setting is not exact
  paper-data provenance.
