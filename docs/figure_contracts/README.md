# Figure reproduction contracts

One file per paper figure, pinning hyperparameters to archive sources by
line number, recording the exact reproduction command, and stating the
expected scientific output. New experiments should add a corresponding
contract before being merged.

These contracts are the bridge between the archived author-specific scripts and
the unified config pipeline. They should distinguish preserved paper artifacts
from fresh reproduction attempts, record which values are known from the
archive, and leave unknown values explicit instead of smoothing them over.

| Paper reference   | Contract |
|-------------------|----------|
| Fig. 1.83         | [`leslie_contraction.md`](leslie_contraction.md) |
| Fig. 1.214        | [`leslie3d_spurious.md`](leslie3d_spurious.md) |
| §1.211 success    | [`leslie3d_success.md`](leslie3d_success.md) |
| §1.256 PDE        | [`chafee_infante.md`](chafee_infante.md) |
| Fig. 1.376        | [`coral_basic.md`](coral_basic.md) |
| Fig. 1.469        | [`coral_data_scaling.md`](coral_data_scaling.md) |
| Fig. 1.528        | [`coral_adaptive.md`](coral_adaptive.md) |

## Contract template

Each contract has the following fixed sections:

1. **Paper figures** — paths under `paper/figures/`.
2. **Source of paper run** — the original training script, CMGDB script,
   `mg_params_log.txt`, saved data/scaler, saved checkpoint, saved
   Morse graph/set artifacts, and any source gaps, with file paths under
   `archive/<who>/`.
3. **Status** — one of:
   - `replay-ready` (saved DOT/CSV/checkpoint sufficient to render +
     metric without re-running CMGDB or training);
   - `partial` (some seeds present, others empty);
   - `scratch-only` (no preserved paper run; needs `--stages all`);
   - `blocked-by-empty-checkpoints`;
   - `blocked-no-source` (training script not in any archive).
4. **Reproduction commands** — both the read-only replay command and the
   AMAREL `--stages all` command.
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

- `leslie3d_spurious` is the only paper figure that the new package
  reproduces exactly today (legacy 3-file checkpoint; CMGDB DOT/CSV
  identical to brittany's archive).
- `leslie_contraction` is the legacy config id for the 10D Embedded Leslie
  example: a 2D Leslie/Ricker map embedded in 10D with eight contracting tail
  coordinates. It and `leslie3d_success` now point to Patrick's archived paper
  artifacts under `archive/patrick/Leslie10D/` and
  `archive/patrick/Leslie3D/`. Their current configs remain fresh-run
  reproduction paths because Patrick's original training scripts/raw CSVs are
  not archived.
- `chafee_infante` has a retrained seed_0 in `code/output/`; the diagnose
  stage shows rich latent dynamics, but its Morse graph still needs CMGDB rerun
  with the right bounds.
- `coral_basic`, `coral_data_scaling`, `coral_adaptive` cannot be
  replayed because the per-seed checkpoints in the `code/output/coral/`
  tree are 0-byte placeholders; fresh runs go via the
  `configs/scratch/coral_*.yaml` siblings to keep the original tree
  intact.
- Patrick's `Leslie10D` and `Leslie3D` checkpoints/CMGDB artifacts are
  archived, but the original training scripts/raw CSVs are still missing.
