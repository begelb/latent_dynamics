# Figure reproduction contracts

One file per paper figure, pinning hyperparameters to archive sources by
line number, recording the exact reproduction command, and stating the
expected scientific output. New experiments should add a corresponding
contract before being merged.

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
   `mg_params_log.txt`, and (where present) saved checkpoint, with file
   paths under `archive/<who>/`.
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
7. **Verification** — concrete shell+grep recipe to confirm reproduction.

## Decisions captured by the contracts

- `leslie3d_spurious` is the only paper figure that the new package
  reproduces exactly today (legacy 3-file checkpoint; CMGDB DOT/CSV
  identical to brittany's archive).
- `leslie3d_success`, `leslie_contraction`, `chafee_infante` each have a
  retrained seed_0 in `code/output/`; the diagnose stage shows
  `chafee_infante` produced rich latent dynamics (its Morse graph just
  needs CMGDB rerun with the right bounds), while `leslie3d_success` and
  `leslie_contraction` have near-identity latent maps that need
  loss-weight re-tuning before another full retrain.
- `coral_basic`, `coral_data_scaling`, `coral_adaptive` cannot be
  replayed because the per-seed checkpoints in the `code/output/coral/`
  tree are 0-byte placeholders; fresh runs go via the
  `configs/scratch/coral_*.yaml` siblings to keep the original tree
  intact.
- `leslie_contraction` is the one figure with no archived training
  script. Its contract holds a stub hyperparameter table that should be
  filled in once Patrick's source surfaces.
