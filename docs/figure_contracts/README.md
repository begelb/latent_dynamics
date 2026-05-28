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
| Fig. 1.83         | [`leslie_2gen_contraction.md`](leslie_2gen_contraction.md) |
| Fig. 1.214        | [`leslie3d_example1.md`](leslie3d_example1.md) |
| §1.211 success    | [`leslie3d_example2.md`](leslie3d_example2.md) |
| §1.256 PDE        | [`chafee_infante.md`](chafee_infante.md) |
| Fig. 1.376        | [`coral_basic.md`](coral_basic.md) |
| Fig. 1.469        | [`coral_data_scaling.md`](coral_data_scaling.md) |
| Fig. 1.528        | [`coral_adaptive.md`](coral_adaptive.md) |

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
   fresh-run or AMAREL command. If fresh reproduction needs a writable copy,
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

- `leslie3d_example1` reproduces exactly today (legacy 3-file checkpoint;
  CMGDB DOT/CSV identical to the preserved replay tree).
- `leslie_2gen_contraction` is the config id for the 10D Embedded Leslie
  example: a 2D Leslie/Ricker map embedded in 10D with eight contracting tail
  coordinates. It is a fresh package retrain (seed 20) computed at subdivision
  27/29/30, fully reproducible from `configs/leslie_2gen_contraction.yaml`; the
  system is defined in code (`src/latentdynamics/systems/leslie.py`,
  `LeslieContraction`) and a read-only replay mirror lives under
  `replay_sources/leslie_2gen_contraction/`. The Morse graph has five nodes,
  with two attractors: an invariant circle `(x-1, x-1, 0)` and a period-six
  orbit `(x^6-1, 0, 0)`.
- `leslie3d_example2` is a read-only replay over the preserved Leslie 3D
  paper artifacts under `replay_sources/leslie3d_example2/`.
- `chafee_infante` has the reference Chafee-Infante data converted into the
  unified CSV layout and uses the reference CMGDB parameters. Replay reads the
  converted data and matched config.
- `coral_basic`, `coral_data_scaling`, `coral_adaptive` are read-only replay
  configs over the preserved coral tree. Replay covers the cells with
  non-empty checkpoints: `train_100` has selected usable seeds, `train_2000`
  has all seeds, and adaptive M = 100/200/300 has all seeds; the remaining
  coral cells are fresh-reproducible from a writable config copy.
