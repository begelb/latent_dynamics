# Reproduction gaps (current status)

What is and isn't reproducible from the artifacts in this repository, and how to
close each gap. Snapshot: 2026-05-26. This supersedes the historical
`FIGURE_PARITY.md` (a 2026-05-03 diagnostic snapshot whose "collapses to 1 sink"
rows are now stale).

Figure numbers follow the canonical `paper/main.tex`. Figures 1–7 are inline
TikZ schematics with no reproduction burden; the computational figures are 8–15.

## Per-figure status

| Fig | Example | Status | Source artifact | Gap / how to close |
|---|---|---|---|---|
| 8 (§5.2) | 10D embedded Leslie | **Replay-ready** | `replay_sources/leslie_contraction` (Patrick; flat, legacy 3-file ckpt) | Fresh *exact* retrain blocked: Patrick's training script + raw CSVs are not archived. Replay matches the paper. |
| 9 (§5.3.1) | 3D Leslie, spurious | **Replay-ready** | `replay_sources/leslie3d_spurious/spurious_attractor_ex` (legacy ckpt) | None for replay (Hasse is byte-identical to Brittany's archive). |
| 10 (§5.3.2) | 3D Leslie, correct | **Replay-ready** | `replay_sources/leslie3d_success` (Patrick) | Fresh exact retrain blocked: script + raw CSVs not archived. |
| 11 (§5.4) | Chafee–Infante bifurcation diagram | **N/A (static asset)** | `paper/figures/ci_bif_diagram.pdf` | A steady-state continuation/BVP figure, not part of the learned-dynamics pipeline. No code reproduction intended. |
| 12 (§5.4) | Chafee–Infante latent Morse graph | **Replay-ready (Marcio's model)** | `replay_sources/chafee_infante/marcio` (his `ci_model_weights.pth` converted via `scripts/convert_marcio_chafee.py`) | RESOLVED 2026-05-26: his weights load cleanly (arch identical); recomputing CMGDB reproduces Fig. 12 exactly — 7 nodes, 2 attractors `(x-1,0,0)`, 3 saddles `(0,x-1,0)`, 2 sources `(0,0,x-1)`. The bare `chafee_infante` config remains a separate fresh-retrain (derived-box) variant. |
| 13 (§5.5) | Red coral, basic Morse graph | **Reproducible (retrain substitute)** | `output/coral_data_scaling/train_500/seed_0` (fresh retrain; passes the §5.5.1 success metric) | Brittany's *preserved* `train_500` checkpoints are 0-byte. The local retrain at the same size reproduces the figure; to recover the exact preserved run, re-sync from the cluster. |
| 14 (§5.5.2) | Coral population histograms | **Reproducible** | data CSVs: `data/coral/train_500.csv`, `replay_sources/coral/data/coral/train_500_300_adaptive.csv` | None — histograms come from the data, independent of checkpoints. |
| 15 (§5.5.2) | Coral success rates (30 seeds × sizes) | **Partial** | `replay_sources/coral` (train_2000×30, train_100×10, adaptive M=100/200/300×30) + `output/coral_data_scaling` (6 sizes × ~4 seeds) | No single tree has 30 seeds across all sizes. Close with a fresh recompute sweep or a cluster re-sync (see below). |

## Detail by example

### Coral — artifacts span three trees

- `output/coral_data_scaling/<size>/seed_k` — **fresh local retrains** (new
  `autoencoder.pt`). All six sizes (`train_100`…`train_5000`) present, but only
  ~4 seeds each are non-empty. `train_500/seed_0` is complete and passes the
  §5.5.1 success metric `{a0: true, a1: true, r: false}` → used as the Fig. 13
  source by `coral.ipynb`.
- `replay_sources/coral/<size>/seed_k` — **Brittany's preserved tree** (legacy
  3-file checkpoints). Complete: `train_2000` (30 seeds), `train_100` (10),
  adaptive M=100/200/300 (30 each). **0-byte**: `train_200/500/1000/5000` and
  adaptive M=400/500.
- `archive/brittany/output/coral` — identical to `replay_sources` (diff is
  empty); it holds nothing the replay tree lacks.

To close Fig. 15: run the data-scaling sweep fresh (180 cells = 30 seeds × 6
sizes) and the adaptive sweep, or re-sync the original cluster tree. The Slurm
array template is `slurm/pipeline_array.sbatch`; get the cell count with
`pipeline.py --config configs/coral_data_scaling.yaml --dry-run`. Recomputing
writes under `output/` (the `replay_sources` config is `read_only`).

### Chafee–Infante — Marcio's model (resolved 2026-05-26)

`scripts/convert_marcio_chafee.py` converts his `ci_model_weights.pth` to the
package checkpoint format — his architecture is identical to the package's chafee
arch, so it is a pure `.net` key rename. The read-only `chafee_infante_marcio`
config now replays **his** model: recomputing CMGDB over his predefined box
`[-3,-2]×[3,2]` reproduces Fig. 12 exactly — 7 nodes, 2 attractors `(x-1,0,0)`,
3 saddles `(0,x-1,0)`, 2 sources `(0,0,x-1)`. Saved under
`replay_sources/chafee_infante/marcio/`.

The bare `chafee_infante` config is a *separate* fresh retrain (derived box); it
produces a richer/different graph (9 nodes, 4 attractors) and is not pinned to
his run. `metrics.json` stays empty for both: the §4 tolerance metric needs a
one-step `x_t → x_{t+τ}` map, and `chafee_infante` is typed as a continuous
integration rather than a `DiscreteMap`.

### Patrick's Leslie runs (Fig. 8, 10)

Replay-ready from the archived checkpoints. Fresh *exact* reproduction is
blocked because the training scripts and raw CSVs were not archived; the configs
encode the recovered hyperparameters but a retrain is not guaranteed to land on
the same Morse graph.

### `*_test_1101` offshoots

The `[1,1,0,1]` loss-weight experiments live in `output/*_seed1_rerun` /
`*_seed0_finer` directories whose configs are not checked into `configs/`, and
four of the five `*_test_1101` configs have no data/output dir. These are
research offshoots, not paper figures; they are out of scope for the replay
notebooks.

## Documentation hygiene

- `FIGURE_PARITY.md` is a 2026-05-03 snapshot; its "far off / 1 sink" rows for
  `chafee_infante`, `leslie_contraction`, and `leslie3d_success` predate the
  current artifacts and are no longer accurate. Use this file instead.
- `code/README.md` and `docs/PAPER_REPRODUCTION.md` still cite "119 tests"; the
  suite is now 272 (including the 8 `test_replay.py` tests).
- `scratch/figure_sources.md` and any remaining "Fig. 4/5/6" references use the
  pre-2026-05-26 numbering; the computational figures are now 8–15.
