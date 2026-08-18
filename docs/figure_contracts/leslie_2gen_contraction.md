# fig_leslie_2gen_contraction

Paper figure `fig:lesliecontraction_dynamics` (§5.2): the 10D Embedded Leslie
example. The first two coordinates follow the 2D Leslie/Ricker map, and the
remaining eight coordinates contract by 0.25, so the relevant invariant
dynamics live on the embedded coordinate plane.

## Paper figures

By published file name:

- `morse_graph_with_zero_index.pdf` (panel a)
- `morse_sets_with_zero_index.pdf` (panel b)
- `morse_graph_flipped.pdf` (panel c)
- `morse_sets_with_overlay.pdf` (panel d)

The current manuscript includes only the four panels above (an unaligned
latent-graph render, `morse_graph.pdf`, also exists but is unused).

## Source of paper run

The latent panels come from a fresh package retrain (seed 20) computed at
subdivision 27/29/30. The finer subdivision recovers the attracting period-six
orbit of the period-doubling cascade; an earlier coarser computation
(subdivision 25/27/28) resolved only a period-three orbit and is no longer
used.

- system definition:  `src/latentdynamics/systems/leslie.py` (`LeslieContraction`)
- training config:    `src/latentdynamics/configs/leslie_2gen_contraction.yaml` (seed 20, writable)
- replay config:      `src/latentdynamics/configs/leslie_2gen_contraction_replay.yaml` (read-only)
- replay mirror:      `replay_sources/leslie_2gen_contraction/` (models, MG, scalers; fetched artifact bundle)

The original latent CMGDB run took 56.5 minutes (`mg_params_log.txt`, `auto`
backend); training took 291.5 s over 179 epochs. Do not quote these as
estimates for re-runs on other backends.

## Status

**Fully reproducible.** The system is defined in code and the run is pinned by
seed, so the data, training, and CMGDB stages all regenerate from
`src/latentdynamics/configs/leslie_2gen_contraction.yaml`. The read-only replay
config replays the saved mirror under
`replay_sources/leslie_2gen_contraction/`.

## Reproduction commands

```bash
# Replay the saved model (no training, no CMGDB recompute):
python pipeline.py --config leslie_2gen_contraction_replay --stages render,metrics

# Fresh retrain end to end:
python pipeline.py --config leslie_2gen_contraction --stages all --max-seeds 1
```

## Expected scientific output

The latent Morse graph has five nodes. The two minimal nodes are the system's
two attractors: an attracting invariant circle with Conley index
`(x-1, x-1, 0)`, and an attracting period-six orbit with Conley index
`(x^6-1, 0, 0)`. The remaining three nodes are transient (saddle/repeller)
sets with indices `(0, x^3+1, 0)`, `(0, x^3-1, 0)`, and `(0, 0, x-1)`. No
trivial-index node appears (recurrence at the origin is not detected in the
latent computation). The eight contracting tail dimensions are projected away
by the encoder, leaving 2D Leslie-like dynamics in the latent space.

## Hyperparameter audit

| param                        | value             | source                                  | notes |
|------------------------------|-------------------|-----------------------------------------|-------|
| system.params.th1            | 23.5              | `src/latentdynamics/configs/leslie_2gen_contraction.yaml`  | `LeslieContraction` default |
| system.params.th2            | 23.5              | `src/latentdynamics/configs/leslie_2gen_contraction.yaml`  | `LeslieContraction` default |
| system.params.survival_p1    | 0.7               | `src/latentdynamics/configs/leslie_2gen_contraction.yaml`  |       |
| system.params.contraction    | 0.25              | `src/latentdynamics/configs/leslie_2gen_contraction.yaml`  | tail contraction |
| arch.high_dims               | 10                | config                                  |       |
| arch.low_dims                | 2                 | config                                  |       |
| arch hidden layers / width   | 4 / 64            | config                                  | all three networks |
| arch out-activations         | tanh/tanh/sigmoid | config                                  | encoder/latent/decoder |
| training.loss_weights        | [100, 10, 20]     | config                                  | reconstruction/prediction/semiconjugacy |
| data.n_samples_train         | 8000              | config (paper `D(20,0,10^4)`, 8000/2000 split) | |
| data.n_samples_val           | 2000              | config                                  | |
| data.n_iterations (T)        | 20                | config                                  | |
| cmgdb.subdiv_init            | 27                | config                                  | finer than the other 2D cases |
| cmgdb.subdiv_min             | 29                | config                                  | |
| cmgdb.subdiv_max             | 30                | config                                  | |
| cmgdb.bounds                 | `[-0.7128866, -0.2896928]` to `[0.9668664, 0.9260910]` | `mg_params_log.txt` | encoded-data bounds, epsilon_frac 0.01 |

### Legacy parameter traps

Two archived author scripts describe different experiments and are
deliberately not used for this baseline (neither is part of this release): an
earlier direct 2D computation used `th1 = th2 = 20.0`, and an earlier 10D
embedding used `theta = (19.6, 23.68)` with different tail bounds. Neither
matches the maintained 10D paper run or its replay manifest. The direct
computation below loads the maintained config instead of copying parameters
from either legacy script.

## Verification

```bash
python pipeline.py --config leslie_2gen_contraction_replay --stages render,metrics
# The Morse graph should have five nodes with two minimal attractors:
# an invariant circle (x-1,x-1,0) and a period-six orbit (x^6-1,0,0).
```

## Postprocessing

### Panel (c): aligned Morse graph

The active graph panel is `morse_graph_flipped.pdf`, rendered from the
aligned DOT shipped at
`artifacts/reference_results/leslie_2gen_contraction/aligned_morse_graph.dot`.
It contains the exact five nodes, four directed edges, labels, and Conley
indices in the saved replay DOT. Two invisible same-rank constraints change
only the left-to-right order so that this panel aligns with the direct-model
panel; no PDF mirroring or arrow reversal is used. The aligned DOT has SHA-256
`c9926e9e92fa3b6433666f51d7bfc20d42b71b2af886379e423abb90faf52517`,
and the rendered paper PDF has SHA-256
`6c5d2d6812fd77f361266a945878e7d1fdb8596e3e6b243cc85c59afae9af8a2`.

```bash
dot -Tpdf \
  artifacts/reference_results/leslie_2gen_contraction/aligned_morse_graph.dot \
  -o morse_graph_flipped.pdf
```

The paper PDF is the unique render of that DOT; re-rendering with a different
Graphviz/Cairo build is content-identical but not guaranteed byte-identical.

### Panel (d): latent Morse sets — filename/content mismatch (open item)

Despite its filename, the published `morse_sets_with_overlay.pdf` contains
**no** trajectory overlay. It is a tick-free re-render (2026-07-22) of the
saved seed-20 Morse-set boxes only, drawn at the exact replay bounds; it
replaced an earlier version that did carry gray trajectory arrows. The
earlier true-overlay render is preserved at
`replay_sources/leslie_2gen_contraction/MG/morse_sets_with_overlay.pdf`.

The published tick-free panel is regenerated by
`scripts/render_leslie_2gen_contraction_morse_sets.py` from
`replay_sources/leslie_2gen_contraction/MG/morse_sets` (11.5 MB CSV) and the
bounds in `mg_params_log.txt`. The pipeline replay render produces the same
boxes with axis ticks.

## Direct-reference computation (panels a, b)

The paper's direct 2D reference is the subdivision `(26,30,40)` run with the
on-demand box-map backend:

```bash
python scripts/compute_original_leslie.py --system 2d --subdiv 26 30 40 \
  --box-map-backend on_demand \
  --output output/original_leslie/leslie_2d_exact_restriction_s26_30_40_on_demand
python scripts/render_original_leslie2d_full_paper_figures.py
```

The compute script loads `leslie_2gen_contraction.yaml`, constructs the
configured 10D `LeslieContraction`, embeds each 2D corner as
`(x0, x1, 0, ..., 0)`, invokes that same 10D `step` implementation, and
projects its first two outputs. Thus the 2D computation uses exactly
`theta = (23.5, 23.5)`, survival `0.7`, and the first-coordinate domain
`[0,90] x [0,70]`, with the padded box map, `subdiv_limit 10^4`, and the
on-demand backend. The CMGDB stage took 3,244.5 s (~54 minutes) on the
development machine (Apple M4 Pro), so the paper panels are rendered from the
saved artifacts. The original run's exact command line was not logged; its
backend and parameters are pinned by the run's `manifest.json` and log.

The render script reads the saved `MG/morse_graph` (DOT) and `MG/morse_sets`
(154 MB CSV) — by default from the recompute location above; pass
`--source-dir` to point it at a fetched copy — verifies the expected six
Conley indices, restores the trivial-index node 4 in teal, and hard-fails on
any index mismatch. The expected graph:

| node | Conley index | role |
|------|--------------|------|
| 0 | `(x^6-1, 0, 0)` | minimal: attracting period-six orbit |
| 1 | `(0, x^3+1, 0)` | unstable period-three orbit |
| 2 | `(x-1, x-1, 0)` | minimal: attracting invariant circle |
| 3 | `(0, x^3-1, 0)` | unstable period-three orbit |
| 4 | `(0, 0, 0)` | trivial index (origin fixed point) |
| 5 | `(0, 0, x-1)` | corner repeller |

Minimal nodes are `{0, 2}`, so the direct reference and the latent graph agree
on the attractor pair (invariant circle plus period-six orbit).

Checksums of the saved reference artifacts and paper renders:

- `MG/morse_graph` (DOT, 716 B): SHA-256
  `93076589e37ff89a3acbc1e458020404f3b4f6a0524c418526ad4a687b86ec98`
- `MG/morse_sets` (CSV, 154 MB): SHA-256
  `0a9686bef097543ab76861d001cd046e89f5f97ff9ae13093d9240ca7e4467df`
- `paper_full/morse_graph.pdf` (= paper `morse_graph_with_zero_index.pdf`): SHA-256
  `8d311f95764ba9ef588ad94b5c706ca3de42e08075eda59b1a511d15d83af29d`
- `paper_full/morse_sets.pdf` (= paper `morse_sets_with_zero_index.pdf`): SHA-256
  `b5c03474791ee5e4a1c814725db307fca48f1c7dec83defaee120c0c081c6e1c`

The 154 MB `MG/morse_sets` CSV is needed only for panel (b); if it is not
fetched with the artifact bundle, the ~54-minute recompute above is the
reproduction path. Other subdivision runs of the same script (for example the
exact-domain `27/29/30` run, five nodes, minimal indices `(x^3-1,0,0)` and
`(x-1,x-1,0)`) are exploratory and are not the paper reference.

An exploratory three-dimensional mode is also available:

```bash
python scripts/compute_original_leslie.py --system 3d --subdiv 20 22 24
```

The 3D default is a preview, not an archived baseline; finer subdivision is
needed before close recurrent sets can be expected to separate.

## Residual/tolerance table rows

The `Extended Leslie (10D)` rows of `tab:sampled_residual_tolerance` are
computed on this model's two minimal-node blocks; see
[`sampled_residual_tolerance.md`](sampled_residual_tolerance.md).
