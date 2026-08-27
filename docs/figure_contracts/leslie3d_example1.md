# fig_leslie3d_example1

Paper figures `fig:3D_Leslie_latent` (a)-(d) and `fig:3D_Leslie_latent_coarse`
(a)-(b): the learned 3D Leslie latent map `g_1` with an additional minimal
latent component, its connection-complete coarsening, and the uniform
fixed-depth comparison.

## Paper figures

The manuscript's six panels, by published file name:

- `adaptive_original_morse_graph.pdf` — fine (23,23,27) Morse graph `MG(G_1)`
- `adaptive_merged_4_5_morse_graph.pdf` — coarsened graph, nodes 4,5 merged
- `adaptive_original_morse_sets_with_separate_4_5_zoom_no_legend.pdf` — fine
  Morse sets with the marker-free separation-zoom inset
- `adaptive_merged_4_5_morse_sets_no_legend.pdf` — merged-region Morse sets
- `morse_graph_coarse22.pdf` — coarse (22,22,24) graph,
  trivial-index nodes omitted
- `morse_sets_coarse22.pdf` — coarse (22,22,24)
  Morse sets

An earlier single-figure layout (`morse_graph.pdf`,
`morse_sets_with_overlay.pdf`) is no longer referenced by the manuscript.

## Source of paper run

The fine graph is the saved wide-domain replay of the author-provided
checkpoint (`spurious_attractor_ex`, dated 2026-05-03), computed at
subdivision 23/23/27 and re-rendered on 2026-05-27. The graph PDF is
byte-identical to
`replay_sources/leslie3d_example1/spurious_attractor_ex/MG/morse_graph.pdf`
(SHA-256
`160ce61e0368f44b687350abd45b585f06b422a7ff1725cc5d8ec2817995500f`).

- mg_params_log:      `replay_sources/leslie3d_example1/spurious_attractor_ex/mg_params_log.txt` (subdiv 23/23/27, 6.02 min run)
- checkpoint:         `replay_sources/leslie3d_example1/spurious_attractor_ex/models/`
- scaler:             `replay_sources/leslie3d_example1/28.9_29.8_22.0/scalers/scaler.gz`
- saved Morse data:   `replay_sources/leslie3d_example1/spurious_attractor_ex/MG/{morse_graph,morse_sets}` (122,346 boxes)

An earlier computation over a narrower latent box
(`[-0.6228695, -0.74216413]` to `[0.30980384, 0.22416562]`) has the same graph
topology but different bounds and annotations; it is not part of this release
and must not be used as metadata for the active rerender. The active run's
bounds are recorded in the audit table below.

Training provenance limits: the checkpoint's training seeds and the exact
generator of the archived `2train.csv`/`2test.csv` are unrecorded, and the
saved scaler was fit on a different 64,000-row draw than the shipped training
CSV. Replay of the saved artifacts is exact; fresh retraining is stochastic
and does not reliably reproduce the third attractor.

## Status

**Artifact replay-ready; derived panels fresh-reproducible.** Panel (a)
renders from the saved fine-run artifacts; panel (c) renders the saved cover
with a zoom inset built from the merged-run CSV. Panels (b), (d), and the two
coarse panels are recorded fresh computations on the same checkpoint (see the
workflow below), with frozen results shipped under
`artifacts/reference_results/leslie3d_example1/`.

## Reproduction commands

Replay the fine run (no CMGDB, seconds):

```bash
python pipeline.py --config leslie3d_example1_replay --stages render,metrics
```

Recompute the fine CMGDB run (~6 min recorded):

```bash
python pipeline.py --config leslie3d_example1_replay --stages morse --force-overwrite
```

## fixed22_vs_merged45 workflow (panels b, c, d and the coarse panels)

The published comparison contrasts the fine adaptive run with a
connection-complete merge of its nodes 4 and 5 and with a uniform fixed-depth
recomputation. Three drivers plus one renderer:

1. **Fine replay + merge** — `scripts/leslie3d_example1_coarsen_morse_graph.py`
   rebuilds the adaptive cell graph live on the saved checkpoint at the
   run-B bounds, matches live recurrent components to the saved node labels,
   and merges nodes 4 and 5 by the connection-complete order-interval
   quotient (`latentdynamics.analysis.morse_coarsening`). The merged fiber is
   174 (node 4) + 123 (node 5) + 25 connection cells = **322 cells** with
   Conley index `(0, x+1, 0)` computed from a verified index pair; the
   literal 297-cell union is not a valid index pair. Quotient minimal nodes
   are `{0, 1}`. The live rebuild differs from the saved cover by 1 cell in
   node 0 and 12 cells in node 1 (backend sensitivity; nodes 4 and 5 match
   exactly); panels (a)/(c) show the saved cover, panels (b)/(d) come from
   the live rebuild. Runtime ~1 min (46 s map build).
2. **Uniform fixed-depth recompute** —
   `scripts/leslie3d_example1_uniform_grid.py --depth 22` runs
   `CMGDB.ComputeMorseGraph` at uniform depth 22 (2048 x 2048 grid,
   4,194,304 cells) on the same checkpoint and bounds. The raw graph has 24
   nodes; the nontrivial-index restriction keeps nodes 0, 1, 9, 23 with
   minimal `{0, 1}` (indices `(x^4-1,0,0)` and `(x^2-1,0,0)`); node 23
   (291 cells) contains the extra fixed point and a period-2 orbit with
   index `(0, x+1, 0)`. Box widths `(8.079e-4, 7.832e-4)`. Recorded graph
   build 27 s.
3. **Paper renders** — `scripts/render_paper_figures.py --only leslie3d_example1`
   re-renders all six panels display-only (0.5% visibility floor in full
   views, exact box sizes in the zoom, no legends/titles/ticks). The zoom
   inset separates node 4 (purple), node 5 (teal), and the 25 connection
   cells (red); the render hard-fails unless the category counts are exactly
   174/123/25/322.

Frozen results (merged-run `result.json` with the merged Conley index and
live-match record, coarse (22,22,24) `result.json` and nontrivial summaries) ship
under `artifacts/reference_results/leslie3d_example1/`.

## Expected scientific output

`replay_sources/leslie3d_example1/spurious_attractor_ex/MG/morse_graph` (DOT)
has six Morse sets with Hasse:

```
{rank=same; 0 1 4};
2 -> 0
2 -> 1
3 -> 1
5 -> 3
5 -> 4
```

Minimal nodes are 0, 1, and 4, with indices `(x^4-1,0,0)`, `(x^4-1,0,0)`, and
`(x-1,0,0)`; nodes 2, 3, 5 carry `(0,x^4-1,0)`, `(0,x^2+1,0)`, and
`(0,x^2-1,0)`. Node 4 is the additional latent minimal component relative to
the two-minimal-node direct computation. All 16 phase points of the recurrent
invariant sets catalogued by the direct computation (P0, P1, S2, S4, p_star,
and the origin) encode outside node 4; the closest is p_star, at distance
`1.369758509e-3` from its box union. Accordingly, node 4 has no counterpart in
that detected recurrent-set inventory. This is numerical evidence, not a proof
that no unidentified recurrent point lies in its ambient preimage. Moreover,
507 centers of the outer direct-computation node-4 boxes encode into the
latent node-4 box union, so the saved boxes cannot establish the requested
nonintersection. A sampled inequality failure is not a nonlifting certificate.

The coarsened graph (panel b) has five nodes `0, 1, 2, 3, [4,5]` with edges
`2 -> 0`, `2 -> 1`, `3 -> 1`, `[4,5] -> 3` and exactly two minimal nodes; it
is isomorphic (after trivial-node omission) to the direct reference graph.

The direct comparison graph contains the edge `3 -> 2` in both its saved
screen and Conley runs, but this is a combinatorial reachability relation, not
a certified connection. An on-demand uniform-grid check finds `S4 -> S2` paths
at levels 29-32. At level 33, a search from all 5,834 saved node-3 boxes to
all 9,217 saved node-2 boxes exhausts a forward closure of 2,924,013 boxes
without a path. CMGDB propagates the coarser relations to the level-33
descendants, so the displayed `3 -> 2` edge is an inherited subdivision
artifact in this corner-sampled computation. Increasing only `subdiv_limit`
cannot remove it. This does not rule out a connection in the continuous map:
the eight-corner box map is not a validated interval enclosure.

## Hyperparameter audit

| param                       | archive value           | YAML value                | source                                                     | notes |
|-----------------------------|-------------------------|---------------------------|------------------------------------------------------------|-------|
| system.params.th1           | 28.9                    | 28.9                      | src/latentdynamics/configs/leslie3d_example1.yaml          | ✓     |
| system.params.th2           | 29.8                    | 29.8                      |                                                            | ✓     |
| system.params.th3           | 22.0                    | 22.0                      |                                                            | ✓     |
| system.params.survival_p1   | 0.7                     | 0.7                       |                                                            | ✓     |
| system.params.survival_p2   | 0.7                     | 0.7                       |                                                            | ✓     |
| arch.num_layers             | 3                       | 3                         | src/latentdynamics/configs/leslie3d_example1.yaml          | ✓     |
| arch.hidden_shape           | 32                      | 32                        |                                                            | ✓     |
| arch.high_dims              | 3                       | 3                         |                                                            | ✓     |
| arch.low_dims               | 2                       | 2                         |                                                            | ✓     |
| arch.encoder_out_activation | tanh                    | tanh (default)            | default                                                    | ✓     |
| arch.latent_out_activation  | tanh                    | tanh (default)            |                                                            | ✓     |
| arch.decoder_out_activation | sigmoid                 | sigmoid (default)         |                                                            | ✓     |
| training.loss_weights       | [10, 10, 1]             | [10, 10, 1]               | src/latentdynamics/configs/leslie3d_example1.yaml          | ✓     |
| data.n_samples_train        | 3200 trajectories       | 4000                      | paper; archived CSV has 64,000 retained transitions        | YAML is a repro config, not paper-data provenance |
| data.n_samples_val          | 800 trajectories        | 5000                      | paper; archived CSV has 16,000 retained transitions        | YAML is a repro config, not paper-data provenance |
| data.skip / retained        | 10 / 20                 | 10 / 20                   | paper and archived row counts (n_iterations 30, skip 10)   | ✓     |
| cmgdb.subdiv_init           | 23                      | 23                        | src/latentdynamics/configs/leslie3d_example1.yaml          | ✓     |
| cmgdb.subdiv_min            | 23                      | 23                        |                                                            | ✓     |
| cmgdb.subdiv_max            | 27                      | 27                        |                                                            | ✓     |
| cmgdb.bounds                | `[-0.6983771920204163, -0.8291957378387451]` -> `[0.9562897086143494, 0.7747613787651062]` | inferred from encoded data | `mg_params_log.txt` | active rerender source |
| cmgdb.bounds_epsilon_frac   | 0.01                    | 0.01                      | `mg_params_log.txt`                                        | ✓     |
| cmgdb.padding               | true                    | true                      | `mg_params_log.txt`                                        | ✓     |

The saved run's 122,346 Morse boxes all sit at effective level 23 (the
`subdiv_limit` of 10^4 prevented refinement to 27); box size
`(4.0397e-4, 7.8318e-4)`.

## Residual/tolerance table rows

The fine rows (`q = 0, 1, 4`) and the coarse rows (`q = 0, 1` at uniform
depth 22, computed by
`scripts/leslie3d_example1_uniform_sampled_metrics.py`, with the exact
forward-closure verification of
`scripts/leslie3d_example1_verify_closures.py`) are documented in
[`sampled_residual_tolerance.md`](sampled_residual_tolerance.md).

## Verification

```bash
python pipeline.py --config leslie3d_example1_replay --stages render,metrics
```

The fine graph must remain byte-identical to
`replay_sources/leslie3d_example1/spurious_attractor_ex/MG/morse_graph.pdf`.
Verify the six graph nodes and the five edges listed above. Treat
`metrics.json` only as a sampled diagnostic for its recorded target node; it
must not be used to label node 4 spurious. For the derived panels, the render
script's own count gates (174/123/25/322; coarse nontrivial rows
2714/85170/1152/291) are the verification.
