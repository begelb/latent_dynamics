# fig_leslie3d_example1

Paper figure `fig:3D_Leslie_latent_dynamics`: the Leslie (g_1) computation
with an additional minimal latent component.

## Paper figures

- `paper/figures/leslie3d_example1/morse_graph.pdf`
- `paper/figures/leslie3d_example1/morse_sets_with_overlay.pdf`

## Source of paper run

The active graph is the May 27 rerender under
`code/replay/leslie3d_example1_brittany/MG/morse_graph.pdf`; its SHA-256
`72137014f4f127157cdee75e8a1fab5817b5197abcba05dd5dc19318c07f0782`
is identical to the paper asset. The rerender reads the saved checkpoint and
raw CMGDB artifacts under
`code/replay_sources/leslie3d_example1/spurious_attractor_ex/`.

- mg_params_log:      `replay_sources/leslie3d_example1/spurious_attractor_ex/mg_params_log.txt` (subdiv 23/23/27)
- legacy checkpoint:  `replay_sources/leslie3d_example1/spurious_attractor_ex/models/{encoder,dynamics,decoder}.pt`

The on-disk copy under `code/replay_sources/` is loaded through `latentdynamics.training.load_legacy_checkpoint`.

The current replay tree was overwritten by commit
`440449ade912744a97aa41a2076ed6717a865718`. The earlier computation is still
recoverable at commit `e6501f1b359e0040d6261f9dc3d84f1c8d729aa5` and must
not be used as metadata for the active rerender. It used bounds
`[-0.6228695,-0.74216413]` to `[0.30980384,0.22416562]`; the active rerender's
saved run uses the bounds recorded below.

## Status

**Artifact replay-ready.** Render and sampled metrics work from the saved DOT,
CSV, and legacy 3-file checkpoint. The generated manifest's data defaults do
not describe the paper's 3,200/800 trajectory split, so this is not a validated
fresh-training reproduction. Re-running CMGDB requires `--force-overwrite`.

## Reproduction commands

Replay (no CMGDB, ~10 s):

```bash
python pipeline.py --config leslie3d_example1 --stages render,metrics
```

Diagnose (cheap, no CMGDB):

```bash
python pipeline.py --config leslie3d_example1 --stages diagnose
```

Note: data/ for spurious uses the legacy `2train.csv`/`2test.csv` naming; the diagnose stage looks for `train.csv` and will fail unless the files are renamed or `data.train_files: ["2train"]` is added to the config.

Forced retrain (recompute data, training, and CMGDB):

```bash
python pipeline.py --config leslie3d_example1 --stages all --force-overwrite
```

## Expected scientific output

`replay_sources/leslie3d_example1/spurious_attractor_ex/MG/morse_graph` (DOT) has six Morse sets with Hasse:

```
{rank=same; 0 1 4};
2 -> 0
2 -> 1
3 -> 1
5 -> 3
5 -> 4
```

Minimal nodes are 0, 1, and 4. Node 4 is the additional latent minimal
component relative to the two-minimal-node ground-truth computation. All 16
phase points of the recurrent invariant sets catalogued by the direct
computation (P0, P1, S2, S4, p_star, and the origin) encode outside node 4; the
closest is p_star, at distance `1.369758509e-3` from its box union. Accordingly,
node 4 has no counterpart in that detected recurrent-set inventory. This is
numerical evidence, not a proof that no unidentified recurrent point lies in
its ambient preimage. Moreover, 507 centers of the outer ground-truth node-4
boxes encode into the latent node-4 box union, so the saved boxes cannot
establish the requested nonintersection. In particular, the current
`metrics.json` targets node 0, not node 4, and a sampled inequality failure is
not a nonlifting certificate.

The direct comparison graph contains the edge `3 -> 2` in both its saved
screen and Conley runs, but this is a combinatorial reachability relation, not
a certified connection. An on-demand uniform-grid check finds `S4 -> S2` paths
at levels 29--32. At level 33, a search from all 5,834 saved node-3 boxes to all
9,217 saved node-2 boxes exhausts a forward closure of 2,924,013 boxes without
a path. CMGDB propagates the coarser relations to the level-33 descendants, so
the displayed `3 -> 2` edge is an inherited subdivision artifact in this
corner-sampled computation. Increasing only `subdiv_limit` cannot remove it.
This does not rule out a connection in the continuous map: the eight-corner
box map is not a validated interval enclosure.

## Hyperparameter audit

| param                       | archive value           | YAML value                | source line                                                | notes |
|-----------------------------|-------------------------|---------------------------|------------------------------------------------------------|-------|
| system.params.th1           | 28.9                    | 28.9                      | src/latentdynamics/configs/leslie3d_example1.yaml:5                           | ✓     |
| system.params.th2           | 29.8                    | 29.8                      |                                                            | ✓     |
| system.params.th3           | 22.0                    | 22.0                      |                                                            | ✓     |
| system.params.survival_p1   | 0.7                     | 0.7                       |                                                            | ✓     |
| system.params.survival_p2   | 0.7                     | 0.7                       |                                                            | ✓     |
| arch.num_layers             | 3                       | 3                         | src/latentdynamics/configs/leslie3d_example1.yaml:11                          | ✓     |
| arch.hidden_shape           | 32                      | 32                        |                                                            | ✓     |
| arch.high_dims              | 3                       | 3                         |                                                            | ✓     |
| arch.low_dims               | 2                       | 2                         |                                                            | ✓     |
| arch.encoder_out_activation | tanh                    | tanh (default)            | default                                                    | ✓     |
| arch.latent_out_activation  | tanh                    | tanh (default)            |                                                            | ✓     |
| arch.decoder_out_activation | sigmoid                 | sigmoid (default)         |                                                            | ✓     |
| training.loss_weights       | [10, 10, 1]             | [10, 10, 1]               | src/latentdynamics/configs/leslie3d_example1.yaml:17                          | ✓     |
| data.n_samples_train        | 3200 trajectories       | 4000                      | paper; archived CSV has 64,000 retained transitions         | generated config is stale |
| data.n_samples_val          | 800 trajectories        | 5000                      | paper; archived CSV has 16,000 retained transitions         | generated config is stale |
| data.skip / retained        | 10 / 20                 | 0 / 30                    | paper and archived row counts                               | generated config is stale |
| cmgdb.subdiv_init           | 23                      | 23                        | src/latentdynamics/configs/leslie3d_example1.yaml:27                          | ✓     |
| cmgdb.subdiv_min            | 23                      | 23                        |                                                            | ✓     |
| cmgdb.subdiv_max            | 27                      | 27                        |                                                            | ✓     |
| cmgdb.bounds                | `[-0.6983771920204163, -0.8291957378387451]` -> `[0.9562897086143494, 0.7747613787651062]` | inferred from encoded data | current `mg_params_log.txt` | active May 27 rerender source |
| cmgdb.bounds_epsilon_frac   | 0.01                    | 0.01                      | current `mg_params_log.txt`                                 | ✓     |
| cmgdb.padding               | true                    | true                      | current `mg_params_log.txt`                                 | ✓     |

## Verification

```bash
python pipeline.py --config leslie3d_example1 --stages render,metrics
```

The active paper graph must remain byte-identical to
`replay/leslie3d_example1_brittany/MG/morse_graph.pdf`. Verify the six graph
nodes and the five edges listed above. Treat `metrics.json` only as a sampled
diagnostic for its recorded target node; it must not be used to label node 4
spurious.
