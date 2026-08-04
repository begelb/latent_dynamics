# fig_leslie3d_example2

Paper figure `fig:3D_Leslie_latent_dynamics_success`: the Leslie \(g_2\)
computation.

## Paper figures

- `paper/figures/leslie3d_example2/morse_graph.pdf`
- `paper/figures/leslie3d_example2/morse_sets_with_overlay.pdf`

## Source of paper run

The active graph is the May 27 render under
`code/replay/leslie3d_example2_patrick/MG/morse_graph.pdf`; its SHA-256
`9eaf1ee72b6e93885acfce78eb3680edbad25a022098b73f5458a6d54e0b968b`
is identical to the paper asset. It was rendered from the preserved checkpoint
and raw CMGDB artifacts mirrored under `replay_sources/leslie3d_example2/`.
This is distinct from the \(g_1\) run documented in
`leslie3d_example1.md`.

- training script:    not archived
- checkpoint:         `replay_sources/leslie3d_example2/models/{encoder,dynamics,decoder}.pt`
- CMGDB artifacts:    `replay_sources/leslie3d_example2/MG/{morse_graph,morse_sets}`
- trajectory render:  `replay_sources/leslie3d_example2/MG/morse_sets_trajectories.png`
- CMGDB params:       `replay_sources/leslie3d_example2/mg_params_log.txt`
- losses:             `replay_sources/leslie3d_example2/{final_losses.txt,logs/*.pkl}`
- scaler:             `replay_sources/leslie3d_example2/data/scalers/scaler`
- tolerance log:      `replay_sources/leslie3d_example2/tolerance_results.txt`
- authoritative reproduction metadata: `replay_sources/leslie3d_example2/run_manifest.json`
- immutable May 27 render record: `replay_sources/leslie3d_example2/run_manifest.render-2026-05-27.json`

## Status

**Artifact-only, read-only replay from the archived paper output.**
`src/latentdynamics/configs/leslie3d_example2_replay.yaml` points at the
artifact mirror under `code/replay_sources/leslie3d_example2/`. The original
training script and raw train/test CSVs are not archived, so the replay path
reads the preserved checkpoint and CMGDB artifacts directly. The generated
May 27 render manifest originally recorded the obsolete parameter inference
`(19.6, 23.68, 23.68)`. Its active `config` and `config_hash` have now been
corrected, while `provenance_corrections` preserves the original value and
explains the post-hoc change. The unedited May 27 record is retained beside it
under the filename above. Neither file is a recovered training log: one is
corrected reproduction metadata and the other is immutable render provenance.
The current `output/leslie3d_example2/` is a later retrain and must not be used
as the original paper source.

## Reproduction commands

Replay the preserved paper artifacts:

```bash
python pipeline.py --config leslie3d_example2_replay --stages render,metrics
```

A fresh retrain uses a writable local copy of the YAML with
`paths.output_dir` outside `replay_sources/` and `paths.read_only: false`
before running `data,scale,train,diagnose,morse`.

## Parameter provenance

The checkpoint does **not** contain `theta`; it contains only the learned
encoder, latent-map, and decoder weights. The corrected tuple was identified
within the known Leslie3D experiment family by combining the checkpoint with
Patrick's scaler and comparing one-step consistency: decode learned latent
transitions (or encode candidate Leslie transitions), then measure their
discrepancy for candidate parameter tuples. This check strongly distinguishes
`(28.9, 29.8, 22.0)` from the obsolete default-based tuple. The exact decimal
tuple is corroborated by the manuscript and the recovered data provenance.

Consequently, this is an inverse-identification result under the assumed
Leslie3D family, not a claim that the exact parameters can be read uniquely
from neural-network weights alone.

The quantitative discriminator replays the legacy loss formulas with the
archived `E`, `G`, `D`, and scaler.  Using 2,000 uniformly sampled initial
conditions (seed `9999`) and 20 transitions gives:

| candidate / reference | decoded one-step loss `loss_ae2` | latent consistency `loss_dyn` |
|---|---:|---:|
| Patrick's archived final validation log | `6.668765e-4` | `2.699535e-4` |
| `(28.9, 29.8, 22.0)` replay | `6.729696e-4` | `2.728845e-4` |
| `(19.6, 23.68, 23.68)` replay | `1.669253e-3` | `6.936188e-4` |

Thus the corrected candidate reproduces both parameter-sensitive losses to
about 1%, whereas the obsolete tuple is roughly 2.5--2.6 times larger.  Across
sampling seeds `1,2,3,42,9999`, mean `loss_dyn` is `2.71137e-4` for the
corrected candidate and `6.93912e-4` for the obsolete one.  This is strong
disambiguation between the two documented candidates; it is not a global
identifiability proof over all possible Leslie parameters or model families.

## Expected scientific output

The raw graph has five Morse sets and the edges

```
2 -> 0
2 -> 1
3 -> 1
4 -> 3
```

Its minimal nodes 0 and 1 both have Conley index `(x^4-1, 0, 0)`. The saved
sampled tolerance tests fail for both nodes: the maximum sampled residuals are
`0.2937395` and `0.1954082`, while the respective sampled tolerances are
`7.628916e-05` and `1.261359e-04`. These finite diagnostics do not certify a
semiconjugacy and must not be described as passing.

## Hyperparameter audit

| param                       | supported value         | YAML value             | severity | notes |
|-----------------------------|-------------------------|------------------------|----------|-------|
| system.params.th1           | 28.9                    | 28.9                   | ✓        | inverse-identified and corroborated; not encoded in checkpoint |
| system.params.th2           | 29.8                    | 29.8                   | ✓        | earlier default-based inference was wrong |
| system.params.th3           | 22.0                    | 22.0                   | ✓        | active manifests corrected; original value retained in correction metadata |
| system.params.survival_p1   | 0.7                     | 0.7                    | ✓        |       |
| system.params.survival_p2   | 0.7                     | 0.7                    | ✓        |       |
| arch.num_layers             | 2                       | 2                      | ✓        | checkpoint has `linear_0` through `linear_2` |
| arch.hidden_shape           | 64                      | 64                     | ✓        | checkpoint tensor shapes imply width 64 |
| arch.high_dims              | 3                       | 3                      | ✓        |       |
| arch.low_dims               | 2                       | 2                      | ✓        |       |
| arch.encoder_out_activation | tanh                    | tanh (default)         | ✓        |       |
| arch.latent_out_activation  | tanh                    | tanh (default)         | ✓        |       |
| arch.decoder_out_activation | sigmoid                 | sigmoid (default)      | ✓        |       |
| training.loss_weights       | [100, 10, 20]           | [100, 10, 20]          | ✓        | recovered from total-loss linear relation |
| data.n_samples_train        | not independently archived | 8000               | unresolved | source CSVs and training script are absent |
| data.n_samples_val          | not independently archived | 2000               | unresolved | source CSVs and training script are absent |
| data.n_iterations           | paper reports 20        | 20                     | manuscript | not independently recoverable from raw data |
| cmgdb.subdiv_init           | 25                      | 25                     | ✓        | from `mg_params_log.txt` |
| cmgdb.subdiv_min            | 28                      | 28                     | ✓        | from `mg_params_log.txt` |
| cmgdb.subdiv_max            | 29                      | 29                     | ✓        | from `mg_params_log.txt` |
| cmgdb.bounds                | `[-0.37490714, -0.4695556]` -> `[0.3535685, 0.455769]` | inferred | ✓ | from `mg_params_log.txt` |
| cmgdb.bounds_epsilon_frac   | 0.01                    | 0.01                   | ✓        | archived generator convention |
| cmgdb.padding               | true                    | true                   | ✓        | replay config |

## Verification

Replay from the archived artifacts:

```bash
python pipeline.py --config leslie3d_example2_replay --stages render,metrics
# Check the five graph nodes, the four edges above, and the two recorded FAILs.
```

The active paper graph must remain byte-identical to
`replay/leslie3d_example2_patrick/MG/morse_graph.pdf`. Treat the corrected
manifest parameters as recovered experiment provenance, not as fields captured
by Patrick's original training run.
