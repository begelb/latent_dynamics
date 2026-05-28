# fig_leslie3d_example2

Paper §1.211: the success case of the 3D Leslie experiment, where the learned latent dynamics correctly resolves the multiple invariant sets.

## Paper figures

- `paper/figures/Leslie3D_correctParam_graph.png`
- `paper/figures/Leslie3D_trajs.png`

## Source of paper run

The preserved replay source. The saved non-spurious Leslie 3D paper run is
mirrored under `replay_sources/leslie3d_example2/`. This is distinct from the
spurious Leslie 3D run, which is documented in `leslie3d_example1.md`.

- training script:    not archived
- checkpoint:         `replay_sources/leslie3d_example2/models/{encoder,dynamics,decoder}.pt`
- CMGDB artifacts:    `replay_sources/leslie3d_example2/MG/{morse_graph,morse_sets}`
- trajectory render:  `replay_sources/leslie3d_example2/MG/morse_sets_trajectories.png`
- CMGDB params:       `replay_sources/leslie3d_example2/mg_params_log.txt`
- losses:             `replay_sources/leslie3d_example2/{final_losses.txt,logs/*.pkl}`
- scaler:             `replay_sources/leslie3d_example2/data/scalers/scaler`
- tolerance log:      `replay_sources/leslie3d_example2/tolerance_results.txt`

## Status

**Read-only replay from the archived paper artifacts.**
`configs/leslie3d_example2.yaml` is read-only and points at the artifact mirror
under `code/replay_sources/leslie3d_example2/`. The original training script
and raw train/test CSVs are not archived, so the replay path reads the
preserved checkpoint and CMGDB artifacts directly. The current
`output/leslie3d_example2/seed_0/` is a retrain produced by an earlier session
and should not be used as the original paper source.

## Reproduction commands

Replay the preserved paper artifacts:

```bash
python pipeline.py --config configs/leslie3d_example2.yaml --stages render,metrics
```

A fresh retrain uses a writable local copy of the YAML with
`paths.output_dir` outside `replay_sources/` and `paths.read_only: false`
before running `data,scale,train,diagnose,morse`.

## Expected scientific output

Paper figure shows the non-spurious/bistable Leslie 3D result from the
preserved checkpoint, with the Hasse diagram and trajectory panel preserved in
`replay_sources/leslie3d_example2/MG/`.

Verification target after retrain: `metrics.json` reports `tau_bar > max_semiconjugacy_error` (i.e., the learned latent dynamics is a faithful semiconjugacy at the data level).

## Hyperparameter audit

| param                       | archive value           | YAML value             | severity | notes |
|-----------------------------|-------------------------|------------------------|----------|-------|
| system.params.th1           | 19.6                    | 19.6                   | ✓        | non-spurious/default Leslie 3D |
| system.params.th2           | 23.68                   | 23.68                  | ✓        | distinct from the spurious run |
| system.params.th3           | 23.68                   | 23.68                  | ✓        | distinct from the spurious run |
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
| data.n_samples_train        | 8000                    | 8000                   | ✓        | inferred from scaler size + paper `D(20,10000)` |
| data.n_samples_val         | 2000                    | 2000                   | ✓        | reconstructed 80/20 split |
| data.n_iterations           | 20                      | 20                     | ✓        | paper `D(20,10000)` |
| cmgdb.subdiv_init           | 25                      | 25                     | ✓        | from `mg_params_log.txt` |
| cmgdb.subdiv_min            | 28                      | 28                     | ✓        | from `mg_params_log.txt` |
| cmgdb.subdiv_max            | 29                      | 29                     | ✓        | from `mg_params_log.txt` |

## Verification

Replay from the archived artifacts:

```bash
python pipeline.py --config configs/leslie3d_example2.yaml --stages render,metrics
# metrics.json should report tau_bar > max_semiconjugacy_error
```
