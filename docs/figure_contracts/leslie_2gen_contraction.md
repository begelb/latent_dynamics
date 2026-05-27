# fig_leslie_2gen_contraction

Paper Fig. 1.83: the 10D Embedded Leslie example. The first two coordinates
follow the 2D Leslie/Ricker map, and the remaining eight coordinates contract
by 0.25, so the relevant invariant dynamics live on the embedded coordinate
plane.

## Paper figures

- `paper/figures/LeslieContraction10D.png`
- `paper/figures/LeslieContraction_10D_10kData_Mgraph.png`
- `paper/figures/LeslieContraction_10D_10kData_Msets.png`

## Source of paper run

Patrick. The saved paper run is now archived under `archive/patrick/Leslie10D/`.
The trained checkpoint and CMGDB artifacts are present; the original training
script and raw train/test CSVs are still not archived.

- training script:    MISSING (originated with Patrick)
- checkpoint:         `archive/patrick/Leslie10D/models/{encoder,dynamics,decoder}.pt`
- CMGDB artifacts:    `archive/patrick/Leslie10D/MG/{morse_graph,morse_sets}`
- CMGDB params:       `archive/patrick/Leslie10D/mg_params_log.txt`
- losses:             `archive/patrick/Leslie10D/final_losses.txt`
- legacy config:      none
- system definition:  `code/src/latentdynamics/systems/leslie.py:13-53` (`LeslieContraction` class)

## Status

**replay-ready for archived paper artifacts; fresh exact reproduction is still
incomplete**. `configs/leslie_2gen_contraction.yaml` is read-only and points at the
Patrick artifact mirror under `code/replay_sources/leslie_2gen_contraction/`.
Patrick's original training script and raw train/test CSVs are still missing,
so a fresh retrain is not the original Patrick run until it passes the expected
Morse-graph and tolerance checks.

Earlier package retrains in `code/output/leslie_2gen_contraction/` are diagnostic
only; they should not be treated as the paper source.

## Reproduction commands

```bash
python pipeline.py --config configs/leslie_2gen_contraction.yaml --stages render,metrics
```

Fresh retrain remains a separate recovery task because Patrick's source data
and training script are missing. Use a writable local copy of the YAML with
`paths.output_dir` outside `replay_sources/` and `paths.read_only: false`
before running `data,scale,train,diagnose,morse`.

## Expected scientific output

Paper figure shows four Morse sets in a chain `3 -> {2, 1, 0}` with Conley indices `(0, x³-1, 0)`, `(0, 0, x-1)`, `(x-1, x-1, 0)`, `(x³-1, 0, 0)` (one repeller, one saddle, two attractors). The 8 contracting tail dimensions should be projected away by the encoder, leaving 2D Leslie-like dynamics in the latent.

## Hyperparameter audit

| param                       | archive value | YAML value              | source line                                | notes |
|-----------------------------|---------------|-------------------------|--------------------------------------------|-------|
| system.params.th1           | 23.5         | 23.5                    | archive/patrick/README.md                  | default `LeslieContraction` |
| system.params.th2           | 23.5         | 23.5                    | archive/patrick/README.md                  | default `LeslieContraction` |
| system.params.survival_p1   | 0.7          | 0.7                     | archive/patrick/README.md                  | default `LeslieContraction` |
| system.params.contraction   | 0.25         | 0.25                    | archive/patrick/README.md                  | tail contraction |
| arch.num_layers             | 4             | 4                       | archive/patrick/Leslie10D/models/*.pt      | checkpoint has `linear_0` through `linear_4` |
| arch.hidden_shape           | 64            | 64                      | archive/patrick/Leslie10D/models/*.pt      | tensor shapes imply width 64 |
| arch.high_dims              | 10            | 10                      |                                            | ✓     |
| arch.low_dims               | 2             | 2                       |                                            | ✓     |
| arch.encoder_out_activation | tanh          | tanh (default)          | archive/patrick/Leslie10D/models/encoder.pt | checkpoint contains final `Tanh` |
| arch.latent_out_activation  | tanh          | tanh (default)          | archive/patrick/Leslie10D/models/dynamics.pt | checkpoint contains final `Tanh` |
| arch.decoder_out_activation | sigmoid       | sigmoid (default)       | archive/patrick/Leslie10D/models/decoder.pt | checkpoint contains final `Sigmoid` |
| training.loss_weights       | [100, 10, 20] | [100, 10, 20]          | archived train/test loss logs              | recovered from total-loss linear relation |
| data.n_samples_train        | 8000          | 8000                   | scaler size + paper `D(20,10000)`          | 80/20 split reconstructed; raw CSV not archived |
| data.n_samples_val         | 2000          | 2000                   | scaler size + paper `D(20,10000)`          | raw CSV not archived |
| data.n_iterations           | 20            | 20                     | paper `D(20,10000)`                        | raw CSV not archived |
| cmgdb.subdiv_init           | 25            | 25                      | archive/patrick/Leslie10D/mg_params_log.txt | ✓ |
| cmgdb.subdiv_min            | 27            | 27                      | archive/patrick/Leslie10D/mg_params_log.txt | ✓ |
| cmgdb.subdiv_max            | 28            | 28                      | archive/patrick/Leslie10D/mg_params_log.txt | ✓ |

## Verification

Replay from the archived artifacts:

```bash
python pipeline.py --config configs/leslie_2gen_contraction.yaml --stages render,metrics
# The rendered Morse graph should preserve the four-node Patrick Hasse diagram.
```
