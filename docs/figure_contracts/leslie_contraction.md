# fig_leslie_contraction

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

**archived paper artifacts located; training script/raw CSVs still missing**.
The paper figure content should be sourced from `archive/patrick/Leslie10D/`.
The YAML now records the recovered architecture, loss weights, dataset split,
and CMGDB settings, but a fresh retrain is still not the original Patrick run
until it passes the expected Morse-graph and tolerance checks.

Diagnose on the current retrain: encoded extent `[1.08, 1.32]` (healthy), `latent_map` iteration does not converge after 200 steps. Same near-identity-latent regime as `leslie3d_success`.

## Reproduction commands

```bash
python pipeline.py --config configs/leslie_contraction.yaml --stages diagnose --max-seeds 1
```

Fresh retrain on AMAREL, after any remaining Patrick hyperparameters are filled
in:

```bash
sbatch --array=0-0 --export=ALL,CONFIG=configs/leslie_contraction.yaml,STAGES=train,diagnose,morse,EXPECTED_CELLS=1 \
  slurm/pipeline_array.sbatch
```

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

After retrain (and once the architecture is reconciled with Patrick's source):

```bash
python pipeline.py --config configs/leslie_contraction.yaml --stages diagnose --max-seeds 1
# Expect frac_unconverged < 0.5 and n_distinct_limit_points >= 2 (two attractors + a saddle).
```
