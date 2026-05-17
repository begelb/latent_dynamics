# fig_leslie3d_success

Paper §1.211: the success case of the 3D Leslie experiment, where the learned latent dynamics correctly resolves the multiple invariant sets.

## Paper figures

- `paper/figures/Leslie3D_correctParam_graph.png`
- `paper/figures/Leslie3D_trajs.png`

## Source of paper run

Patrick. The saved non-spurious Leslie 3D paper run is archived under
`archive/patrick/Leslie3D/`. This is distinct from Brittany's spurious Leslie
3D run, which is documented in `leslie3d_spurious.md`.

- training script:    MISSING (originated with Patrick)
- checkpoint:         `archive/patrick/Leslie3D/models/{encoder,dynamics,decoder}.pt`
- CMGDB artifacts:    `archive/patrick/Leslie3D/MG/{morse_graph,morse_sets}`
- trajectory render:  `archive/patrick/Leslie3D/MG/morse_sets_trajectories.png`
- CMGDB params:       `archive/patrick/Leslie3D/mg_params_log.txt`
- losses:             `archive/patrick/Leslie3D/{final_losses.txt,logs/*.pkl}`
- scaler:             `archive/patrick/Leslie3D/data/scalers/scaler`
- tolerance log:      `archive/patrick/Leslie3D/tolerance_results.txt`

## Status

**archived paper artifacts located; training script/raw CSVs still missing**.
The paper figure content should be sourced from `archive/patrick/Leslie3D/`.
The YAML now records Patrick's non-spurious Leslie parameters, recovered loss
weights, reconstructed dataset split, and CMGDB settings. The current
`output/leslie3d_success/seed_0/` is a retrain produced by an earlier session
and should not be used as the original paper source.

Diagnose on the current retrain reports: encoded extent `[1.32, 1.03]` (healthy spread), `latent_map` iteration does NOT converge after 200 steps (latent_map is too close to identity). The saved `MG/morse_graph` shows 1 Morse set with index `(x-1, 0, 0)`. Conclusion: training converged the encoder but not the latent_map; the prediction/semiconjugacy term needs more weight or longer training.

## Reproduction commands

Diagnose the existing retrain (no CMGDB):

```bash
python pipeline.py --config configs/leslie3d_success.yaml --stages diagnose --max-seeds 1
```

Forced retrain on the AMAREL cluster after re-tuning:

```bash
sbatch --array=0-0 --export=ALL,CONFIG=configs/leslie3d_success.yaml,STAGES=train,diagnose,morse,EXPECTED_CELLS=1 \
  slurm/pipeline_array.sbatch
```

## Expected scientific output

Paper figure shows the non-spurious/bistable Leslie 3D result from Patrick's
checkpoint, with the Hasse diagram and trajectory panel preserved in
`archive/patrick/Leslie3D/MG/`.

Verification target after retrain: `metrics.json` reports `tau_bar > max_semiconjugacy_error` (i.e., the learned latent dynamics is a faithful semiconjugacy at the data level).

## Hyperparameter audit

| param                       | archive value           | YAML value             | severity | notes |
|-----------------------------|-------------------------|------------------------|----------|-------|
| system.params.th1           | 19.6                    | 19.6                   | ✓        | Patrick non-spurious/default Leslie 3D |
| system.params.th2           | 23.68                   | 23.68                  | ✓        | distinct from Brittany's spurious run |
| system.params.th3           | 23.68                   | 23.68                  | ✓        | distinct from Brittany's spurious run |
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
| data.n_samples_train        | 8000                    | 8000                   | ✓        | scaler size + paper `D(20,10000)`; raw CSV not archived |
| data.n_samples_val         | 2000                    | 2000                   | ✓        | reconstructed 80/20 split |
| data.n_iterations           | 20                      | 20                     | ✓        | paper `D(20,10000)` |
| cmgdb.subdiv_init           | 25                      | 25                     | ✓        | from `mg_params_log.txt` |
| cmgdb.subdiv_min            | 28                      | 28                     | ✓        | from `mg_params_log.txt` |
| cmgdb.subdiv_max            | 29                      | 29                     | ✓        | from `mg_params_log.txt` |

## Verification

After retraining:

```bash
python pipeline.py --config configs/leslie3d_success.yaml --stages diagnose --max-seeds 1
# diagnose.json should report frac_unconverged < 0.5 and n_distinct_limit_points >= 3
python pipeline.py --config configs/leslie3d_success.yaml --stages metrics --max-seeds 1
# metrics.json should report tau_bar > max_semiconjugacy_error
```
