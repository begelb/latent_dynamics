# fig_leslie3d_success

Paper §1.211: the success case of the 3D Leslie experiment, where the learned latent dynamics correctly resolves the multiple invariant sets.

## Paper figures

- `paper/figures/Leslie_3D_10KData_Mgraph.png`
- `paper/figures/Leslie_3D_10kData_Msets.png`
- `paper/figures/3D_success_tolerance.png`

## Source of paper run

Brittany. **No saved CMGDB artefacts in archive** — `archive/brittany/output/Leslie_3D/28.9_29.8_22.0/` contains only `scalers/`. The original run was either lost or was performed without committing `MG/`/`models/` to git.

- training script:    `archive/brittany/main_scripts/train.py`
- CMGDB script:       `archive/brittany/main_scripts/morse_graph.py`
- legacy config:      `archive/brittany/config/Leslie_3D_larger_domain_tail_only.yaml`

## Status

**scratch-only** until the source paper run is recovered. The current `output/leslie3d_success/seed_0/` is a retrain produced by an earlier session.

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

Paper figure shows six Morse sets in a chain `5 -> 4 -> 3 -> 2 -> {0,1}` with attractors at the leaves. Conley indices reported in the paper text indicate attracting fixed points and a non-trivial saddle structure. Exact replication requires recovering Brittany's checkpoint.

Verification target after retrain: `metrics.json` reports `tau_bar > max_semiconjugacy_error` (i.e., the learned latent dynamics is a faithful semiconjugacy at the data level).

## Hyperparameter audit

Drift identified between the legacy YAML and the current config:

| param                       | archive value           | YAML value             | severity | notes |
|-----------------------------|-------------------------|------------------------|----------|-------|
| arch.num_layers             | 3                       | 3                      | ✓        |       |
| arch.hidden_shape           | 32                      | 32                     | ✓        |       |
| arch.encoder_out_activation | tanh                    | tanh (default)         | ✓        |       |
| arch.latent_out_activation  | tanh                    | tanh (default)         | ✓        | suspected near-identity collapse for 2D Leslie targets in this regime |
| training.loss_weights       | [10, 10, 1]             | [10, 10, 1]            | ✓        |       |
| data.n_samples_train        | 4000                    | 10000                  | HIGH     | drift introduced when reverse-engineering YAML |
| data.n_iterations           | 30                      | 20                     | HIGH     | drift; archive trains over longer trajectories |
| data.sampling_method        | uniform                 | uniform                | ✓        |       |

Recommended pre-retrain edits to `configs/leslie3d_success.yaml`:

```yaml
data:
  n_samples_train: 4000
  n_iterations: 30
```

If retraining still produces a near-identity latent map, raise the prediction-loss weight (currently `loss_weights[1] = 10`) or increase `loss_weights[2]` (the latent-semiconjugacy term, currently `1`).

## Verification

After retraining:

```bash
python pipeline.py --config configs/leslie3d_success.yaml --stages diagnose --max-seeds 1
# diagnose.json should report frac_unconverged < 0.5 and n_distinct_limit_points >= 3
python pipeline.py --config configs/leslie3d_success.yaml --stages metrics --max-seeds 1
# metrics.json should report tau_bar > max_semiconjugacy_error
```
