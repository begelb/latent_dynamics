# fig_chafee_infante

Paper §1.256: Chafee-Infante PDE in spectral coordinates (64 Fourier modes), super-critical regime `α = 28`. Demonstrates the framework on a stiff PDE with multiple equilibria and saddle structure.

## Paper figures

- `paper/figures/ci_morse_graph.pdf`
- `paper/figures/ci_morse_sets.pdf`
- `paper/figures/ci_bif_diagram.pdf`

## Source of paper run

Marcio. Self-contained reference implementation in `archive/marcio/scripts/`.

- model + system:     `archive/marcio/scripts/autoencoder_model.py` (`DynamicsAutoencoder` class, lines 76-102; PDE in lines 19-32; data generator in lines 37-61)
- training script:    `archive/marcio/scripts/train_model.py`
- CMGDB script:       `archive/marcio/scripts/compute_dynamics.py:25-30` (subdiv 14/16/28; bounds `[-3, 3] × [-2, 2]`)
- weights:            `archive/marcio/scripts/ci_model_weights.pth` (single state_dict)

## Status

**partial: scratch retrain produced an encoder with rich dynamics but a Morse graph that under-resolves it**.

Diagnose on the current `output/chafee_infante/seed_0/`:

- encoded extent `[1.40, 2.03]` (healthy spread)
- `n_distinct_limit_points = 19` (rich attractor structure)
- `frac_unconverged = 0.00`
- but the saved `MG/morse_graph` reports a single Morse set `(x-1, 0, 0)`

Cross-check verdict: `morse_underresolves`. The training is OK; the saved Morse graph was computed with too-coarse subdivisions (current YAML inherits the default `subdiv_min=8, subdiv_max=10`). Marcio's archive uses `subdiv_min=14, subdiv_max=28` and explicit bounds `[-3, 3] × [-2, 2]`.

## Reproduction commands

Diagnose (no CMGDB):

```bash
python pipeline.py --config configs/chafee_infante.yaml --stages diagnose --max-seeds 1
```

Re-run CMGDB only with Marcio's subdivisions (replaces the existing 1-Morse-set output; long):

```bash
# Edit configs/chafee_infante.yaml so cmgdb block reads:
#   subdiv_init: 14
#   subdiv_min: 14
#   subdiv_max: 28
#   lower_bounds: [-3.0, -2.0]
#   upper_bounds: [3.0, 2.0]
#   padding: false
sbatch --array=0-0 --export=ALL,CONFIG=configs/chafee_infante.yaml,STAGES=morse,EXPECTED_CELLS=1 \
  slurm/pipeline_array.sbatch
```

`--stages morse` will refuse with the legacy-checkpoint guard if the model dir contains 3-file format files, but the new format `autoencoder.pt`+`autoencoder.json` won't trigger it. The morse-artefact guard will refuse a second run unless `--force-overwrite` is passed.

## Expected scientific output

Paper figure shows seven Morse sets: two attractors `(x-1, 0, 0)`, three saddles `(0, x-1, 0)`, two repellers `(0, 0, x-1)`. The Hasse diagram is non-trivial.

## Hyperparameter audit

| param                       | archive value           | YAML value             | source line                                                | notes |
|-----------------------------|-------------------------|------------------------|------------------------------------------------------------|-------|
| arch.encoder hidden_shapes  | [64, 32]                | [64, 32]               | configs/chafee_infante.yaml                                | ✓     |
| arch.latent_map hidden_shapes | [32, 32]              | [32, 32]               |                                                            | ✓     |
| arch.decoder hidden_shapes  | [32, 64]                | [32, 64]               |                                                            | ✓     |
| arch.activation             | tanh                    | tanh                   | autoencoder_model.py:82,84,90,92,98,100                    | ✓     |
| arch.encoder_out_activation | none                    | none                   | autoencoder_model.py:85 (Linear, no terminal)              | ✓     |
| arch.latent_out_activation  | none                    | none                   | autoencoder_model.py:93                                    | ✓     |
| arch.decoder_out_activation | none                    | none                   | autoencoder_model.py:101                                   | ✓     |
| arch.high_dims              | 64                      | 64                     |                                                            | ✓     |
| arch.low_dims               | 2                       | 2                      |                                                            | ✓     |
| training.epochs             | 4000                    | 4000                   | autoencoder_model.py:73                                    | ✓     |
| training.learning_rate      | 0.003                   | 0.003                  | autoencoder_model.py:72                                    | ✓     |
| training.scheduler_factor   | 0.5                     | 0.5                    | autoencoder_model.py:113                                   | ✓     |
| training.scheduler_min_lr   | 1e-6                    | 1e-6                   | autoencoder_model.py:115                                   | ✓     |
| training.loss_weights       | recon + pred only       | [1, 1, 0]              | train_model.py:32,37 (no semiconjugacy term)               | ✓     |
| data.scaling                | none                    | none                   | no scaler in archive                                       | ✓     |
| data.n_samples_train        | 1000                    | 1000                   | autoencoder_model.py:13                                    | ✓     |
| data.n_iterations           | 30                      | 30                     | autoencoder_model.py:14 (time_steps)                       | ✓     |
| data.tau                    | 0.1                     | 0.1                    | autoencoder_model.py:12                                    | ✓     |
| data.sampling_method        | sobol (seed 7206)       | sobol (seed 7206)      | autoencoder_model.py:40                                    | ✓     |
| cmgdb.subdiv_init           | 14                      | 14                     | compute_dynamics.py:25                                     | needs update; current YAML may default to 8 |
| cmgdb.subdiv_min            | 14                      | 14                     |                                                            | "  |
| cmgdb.subdiv_max            | 28                      | 28                     | compute_dynamics.py:26                                     | "  |
| cmgdb.lower_bounds          | [-3, -2]                | [-3, -2]               | compute_dynamics.py:29                                     | "  |
| cmgdb.upper_bounds          | [3, 2]                  | [3, 2]                 | compute_dynamics.py:30                                     | "  |
| cmgdb.padding               | false                   | false                  | compute_dynamics.py:24                                     | "  |

The training-side config matches Marcio. The CMGDB-side block of the YAML should be verified to match the archive (needs read of current YAML).

## Verification

```bash
python pipeline.py --config configs/chafee_infante.yaml --stages metrics --max-seeds 1
# After CMGDB rerun, metrics.json should report
#   diagnose_morse_cross_check.agreement: agree
#   n_morse_sets: ~ 7  (two attractors, three saddles, two repellers)
```
