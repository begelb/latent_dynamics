# fig_chafee_infante

Paper §1.256: Chafee-Infante PDE in spectral coordinates (64 Fourier modes), super-critical regime `α = 28`. Demonstrates the framework on a stiff PDE with multiple equilibria and saddle structure.

## Paper figures

- `paper/figures/ci_morse_graph.pdf`
- `paper/figures/ci_morse_sets.pdf`
- `paper/figures/ci_bif_diagram.pdf`

## Source of paper run

The reference Chafee-Infante model. Self-contained reference implementation in
`archive/marcio/scripts/`.

- model + system:     `archive/marcio/scripts/autoencoder_model.py` (`DynamicsAutoencoder` class, lines 76-102; PDE in lines 19-32; data generator in lines 37-61)
- training script:    `archive/marcio/scripts/train_model.py`
- CMGDB script:       `archive/marcio/scripts/compute_dynamics.py:25-30` (subdiv 10/14/28; bounds `[-3, 3] × [-2, 2]`)
- weights:            `archive/marcio/scripts/ci_model_weights.pth` (single state_dict)
- data:               `archive/marcio/scripts/train_data.csv` converted to `code/data/chafee_infante/{train,val}.csv` for the current pipeline; an exact header-added archive mirror also exists under `code/replay_sources/chafee_infante/data/`
- rendered figures:   `archive/marcio/scripts/ci_morse_graph.pdf`, `archive/marcio/scripts/ci_morse_sets.pdf`

## Status

**Reference data and config values are matched.** The converted train/val CSVs
are in the expected artifact structure, and `configs/chafee_infante.yaml` uses
the reference CMGDB settings: `subdiv_init=10`, `subdiv_min=14`,
`subdiv_max=28`, explicit bounds `[-3, -2] -> [3, 2]`, and `padding: false`.

The reference `ci_model_weights.pth` uses the original state_dict key names; a
fresh CMGDB run uses the current `autoencoder.pt` + `autoencoder.json`
checkpoint format. The archive carries rendered PDFs, so a fresh CMGDB run
regenerates the raw DOT/CSV artifacts.

Earlier package retrains in `code/output/chafee_infante/` are exploratory and
should not be treated as the paper source.

## Reproduction commands

Inspect the current converted-data config without recomputing CMGDB:

```bash
python pipeline.py --config configs/chafee_infante.yaml --stages diagnose --max-seeds 1
```

After converting the reference `ci_model_weights.pth` into the current
checkpoint format, rerun CMGDB with the reference archived settings (long):

```bash
CONFIG=configs/chafee_infante.yaml STAGES=morse EXPECTED_CELLS=1 \
  sbatch --array=0-0 --export=ALL,CONFIG,STAGES,EXPECTED_CELLS \
  slurm/pipeline_array.sbatch
```

`--stages morse` will refuse with the legacy-checkpoint guard if the model dir
contains 3-file format files, but the new format
`autoencoder.pt`+`autoencoder.json` will not trigger it. The morse-artifact
guard will refuse a second run unless `--force-overwrite` is passed.

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
| data.sampling_method        | uniform (seed 7206)     | uniform                | autoencoder_model.py:40,55                                 | ✓     |
| cmgdb.subdiv_init           | 10                      | 10                     | compute_dynamics.py:25                                     | ✓     |
| cmgdb.subdiv_min            | 14                      | 14                     | compute_dynamics.py:25                                     | ✓     |
| cmgdb.subdiv_max            | 28                      | 28                     | compute_dynamics.py:26                                     | ✓     |
| cmgdb.lower_bounds          | [-3, -2]                | [-3, -2]               | compute_dynamics.py:29                                     | ✓     |
| cmgdb.upper_bounds          | [3, 2]                  | [3, 2]                 | compute_dynamics.py:30                                     | ✓     |
| cmgdb.padding               | false                   | false                  | compute_dynamics.py:24                                     | ✓     |

The data, training-side hyperparameters, and CMGDB-side config values match the
reference archived scripts. A fresh run performs the checkpoint-format
conversion and regenerates the raw CMGDB artifacts.

## Verification

```bash
python pipeline.py --config configs/chafee_infante.yaml --stages metrics --max-seeds 1
# After CMGDB rerun, metrics.json should report
#   diagnose_morse_cross_check.agreement: agree
#   n_morse_sets: ~ 7  (two attractors, three saddles, two repellers)
```
