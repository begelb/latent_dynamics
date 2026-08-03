# fig_coral_basic

Paper figure `fig:coral_latent_dynamics`: 13D coral population system, 1D latent,
bistable extinction-versus-recovery dynamics.

## Paper figures

- `paper/figures/coral_morse_graph.pdf`
- `paper/figures/morse_sets_1D.pdf` (custom 1D segment plot with encoded fixed points)

## Source of paper run

The featured paper model is specifically
`code/replay_sources/coral/train_500/seed_16/`, as recorded in the replay-tree
README and hard-coded in the archived 1D figure generator. The basic
`train_500` sweep is fresh-reproducible from a writable config copy, but a new
training run is not provenance for this selected paper asset.

- parameter log: `replay_sources/coral/train_500/seed_16/mg_params_log.txt`
- checkpoint: `replay_sources/coral/train_500/seed_16/models/{encoder,dynamics,decoder}.pt`
- raw CMGDB artifacts: `replay_sources/coral/train_500/seed_16/MG/{morse_graph,morse_sets}`
- paper-data metadata: `archive/brittany/data/coral/train_500_metadata.json`
- 1D plot source: `archive/brittany/coral_experiment_scripts/1D_morse_set_plot_for_coral.py`

## Status

**Read-only artifact replay; basic `train_500` is fresh-reproducible from a
writable config copy.** `src/latentdynamics/configs/coral_basic.yaml` points at
the preserved replay tree. Its `data.sampling_method: uniform` does not match
the archived paper dataset metadata, which records scrambled Sobol sampling,
so that YAML must not be presented as exact data-generation provenance for the
paper run.

## Reproduction commands

Replay the selected paper cell from the preserved tree:

```bash
python pipeline.py --config coral_basic --stages render,metrics \
  --cell-index 16 --expected-cells 30
```

Fresh retraining requires copying the YAML to a writable output location and
setting `paths.read_only: false`; do not write into `replay_sources/coral/`.

## Expected scientific output

The seed-16 graph has three Morse sets in 1D: two attractors at the encoded
`E(a₀)` (extinction) and `E(a₁)` (healthy steady state), separated by one
repeller at `E(r)` (the population separatrix). Its edges are `2 -> 0` and
`2 -> 1`, with indices `(x-1, 0)`, `(x-1, 0)`, and `(0, x-1)`.

For the featured seed, the exact latent interval is
`[-1.0138611376285553, 0.4264032423496246]`. CMGDB uses
`subdiv_init=8`, `subdiv_min=8`, `subdiv_max=12`, `subdiv_limit=10000`, and
padding. The archived generator first expands the encoded-data interval by
`0.01 * width`; CMGDB padding is a separate setting.

## Hyperparameter audit

| param                       | archive value           | YAML value             | source | notes |
|-----------------------------|-------------------------|------------------------|--------|-------|
| arch.num_layers             | 3                       | 3                      | replay config | ✓ |
| arch.hidden_shape           | 64                      | 64                     | replay config | ✓ |
| arch.high_dims              | 13                      | 13                     | replay config | ✓ |
| arch.low_dims               | 1                       | 1                      | replay config | ✓ |
| arch.encoder_out_activation | tanh                    | tanh                   | replay config | ✓ |
| arch.latent_out_activation  | tanh                    | tanh                   | replay config | ✓ |
| arch.decoder_out_activation | sigmoid                 | sigmoid                | replay config | ✓ |
| training.loss_weights       | [10, 10, 1]             | [10, 10, 1]            | replay config | ✓ |
| training.epochs             | 1000                    | 1000                   | replay config | ✓ |
| training.learning_rate      | 0.001                   | 0.001                  | replay config | ✓ |
| training.batch_size         | 1024                    | 1024                   | replay config | ✓ |
| training.patience           | 100                     | 100                    | replay config | ✓ |
| data.n_samples_train        | 500                     | [500]                  | archived metadata | ✓ |
| data.n_samples_val          | 10000                   | 10000                  | replay config | not independently recorded in train metadata |
| data.n_iterations           | 20                      | 20                     | archived metadata | ✓ |
| data.sampling_method        | scrambled Sobol         | uniform                | archived metadata and `make_data.py` | YAML is stale for paper data |
| data.train_seed             | 42                      | 42                     | archived `make_data.py` | Sobol scramble seed |
| data.initial-condition box  | `[0]*13` -> `[1300,1150,750,520,270,120,35,20,7,5,5,2,2]` | not encoded in YAML | archived metadata | exact paper-data domain |
| cmgdb.subdiv_init           | 8                       | 8                      | seed-16 parameter log | ✓ |
| cmgdb.subdiv_min            | 8                       | 8                      | seed-16 parameter log | ✓ |
| cmgdb.subdiv_max            | 12                      | 12                     | seed-16 parameter log | ✓ |
| cmgdb.subdiv_limit          | 10000                   | 10000                  | seed-16 parameter log | ✓ |
| cmgdb.lower_bounds          | `[-1.0138611376285553]` | inferred               | seed-16 parameter log | featured model |
| cmgdb.upper_bounds          | `[0.4264032423496246]`  | inferred               | seed-16 parameter log | featured model |
| cmgdb.bounds_epsilon_frac   | 0.01                    | 0.01                   | archived `morse_graph.py` | applied before CMGDB padding |
| cmgdb.padding               | true                    | true                   | replay config | ✓ |

## Verification

```bash
python pipeline.py --config coral_basic --stages render,metrics \
  --cell-index 16 --expected-cells 30
# metrics.json should assign distinct labels to E(a0), E(r), and E(a1).
```

All six non-adaptive sample-size directories currently contain raw graphs,
raw Morse sets, parameter logs, and checkpoint triplets for all 30 seeds.
Rendered PDFs are not present in every cell and may be regenerated from those
saved raw artifacts.
