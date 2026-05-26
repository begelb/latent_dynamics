# fig_coral_basic

Paper Fig. 1.376: 13D coral population system, 1D latent, bistable extinction-vs-recovery dynamics.

## Paper figures

- `paper/figures/coral_morse_graph.pdf`
- `paper/figures/coral_morse_sets.pdf`
- `paper/figures/morse_sets_1D.pdf` (custom 1D segment plot with encoded fixed points)
- `paper/figures/morse_sets_1D_old.pdf`
- `paper/figures/1D_Coral_MG.pdf`

## Source of paper run

Brittany. Preserved partially under `code/replay_sources/coral/train_500/seed_*/`. **Note:** in the current on-disk tree the `train_500` per-seed checkpoints and Morse graph files are 0 bytes (incomplete uploads), so this figure cannot be replayed until the source tree is re-synced.

- training script:        `archive/brittany/main_scripts/train.py`
- CMGDB script:           `archive/brittany/main_scripts/morse_graph.py`
- 1D plotter:             `archive/brittany/coral_experiment_scripts/1D_morse_set_plot_for_coral.py`
- legacy config:          `archive/brittany/config/coral.yaml`
- mg_params_log per seed: `archive/brittany/output/coral/train_500/seed_*/mg_params_log.txt`
- known fixed points (`a₀`, `a₁`, `r`): `archive/brittany/coral_experiment_scripts/1D_morse_set_plot_for_coral.py:18-22`

## Status

**blocked-by-empty-checkpoints**. `configs/coral_basic.yaml` is
`read_only: true` and points at Brittany's preserved tree. Full coral
recomputation is intentionally outside the default paper replay path; use a
writable local copy of the YAML if a fresh retrain is needed.

## Reproduction commands

Replay (currently fails per-seed because `train_500` checkpoints are empty;
left in place so a future re-sync from the cluster lights it up):

```bash
python pipeline.py --config configs/coral_basic.yaml --stages render,metrics
```

Fresh retrain requires copying the YAML to a writable output location and
setting `paths.read_only: false`; do not write into `replay_sources/coral/`.

## Expected scientific output

Paper figure shows three Morse sets in 1D: two attractors at the encoded `E(a₀)` (extinction) and `E(a₁)` (healthy steady state), separated by one repeller at `E(r)` (the population separatrix).

Per-seed bounds vary (see archive `mg_params_log.txt`s); CMGDB subdivisions are `subdiv_init=8, subdiv_min=8, subdiv_max=12, subdiv_limit=10000` for every seed.

## Hyperparameter audit

| param                       | archive value           | YAML value             | source line                                | notes |
|-----------------------------|-------------------------|------------------------|--------------------------------------------|-------|
| arch.num_layers             | 3                       | 3                      | configs/coral_basic.yaml                   | ✓     |
| arch.hidden_shape           | 64                      | 64                     |                                            | ✓     |
| arch.high_dims              | 13                      | 13                     |                                            | ✓     |
| arch.low_dims               | 1                       | 1                      |                                            | ✓     |
| arch.encoder_out_activation | tanh                    | tanh (default)         | archive/brittany/src/models.py             | ✓     |
| arch.latent_out_activation  | tanh                    | tanh (default)         |                                            | ✓     |
| arch.decoder_out_activation | sigmoid                 | sigmoid (default)      |                                            | ✓     |
| training.loss_weights       | [10, 10, 1]             | [10, 10, 1]            |                                            | ✓     |
| training.epochs             | 1000                    | 1000 (default)         | archive/brittany/config/coral.yaml         | ✓     |
| training.learning_rate      | 0.001                   | 0.001 (default)        |                                            | ✓     |
| training.batch_size         | 1024                    | 1024 (default)         |                                            | ✓     |
| training.patience           | 100                     | 100 (default)          |                                            | ✓     |
| data.n_samples_train        | 500                     | [500]                  |                                            | ✓     |
| data.n_samples_val         | 10000                   | 10000                  |                                            | ✓     |
| data.n_iterations           | 20                      | 20                     |                                            | ✓     |
| data.sampling_method        | uniform                 | uniform                |                                            | ✓     |
| cmgdb.subdiv_init           | 8                       | 8                      | mg_params_log.txt per seed                 | ✓     |
| cmgdb.subdiv_min            | 8                       | 8                      |                                            | ✓     |
| cmgdb.subdiv_max            | 12                      | 12                     |                                            | ✓     |
| cmgdb.lower_bounds          | per-seed inferred       | inferred               | mg_params_log.txt                          | ✓     |
| cmgdb.upper_bounds          | per-seed inferred       | inferred               |                                            | ✓     |

## Verification

```bash
# After re-syncing non-empty train_500 checkpoints, verify the 1D Morse set
# produces three intervals bracketing E(a0), E(r), E(a1):
python pipeline.py --config configs/coral_basic.yaml --stages render,metrics --max-seeds 1
# metrics.json should contain
#   labels.a0_(Extinction): label of the leftmost Morse set
#   labels.r_(Repeller):    label of the middle Morse set
#   labels.a1_(Healthy):    label of the rightmost Morse set
# All three must be distinct.
```
