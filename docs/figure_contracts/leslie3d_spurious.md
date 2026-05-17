# fig_leslie3d_spurious

Paper Fig. 1.214 (chapter 1.2.10): the spurious-attractor case.

## Paper figures

- `paper/figures/morse_graph_new_colors.pdf` (Hasse diagram)
- `paper/figures/Leslie3D.png` (latent trajectory overlay)
- `paper/figures/latent_trajectory.png` / `latent_trajectory.PDF`

## Source of paper run

Brittany. Preserved verbatim under `code/output/Leslie_3D/spurious_attractor_ex/`.

- training script:    `archive/brittany/main_scripts/train.py`
- CMGDB script:       `archive/brittany/main_scripts/morse_graph.py`
- mg_params_log:      `archive/brittany/output/Leslie_3D/spurious_attractor_ex/mg_params_log.txt` (94 minutes; subdiv 23/23/27)
- legacy checkpoint:  `archive/brittany/output/Leslie_3D/spurious_attractor_ex/models/{encoder,dynamics,decoder}.pt`

The on-disk copy under `code/output/` was carried over to the new package; loaded through `latentdynamics.training.load_legacy_checkpoint`.

## Status

**replay-ready**. Config is `read_only: true`. Render and metrics work end-to-end from saved DOT, CSV, and the legacy 3-file checkpoint. Re-running CMGDB requires `--force-overwrite`.

## Reproduction commands

Replay (no CMGDB, ~10 s):

```bash
python pipeline.py --config configs/leslie3d_spurious.yaml --stages render,metrics
```

Diagnose (cheap, no CMGDB):

```bash
python pipeline.py --config configs/leslie3d_spurious.yaml --stages diagnose
```

Note: data/ for spurious uses Brittany's legacy `2train.csv`/`2test.csv` naming; the diagnose stage looks for `train.csv` and will fail unless the files are renamed or `data.train_files: ["2train"]` is added to the config.

Forced retrain (~1.5 h CMGDB on AMAREL; do not run locally):

```bash
python pipeline.py --config configs/leslie3d_spurious.yaml --stages all --force-overwrite
```

## Expected scientific output

`output/Leslie_3D/spurious_attractor_ex/MG/morse_graph` (DOT) has six Morse sets with Hasse:

```
{rank=same; 0 1 4};
2 -> 0
2 -> 1
3 -> 1
5 -> 3
5 -> 4
```

Leaves (attractors): 0, 1, 4. Saddles: 2, 3. Source: 5. The "spurious" attractor is node 0; semiconjugacy verification (in `metrics.json`) reports `is_spurious_attractor: true` because `tau_bar < max_semiconjugacy_error` — the attractor in the latent map does not lift to an invariant set of the high-dimensional Leslie 3D system.

## Hyperparameter audit

| param                       | archive value           | YAML value                | source line                                                | notes |
|-----------------------------|-------------------------|---------------------------|------------------------------------------------------------|-------|
| system.params.th1           | 28.9                    | 28.9                      | configs/leslie3d_spurious.yaml:5                           | ✓     |
| system.params.th2           | 29.8                    | 29.8                      |                                                            | ✓     |
| system.params.th3           | 22.0                    | 22.0                      |                                                            | ✓     |
| system.params.survival_p1   | 0.7                     | 0.7                       |                                                            | ✓     |
| system.params.survival_p2   | 0.7                     | 0.7                       |                                                            | ✓     |
| arch.num_layers             | 3                       | 3                         | configs/leslie3d_spurious.yaml:11                          | ✓     |
| arch.hidden_shape           | 32                      | 32                        |                                                            | ✓     |
| arch.high_dims              | 3                       | 3                         |                                                            | ✓     |
| arch.low_dims               | 2                       | 2                         |                                                            | ✓     |
| arch.encoder_out_activation | tanh                    | tanh (default)            | archive/brittany/src/models.py legacy default              | ✓     |
| arch.latent_out_activation  | tanh                    | tanh (default)            |                                                            | ✓     |
| arch.decoder_out_activation | sigmoid                 | sigmoid (default)         |                                                            | ✓     |
| training.loss_weights       | [10, 10, 1]             | [10, 10, 1]               | configs/leslie3d_spurious.yaml:17                          | ✓     |
| data.n_samples_train        | 4000                    | 4000                      | configs/leslie3d_spurious.yaml:21                          | ✓     |
| data.n_samples_val         | 5000                    | 5000                      |                                                            | ✓     |
| data.n_iterations           | 30                      | 30                        |                                                            | ✓     |
| cmgdb.subdiv_init           | 23                      | 23                        | configs/leslie3d_spurious.yaml:27                          | ✓     |
| cmgdb.subdiv_min            | 23                      | 23                        |                                                            | ✓     |
| cmgdb.subdiv_max            | 27                      | 27                        |                                                            | ✓     |
| cmgdb.bounds                | `[-0.6228695, -0.7421641]` -> `[0.30980384, 0.22416562]` | inferred from encoded data | mg_params_log.txt | bounds are reproducible from the legacy encoder image |

## Verification

```bash
python pipeline.py --config configs/leslie3d_spurious.yaml --stages render,metrics
diff <(grep -E '^[0-9] |->' output/Leslie_3D/spurious_attractor_ex/MG/morse_graph) \
     <(grep -E '^[0-9] |->' archive/brittany/output/Leslie_3D/spurious_attractor_ex/MG/morse_graph)
```

The diff should be empty (Hasse diagram structure is byte-identical between the on-disk copy and brittany's archive). `metrics.json` should contain `is_spurious_attractor: true`.
