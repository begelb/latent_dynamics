# fig_leslie_2gen_contraction

Paper figure `fig:lesliecontraction_dynamics` (§5.2): the 10D Embedded Leslie
example. The first two coordinates follow the 2D Leslie/Ricker map, and the
remaining eight coordinates contract by 0.25, so the relevant invariant
dynamics live on the embedded coordinate plane.

## Paper figures

- `paper/figures/leslie_2gen_contraction/morse_graph.pdf`
- `paper/figures/leslie_2gen_contraction/morse_sets_with_overlay.pdf`

## Source of paper run

A fresh package retrain (seed 20) computed at subdivision 27/29/30. The finer
subdivision recovers the attracting period-six orbit of the period-doubling
cascade; an earlier coarser computation (subdivision 25/27/28) resolved only a
period-three orbit and is no longer used.

- system definition:  `src/latentdynamics/systems/leslie.py` (`LeslieContraction`)
- training config:    `configs/leslie_2gen_contraction.yaml` (seed 20, writable)
- replay config:      `configs/leslie_2gen_contraction_replay.yaml` (read-only)
- replay mirror:      `replay_sources/leslie_2gen_contraction/` (models, MG, scalers)

## Status

**Fully reproducible.** The system is defined in code and the run is pinned by
seed, so the data, training, and CMGDB stages all regenerate from
`configs/leslie_2gen_contraction.yaml`. The read-only replay config replays the
saved mirror under `replay_sources/leslie_2gen_contraction/`.

## Reproduction commands

```bash
# Replay the saved model (no training, no CMGDB recompute):
python pipeline.py --config configs/leslie_2gen_contraction_replay.yaml --stages render,metrics

# Fresh retrain end to end:
python pipeline.py --config configs/leslie_2gen_contraction.yaml --stages all --max-seeds 1
```

## Expected scientific output

The Morse graph has five nodes. The two minimal nodes are the system's two
attractors: an attracting invariant circle with Conley index `(x-1, x-1, 0)`,
and an attracting period-six orbit with Conley index `(x^6-1, 0, 0)`. The
remaining three nodes are transient (saddle/repeller) sets. The eight
contracting tail dimensions are projected away by the encoder, leaving 2D
Leslie-like dynamics in the latent space.

## Hyperparameter audit

| param                        | value             | source                                  | notes |
|------------------------------|-------------------|-----------------------------------------|-------|
| system.params.th1            | 23.5              | `configs/leslie_2gen_contraction.yaml`  | `LeslieContraction` default |
| system.params.th2            | 23.5              | `configs/leslie_2gen_contraction.yaml`  | `LeslieContraction` default |
| system.params.survival_p1    | 0.7               | `configs/leslie_2gen_contraction.yaml`  |       |
| system.params.contraction    | 0.25              | `configs/leslie_2gen_contraction.yaml`  | tail contraction |
| arch.high_dims               | 10                | config                                  |       |
| arch.low_dims                | 2                 | config                                  |       |
| arch hidden layers / width   | 4 / 64            | config                                  | all three networks |
| arch out-activations         | tanh/tanh/sigmoid | config                                  | encoder/latent/decoder |
| training.loss_weights        | [100, 10, 20]     | config                                  | reconstruction/prediction/semiconjugacy |
| data.n_samples_train         | 8000              | config (paper `D(20,0,10^4)`, 8000/2000 split) | |
| data.n_samples_val           | 2000              | config                                  | |
| data.n_iterations (T)        | 20                | config                                  | |
| cmgdb.subdiv_init            | 27                | config                                  | finer than the other 2D cases |
| cmgdb.subdiv_min             | 29                | config                                  | |
| cmgdb.subdiv_max             | 30                | config                                  | |

## Verification

```bash
python pipeline.py --config configs/leslie_2gen_contraction_replay.yaml --stages render,metrics
# The Morse graph should have five nodes with two minimal attractors:
# an invariant circle (x-1,x-1,0) and a period-six orbit (x^6-1,0,0).
```
