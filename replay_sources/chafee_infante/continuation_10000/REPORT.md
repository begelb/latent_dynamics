# Chafee--Infante 10,000-trajectory continuation

The archived label `0` was only a finite-time nonclassification: the t=6 endpoint was not within `1e-8` of either saved stable equilibrium in the first 16 spectral modes. It was not a label for the unstable zero equilibrium.

## Outcome

All 2,138 previously unresolved trajectories met the original convergence criterion by t=660.892. They add 1,121 negative and 1,017 positive outcomes, giving completed totals of 5,030 negative and 4,970 positive trajectories.

LSODA and BDF agreed on 2,138/2,138 labels; disagreements: 0. Both solvers were run independently from each exact archived initial condition.

## Staged resolution

| Physical time | Resolved cumulatively | Still unresolved |
|---:|---:|---:|
| 6 | 0 | 2,138 |
| 8 | 190 | 1,948 |
| 10 | 351 | 1,787 |
| 12 | 468 | 1,670 |
| 16 | 653 | 1,485 |
| 24 | 904 | 1,234 |
| 40 | 1,249 | 889 |
| 64 | 1,479 | 659 |
| 100 | 1,730 | 408 |
| 150 | 1,905 | 233 |
| 200 | 2,007 | 131 |
| 300 | 2,085 | 53 |
| 500 | 2,130 | 8 |
| 800 | 2,138 | 0 |
| 1600 | 2,138 | 0 |
| 3200 | 2,138 | 0 |
| 6400 | 2,138 | 0 |

The slowest trajectory passed near an index-one saddle with a weak unstable direction; this explains its long transient without introducing a third attractor. Its saddle diagnostic is recorded in `summary.json`.

## Rescored paper headline

The encoder outputs and CMGDB regions were not rerun. Their saved 10,000 point classifications were verified against the old counts and rescored against the completed truth labels.

| Latent dimension | Old mean correct (n=7,862) | Completed mean correct (n=10,000) | Completed outside both |
|---:|---:|---:|---:|
| 1 | 52.78% | 42.82% | 56.60% |
| 2 | 71.89% | 56.93% | 42.75% |
| 3 | 76.23% | 60.11% | 39.82% |

## Suggested manuscript replacement

Starting from 10,000 initial conditions, we integrated each trajectory until its first 16 spectral coefficients entered a $10^{-8}$ neighborhood of one of the two stable equilibria. At the original cutoff $t=6$ (60 applications of the time-$0.1$ map), 7,862 trajectories met this criterion. We continued the remaining 2,138 trajectories; all subsequently met the criterion, with 1,121 converging to the negative equilibrium and 1,017 to the positive equilibrium. A diagnostic of the slowest trajectory shows it passing near a weakly unstable equilibrium. We therefore report the basin-classification percentages over all 10,000 initial conditions.

## Artifacts

- `summary.json`: provenance, solver agreement, trajectory counts, and saddle diagnostic.
- `continuation_results.npz`: accepted full 10,000 labels and per-trajectory resolution data.
- `lsoda_continuation.npz`, `bdf_continuation.npz`: independent raw solver results.
- `updated_paper_statistics.json` and `.csv`: all 45 rescored canonical runs.
- `resolution_times.png`: staged continuation plot.
