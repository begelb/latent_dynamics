# Paper figures (staging)

Isolated, paper-ready figures collected by `scratch/assemble_paper_figures.py`.
Re-run that script to refresh after new good runs land (edit its MANIFEST).
Each example has PDF (vector, for the paper) + PNG (preview): `morse_graph`,
`morse_sets`, `morse_sets_with_overlay` (grey orbit arrows), `regions_of_attraction`.

Ground truth: every paper example has >=2 attractors; Chafee-Infante has exactly 2
(u+/u-), so a chafee run with 1 or >=3 attractor Morse sets is a failed reconstruction.

## replay_sources/ (the fixed models replayed for the paper figures)

| example | attractor Morse sets | note |
|---|---|---|
| leslie3d_example1 | 3 | Author model (Brittany). Tristable: two period-4 spurious attractors + one fixed point. |
| leslie3d_example2 | 2 | Author model (Patrick). Two period-4 attractors. |
| chafee_infante | 2 | Author model (Marcio). Correct: 2 attractors (u+/u-). |
| leslie_2gen_contraction | 2 | OFFICIAL contraction model (promoted from retrain seed 20, replacing Patrick's run, archived in scratch/archive_contraction_patrick/). 10D->2D, finer subdiv 27/29/30. Bistable: period-6 orbit + annulus -- recovers the true period-six cascade orbit (main.tex:1453) that Patrick's period-3 computation did not detect. |

## retrains/ (fresh reproductions)

| example | attractor Morse sets | note |
|---|---|---|
| leslie3d_example1 | 2 | Fresh retrain seed 2. Bistable: spurious period-4 attractor + fixed point. Tristability (author=3) NOT reproducible: 0/19 retrains across 3 architectures reached 3 -> use the replay (replay_sources) for the tristable figure. |
| leslie3d_example2 | 2 | Fresh retrain seed 2. Bistable (period-2 + fixed point). |
| chafee_infante_dyn_heavy | 3 | CONSISTENT @14/20/30 -> 3 minimal attractors (truth is 2). OVERFIT: one spurious attractor. Cleanest retrain-failure illustration. |
| chafee_infante_relu_mlp | 8 | CONSISTENT @18/22/30 -> 8 minimal attractors (truth is 2). Grossly OVERFIT under-constrained latent map; extreme retrain-failure example. |

Chafee note: the baseline arch (`marcio_base`) always overcounts (4-5 spurious attractors). The architecture sweep (`scratch/chafee_arch_sweep.py`, results in `scratch/chafee_arch_sweep_results.csv`) found that a relu MLP gives the correct 2 attractors in all 4 seeds, and dynamics-weighted training gives the cleanest single figure. Two candidates are staged above; pick one for the paper.
