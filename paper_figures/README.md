# Paper figures (staging)

Isolated, paper-ready figures collected by `scratch/assemble_paper_figures.py`.
Re-run that script to refresh after new good runs land (edit its MANIFEST).
Each example has PDF (vector, for the paper) + PNG (preview): `morse_graph`,
`morse_sets`, `morse_sets_with_overlay` (grey orbit arrows), `regions_of_attraction`.

Ground truth: every paper example has >=2 attractors; Chafee-Infante has exactly 2
(u+/u-), so a chafee run with 1 or >=3 attractor Morse sets is a failed reconstruction.

## replay_sources/ (author models -- the reference structure)

| example | attractor Morse sets | note |
|---|---|---|
| leslie3d_spurious | 3 | Author model (Brittany). Tristable: two period-4 spurious attractors + one fixed point. |
| leslie3d_success | 2 | Author model (Patrick). Two period-4 attractors. |
| leslie_contraction | 2 | Author model (Patrick). 10D->2D; period-3 + annulus. |
| chafee_infante | 2 | Author model (Marcio). Correct: 2 attractors (u+/u-). |

## retrains/ (fresh reproductions)

| example | attractor Morse sets | note |
|---|---|---|
| leslie3d_spurious | 2 | Fresh retrain seed 2. Bistable: spurious period-4 attractor + fixed point. Tristability (author=3) NOT reproducible: 0/19 retrains across 3 architectures reached 3 -> use the replay (replay_sources) for the tristable figure. |
| leslie3d_success | 2 | Fresh retrain seed 2. Bistable (period-2 + fixed point). |
| leslie_contraction | 2 | Fresh retrain seed 20, finer subdiv 27/29/30 (KEPT). Bistable: period-6 + annulus. |
| chafee_infante_dyn_heavy | 2 | Arch sweep winner (marcio arch + 5x dynamics-loss weight, seed 0). CLEANEST faithful: 8 Morse sets ~ the true 7-equilibrium structure; validated u+/u- -> 2 distinct attractors. (only 1/4 seeds correct -> less robust). |
| chafee_infante_relu_mlp | 2 | Arch sweep winner (relu MLPs [64,64]/[32,32]/[64,64], seed 3). ROBUST: 4/4 seeds give the correct 2 attractors; validated u+/u- -> 2 distinct attractors. Busier figure (~12 Morse sets). |

Chafee note: the baseline arch (`marcio_base`) always overcounts (4-5 spurious attractors). The architecture sweep (`scratch/chafee_arch_sweep.py`, results in `scratch/chafee_arch_sweep_results.csv`) found that a relu MLP gives the correct 2 attractors in all 4 seeds, and dynamics-weighted training gives the cleanest single figure. Two candidates are staged above; pick one for the paper.
