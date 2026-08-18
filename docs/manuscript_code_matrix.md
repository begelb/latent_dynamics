# Manuscript-to-code matrix

Scope: the current manuscript. Every computational figure panel and table of
the manuscript is listed with its generating code, configuration, required
artifacts, expected scientific invariants, and provenance status.

Status legend:

- **replay** — the exact published artifact is on disk with a byte-identical
  (checksum-verified) source; the public command re-renders from saved results.
- **fresh** — the published artifact was produced by a recorded fresh
  computation that the public command repeats (stochastic steps excepted).
- **static** — hand-made or author-provided asset with no in-repo generator.
- **open** — provenance gap: the exact published bytes have no on-disk
  source; each such row states the gap and the available regeneration path.

Runtime tiers: `sec` (< 1 min), `min` (1–15 min), `long` (> 15 min; wall-clock
from recorded run logs on an Apple M4 Pro, CPU unless noted).

## 1. Extended two-dimensional Leslie model in 10D (`sec:2D_Leslie`)

| Item | Status | Public command | Key artifacts | Invariants | Runtime |
|---|---|---|---|---|---|
| Fig `lesliecontraction_dynamics`(a) Morse graph, direct 2D reference | replay | notebook `01_leslie_2d_contraction.ipynb`; `scripts/render_original_leslie2d_full_paper_figures.py` (render from saved MG); fresh: `scripts/compute_original_leslie.py` subdiv (26,30,40), on-demand backend | `original_leslie/leslie_2d_exact_restriction_s26_30_40_on_demand/MG/` (morse_graph DOT 716 B; morse_sets CSV 154 MB) | 6 nodes: 0 (x^6−1,0,0), 1 (0,x^3+1,0), 2 (x−1,x−1,0), 3 (0,x^3−1,0), 4 trivial, 5 (0,0,x−1); minimal {0,2}; render hard-fails on index mismatch | render `sec`; recompute ~54 min |
| Fig (b) Morse sets, direct reference | replay | same render script | same MG CSV | same node set; teal trivial node at origin | render `min` (154 MB CSV) |
| Fig (c) latent Morse graph (flipped alignment) | replay | `dot -Tpdf` on `aligned_morse_graph.dot`; unaligned: `pipeline.py --config leslie_2gen_contraction_replay --stages render,metrics` | `replay_sources/leslie_2gen_contraction/` (autoencoder.pt 168 KB, MG DOT, scaler); `aligned_morse_graph.dot` | 5 nodes, 4 edges; minimal: 0 (x−1,x−1,0) invariant circle, 1 (x^6−1,0,0) period-6 orbit; transient (0,x^3+1,0), (0,x^3−1,0), (0,0,x−1); no trivial node | `sec`; original CMGDB run 56.5 min |
| Fig (d) latent Morse sets | replay | `scripts/render_leslie_2gen_contraction_morse_sets.py` | `replay_sources/leslie_2gen_contraction/MG/morse_sets` (11.5 MB) | same 5 sets; **open**: filename says `with_overlay` but published bytes contain no trajectory overlay | `sec` |
| Table `sampled_residual_tolerance` rows q=0,1 | fresh | `scripts/compute_sampled_residual_tolerance.py leslie_2gen_contraction` | model + MG + `data_pairs` train/val CSVs (44.5 MB); frozen values `artifacts/reference_results/sampled_residual_tolerance/leslie_2gen_contraction/` | q=0 (annulus, 117,331 boxes): R=6.80e−2 > τ=5.20e−5; q=1 (period-6, 9,792 boxes): R=5.31e−2 > τ=5.41e−5; ≥2^23 latent samples/node | ~40 min |

Training config: `leslie_2gen_contraction.yaml` — θ=(23.5,23.5), 10D→2D,
4×64 hidden, loss weights (100,10,20), 8,000/2,000 trajectories × 20 iterates,
CMGDB subdiv (27,29,30), padding 0.01. The shipped model is the seed-20 run
(training 291 s / 179 epochs; run manifest 2026-05-27).

## 2. Three-dimensional Leslie model (`sec:3d_leslie`)

| Item | Status | Public command | Key artifacts | Invariants | Runtime |
|---|---|---|---|---|---|
| Fig `3D_Leslie_direct`(a) reference Morse graph | replay | `scripts/plot_original_leslie3d_ground_truth_morse_graph.py --graph pruned --include-zero-index`; fresh: `scripts/screen_original_leslie3d_initial.py --domain absorbing` | `ground_truth/absorbing_B_i29_m33_M36_L10000/screen/MG/morse_sets` (158 MB), `saved_set_conley/`, level-33 closure dir | 6 nodes; edges exactly {2→1, 3→0, 3→1, 4→2, 5→3, 5→4}; two minimal nodes both (x^4−1,0,0,0); P0/P1 period-4 orbits as printed | render `sec`; recompute 5,813 s |
| Fig (b) reference Morse sets, 3D cubical no-legend | **open** | `scripts/render_original_leslie3d_morse_sets_cubical.py --no-legend` (visually identical regeneration; exact published bytes have no on-disk source) | level-24 display cover CSV (648 KB) + manifest | 10,498 display cells; per-node counts (141, 10125, 81, 66, 84, 1) | `min` |
| Fig `3D_Leslie_latent`(a) fine Morse graph (23,23,27) | replay | replay bundle PDF; recompute: `pipeline.py --config leslie3d_example1_replay` / notebook 02 | `replay_sources/leslie3d_example1/spurious_attractor_ex/` (legacy model migrated to state_dict; MG 10.4 MB, 122,346 boxes) | 6 nodes; minimal {0,1,4}; 0,1: (x^4−1,0,0); 4: (x−1,0,0); 5: (0,x^2−1,0); edges 2→0,2→1,3→1,5→3,5→4 | replay `sec`; CMGDB ~6 min |
| Fig (b) coarsened Morse graph, nodes 4,5 merged | fresh | `scripts/leslie3d_example1_coarsen_morse_graph.py` | fine MG + model; frozen result `artifacts/reference_results/leslie3d_example1/` | merged fiber = 174+123+25 connection cells = 322; Conley index (0,x+1,0); literal 297-cell union is NOT a valid index pair; quotient minimal {0,1} | ~1 min (46 s map build) |
| Fig (c) fine Morse sets + separation zoom | replay | `scripts/render_leslie3d_example1_figures.py` | fine morse_sets CSV + merged CSV | zoom category counts exactly 174/123/25/322 (render hard-fails otherwise) | `sec` |
| Fig (d) merged Morse sets | replay | same render script | connection-complete CSV (322 cells label 4) | merged region purple; quotient minimal {0,1} | `sec` |
| Fig `3D_Leslie_latent_coarse`(a) uniform (22,22,22) graph | fresh | `scripts/leslie3d_example1_uniform_grid.py --depth 22` | model + bounds; frozen `result.json` | 24 raw nodes; nontrivial 0:(x^4−1,0,0), 1:(x^2−1,0,0), 9:(0,x^4−1,0), 23:(0,x+1,0); minimal {0,1}; node 23 (291 cells) holds fixed point + period-2 orbit; box widths (8.08e−4, 7.83e−4) | 27 s + render |
| Fig (b) uniform-22 Morse sets | fresh | same + render script | nontrivial CSV (89,449-box raw cover) | 4 nontrivial regions; bistability visible | `sec` |
| Table rows 3D Leslie fine q=0,1,4 | fresh | `scripts/compute_sampled_residual_tolerance.py leslie3d_example1` | model, blocks, pair CSVs; frozen `dense_sampling.json` | R=1.07/6.97e−1/2.31e−1 > τ=4.25e−4/4.06e−4/4.62e−4; \|S\|=5.68e5/3.78e6/2.84e4 | ~1 h scale |
| Table rows 3D Leslie coarse q=0,1 | fresh | `scripts/leslie3d_example1_uniform_sampled_metrics.py --depth 22 --stage all` | uniform-22 blocks; frozen JSON + forward-closure verification | R identical to fine rows (same witnesses); τ=8.01e−4/7.92e−4 ≈ one box width; minimal components equal their forward closures (exact check) | ~3 min |

Model: author-provided checkpoint (`spurious_attractor_ex`, 2026-05-03),
trained on 3,200/800 trajectories × 20 iterates after 10 transient steps at
θ=(28.9,29.8,22.0). Training seeds are unrecorded; the replay is exact, fresh
retraining is stochastic and does not reliably reproduce the third attractor.

## 3. Red coral population model (`sec:red_coral`)

| Item | Status | Public command | Key artifacts | Invariants | Runtime |
|---|---|---|---|---|---|
| Fig `coral_latent_dynamics`(a) Morse graph | replay/**open** | notebook `03_coral.ipynb`; `pipeline.py --config coral_basic --stages render,metrics --cell-index 16` re-renders the identical DOT; the published PDF was rendered by an author with an unidentified toolchain (content verified identical) | `replay_sources/coral/train_500/seed_16/` (models migrated to state_dict, MG DOT 348 B, morse_sets 423 B, scaler) | 3 nodes: 0,1 index (x−1,0) minimal; 2 index (0,x−1); edges 2→0, 2→1; metrics: consistent=true | `sec` (original CMGDB < 1 min) |
| Fig (b) 1D Morse-set bands + fixed points | replay | `scripts/render_coral_morse_sets_1d.py` | seed_16 morse_sets + encoder + scaler | bands at node boxes on [−1.0139, 0.4264]; E(a0)∈\|π⁻¹(0)\|, E(a1)∈\|π⁻¹(1)\|, E(r)≈−0.27 outside all bands | `sec` |
| Table rows red coral q=0,1 | fresh | `scripts/compute_sampled_residual_tolerance.py coral_candidate_train500_seed16` | seed_16 bundle + `train_500.csv` (3.1 MB) + `test.csv` (62 MB residual pool) | R=5.40e−2/2.48e−1 > τ=7.79e−3/7.96e−3; \|S\|=1.94e6/7.16e5; ≥2^23 samples/node | `min` |
| Table `coral_data` | static | values asserted equal to `systems/coral.py` defaults | — | b_i, s_i, Ω=36, c1=2.94, c2=520, α=0.14 match code exactly | — |

Data: 500 Sobol' ICs × 20 iterates (scrambled, archived metadata seed 42);
`coral_basic.yaml` sampling_method says `sobol`, matching the manuscript and
the archived metadata (an earlier copy carried a stale `uniform` value).

## 4. Chafee–Infante equation (`sec:chafee_infante`)

| Item | Status | Public command | Key artifacts | Invariants | Runtime |
|---|---|---|---|---|---|
| Fig `ci_bif_diagram` | **static / not reproduced** | none in repo (matplotlib PDF; no generating script or source data survives; it is a static asset) | the manuscript PDF itself | contextual diagram: 11 equilibria at λ=28; 2 stable; no companion-code regeneration claim | — |
| Fig `ci_MRfull`(a,b) Hasse diagrams | replay | `scripts/render_chafee_theoretical_morse.py` | script only (pure theory rendering) | full: 11 nodes by unstable dimension; coarse: 3 nodes M(0−),M(0+),M(1) | `sec` |
| Fig `ci_latent_1d`(a,b) | replay | notebook `04_chafee_infante.ipynb` (d=1,2,3 tour); `scripts/chafee_latent_dimension_study.py` outputs (saved run shipped) | `latent_dimension_study/latent_1d/seed_0/` (state_dict model, MG_adaptive) | 3 nodes; minimal {0,1} both (x−1,0); node 2 (0,x−1); subdiv (7,8,11) | replay `sec`; retrain `long`, stochastic |
| Fig `ci_morse_graph_dynamics`(a,b) fine d=2 | replay | recolored author PDFs (`scripts/recolor_chafee_pdf.py`); **not recomputable byte-exactly** (author DOT/CSV never persisted) | author weights `ci_model_weights.pth` (62 KB) + recolor masters | 7 Morse sets: 2 attractors (x−1,0,0), 3 saddles (0,x−1,0), 2 repellers (0,0,x−1); subdiv (14,16,22), data bounds +10% | `sec` (recolor) |
| Fig (c,d) coarsened d=2 | fresh | `scripts/coarsen_chafee_infante.py` + `scripts/render_chafee_infante_standardized.py` | author weights + train data; quotient.json | M(1) = 1,515 recurrent + 2,702 connection = 4,217 cells; no overlap with the two attractor cells; quotient 3 nodes | `min` |
| Fig `ci_latent_3d`(a) | replay | `scripts/render_chafee_infante_3d_graph_palette.py` | `latent_3d/seed_0/MG_adaptive` (12 MB) | 11 nodes, 14 edges; minimal {0,1} (x−1,0,0,0); 2,3,7 (0,x−1,0,0); 6,8,9 (0,0,x−1,0); 10 (0,0,0,x−1); 4,5 trivial; subdiv (21,24,33) | `sec` render |
| Fig (b) d=3 cubical no-legend | **open** | same script `--no-legend` variant (visually identical regeneration; published bytes unmatched on disk) | same morse_sets CSV | same 11 sets | `min` |
| Fig `ci_attractor_basins`(a,b) | replay | `scripts/plot_chafee_coarse_morse_roa_overlay.py` | author weights + data; `morse_roa_overlay.json` manifest | exactly 2 uniform attractors on 256×256 grid; 137 uniform Morse nodes; basin cells 15,941 / 15,764 | `min` |
| Table `ci_dimension_roa_statistics` (45 runs) | replay | rescore from saved runs; per-run replay needs the 45-run bundle | `updated_paper_statistics.{csv,json}`; run dirs (d1 20 MB, d2 6 MB, d3 58 MB); truth = author 10k trajectories + continuation record | 45 runs, all with exactly 2 attracting minimal sets; means 42.82±17.89 / 56.93±11.36 / 60.11±7.77 | rescore `sec`; d2 basins 36 s/run; d1/d3 retrain `long`, stochastic |
| Table `basins_attraction` | replay | `scripts/chafee_basin_table.py` (validated: reproduces every printed value from the 450,000-row per-IC record) | `ci_completed_10k_raw_classifications_45_runs.csv` (18.9 MB) + undetermined split | column sums 100 % per d; precision d3 99.93/99.84; recall d3 61.00/59.21; FN(sol)=FP(other)+undetermined→sol | `sec` |
| Table rows Chafee d=1,2,3 | fresh | `scripts/compute_sampled_residual_tolerance.py chafee_infante_current` / `chafee_latent_dimensions --dimension {1,3}` | d1/d3 study checkpoints (sha-pinned), d2 converted author weights; train pairs CSV (98 MB) | d1: R=6.58/6.11 > τ; d2: R=3.52e−2/1.60e−2 < τ=3.95e−2/4.25e−2 (Yes); d3: R=4.31e−3/4.73e−3 < τ (Yes) | d2 ~1.8 h; d1+d3 ~13 min wall (batches parallel) |

Truth labels for the classification tables: author-provided 10,000-trajectory
integration (7,862 converged by t=6) completed by a recorded LSODA+BDF
continuation of the 2,138 stragglers (all converged by t=660.9; 5,030 negative
/ 4,970 positive; both solvers agreed on all 2,138).

## 5. Appendix parameter tables

`tab:data`, `tab:architecture`, `tab:hyperparameters`, `tab:cmgdb` are
typeset by hand; each value matches the shipped configs and run logs.
`tab:coral_data` matches `systems/coral.py` defaults exactly.

## Cross-cutting notes

- The published residual/tolerance table was produced by the protocol now in
  `latentdynamics.analysis.sampled_metrics`; the frozen per-row JSON results
  ship in `artifacts/reference_results/sampled_residual_tolerance/` and carry
  the exact published values.
- Six referenced figure files have no byte-identical on-disk source
  (`ci_bif_diagram.pdf`, `coral_morse_graph.pdf`, `morse_sets_1D.pdf`,
  `morse_graph_flipped.pdf`, and the two `_no_legend` cubical panels). Five
  of the six have verified content-identical regeneration paths
  (pixel-identical for `morse_sets_1D.pdf` and both `_no_legend` panels;
  same-DOT re-renders for the two Morse graphs); only `ci_bif_diagram.pdf`
  has none: no generating script for it survives; it is a static asset.
- Estimates R̂ and τ̂ are sampled quantities (lower estimate of the residual
  supremum, upper estimate of the tolerance), not certified bounds; no result
  in the repository is a computer-assisted proof.
