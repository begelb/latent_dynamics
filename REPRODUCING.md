# Reproducing the paper's computations

One documented entry point per computational figure/table of the manuscript.
Authority for what each command must produce (expected Morse graphs, Conley
indices, statistics) is `docs/manuscript_code_matrix.md`; per-figure
provenance details are in `docs/figure_contracts/`.

Conventions:

- Run everything from the repository root with the project environment
  active (see the installation notes in `README.md` and section 0 below).
- Saved inputs live under `replay_sources/` (fetched artifact bundles,
  read-only). All outputs go under `output/` or an explicit `--output`.
- Runtime tiers measured on an Apple M4 Pro (CPU): `sec` < 1 min,
  `min` 1–15 min, `long` > 15 min.
- `python reproduce_paper.py --list` shows the same plan programmatically;
  `python reproduce_paper.py --only <family> --tiers replay` runs a whole
  family's replay tier.

## 0. Fetch artifacts

```sh
python -c "from latentdynamics.replay import fetch_artifacts as f; \
           [f(k) for k in ('leslie_2gen_contraction','leslie3d_example1','coral','chafee_infante')]"
# reference-computation bundles (any manifest key):
python -c "from latentdynamics.replay import fetch_bundle as f; \
           [f(k) for k in ('original_leslie_2d_reference','original_leslie3d_reference')]"
```

The reference bundles are only needed to re-render the direct-computation
panels without recomputing them (~54 min / ~97 min). CMGDB
`>=1.5.0` is installed from PyPI; see `README.md`. Replay outputs of `pipeline.py` land under
`<repo>/replay/<config>/`; everything else goes under `output/`.

## 1. Extended two-dimensional Leslie model (10-D)

| What | Command | Tier |
|---|---|---|
| Direct 2-D reference, Morse graph + sets (fig. panels a,b) | `python scripts/render_original_leslie2d_full_paper_figures.py --source-dir replay_sources/original_leslie_2d_reference/MG --output-dir output/leslie2d_reference_figures` | sec–min |
| Direct 2-D reference, full recompute | `python scripts/compute_original_leslie.py --system 2d --subdiv 26 30 40 --box-map-backend on_demand --output output/original_leslie_2d` | ~54 min |
| Latent Morse graph replay (panel c) | `python pipeline.py --config leslie_2gen_contraction_replay --stages render,metrics` | sec |
| Aligned (paper-layout) Morse graph | `dot -Tpdf artifacts/reference_results/leslie_2gen_contraction/aligned_morse_graph.dot -o output/morse_graph_flipped.pdf` | sec |
| Latent Morse sets (panel d) | `python scripts/render_leslie_2gen_contraction_morse_sets.py` | sec |
| Latent CMGDB recompute | `python pipeline.py --config leslie_2gen_contraction_replay --stages morse,render,metrics` | ~1 h |
| Retrain from scratch (stochastic) | `python pipeline.py --config leslie_2gen_contraction --stages all --max-seeds 1` | long |
| Residual/tolerance table rows | `python scripts/compute_sampled_residual_tolerance.py leslie_2gen_contraction` | ~40 min |

## 2. Three-dimensional Leslie model

| What | Command | Tier |
|---|---|---|
| Reference Morse graph (direct computation) | `python scripts/plot_original_leslie3d_ground_truth_morse_graph.py --graph pruned --include-zero-index --source replay_sources/original_leslie3d_reference/absorbing_B_uniform_level33_recurrent_closure/paper_figure_pruned/morse_graph` (see `--help` for `--indices`/output flags) | sec |
| Reference Morse sets, 3-D cubical (incl. no-legend paper panel) | `python scripts/render_original_leslie3d_morse_sets_cubical.py --source <fetched display-cover CSV> --manifest <its manifest.json>` (both under `replay_sources/original_leslie3d_reference/absorbing_B_uniform_level33_recurrent_closure/cubical_3d_level24_display_cover/`) | min |
| Reference full recompute | `python scripts/screen_original_leslie3d_initial.py 29 --domain absorbing` then `compute_original_leslie3d_conley_from_saved_sets.py`, `analyze_original_leslie3d_uniform_level33.py` | ~97 min + min |
| Latent fine Morse graph replay (subdiv 23,23,27) | `python pipeline.py --config leslie3d_example1_replay --stages render,metrics` or notebook `02_leslie3d_example1.ipynb` | sec |
| Morse-graph coarsening: merge nodes 4,5 | `python scripts/leslie3d_example1_coarsen_morse_graph.py` | ~1 min |
| Coarse (22,22,24) grid recompute | `python scripts/leslie3d_example1_uniform_grid.py --depth 22` | ~45 s |
| Verify minimal components = forward closures | `python scripts/leslie3d_example1_verify_closures.py` | ~1 min |
| Paper figure panels (a–d + coarse a,b) | `python scripts/render_paper_figures.py --only leslie3d_example1` | min |
| Residual/tolerance, fine rows | `python scripts/compute_sampled_residual_tolerance.py leslie3d_example1` | ~1 h |
| Residual/tolerance, coarse rows | `python scripts/leslie3d_example1_uniform_sampled_metrics.py --depth 22 --stage all` | ~3 min |

## 3. Red coral population model

| What | Command | Tier |
|---|---|---|
| Morse graph replay + metrics (panel a) | `python pipeline.py --config coral_basic --stages render,metrics --cell-index 16 --expected-cells 30` | sec |
| Morse-set bands with fixed-point overlay (panel b) | `python scripts/render_coral_morse_sets_1d.py` | sec |
| CMGDB recompute from the shipped model | `python pipeline.py --config coral_basic --stages morse,render,metrics --cell-index 16` | min |
| Residual/tolerance rows | `python scripts/compute_sampled_residual_tolerance.py coral_candidate_train500_seed16` | min |

## 4. Chafee–Infante

| What | Command | Tier |
|---|---|---|
| Theoretical Morse representations (Hasse diagrams) | `python scripts/render_chafee_theoretical_morse.py` | sec |
| d=1 and d=3 latent models + Morse graphs (replayed from the shipped study) | figures re-render via `python scripts/render_chafee_infante_standardized.py` (d=1) and `python scripts/render_chafee_infante_3d_graph_palette.py` (d=3, incl. no-legend cubical panel) | sec–min |
| d=1/d=3 full study recompute (train + CMGDB; stochastic) | `python scripts/chafee_latent_dimension_study.py` | long |
| d=2 fine panels (recolored author computation) | `python scripts/recolor_chafee_pdf.py <reference-pdf> <dest> --mapping reference_d2` | sec |
| d=2 coarsened Morse graph + sets (panels c,d) | `python scripts/coarsen_chafee_infante.py` then `render_chafee_infante_standardized.py` | min |
| Basins + Morse/RoA overlay (fig. panels) | `python scripts/plot_chafee_coarse_morse_roa.py` | min |
| Classification statistics tables (45 runs) | `python scripts/chafee_basin_table.py` (validated to reproduce every printed value from the shipped per-IC record) | sec |
| Per-run 45-computation regeneration | d=2: `python scripts/analyze_chafee_d2_archive.py`; d=1: `python scripts/run_chafee_d1_matched_5x3.py`; d=3: `python scripts/run_chafee_d3_matched_5x3_training.py` + `run_chafee_d3_ondemand_5x3_controller.py` (training stochastic) | long |
| Residual/tolerance rows d=2 | `python scripts/compute_sampled_residual_tolerance.py chafee_infante_current` (base seed; see note below) | ~30 min |
| Residual/tolerance rows d=1, d=3 | `python scripts/compute_sampled_residual_tolerance.py chafee_latent_dimensions --dimension 1 --stage tolerance` then `--stage stored`, the `--stage fresh`/`--stage decoder` batches, and `--stage merge` (same for `--dimension 3`) | ~15 min wall |

The d=2 command above runs the **base seed only**. The shipped
`artifacts/reference_results/sampled_residual_tolerance/chafee_infante_current/dense_sampling.json`
is a nine-seed ensemble: fresh-trajectory seeds 20260727-20260731 (the first
with 1024 initial conditions, the rest with 2048) plus decoder-guided seeds
20260732-20260735, folded together with `--stage merge`. Its
`fresh_trajectory_ensemble_seeds` and `supplemental_runs` fields record the
whole set. To reproduce that number rather than a single-seed lower bound, run
the supplemental seeds with `--stage residual --seed <S> --output-suffix
seed<S>`, then `--stage merge --merge-suffixes ...`. A single seed agrees with
its published counterpart to ~1e-5 but samples one of nine point sets, so its
maximum is a weaker lower bound.

## 5. Checking results

Every published number has a frozen, checksummed record under
`artifacts/reference_results/`:

- `sampled_residual_tolerance/` — the 15 rows of the residual/tolerance
  table (per-experiment `dense_sampling.json`).
- `leslie3d_example1/` — coarsening result (merged Conley index, 322-cell
  count), coarse (22,22,24) result, coarse-row metrics, forward-closure
  verification.
- `chafee_infante/` — `updated_paper_statistics.{csv,json}` (both appendix
  tables) and the undetermined split.
- `coral/`, `leslie_2gen_contraction/` — render-input checksums and the
  aligned Morse-graph DOT.

Compare recomputed outputs against these records (Morse-node counts, Conley
indices, cell counts, statistics values) rather than against PDF bytes.

## 6. What cannot be regenerated

- `ci_bif_diagram.pdf` is a static manuscript asset. Its source
  data and generator were not preserved, so this repository does not claim to
  reproduce it. The exact bytes of a handful of other author-rendered panels
  are also unavailable, although content-equivalent regeneration paths exist
  for those panels (see `docs/manuscript_code_matrix.md`).
- The exact author checkpoints: replaying them is exact; retraining is
  stochastic (see `README.md`, tier 3).
