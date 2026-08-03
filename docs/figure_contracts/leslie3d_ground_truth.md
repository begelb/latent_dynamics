# fig_leslie3d_ground_truth

Paper figure `fig:3D_Leslie_ground_truth`: the direct Leslie-map graph used as
the ground-truth comparison for the two learned latent graphs.

## Paper figures

- `paper/figures/leslie3d/morse_graph_pruned.pdf`

The current PDF has SHA-256
`609679cf353739c18413cc19ddbbf8ded736ab9dcbcba9cc4d5962a4d83b9fd5`.

## Source of paper run

The six saved Morse sets come from the adaptive CMGDB run at
`output/original_leslie/ground_truth/absorbing_B_i29_m33_M36_L10000/screen/`.
Its raw Morse-set file has SHA-256
`91dce56924ebaa263d4b30b9ae4ce23656d40c71f3a5c29751af7c0e2e0b6fcd`.

The displayed order is recorded in
`output/original_leslie/ground_truth/absorbing_B_uniform_level33_recurrent_closure/manifest.json`.
Every forward closure from the saved sets was exhausted in the replicated
level-33 graph. The verified local Conley indices are recorded in
`output/original_leslie/ground_truth/absorbing_B_i29_m33_M36_L10000/saved_set_conley/summary.json`.
The graph renderer is
`scripts/plot_original_leslie3d_ground_truth_morse_graph.py`.

The broad layout follows the preserved learned-map comparison at
`replay_sources/leslie3d_example2/MG/morse_graph`: blue upper-left, purple
upper-right, orange below purple, and the two attractors along the bottom.

## Three-dimensional Morse-set display

The optional cubical display is generated directly from the same saved
`morse_sets` file by
`scripts/render_original_leslie3d_morse_sets_cubical.py`. The 1,955,948 exact
level-33 cells are too dense for a useful vector rendering, so each occupied
cell is mapped to its containing level-24 cell (a factor of eight per axis).
This produces a 10,498-cell labeled outer display cover with per-node counts
`(141, 10125, 81, 66, 84, 1)`. No level-24 cell receives more than one node
label.

This coarsening is render-only: it is not a recomputed level-24 Morse
decomposition and does not alter the verified graph. The labeled and
near-`x_1`--`x_3` views redraw the single origin cell at its exact bounds with
a stronger edge; no marker or geometric enlargement is used. Full provenance,
checksums, counts, and camera settings are recorded in
`output/original_leslie/ground_truth/absorbing_B_uniform_level33_recurrent_closure/cubical_3d_level24_display_cover/manifest.json`.

## Status

**Saved-computation, render-ready.** The level-33 calculation exactly decides
reachability among the six saved recurrent sets for the replicated
floating-point corner-sampled graph. It does not enumerate every SCC in all
`2^33` level-33 cells and is not a validated interval proof about the
continuous Leslie map.

## Reproduction commands

From `code/`, render the verified saved-set graph without recomputing its
forward closures:

```bash
python scripts/plot_original_leslie3d_ground_truth_morse_graph.py \
  --graph pruned --include-zero-index
cp output/original_leslie/ground_truth/absorbing_B_uniform_level33_recurrent_closure/paper_figure_pruned/morse_graph.pdf \
  ../paper/figures/leslie3d/morse_graph_pruned.pdf
```

The closure-analysis script can reproduce only the rendering layer from the
saved manifest with:

```bash
python scripts/analyze_original_leslie3d_uniform_level33.py --reuse-output
```

Render the three-dimensional display without recomputing CMGDB with:

```bash
python scripts/render_original_leslie3d_morse_sets_cubical.py
```

This writes paper-style, labeled, and near-`x_1`--`x_3` PDF/PNG views under
`output/original_leslie/ground_truth/absorbing_B_uniform_level33_recurrent_closure/cubical_3d_level24_display_cover/`.

## Expected scientific output

The saved-set transitive reduction has six nodes and the edges

```text
2 -> 1
3 -> 0
3 -> 1
4 -> 2
5 -> 3
5 -> 4
```

In particular, `3 -> 2` is absent. The verified local Conley indices are:

| node | Conley index |
|------|--------------|
| 0 | `(x^4-1, 0, 0, 0)` |
| 1 | `(x^4-1, 0, 0, 0)` |
| 2 | `(0, x^2+1, 0, 0)` |
| 3 | `(0, x^4-1, 0, 0)` |
| 4 | `(0, x+1, 0, 0)` |
| 5 | `(0, 0, 0, 0)` |

These indices belong to the saved recurrent sets, not their forward closures.

## Hyperparameter audit

| parameter | value | authoritative source |
|-----------|-------|----------------------|
| Leslie parameters | `(28.9, 29.8, 22.0)`; survival `(0.7, 0.7)` | uniform-run manifest |
| domain `B` | `[0,0,0]` to `[110,77,54]` | uniform-run manifest |
| adaptive run | init/min/max `29/33/36`; box limit `10000` | saved adaptive run |
| reachability grid | uniform level `33`, axis splits `(11,11,11)` | uniform-run manifest |
| box map | eight corner samples; no padding | uniform-run manifest |

## Postprocessing notes

The paper palette is preserved by Conley-index role. The only layout constraint
added to the native DOT is `{rank=same; 3 4};`, in addition to the two bottom
nodes sharing a rank. No PDF mirroring, fixed coordinates, invisible edges, or
topology-changing layout edges are used. The renderer supplies a fixed
`SOURCE_DATE_EPOCH` when the caller has not set one, so the generated PDF is
byte-reproducible under the same Graphviz/Cairo versions.

## Verification

After rendering, verify that the generated and paper PDFs are identical:

```bash
shasum -a 256 \
  output/original_leslie/ground_truth/absorbing_B_uniform_level33_recurrent_closure/paper_figure_pruned/morse_graph.pdf \
  ../paper/figures/leslie3d/morse_graph_pruned.pdf
```

Also inspect the raw DOT for exactly the six edges above. The manifest's
`uniform_reachability_3_to_2_present` and
`uniform_reduced_edge_3_to_2_present` fields must both remain `false`.
