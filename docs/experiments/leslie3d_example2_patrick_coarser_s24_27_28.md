# Leslie3D Example2 (Patrick paper run): one-level-coarser subdivision

Date: 2026-08-03  
Status: complete (`morse`, `render`, and `metrics`)  
Scope: code-local experiment only; no file under `../paper/` was changed.

## Result in one sentence

For Patrick's canonical Leslie3D Example2 network used in the paper's second
Leslie3D section, changing CMGDB from `25/28/29` to `24/27/28` leaves the
five-node Morse poset and every Conley-index label unchanged; only the outer
Morse-set covers get geometrically coarser, so this subdivision change does
not recover the missing sixth direct-map set or its differing non-attractor
indices.

## Which Example2 this is

Here, **Leslie3D Example2 means Patrick's archived paper computation**, whose
checkpoint is
`replay_sources/leslie3d_example2/models/{encoder,dynamics,decoder}.pt` and
whose baseline graph has five nodes. It does not mean the later fresh retrain
under `output/leslie3d_example2/seed_2`.

The earlier, mistakenly targeted coarse experiment remains untouched at
`output/leslie3d_example2_coarser_s24_27_28/`; none of its artifacts or
conclusions are used here.

The corrected experiment is configured by
`src/latentdynamics/configs/leslie3d_example2_patrick_coarser_s24_27_28.yaml`
and is entirely contained under
`output/leslie3d_example2_patrick_coarser_s24_27_28/`.

## Network, scaler, bounds, and isolation

All three archived checkpoint files were copied into the corrected output
before CMGDB was run. Their source and destination SHA-256 hashes agree
byte-for-byte:

| file | bytes | SHA-256 |
|---|---:|---|
| `encoder.pt` | 21,847 | `e581773b1ab0dfdb1002ffc1542331b71398b4c7cb37e323c653f47c4fb67255` |
| `dynamics.pt` | 21,919 | `b062ae69cd855f3ff304a46a3532b45048f9628f5990032df400831821d92d60` |
| `decoder.pt` | 21,847 | `855a1eee3bfa6f57935cd58b9241725c70eca04ef8c1aadee267b04fbff0b57f` |

The run directly reused Patrick's archived scaler at
`replay_sources/leslie3d_example2/data/scalers/scaler`, SHA-256
`bb908b946d259fd6aa6a716cc003f789631e21bc7c9aa0a6a64c09ac629aa5e1`.
The source/copy audit is persisted in
`output/leslie3d_example2_patrick_coarser_s24_27_28/analysis/source_provenance.json`.

The comparison freezes Patrick's exact archived latent domain and changes
only the three subdivision depths:

| setting | Patrick baseline | corrected coarse run |
|---|---:|---:|
| `subdiv_init` | 25 | 24 |
| `subdiv_min` | 28 | 27 |
| `subdiv_max` | 29 | 28 |
| `subdiv_limit` | 10000 | 10000 |
| lower bounds | `[-0.37490714, -0.4695556]` | identical |
| upper bounds | `[0.3535685, 0.455769]` | identical |
| padding | true | true |
| requested backend | auto | auto |

Patrick's original train/validation CSVs were not archived. This does not
affect the CMGDB graph because the bounds are fixed in the config. The
trajectory render uses today's `data/leslie3d_example2` CSVs with Patrick's
archived scaler, and therefore should be treated as a diagnostic render rather
than an archived-training trajectory reproduction. The finite-sample metric
generates points directly from the configured Leslie map and uses the archived
scaler; it does not read those CSV values.

## Precomputation and resources

Yes, this run used precomputation. `box_map_backend: auto` resolves to
`adaptive_precomputed` for the nonuniform `24/27/28` ladder. At latent
dimension 2 and `subdiv_max=28`, the backend pre-evaluates the network on a
`16385 x 16385` corner lattice, or 268,468,225 corner points, in automatically
selected CPU batches. The table cap was 1.2 billion points.

The successful command ran on CPU with the explicit graph-allocation guards
`CMGDB_MAPGRAPH_MAX_VERTICES=20000000` and
`CMGDB_MAPGRAPH_MAX_EDGES=1200000000`. These are safety ceilings, not changes
to the dynamics or subdivision. CMGDB 1.3.3 completed the graph in 8.1408
minutes. The full terminal log is
`output/leslie3d_example2_patrick_coarser_s24_27_28/logs/pipeline_morse_render_metrics.log`.

## Morse graph and Conley indices

The coarse run is semantically identical to Patrick's five-node baseline:

| node | Conley index | baseline outgoing | coarse outgoing | minimal in both |
|---:|---|---|---|---|
| 0 | `(x^4-1, 0, 0)` | none | none | yes |
| 1 | `(x^4-1, 0, 0)` | none | none | yes |
| 2 | `(0, x^4-1, 0)` | `2 -> 0`, `2 -> 1` | `2 -> 0`, `2 -> 1` | no |
| 3 | `(0, x^4-1, 0)` | `3 -> 1` | `3 -> 1` | no |
| 4 | `(0, 0, x-1)` | `4 -> 3` | `4 -> 3` | no |

The raw DOT hashes differ because CMGDB serialized node 2's two edges in the
opposite line order; the edge sets, five labels, minimal nodes, and indices are
identical. A programmatic parse asserted exact equality of the node-to-index
map, edge set, and minimal-node set across the two DOT files. The current
consistency check reports two minimal attractors, no
attractor-type nonminimal set, no trivial-index node, and `consistent=true`.

## Morse-set geometry

Baseline boxes have width approximately
`(4.44626239e-5, 5.64773309e-5)`. Coarse boxes have width approximately
`(4.44626245e-5, 1.12954663e-4)`: the `z2` side doubles while `z1` remains at
the same binary-grid scale. Consequently, box counts and summed cover areas
must be interpreted together rather than treating box counts as physical
size.

| node | baseline boxes | coarse boxes | count change | baseline extent `(z1,z2)` | coarse extent `(z1,z2)` | baseline area | coarse area | area ratio |
|---:|---:|---:|---:|---|---|---:|---:|---:|
| 0 | 6,581 | 9,102 | +38.31% | `(0.39620644, 0.22596580)` | `(0.39900759, 0.22760365)` | `1.652575e-5` | `4.571262e-5` | 2.766x |
| 1 | 26,257 | 30,890 | +17.64% | `(0.12040479, 0.09657624)` | `(0.12974194, 0.10335352)` | `6.593475e-5` | `1.551376e-4` | 2.353x |
| 2 | 14,010 | 16,738 | +19.47% | `(0.20652889, 0.16203346)` | `(0.20839632, 0.16355835)` | `3.518094e-5` | `8.406260e-5` | 2.389x |
| 3 | 15,266 | 12,109 | -20.68% | `(0.03303573, 0.02479355)` | `(0.03379159, 0.02541480)` | `3.833492e-5` | `6.081456e-5` | 1.586x |
| 4 | 61 | 76 | +24.59% | `(0.00071140, 0.00033886)` | `(0.00115603, 0.00045182)` | `1.531789e-7` | `3.816918e-7` | 2.492x |

Node 4 is most sensitive in relative bounding-box extent: `z1` grows 62.5%
and `z2` grows 33.3%. Node 1 is the next-most changed geometrically, with
extent growth of 7.75% and 7.02%. Every other node's bounding extent changes
by at most about 2.51%, even though the larger cells inflate summed cover area.
All exact ranges, widths, areas, and percentage changes are saved in
`output/leslie3d_example2_patrick_coarser_s24_27_28/analysis/morse_set_comparison.csv`.

## Comparison with the direct Leslie-map ground truth

The verified direct-map graph has six nodes, minimal nodes 0 and 1, and the
transitive-reduction edges
`2 -> 1`, `3 -> 0`, `3 -> 1`, `4 -> 2`, `5 -> 3`, and `5 -> 4`. Its local
indices are:

| direct-map node | Conley index |
|---:|---|
| 0 | `(x^4-1, 0, 0, 0)` |
| 1 | `(x^4-1, 0, 0, 0)` |
| 2 | `(0, x^2+1, 0, 0)` |
| 3 | `(0, x^4-1, 0, 0)` |
| 4 | `(0, x+1, 0, 0)` |
| 5 | `(0, 0, 0, 0)` |

Role-aligning the learned colors/sets to the direct graph, the learned graph
recovers both period-4 sinks and the blue degree-one period-4 set, along with
the same five-node induced order after the direct graph's top zero-index set 5
is omitted. The remaining precise-index differences are still present: the
learned orange set has `(0, x^4-1, 0)` instead of direct-map
`(0, x^2+1, 0, 0)`, the learned purple set has `(0, 0, x-1)` instead of
`(0, x+1, 0, 0)`, and the learned graph lacks direct-map node 5 entirely.

Because the corrected coarser run exactly preserves Patrick's learned labels
and order, coarsening by this one level neither improves nor worsens that
precise Conley-index comparison. It only enlarges the outer covers.

The machine-readable comparison, including all three graphs and the resolved
precomputed backend, is saved in
`output/leslie3d_example2_patrick_coarser_s24_27_28/analysis/comparison_summary.json`.

## Finite-sample tolerance diagnostic

Patrick's baseline artifacts were evaluated in memory with the same current
metrics code, corrected Leslie parameters, and archived scaler, without
overwriting the archive. That controlled comparison is saved at
`output/leslie3d_example2_patrick_coarser_s24_27_28/analysis/baseline_metrics_current_code.json`:

| run / minimal node | boxes | `tau_bar` | generated samples in set | max residual | diagnostic |
|---|---:|---:|---:|---:|---|
| baseline / 0 | 6,581 | `4.572454e-5` | 0 | n/a | indeterminate |
| coarse / 0 | 9,102 | `7.539749e-9` | 0 | n/a | indeterminate |
| baseline / 1 | 26,257 | `4.486434e-5` | 2 | `0.0214613` | fails |
| coarse / 1 | 30,890 | `6.167412e-10` | 23 | `0.0232665` | fails |

Coarsening reduces `tau_bar` by factors of about 6,064 for node 0 and 72,744
for node 1. The populated node-1 diagnostic fails in both runs, with similar
maximum residuals; the much smaller coarse `tau_bar` gives an even weaker
tolerance margin. Node 0 remains indeterminate because none of the 4,096
generated samples fell inside that set. Patrick's archived historical
tolerance log also failed both baseline attractors, but it used a different
sampling run and is not used for the controlled numerical table above.

Loading Patrick's archived scaler emitted scikit-learn's
`InconsistentVersionWarning`: it was pickled by scikit-learn 1.7.1 and this run
used 1.8.0. The warning is preserved in the terminal log. It cannot affect the
CMGDB graph because fixed bounds bypass encoded-data bound inference and the
graph evaluates only the copied latent network. It can affect the diagnostic
trajectory and finite-sample tolerance values, so those should retain this
version caveat.

## Reproduction

Run from `code/` after copying the three archived checkpoint files into the
new output's `models/` directory:

```bash
CMGDB_MAPGRAPH_MAX_EDGES=1200000000 \
CMGDB_MAPGRAPH_MAX_VERTICES=20000000 \
../.venv/bin/python pipeline.py \
  --config leslie3d_example2_patrick_coarser_s24_27_28 \
  --stages morse,render,metrics \
  --device cpu
```

## Saved outputs

The corrected output contains nonempty raw DOT/CSV artifacts, all three
legacy neural-network files, PDF/PNG renders, metrics, a run manifest, a
pipeline summary, the CMGDB parameter log, the terminal log, and the two
machine-readable comparison files. Key entry points are:

- `output/leslie3d_example2_patrick_coarser_s24_27_28/MG/morse_graph`
- `output/leslie3d_example2_patrick_coarser_s24_27_28/MG/morse_sets`
- `output/leslie3d_example2_patrick_coarser_s24_27_28/MG/morse_graph.png`
- `output/leslie3d_example2_patrick_coarser_s24_27_28/MG/morse_sets.png`
- `output/leslie3d_example2_patrick_coarser_s24_27_28/models/`
- `output/leslie3d_example2_patrick_coarser_s24_27_28/metrics.json`
- `output/leslie3d_example2_patrick_coarser_s24_27_28/run_manifest.json`
- `output/leslie3d_example2_patrick_coarser_s24_27_28/analysis/`

The graph and Morse-set PNGs were visually inspected and are complete and
legible. The four MG PDFs and the trajectory PDF are valid one-page PDFs.
