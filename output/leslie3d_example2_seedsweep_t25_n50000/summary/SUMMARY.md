# Leslie3D Example 2 - T=25, N=50,000 Marcio-style 5x3 summary

**Report status:** COMPLETE AND VERIFIED

All 15 expected cells are represented, complete, and strictly verified. Under
the historical Marcio-style classification, **7/15 (46.7%)** have exactly two
Morse nodes with a nonzero degree-0 Conley component. Under the system-specific
target, **1/15 (6.7%)** has exactly two graph sinks and both sink indices are
`(x^4-1, 0, 0)`.

## Two reported criteria

The Marcio-style criterion counts Morse nodes anywhere in the graph, not only
sinks. A node counts once when the degree-0 component of its full Conley index
is nonzero. Higher-degree components do not disqualify it: for example,
`(x-1, x-1, 0)` is one stable-index node, not two. A cell passes exactly when
the entire Morse graph contains two such nodes.

The exact Example 2 criterion is stricter: the graph must have exactly two
sinks and each full sink index must be `(x^4-1, 0, 0)`. This is the precise
period-four sink target. Tolerance and diagnosis are independent numerical
diagnostics and do not change either topology flag.

The sweep-level runner's earlier `10/15` label used a different rule: exactly
two periodic-attractor sinks, allowing different periods. It is not the
Marcio-style count. In particular, `dataset_01/seed_0`, `dataset_04/seed_1`,
and `dataset_05/seed_0` each have a third nonminimal degree-0 node, so they fail
the Marcio-style criterion even though their two sinks are periodic.

## Main findings

- Marcio-style two-stable-index-node recovery: **7/15 (46.7%)**.
- Exact two-period-four sink recovery: **1/15 (6.7%)**.
- The broad successes are `dataset_02/seed_0`, `dataset_02/seed_1`,
  `dataset_02/seed_2`, `dataset_03/seed_0`, `dataset_03/seed_1`,
  `dataset_04/seed_0`, and `dataset_05/seed_1`.
- The sole exact success is `dataset_02/seed_2`. Its two sinks both have
  `(x^4-1, 0, 0)`, but both have zero semiconjugacy samples, so the tolerance
  diagnostic is inconclusive rather than passed.
- The 15 runs realize 14 distinct topology signatures. Eleven graphs have two
  sinks and four have three; graph sizes range from 4 to 11 Morse nodes.
- The lowest selected validation loss is `0.018010` at
  `dataset_04/seed_1`, which fails both topology criteria. The exact success has
  selected validation loss `0.024973`; loss ranking and topology recovery are
  not aligned.
- All 15 diagnoses are `ok`; none flags encoder collapse or latent-map
  overcontraction.

## Cells

`Train total` is the final training total. `Best val total` is the validation
total at the selected best epoch. The degree-0 column lists every qualifying
node as `label:(full Conley index)`, including nonminimal nodes. A tolerance
entry containing `zero-sample` is explicitly inconclusive for that minimal
set.

| Dataset | Model seed | Epochs | Train total | Best val total | Nodes/edges/sinks | Degree-0 nodes (label: full index) | Marcio-style | Exact target | Tolerance | Diagnosis | CMGDB min |
|---:|---:|---:|---:|---:|---|---|:---:|:---:|---|---|---:|
| 01 | 0 | 124 | 0.062933 | 0.051639 | 8/7/2 | 3 - 0:`(x^4-1,0,0)`; 1:`(x^2-1,0,0)`; 5:`(x^4-1,0,0)` | no | no | fail + 1 zero-sample | ok | 16.59 |
| 01 | 1 | 151 | 0.025168 | 0.025310 | 6/6/3 | 3 - 0:`(x^4-1,0,0)`; 1:`(x^4-1,0,0)`; 3:`(x-1,0,0)` | no | no | fail + 1 zero-sample | ok | 16.37 |
| 01 | 2 | 227 | 0.021987 | 0.022062 | 6/6/3 | 3 - 0:`(x^4-1,0,0)`; 1:`(x^4-1,0,0)`; 3:`(x-1,0,0)` | no | no | fail + 1 zero-sample | ok | 3.81 |
| 02 | 0 | 161 | 0.038394 | 0.036382 | 5/4/2 | 2 - 0:`(x^4-1,0,0)`; 1:`(x-1,0,0)` | **yes** | no | fail + 1 zero-sample | ok | 4.16 |
| 02 | 1 | 222 | 0.019730 | 0.020336 | 8/7/2 | 2 - 0:`(x^2-1,0,0)`; 5:`(x^4-1,0,0)` | **yes** | no | fail | ok | 7.45 |
| 02 | 2 | 198 | 0.024645 | 0.024973 | 5/4/2 | 2 - 0:`(x^4-1,0,0)`; 1:`(x^4-1,0,0)` | **yes** | **yes** | inconclusive - 2 zero-sample | ok | 3.40 |
| 03 | 0 | 117 | 0.050446 | 0.047690 | 4/3/2 | 2 - 0:`(x^2-1,0,0)`; 2:`(x^4-1,0,0)` | **yes** | no | inconclusive - 2 zero-sample | ok | 3.28 |
| 03 | 1 | 201 | 0.021728 | 0.021764 | 10/9/2 | 2 - 0:`(x^4-1,0,0)`; 2:`(x-1,0,0)` | **yes** | no | fail + 1 zero-sample | ok | 3.46 |
| 03 | 2 | 142 | 0.055838 | 0.036668 | 7/6/3 | 3 - 0:`(x^4-1,0,0)`; 1:`(x-1,0,0)`; 2:`(x^2-1,0,0)` | no | no | fail + 1 zero-sample | ok | 3.19 |
| 04 | 0 | 140 | 0.029322 | 0.029728 | 5/4/2 | 2 - 0:`(x^2-1,0,0)`; 2:`(x^4-1,0,0)` | **yes** | no | fail | ok | 3.16 |
| 04 | 1 | 272 | 0.017105 | 0.018010 | 11/10/2 | 3 - 0:`(x^8-1,0,0)`; 2:`(x^2-1,0,0)`; 6:`(x^4-1,0,0)` | no | no | fail + 1 zero-sample | ok | 4.87 |
| 04 | 2 | 229 | 0.020314 | 0.020496 | 10/8/3 | 3 - 1:`(x^4-1,0,0)`; 2:`(x^2-1,0,0)`; 6:`(x^4-1,0,0)` | no | no | fail + 1 zero-sample | ok | 3.02 |
| 05 | 0 | 141 | 0.060188 | 0.038751 | 7/6/2 | 3 - 0:`(x^8-1,0,0)`; 2:`(x^2-1,0,0)`; 5:`(x^4-1,0,0)` | no | no | fail | ok | 2.96 |
| 05 | 1 | 198 | 0.023394 | 0.023509 | 5/4/2 | 2 - 0:`(x^2-1,0,0)`; 2:`(x^4-1,0,0)` | **yes** | no | fail + 1 zero-sample | ok | 3.05 |
| 05 | 2 | 160 | 0.044024 | 0.031412 | 7/6/2 | 3 - 0:`(x-1,x-1,0)`; 2:`(x^4-1,0,0)`; 4:`(x^8-1,0,0)` | no | no | fail + 1 zero-sample | ok | 2.99 |

## Tolerance diagnostic

The tolerance diagnostic is separate from both topology criteria. Across the
34 minimal sets, 20 have semiconjugacy samples and all 20 fail their reported
tolerance; the remaining 14 have zero samples and are inconclusive. Only three
cells have every minimal set sampled, and all three fail. There are no
tolerance passes. The exact topology success, `dataset_02/seed_2`, has zero
samples on both minimal sets and therefore has no independent numerical
tolerance confirmation.

## Design and provenance

- Leslie map:
  `((28.9*x1 + 29.8*x2 + 22.0*x3)*exp(-0.1*(x1+x2+x3)), 0.7*x1, 0.7*x2)`.
- Five independent training datasets use seeds 1-5. Each has 40,000 initial
  conditions sampled uniformly on `[0,220] x [0,154] x [0,108]`, 25 retained
  map steps, and 1,000,000 transition pairs.
- All datasets share validation seed 9999: 10,000 initial conditions and
  250,000 transition pairs. The five training CSV hashes are distinct; all
  validation copies have SHA-256
  `f55553a78bffa5edec2200dcd676b2efb81e2a443b6f4d072795a3976caefbd5`.
- Three model seeds (0, 1, 2) are trained per dataset. The latent dimension is
  2; encoder, latent map, and decoder each use two width-64 hidden layers.
  Adam uses learning rate 0.001 and batch size 1024; loss weights are
  `(100,10,20)`. All 15 checkpoint hashes are distinct.
- CMGDB uses subdivision `25/28/29`, limit 10,000, encoded-data bounds with a
  1% margin, and explicit `adaptive_precomputed` evaluation. All 15 cells use
  the same 32,769 x 32,769 corner lattice (1,073,807,361 points), and manifest
  and log backend records agree.
- All 15 CMGDB logs record `compute_roa: False`; no regions-of-attraction stage
  or artifact is part of this package.
- Strict analysis is non-provisional and read-only with respect to source
  cells. It reports 15/15 verified cells, five verified dataset trees, and no
  errors or warnings. Analyzer SHA-256:
  `66a2f238772ae949a8bf8989fb39eca2af248eeba1492afad75770d47af44113`.

Mean final training total is `0.034348`. Mean selected validation total is
`0.029915`. Recorded training time totals 20,693.77 seconds (5.75 hours), and
CMGDB time totals 4,906.692 seconds (1.36 hours), for 7.11 recorded combined
hours.

## Visualization note

The six-page PDF uses the exact Marcio-style layout and classification. Its
Morse-set panels are summary-local derived renders under
`rendered_morse_sets/`; the source `dataset_*/seed_*/MG/morse_sets` files are
not modified. `box_scale=auto` inflates only sets below the 4% visibility
floor, with a maximum factor of 30, and `min_box_side_frac=0.004` applies a
display-only 0.4% side-length floor. These settings change visibility only;
all source boxes, labels, graph topology, and reported indices remain intact.

## Operationally incomplete or invalid cells

None.

## Derived artifacts

- `cells.csv` - strict flat inventory for all 15 cells.
- `cells.json` - exact parsed records, topology, losses, tolerance, diagnosis,
  hashes, and provenance.
- `aggregate_summary.json` - strict exact-target aggregation and design checks.
- `summary.pdf` - six-page Marcio-style graph and visible-Morse-set report.
- `rendered_morse_sets/` - 15 summary-local display-only PNG renders.
- `../../pdf/leslie3d_example2_t25_n50000_5x3_summary.pdf` - delivery copy of
  `summary.pdf`.
