# Leslie3D Example 2 - T=40 5x3 Marcio-style summary

**Report status:** COMPLETE AND VERIFIED

All 15 expected cells are present and pass the strict artifact and sweep-design
checks. Under the primary Marcio-style classification, **10/15 (66.7%)** cells
have exactly two Morse nodes whose Conley index has a nonzero degree-0
component. Under the system-specific exact criterion, **1/15 (6.7%)** cells
have exactly two graph sinks and both are `(x^4-1, 0, 0)`.

## Two reported criteria

The six-page PDF uses the historical Marcio-style classification: a cell is
green when exactly two Morse nodes anywhere in the graph have a nonzero
degree-0 Conley component. Graph minimality is not required, and the actual
degree-0 polynomial may be `x-1`, `x^2-1`, `x^4-1`, `x^8-1`, or another
nonzero value. This broad criterion is the primary green/red flag in the PDF.

The exact Leslie3D Example 2 criterion is stricter: the graph must have exactly
two sinks and each sink must have Conley index `(x^4-1, 0, 0)`. This criterion
is reported separately in the PDF and machine-readable files. Sampled
semiconjugacy tolerance is an independent diagnostic and changes neither
topology flag. A minimal set with zero encoded samples is **inconclusive**, not
a pass.

**Visualization note:** the PDF's Morse-set panels are derived, summary-local
renders under `rendered_morse_sets/`. They use automatic display scaling with
minimum scale fraction `0.04`, maximum scale `30`, and a display-only minimum
box side of `0.004` (0.4%) of each plotted axis span. The source Morse boxes,
source cell artifacts, and all reported topology are unchanged.

## Main findings

- Marcio-style recovery is **10/15**. Broad successes are all three models in
  datasets 1 and 4; models 0 and 1 in dataset 2; model 0 in dataset 3; and
  model 0 in dataset 5.
- Exact two-period-four recovery is **1/15**, solely
  `dataset_5/seed_0`. Its graph has seven nodes, six edges, exactly two sinks,
  and both sink indices are `(x^4-1, 0, 0)`.
- Five cells fail the broad criterion: `dataset_2/seed_2`,
  `dataset_3/seed_2`, and `dataset_5/seed_2` have one degree-0 attractor-type
  node, while `dataset_3/seed_1` and `dataset_5/seed_1` have three. “Exactly
  two” is required, so both under- and over-resolution fail.
- Fourteen cells have exactly two graph sinks; `dataset_3/seed_1` has three.
  All 15 complete graph/topology signatures are distinct.
- The lowest selected validation total is `0.0170275` at
  `dataset_4/seed_2`; that cell passes the broad criterion but not the exact
  one. The exact-success cell's selected validation total is `0.0231912`, so
  validation-loss and exact-topology rankings are not aligned.
- Across the 31 graph-minimal sets, 22 sampled sets fail the reported
  tolerance and nine are inconclusive because they contain zero encoded
  samples. No minimal set passes, and therefore no cell passes all of its
  tolerance checks.
- All 15 diagnoses are `ok`; none flags encoder collapse or latent-map
  overcontraction.

## Cells

The attractor-type column lists **all** graph nodes with nonzero degree-0
Conley component, including nonminimal nodes. Tolerance entries are keyed by
graph-minimal label and retain mixed fail/inconclusive outcomes.

| Dataset | Model seed | Epochs | Final train total | Best val total | Nodes/edges/sinks | Degree-0 attractor nodes (label:H0) | Broad | Exact | Tolerance by minimal label | Diagnosis | CMGDB min |
|---:|---:|---:|---:|---:|---|---|:---:|:---:|---|---|---:|
| 01 | 0 | 1000 | 0.023417 | 0.025010 | 10/10/2 | 2 [0:`x^4-1`, 3:`x^2-1`] | **yes** | no | 0 fail; 3 fail | ok | 16.42 |
| 01 | 1 | 183 | 0.022121 | 0.023976 | 6/5/2 | 2 [0:`x^8-1`, 2:`x^2-1`] | **yes** | no | 0 fail; 2 fail | ok | 16.49 |
| 01 | 2 | 332 | 0.023827 | 0.026890 | 11/11/2 | 2 [0:`x^4-1`, 1:`x^2-1`] | **yes** | no | 0 fail; 1 fail | ok | 16.34 |
| 02 | 0 | 1000 | 0.018905 | 0.019962 | 4/3/2 | 2 [0:`x^2-1`, 2:`x^4-1`] | **yes** | no | 0 fail; 2 fail | ok | 19.16 |
| 02 | 1 | 268 | 0.017610 | 0.017701 | 8/8/2 | 2 [0:`x^8-1`, 2:`x^2-1`] | **yes** | no | 0 inconclusive; 2 inconclusive | ok | 18.11 |
| 02 | 2 | 460 | 0.018622 | 0.020128 | 6/5/2 | 1 [0:`x^4-1`] | no | no | 0 fail; 1 fail | ok | 20.09 |
| 03 | 0 | 182 | 0.033483 | 0.033824 | 4/3/2 | 2 [0:`x^4-1`, 1:`x-1`] | **yes** | no | 0 inconclusive; 1 fail | ok | 16.72 |
| 03 | 1 | 1000 | 0.018475 | 0.019333 | 7/7/3 | 3 [0:`x^4-1`, 1:`x^2-1`, 4:`x^4-1`] | no | no | 0 fail; 1 inconclusive; 4 fail | ok | 16.72 |
| 03 | 2 | 1000 | 0.020690 | 0.022470 | 4/3/2 | 1 [1:`x^4-1`] | no | no | 0 fail; 1 fail | ok | 17.83 |
| 04 | 0 | 207 | 0.028302 | 0.028560 | 5/4/2 | 2 [0:`x^2-1`, 3:`x^4-1`] | **yes** | no | 0 fail; 3 inconclusive | ok | 16.43 |
| 04 | 1 | 348 | 0.018232 | 0.020166 | 6/5/2 | 2 [0:`x^4-1`, 2:`x^2-1`] | **yes** | no | 0 fail; 2 inconclusive | ok | 31.31* |
| 04 | 2 | 435 | 0.016452 | 0.017027 | 8/9/2 | 2 [0:`x^4-1`, 4:`x^2-1`] | **yes** | no | 0 fail; 4 inconclusive | ok | 26.52 |
| 05 | 0 | 332 | 0.020946 | 0.023191 | 7/6/2 | 2 [0:`x^4-1`, 5:`x^4-1`] | **yes** | **yes** | 0 fail; 5 fail | ok | 17.12 |
| 05 | 1 | 261 | 0.016782 | 0.018895 | 6/5/2 | 3 [0:`x^4-1`, 1:`x^2-1`, 3:`x^4-1`] | no | no | 0 inconclusive; 1 inconclusive | ok | 17.08 |
| 05 | 2 | 358 | 0.017676 | 0.018895 | 7/6/2 | 1 [0:`x^4-1`] | no | no | 0 fail; 1 fail | ok | 18.24 |

The asterisked `dataset_4/seed_1` CMGDB duration includes an approximately
14-minute deliberate process pause while an unrelated CMGDB job finished, so
that recorded wall duration is provenance rather than a clean performance
measurement.

## Run profile, parameters, and provenance

This sweep uses five independent training datasets (seeds 1-5), three model
initializations per dataset (seeds 0-2), and one shared validation set (seed
9999). Each dataset has 8,000 training and 2,000 validation initial conditions,
`T=40`, and no discarded steps, producing 320,000 training and 80,000
validation transition pairs per dataset. The five training CSV hashes are
distinct; the validation CSV hash is identical across all five dataset trees.

Initial conditions are sampled uniformly from
`[0,220] x [0,154] x [0,108]`. The Leslie map uses
`theta=(28.9,29.8,22.0)`, survivals `(0.7,0.7)`, and density-decay coefficient
`0.1`. The encoder, latent map, and decoder have dimensions
`3-64-64-2`, `2-64-64-2`, and `2-64-64-3`, respectively. Training uses Adam
at learning rate `0.001`, batch size 1024, at most 1,000 epochs, early-stopping
patience 100, loss weights `(100,10,20)`, and gradient clipping at 1.0.

Four models reach the 1,000-epoch ceiling. Mean selected validation total is
`0.0224019`, with range `0.0170275-0.0338240`; mean final training total is
`0.0210360`. Recorded training time totals 348.05 minutes.

Every cell uses CMGDB subdivision `25/28/29`, limit 10,000, padding, and the
explicit `adaptive_precomputed` backend over the encoded-data box plus a 1%
margin. Each two-dimensional adaptive lookup has axis depth 15, shape
`32769 x 32769`, and exactly `1,073,807,361` table points, under the configured
1.2-billion-point cap. Recorded CMGDB time totals 284.57 minutes. The pause
caveat above means this aggregate should not be used as a clean timing
benchmark.

The strict analyzer verifies all 15 cell inventories, manifests, hashes,
Morse graphs, Morse-set files, metrics, diagnoses, explicit backends,
subdivision parameters, and requested dataset counts. The source datasets,
checkpoints, graphs, Morse boxes, metrics, and per-cell renders were treated as
read-only. This package and its visibility-enhanced renders are derived outputs
under `summary/`; nothing was written under `paper/`.

## Operationally incomplete or invalid cells

None.

## Derived artifacts

- `summary.pdf` - six-page Marcio-style visual report; green means exactly two
  degree-0 attractor-type nodes, and the separate exact column marks the
  two-`x^4-1` sink target.
- `rendered_morse_sets/` - summary-local, visibility-enhanced Morse-set PNGs;
  source boxes are unchanged.
- `cells.csv` - strict flat per-cell inventory with broad and exact flags.
- `cells.json` - strict parsed cell records, topology, tolerance, and
  provenance.
- `aggregate_summary.json` - strict aggregate analysis for both criteria.
- `output/pdf/leslie3d_example2_t40_5x3_summary.pdf` - identical convenience
  copy of `summary.pdf`.
