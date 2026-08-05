# Leslie3D Example 2 - legacy minibatch 5x3 summary

**Report status:** COMPLETE AND VERIFIED

All 15 expected cells and all five datasets pass the strict artifact, provenance, and sweep-design checks with no errors or warnings.

## Topology criteria

This report keeps four related results separate:

- **Marcio-style H0 count:** exactly two Morse nodes anywhere in the graph have nonzero degree-0 Conley index. Result: **12/15 (80.0%)**.
- **Two minimal nodes:** the Morse graph has exactly two graph sinks, regardless of index. Result: **11/15 (73.3%)**.
- **Periodic bistability:** the graph has exactly two sinks, each with a pure degree-0 index `(x^p-1, 0, 0)` for some `p >= 1`. Result: **10/15 (66.7%)**. The PDF's green/red page frames use this criterion.
- **Exact Example 2 target:** the graph has exactly two sinks and both are `(x^4-1, 0, 0)`. Result: **2/15 (13.3%)**.

Sampled tolerance is a separate diagnostic and does not change any topology classification. A zero-sample check is inconclusive, never a pass.

## Main findings

- The exact target is recovered only by `dataset_2/seed_2` and `dataset_5/seed_2`. The stronger of these by selected validation total is `dataset_5/seed_2` (`0.038935`).
- The lowest selected validation total overall is `dataset_5/seed_1` (`0.035571`), but its sinks are `x^4-1` and `x^3-1`, so it is periodic-bistable but not the exact period-four target. Loss and topology ranking are not aligned.
- The broad H0 count is balanced across model seeds: each model seed succeeds in 4/5 datasets. Both exact successes happen at model seed 2.
- `dataset_4/seed_0` shows why the H0 and minimal-node criteria must remain separate: it has two H0 nodes but only one graph sink because one attractor-type node is nonminimal.
- `dataset_2/seed_1` has two H0 nodes and two sinks, but one sink is `(x-1, x-1, 0)`; its nonzero higher component makes it fail the pure periodic-index criterion.
- `dataset_1/seed_2` over-splits into four sinks and is the only tolerance-inconclusive cell.
- No cell passes sampled tolerance. Fourteen cells fail. In `dataset_1/seed_2`, two sampled minimal sets fail and two additional sets have zero samples, so the overall cell result is inconclusive.
- All 15 diagnoses are `ok`; none flags encoder collapse or latent-map overcontraction. The sweep contains 14 distinct full topology signatures.

## Cells

`min` in the topology column means graph-minimal Morse nodes (graph sinks).

| Dataset | Model seed | Epochs | Final train total | Best val total | Nodes/edges/min | H0 nodes (labels: H0) | 2 H0 | 2 min | Periodic | Exact | Tolerance | CMGDB min |
|---:|---:|---:|---:|---:|---|---|:---:|:---:|:---:|:---:|---|---:|
| 01 | 0 | 289 | 0.036381 | 0.041302 | 5/4/2 | 2 [0,2]: x-1, x^4-1 | yes | yes | yes | no | fail | 7.56 |
| 01 | 1 | 375 | 0.042737 | 0.045710 | 6/5/2 | 2 [0,1]: x^4-1, x-1 | yes | yes | yes | no | fail | 7.69 |
| 01 | 2 | 1000 | 0.038827 | 0.043625 | 7/6/4 | 4 [0,1,2,5]: x^4-1, x^4-1, x^4-1, x-1 | no | no | no | no | inconclusive | 7.70 |
| 02 | 0 | 216 | 0.051019 | 0.054302 | 3/2/2 | 2 [0,1]: x-1, x^4-1 | yes | yes | yes | no | fail | 7.53 |
| 02 | 1 | 323 | 0.038090 | 0.039422 | 5/4/2 | 2 [0,1]: x^4-1, x-1 | yes | yes | no | no | fail | 7.72 |
| 02 | 2 | 301 | 0.049129 | 0.052676 | 4/3/2 | 2 [0,2]: x^4-1, x^4-1 | yes | yes | yes | **yes** | fail | 7.60 |
| 03 | 0 | 406 | 0.032762 | 0.037565 | 2/1/1 | 1 [0]: x^4-1 | no | no | no | no | fail | 7.95 |
| 03 | 1 | 452 | 0.040194 | 0.042528 | 7/9/3 | 3 [0,1,4]: x^4-1, x^4-1, x-1 | no | no | no | no | fail | 7.59 |
| 03 | 2 | 349 | 0.034612 | 0.036679 | 3/2/2 | 2 [0,1]: x-1, x^4-1 | yes | yes | yes | no | fail | 7.62 |
| 04 | 0 | 1000 | 0.040858 | 0.044559 | 6/5/1 | 2 [0,1]: x^4-1, x^4-1 | yes | no | no | no | fail | 7.67 |
| 04 | 1 | 263 | 0.035970 | 0.041635 | 4/3/2 | 2 [0,2]: x-1, x^4-1 | yes | yes | yes | no | fail | 7.66 |
| 04 | 2 | 1000 | 0.038885 | 0.042939 | 4/3/2 | 2 [0,1]: x^4-1, x-1 | yes | yes | yes | no | fail | 7.65 |
| 05 | 0 | 210 | 0.065110 | 0.057681 | 4/3/2 | 2 [0,1]: x-1, x^4-1 | yes | yes | yes | no | fail | 7.70 |
| 05 | 1 | 380 | 0.033155 | 0.035571 | 4/3/2 | 2 [0,1]: x^4-1, x^3-1 | yes | yes | yes | no | fail | 7.60 |
| 05 | 2 | 320 | 0.035299 | 0.038935 | 4/3/2 | 2 [0,1]: x^4-1, x^4-1 | yes | yes | yes | **yes** | fail | 7.65 |

## Run profile and timing

This is the older `T=20` minibatch sweep: five independent training datasets (seeds 1-5), one shared validation set (seed 9999), 8,000/2,000 train/validation initial conditions, and 160,000/40,000 transition pairs per dataset. The data are byte-identical to the matching legacy full-batch sweep; the main optimization difference is batch size 1,024 instead of 160,000.

Training uses the legacy weighted objective, validation-based model selection, and early stopping. Three cells reach the 1,000-epoch ceiling and 12 stop earlier. Mean selected validation total is `0.043675` (range `0.035571-0.057681`), and mean final training total is `0.040869`.

Training totals 130.73 minutes, CMGDB totals 114.87 minutes, and their recorded combined time is about 4.09 hours. CMGDB uses subdivision `24/25/29`, encoded-data bounds with a 1% margin, padding, and the legacy `auto` backend setting. Exact region-of-attraction computation was disabled.

## Operationally incomplete or invalid cells

None.

## Derived artifacts

- `summary.pdf` - six-page visual report; green page frames indicate periodic bistability.
- `cells.csv` - strict flat cell inventory with all four topology flags.
- `cells.json` - strict parsed per-cell records and provenance.
- `aggregate_summary.json` - strict aggregate analysis for all topology and tolerance criteria.

