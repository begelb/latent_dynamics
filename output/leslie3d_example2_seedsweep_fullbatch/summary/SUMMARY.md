# Leslie3D Example 2 - legacy full-batch 5x3 summary

**Report status:** COMPLETE AND VERIFIED

All 15 expected cells are present and pass the legacy sweep's strict artifact and design checks. In the Marcio-style visual classification, **8/15** cells have exactly two Morse nodes with nonzero degree-0 Conley index. Under the more specific Example 2 target, **4/15** cells have exactly two graph sinks and both are `(x^4-1, 0, 0)`.

## Two reported criteria

The six-page PDF uses the historical Marcio-style classification: a cell is green when exactly two Morse nodes have nonzero degree-0 Conley index. This broad classification accepts both `x^4-1` and `x-1` in degree 0 and is directly comparable with the older sweep summaries.

The exact Example 2 criterion is stricter: the graph must have exactly two sinks and each sink must have Conley index `(x^4-1, 0, 0)`. This is the primary system-specific recovery diagnostic. Tolerance is reported separately and does not alter either topology classification.

## Main findings

- Marcio-style two-attractor recovery: **8/15 (53.3%)**.
- Exact two-period-four recovery: **4/15 (26.7%)**.
- The exact successes are `dataset_1/seed_1`, `dataset_2/seed_1`, `dataset_3/seed_1`, and `dataset_4/seed_1`. Thus model seed 1 succeeds in 4/5 datasets; seeds 0 and 2 have no exact successes.
- Four additional cells pass the broad criterion but miss the exact target: `dataset_1/seed_0`, `dataset_2/seed_0`, `dataset_3/seed_0`, and `dataset_5/seed_0` each recover one `x^4-1` and one `x-1` attractor-type node.
- `dataset_4/seed_1` is the best exact success by selected validation total (`0.158580`). The lowest selected validation total overall is `dataset_1/seed_0` (`0.135136`), which is not an exact topology success; loss and topology ranking are therefore not aligned.
- No cell passes the sampled tolerance diagnostic. Nine cells have a recorded failure and six are inconclusive because at least one minimal set has no encoded samples. Across 23 minimal sets, 16 are sampled and all 16 fail, by `30.38x` to `416.18x` their reported tolerance.
- All 15 diagnoses are `ok`; none flags encoder collapse or latent-map overcontraction.

## Cells

| Dataset | Model seed | Epochs | Final train total | Best val total | Nodes/edges/sinks | Attractor-type nodes (labels: H0) | Marcio-style | Exact target | Tolerance | CMGDB min |
|---:|---:|---:|---:|---:|---|---|:---:|:---:|---|---:|
| 01 | 0 | 907 | 0.134099 | 0.135136 | 3/2/2 | 2 [0,1]: x^4-1, x-1 | yes | no | fail | 7.57 |
| 01 | 1 | 1000 | 0.200409 | 0.197875 | 5/4/2 | 2 [0,1]: x^4-1, x^4-1 | yes | **yes** | inconclusive | 7.65 |
| 01 | 2 | 838 | 0.182269 | 0.177957 | 1/0/1 | 1 [0]: x-1 | no | no | fail | 7.72 |
| 02 | 0 | 1000 | 0.241072 | 0.234844 | 3/2/2 | 2 [0,1]: x^4-1, x-1 | yes | no | inconclusive | 7.63 |
| 02 | 1 | 1000 | 0.173863 | 0.170884 | 5/4/2 | 2 [0,2]: x^4-1, x^4-1 | yes | **yes** | inconclusive | 7.68 |
| 02 | 2 | 1000 | 0.174266 | 0.176180 | 2/1/1 | 1 [0]: x-1 | no | no | fail | 8.23 |
| 03 | 0 | 1000 | 0.155441 | 0.139836 | 3/2/2 | 2 [0,1]: x-1, x^4-1 | yes | no | fail | 7.59 |
| 03 | 1 | 628 | 0.209429 | 0.206721 | 5/4/2 | 2 [0,2]: x^4-1, x^4-1 | yes | **yes** | inconclusive | 7.62 |
| 03 | 2 | 1000 | 0.189833 | 0.190806 | 3/2/1 | 1 [0]: x-1 | no | no | fail | 8.14 |
| 04 | 0 | 1000 | 0.165004 | 0.164106 | 2/1/1 | 1 [0]: x-1 | no | no | fail | 7.57 |
| 04 | 1 | 1000 | 0.156895 | 0.158580 | 5/4/2 | 2 [0,2]: x^4-1, x^4-1 | yes | **yes** | inconclusive | 7.64 |
| 04 | 2 | 1000 | 0.239897 | 0.233649 | 3/2/1 | 1 [0]: x^4-1 | no | no | inconclusive | 7.68 |
| 05 | 0 | 1000 | 0.149867 | 0.146922 | 3/2/2 | 2 [0,1]: x-1, x^4-1 | yes | no | fail | 7.61 |
| 05 | 1 | 734 | 0.197339 | 0.194711 | 4/3/1 | 1 [0]: x^4-1 | no | no | fail | 7.72 |
| 05 | 2 | 601 | 0.219081 | 0.217229 | 2/1/1 | 1 [0]: x-1 | no | no | fail | 7.82 |

## Run profile and timing

This is the older `T=20` full-batch sweep: five independent training datasets (seeds 1-5), one shared validation set (seed 9999), 8,000/2,000 train/validation initial conditions, 20 map steps, and 160,000 training pairs per dataset. The models use the legacy weighted objective, validation-based model selection, and early stopping; this is not the newer exact Marcio two-term, fixed-epoch training loop.

Ten cells reach the 1,000-epoch ceiling and five stop earlier. Mean selected validation total is `0.183029` (range `0.135136-0.234844`). Training totals 125.29 minutes, CMGDB totals 115.89 minutes, and their recorded combined time is about 4.02 hours. CMGDB uses subdivision `24/25/29`, encoded-data bounds with a 1% margin, and the legacy `auto` backend setting.

## Operationally incomplete or invalid cells

None.

## Derived artifacts

- `summary.pdf` - six-page Marcio-style visual report; green means exactly two degree-0 attractor-type nodes.
- `cells.csv` - strict flat cell inventory.
- `cells.json` - strict parsed per-cell records and provenance.
- `aggregate_summary.json` - strict aggregate analysis using the exact two-period-four sink criterion.

