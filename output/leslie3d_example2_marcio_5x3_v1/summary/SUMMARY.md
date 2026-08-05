# Leslie3D Example 2 - Marcio-style 5x3 summary

**Report status:** COMPLETE AND VERIFIED

All 15 expected cells are represented. 15/15 are complete, 15/15 passed artifact verification, and 6/15 meet the bistability criterion.

## Bistability criterion

A cell passes the bistability criterion when exactly two Morse nodes have a stable Conley-index signature, identified by a nonzero degree-0 component. Higher-degree components may be nonzero; for example, both (x^4-1, 0, 0) and (x-1, x-1, 0) qualify. Graph edges/minimality and sampled tolerance metrics are diagnostics and do not affect this classification.

Graph sink/minimal status and sampled tolerance are shown separately; neither changes the index-based pass.

**Visualization note:** Morse-set panels apply a display-only minimum box side of 0.75% of each plotted axis span. The saved CMGDB boxes and all reported topology are unchanged.

## Cells

| Dataset | Model seed | Status | Final train total | Nodes/edges/sinks | Stable-index nodes (labels: H0) | Tolerance | Diagnosis | CMGDB min |
|---:|---:|---|---:|---|---|---|---|---:|
| 01 | 0 | verified_criterion_failure | 0.000953 | 4/4/1 | 1 [0]: x^4-1 | fail | ok | 17.2 |
| 01 | 1 | verified_criterion_failure | 0.00106 | 3/2/1 | 1 [0]: x^4-1 | fail | ok | 18.1 |
| 01 | 2 | verified_criterion_failure | 0.000913 | 3/2/1 | 1 [0]: x^4-1 | unknown | ok | 17.7 |
| 02 | 0 | verified_success | 0.000957 | 5/4/2 | 2 [0,1]: x^4-1,x^4-1 | fail | ok | 17.6 |
| 02 | 1 | verified_criterion_failure | 0.00101 | 3/2/1 | 1 [0]: x^4-1 | fail | ok | 17.3 |
| 02 | 2 | verified_criterion_failure | 0.000915 | 3/2/1 | 1 [0]: x^4-1 | unknown | ok | 17.8 |
| 03 | 0 | verified_criterion_failure | 0.001 | 3/2/1 | 1 [0]: x^4-1 | fail | ok | 17.6 |
| 03 | 1 | verified_success | 0.00102 | 4/3/1 | 2 [0,1]: x^4-1,x^4-1 | fail | ok | 18 |
| 03 | 2 | verified_criterion_failure | 0.000934 | 3/2/1 | 1 [0]: x^4-1 | unknown | ok | 18.8 |
| 04 | 0 | verified_criterion_failure | 0.00101 | 3/2/1 | 1 [0]: x^4-1 | fail | ok | 16.4 |
| 04 | 1 | verified_success | 0.00111 | 5/4/2 | 2 [0,1]: x^4-1,x^4-1 | fail | ok | 16.3 |
| 04 | 2 | verified_success | 0.00105 | 5/4/1 | 2 [0,1]: x^4-1,x^4-1 | unknown | ok | 18.4 |
| 05 | 0 | verified_success | 0.00143 | 5/6/2 | 2 [0,1]: x^4-1,x^4-1 | fail | ok | 16.3 |
| 05 | 1 | verified_success | 0.00118 | 4/3/2 | 2 [0,1]: x-1,x^4-1 | fail | ok | 17.6 |
| 05 | 2 | verified_criterion_failure | 0.00108 | 4/3/1 | 1 [0]: x^4-1 | unknown | ok | 17.8 |

## Operationally incomplete or invalid cells

None.

## Verified cells that fail the bistability criterion

- `dataset_01/seed_0`: stable-index nodes=1 [0]; graph sinks=[0]
- `dataset_01/seed_1`: stable-index nodes=1 [0]; graph sinks=[0]
- `dataset_01/seed_2`: stable-index nodes=1 [0]; graph sinks=[0]
- `dataset_02/seed_1`: stable-index nodes=1 [0]; graph sinks=[0]
- `dataset_02/seed_2`: stable-index nodes=1 [0]; graph sinks=[0]
- `dataset_03/seed_0`: stable-index nodes=1 [0]; graph sinks=[0]
- `dataset_03/seed_2`: stable-index nodes=1 [0]; graph sinks=[0]
- `dataset_04/seed_0`: stable-index nodes=1 [0]; graph sinks=[0]
- `dataset_05/seed_2`: stable-index nodes=1 [0]; graph sinks=[0]

## Derived artifacts

- `cells.csv` - flat cell inventory
- `cells.json` - exact parsed per-cell records
- `aggregate_summary.json` - aggregate counts and distributions
- `summary.pdf` - six-page visual report
