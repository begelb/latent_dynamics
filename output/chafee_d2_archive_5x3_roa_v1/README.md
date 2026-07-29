# Archived Chafee-Infante 2-D 5x3 basin audit

No model was retrained. Each row recomputes Marcio's separate uniform
level-16 strict-singleton basin analysis from one archived checkpoint.
The archived adaptive PDFs are preserved by reference but are not used
to infer regions of attraction.

| Dataset | IC seed | Trial | Correct (%) | Outside (%) | Misclassified | Morse nodes |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2158 | 1 | 82.345459 | 17.552786 | 8 | 110 |
| 1 | 2158 | 2 | 42.482829 | 57.504452 | 1 | 96 |
| 1 | 2158 | 3 | 79.776138 | 20.109387 | 9 | 81 |
| 2 | 4792 | 1 | 83.108624 | 16.891376 | 0 | 96 |
| 2 | 4792 | 2 | 83.413890 | 16.573391 | 1 | 108 |
| 2 | 4792 | 3 | 83.960824 | 16.001018 | 3 | 126 |
| 3 | 3174 | 1 | 44.721445 | 55.278555 | 0 | 150 |
| 3 | 3174 | 2 | 82.561689 | 17.412872 | 2 | 114 |
| 3 | 3174 | 3 | 81.709489 | 18.265073 | 2 | 62 |
| 4 | 688 | 1 | 63.075553 | 36.899008 | 2 | 120 |
| 4 | 688 | 2 | 62.808446 | 37.127957 | 5 | 100 |
| 4 | 688 | 3 | 63.406258 | 36.581023 | 1 | 143 |
| 5 | 5727 | 1 | 62.147036 | 37.840244 | 1 | 160 |
| 5 | 5727 | 2 | 81.505978 | 18.468583 | 2 | 112 |
| 5 | 5727 | 3 | 81.264309 | 18.621216 | 9 | 132 |

Across the 15 archived checkpoints (descriptive, not an
independence claim):

| Metric | Mean | Sample SD | Median | Range |
|---|---:|---:|---:|---:|
| Correct (%) | 71.885864 | 14.383776 | 81.264309 | 42.482829-83.960824 |
| Outside (%) | 28.075129 | 14.398760 | 18.621216 | 16.001018-57.504452 |
| Total misclassified (%) | 0.039006 | 0.040069 | 0.025439 | 0.000000-0.114475 |
| Uniform Morse nodes | 114.000 | 25.981 | 112.000 | 62-160 |

Pooled descriptive counts (the same evaluation archive is
reused 15 times, so this is not an independent-observation pool):

- Conditioned rows: 117,930; outside 33,109, misclassified-negative 16, misclassified-positive 30, correct-negative 40,124, correct-positive 44,651.
- Pooled correct: 71.885864%; pooled outside: 28.075129%; pooled total misclassified: 0.039006%.

Retrospectively, 9/15 checkpoints exceed the known archived Marcio score of 78.389723%. This is a descriptive post-hoc comparison, not a prospective test.

Initial pipeline wall time: 36.13 seconds; sum of per-run times: 32.61 seconds.

Grouped metric summaries by dataset and by unseeded training
trial are recorded in `aggregate_statistics.json`.

Each `dataset_N/trial_M/` directory contains the uniform Morse
DOT/CSV, the full strict-singleton RoA grid, root association,
trajectory labels, detailed statistics, hashes, and timings.
