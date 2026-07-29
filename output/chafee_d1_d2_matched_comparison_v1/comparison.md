# Matched-dataset Chafee-Infante D1 versus D2 comparison

## Interpretation boundary

The five training datasets are matched across dimensions, but D1 training seeds are not paired to D2 trial IDs. D2 trial IDs are archival labels, not recorded RNG seeds; all D1-D2 deltas are dataset- or dimension-level aggregate contrasts, never cellwise paired differences.

Each run cell below is `correct / outside / total misclassified` in percent. A dagger (`†`) marks a statistics-valid row whose topology check failed. Only rows passing both statistics and topology checks enter reported means.

## Run validity

| Dimension | Planned | Present | Statistics valid | Statistics failed | Topology valid | Topology failed | Reportable valid |
|---|---:|---:|---:|---:|---:|---:|---:|
| D1 | 15 | 15 | 15 | 0 | 15 | 0 | 15 |
| D2 | 15 | 15 | 15 | 0 | 15 | 0 | 15 |

## Per-dataset three-run cells and means

| Dataset | IC seed | D1 seeds 0/1/2: correct / outside / misclass (%) | D2 trials 1/2/3: correct / outside / misclass (%) | D1 mean correct | D2 mean correct | Delta D1-D2 |
|---:|---:|---|---|---:|---:|---:|
| 1 | 2158 | s0: 80.641 / 19.257 / 0.102; s1: 81.226 / 18.723 / 0.051; s2: 45.981 / 53.790 / 0.229 | t1: 82.345 / 17.553 / 0.102; t2: 42.483 / 57.504 / 0.013; t3: 79.776 / 20.109 / 0.114 | 69.283 | 68.201 | 1.081 |
| 2 | 4792 | s0: 57.899 / 41.961 / 0.140; s1: 58.916 / 40.918 / 0.165; s2: 54.719 / 45.243 / 0.038 | t1: 83.109 / 16.891 / 0.000; t2: 83.414 / 16.573 / 0.013; t3: 83.961 / 16.001 / 0.038 | 57.178 | 83.494 | -26.316 |
| 3 | 3174 | s0: 54.515 / 45.408 / 0.076; s1: 60.226 / 39.519 / 0.254; s2: 57.492 / 42.381 / 0.127 | t1: 44.721 / 55.279 / 0.000; t2: 82.562 / 17.413 / 0.025; t3: 81.709 / 18.265 / 0.025 | 57.411 | 69.664 | -12.253 |
| 4 | 688 | s0: 65.899 / 33.999 / 0.102; s1: 78.415 / 21.432 / 0.153; s2: 32.027 / 67.973 / 0.000 | t1: 63.076 / 36.899 / 0.025; t2: 62.808 / 37.128 / 0.064; t3: 63.406 / 36.581 / 0.013 | 58.781 | 63.097 | -4.316 |
| 5 | 5727 | s0: 3.778 / 96.197 / 0.025; s1: 38.603 / 61.358 / 0.038; s2: 21.330 / 78.491 / 0.178 | t1: 62.147 / 37.840 / 0.013; t2: 81.506 / 18.469 / 0.025; t3: 81.264 / 18.621 / 0.114 | 21.237 | 74.972 | -53.735 |

## Dimension-level statistics across valid runs

| Metric (%) | D1 mean | D1 sample SD | D1 median | D2 mean | D2 sample SD | D2 median | Delta mean D1-D2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Correct | 52.778 | 21.764 | 57.492 | 71.886 | 14.384 | 81.264 | -19.108 |
| Correct negative | 25.481 | 12.576 | 20.211 | 34.024 | 9.567 | 40.181 | -8.542 |
| Correct positive | 27.297 | 14.448 | 26.545 | 37.862 | 10.818 | 41.758 | -10.566 |
| Misclassified total | 0.112 | 0.076 | 0.102 | 0.039 | 0.040 | 0.025 | 0.073 |
| Misclassified negative | 0.054 | 0.060 | 0.038 | 0.014 | 0.021 | 0.000 | 0.041 |
| Misclassified positive | 0.058 | 0.055 | 0.064 | 0.025 | 0.027 | 0.013 | 0.032 |
| Outside both basins | 47.110 | 21.780 | 42.381 | 28.075 | 14.399 | 18.621 | 19.035 |

## Per-dataset mean decomposition

| Dataset | Dimension | Valid n | Correct negative (%) | Correct positive (%) | Misclass negative (%) | Misclass positive (%) | Outside (%) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | D1 | 3 | 34.809 | 34.474 | 0.076 | 0.051 | 30.590 |
| 1 | D2 | 3 | 40.070 | 28.131 | 0.038 | 0.038 | 31.722 |
| 2 | D1 | 3 | 24.540 | 32.638 | 0.017 | 0.098 | 42.708 |
| 2 | D2 | 3 | 41.440 | 42.055 | 0.000 | 0.017 | 16.489 |
| 3 | D1 | 3 | 22.632 | 34.779 | 0.059 | 0.093 | 42.436 |
| 3 | D2 | 3 | 34.003 | 35.661 | 0.000 | 0.017 | 30.319 |
| 4 | D1 | 3 | 31.476 | 27.304 | 0.042 | 0.042 | 41.135 |
| 4 | D2 | 3 | 20.635 | 42.462 | 0.013 | 0.021 | 36.869 |
| 5 | D1 | 3 | 13.949 | 7.288 | 0.076 | 0.004 | 78.682 |
| 5 | D2 | 3 | 33.969 | 41.003 | 0.017 | 0.034 | 24.977 |

## Pooled descriptive counts

These counts reuse the same evaluation archive across model runs and are descriptive, not independent observations.

| Dimension | Valid runs | Conditioned rows | Correct negative | Correct positive | Misclass negative | Misclass positive | Outside |
|---|---:|---:|---:|---:|---:|---:|---:|
| D1 | 15 | 117930 | 30050 | 32191 | 64 | 68 | 55557 |
| D2 | 15 | 117930 | 40124 | 44651 | 16 | 30 | 33109 |

## Failure accounting

- No statistics or topology failures.

## Dataset alignment

Overall alignment status: `verified_match`.

| Dataset | Overall | Training-data hash | IC seed |
|---:|---|---|---|
| 1 | verified_match | verified_match | verified_match |
| 2 | verified_match | verified_match | verified_match |
| 3 | verified_match | verified_match | verified_match |
| 4 | verified_match | verified_match | verified_match |
| 5 | verified_match | verified_match | verified_match |
