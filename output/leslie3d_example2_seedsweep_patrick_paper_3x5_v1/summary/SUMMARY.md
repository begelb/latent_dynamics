# Leslie 3D Example 2 - Patrick-style paper 3x5 replication

## Report status

**COMPLETE AND STRICTLY VERIFIED.** All 15 expected cells (5 independently sampled training datasets x 3 model seeds) contain the required training, diagnostic, Morse-graph, Morse-set, and provenance artifacts. The legacy sweep did not originally run the sampled metrics stage, so metrics were replayed read-only into a separate derived directory. All 257 source sweep files were hash-checked before and after that replay and were unchanged. The strict analyzer verified 15/15 cells, all five dataset records, and the shared validation holdout with no errors or warnings.

## Reported criteria

The classifications below are intentionally separate:

1. **Two H0-nonzero Morse nodes:** exactly two Morse nodes have a nonzero degree-0 Conley-index component. This counts every Morse node, not only graph-minimal nodes, and does not reject a node for also having a nonzero higher-degree component.
2. **Two graph-minimal nodes:** exactly two Morse-graph sinks (nodes with no outgoing edge), irrespective of index type.
3. **Requested periodic bistability pass:** exactly two graph-minimal nodes and both indices belong to `(x^p-1, 0, 0)` for an integer `p >= 1`; the two values of `p` may differ.

Sampled tolerance is reported independently and never changes any topology classification. A minimal set with zero semiconjugacy samples makes the cell **inconclusive**, not a tolerance pass. The retained legacy fixed-`x^4` field in the machine-readable files is compatibility-only and is not the requested periodic-bistability criterion.

## Main findings

- **5/15 (33.3%)** cells have exactly two H0-nonzero Morse nodes.
- **10/15 (66.7%)** cells have exactly two graph-minimal nodes.
- **8/15 (53.3%)** meet the requested variable-`p` periodic-bistability criterion: `d1/s1`, `d1/s2`, `d2/s2`, `d3/s1`, `d3/s2`, `d4/s2`, `d5/s0`, and `d5/s2`.
- The sampled tolerance result is **0 pass, 9 fail, and 6 inconclusive**. Each inconclusive cell had at least one minimal Morse set with zero samples.
- The compatibility-only requirement of two `(x^4-1, 0, 0)` sinks holds in **1/15** cells.
- Morse graphs range from **4 to 9 nodes**. Sink counts are 1 in one cell, 2 in ten cells, and 3 in four cells.

## Cells

Loss columns are the final weighted training total and the validation total at the selected best epoch. `N/E/M` means Morse nodes / edges / graph-minimal nodes. H0 entries are `label:index`. CMGDB time excludes training and rendering.

| Dataset | Model seed | Epochs | Final train total | Best val total | N/E/M | H0 nodes (label:index) | 2 H0 | 2 minimal | Periodic pass | Tolerance | Diagnosis | CMGDB min |
|---:|---:|---:|---:|---:|:---:|:---|:---:|:---:|:---:|:---:|:---:|---:|
| 1 | 0 | 440 | 0.037119 | 0.039979 | 6/5/3 | 0:x^4-1, 1:x-1, 2:x^4-1 | fail | fail | fail | fail | ok | 1.7491 |
| 1 | 1 | 283 | 0.041713 | 0.048029 | 7/6/2 | 0:x^4-1, 1:x^4-1 | PASS | PASS | PASS | inconclusive | ok | 1.7501 |
| 1 | 2 | 393 | 0.031952 | 0.037333 | 4/3/2 | 0:x-1, 2:x^4-1 | PASS | PASS | PASS | inconclusive | ok | 1.7323 |
| 2 | 0 | 252 | 0.057634 | 0.065059 | 6/5/2 | 0:x-1, 2:x^4-1 | PASS | PASS | fail | fail | ok | 1.7748 |
| 2 | 1 | 331 | 0.034841 | 0.037364 | 8/7/2 | 1:x-1, 3:x^4-1, 6:x^4-1 | fail | PASS | fail | inconclusive | ok | 1.7017 |
| 2 | 2 | 311 | 0.055296 | 0.061297 | 7/7/2 | 0:x^4-1, 1:x^2-1, 3:x^4-1 | fail | PASS | PASS | fail | ok | 1.7019 |
| 3 | 0 | 366 | 0.038554 | 0.040598 | 4/3/1 | 0:x^4-1 | fail | fail | fail | fail | ok | 4.2588 |
| 3 | 1 | 296 | 0.042256 | 0.043270 | 7/6/2 | 0:x^4-1, 1:x-1, 2:x^4-1, 4:x^4-1 | fail | PASS | PASS | fail | ok | 1.6973 |
| 3 | 2 | 266 | 0.038838 | 0.042132 | 9/9/2 | 0:x-1, 2:x^4-1, 6:x^8-1 | fail | PASS | PASS | inconclusive | ok | 1.7153 |
| 4 | 0 | 392 | 0.040007 | 0.044703 | 9/9/3 | 0:x^4-1, 3:x^4-1, 7:x-1 | fail | fail | fail | inconclusive | ok | 3.1361 |
| 4 | 1 | 369 | 0.038313 | 0.042157 | 6/6/3 | 0:x^4-1, 1:x^4-1, 3:x-1 | fail | fail | fail | fail | ok | 1.7983 |
| 4 | 2 | 345 | 0.035968 | 0.038807 | 5/4/2 | 0:x-1, 1:x^4-1, 2:x^4-1 | fail | PASS | PASS | inconclusive | ok | 1.7338 |
| 5 | 0 | 473 | 0.042244 | 0.046580 | 6/5/2 | 0:x^2-1, 3:x^4-1 | PASS | PASS | PASS | fail | ok | 2.1645 |
| 5 | 1 | 427 | 0.034955 | 0.038033 | 7/6/3 | 0:x-1, 1:x^3-1, 3:x^4-1, 5:x^4-1 | fail | fail | fail | fail | ok | 1.6714 |
| 5 | 2 | 1000 | 0.035500 | 0.038670 | 5/5/2 | 0:x^4-1, 1:x-1 | PASS | PASS | PASS | fail | ok | 1.6642 |

## Run profile and timing

- Data: `T=20`, `T0=0`; 8,000 training initial conditions (160,000 transition pairs) and 2,000 validation initial conditions (40,000 pairs) per dataset.
- Dataset design: training seeds 1-5 with five distinct training CSV hashes; one shared validation holdout with seed 9999 and one shared hash.
- Network: encoder `3 -> 64 -> 64 -> 2`, latent map `2 -> 64 -> 64 -> 2`, decoder `2 -> 64 -> 64 -> 3`.
- Legacy training: Adam at `1e-3`, batch 1024, maximum 1,000 epochs, early-stop patience 100, weighted losses `[100, 10, 20]`, gradient clipping 1.0. Cells ran 252-1,000 epochs (mean 396.3).
- CMGDB: subdivisions `25/28/29` (init/min/max), `adaptive_precomputed`, bounds from encoded training pairs plus a 1% margin, precomputed at `init=25`, and no regions of attraction. The precomputed alternating-axis lattice is `8193 x 4097` = 33,566,721 points; later refinements are evaluated in batches on demand.
- Training time: 7,290.95 seconds total (121.52 minutes), mean 486.06 seconds per cell.
- CMGDB time: 1,814.98 seconds total (30.25 minutes), mean 121.00 seconds (2.02 minutes) per cell; range 99.85-255.53 seconds.

## Operationally incomplete or invalid cells

None. All 15 cells are complete and strictly verified, all diagnostics are `ok`, and there are no analyzer errors or warnings.

## Derived artifacts

- `summary.pdf`: six-page landscape PDF; overview plus one graph/set page per dataset.
- `cells.csv`: one flattened row per cell, including losses, topology, both requested topology flags, periodic-bistability pass, tolerance, diagnosis, CMGDB timing, and hashes.
- `cells.json`: detailed per-cell records and source provenance.
- `aggregate_summary.json`: sweep-level counts, rates, timing, dataset-design checks, tolerance statuses, and topology distributions.

All files in this directory are derived reports. Training data, checkpoints, and CMGDB run artifacts were treated as read-only.
