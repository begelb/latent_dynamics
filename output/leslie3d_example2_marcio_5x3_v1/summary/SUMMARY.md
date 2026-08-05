# Leslie3D Example 2 - Marcio-style 5x3 summary

**Report status:** COMPLETE AND VERIFIED

All 15 expected cells are represented. 15/15 are complete, and 15/15 passed strict artifact and provenance verification.

## Topology criteria

This report keeps four related results separate:

- **Marcio-style H0 count:** exactly two H0-type Morse nodes. Result: **6/15 (40.0%)**.
- **Two minimal nodes:** exactly two graph sinks. Result: **4/15 (26.7%)**.
- **Periodic bistability:** two pure `(x^p-1, 0, 0)` sinks. Result: **3/15 (20.0%)**.
- **Exact Example 2 target:** two `(x^4-1, 0, 0)` sinks. Result: **3/15 (20.0%)**.

The PDF's green/red dataset frames use periodic bistability. Sampled tolerance is separate from topology: a zero-sample result is inconclusive, while a known failure still makes the tolerance status fail.

## Main findings

- Exact-target cells: `dataset_02/seed_0`, `dataset_04/seed_1`, `dataset_05/seed_0`.
- H0 count and graph minimality diverge at `dataset_03/seed_1`, `dataset_04/seed_2`: each has two H0-type nodes but only one sink.
- Two-sink cells failing the pure periodic-index criterion because a sink has nonzero higher homology: `dataset_05/seed_1` ((x-1, x-1, 0), (x^4-1, 0, 0)).
- Every periodic-bistable cell is also an exact period-four success in this sweep.
- Loss and topology ranking are not aligned. The lowest fixed-holdout total is `0.000932498` at `dataset_02/seed_0`; the highest is `0.00127673` at `dataset_05/seed_0`.
- The lowest final-epoch train total belongs to `dataset_01/seed_2`, which has 1 graph sink.
- Broad H0 successes by model seed: seed 0=2/5, seed 1=3/5, seed 2=1/5. Exact successes: seed 0=2/5, seed 1=1/5, seed 2=0/5.
- Sampled tolerance: 0 pass, 10 fail, and 5 inconclusive cells.
- Every cell at model seed 2 is tolerance-inconclusive because its minimal-set checks have zero semiconjugacy samples.
- Diagnosis counts: ok=15; the sweep contains 9 distinct full topology signatures.

## Cells

`min` means graph-minimal Morse nodes (graph sinks).

| Dataset | Model seed | Epochs | Final-epoch train total | Post-run holdout total | Nodes/edges/min | H0 nodes (labels: H0) | 2 H0 | 2 min | Periodic | Exact | Tolerance | CMGDB min |
|---:|---:|---:|---:|---:|---|---|:---:|:---:|:---:|:---:|---|---:|
| 01 | 0 | 4000 | 0.000953405 | 0.00101011 | 4/4/1 | 1 [0]: x^4-1 | no | no | no | no | fail | 17.2 |
| 01 | 1 | 4000 | 0.00106108 | 0.00105572 | 3/2/1 | 1 [0]: x^4-1 | no | no | no | no | fail | 18.1 |
| 01 | 2 | 4000 | 0.000913063 | 0.00100213 | 3/2/1 | 1 [0]: x^4-1 | no | no | no | no | inconclusive | 17.7 |
| 02 | 0 | 4000 | 0.000957058 | 0.000932498 | 5/4/2 | 2 [0,1]: x^4-1, x^4-1 | yes | yes | yes | yes | fail | 17.6 |
| 02 | 1 | 4000 | 0.00100941 | 0.00100464 | 3/2/1 | 1 [0]: x^4-1 | no | no | no | no | fail | 17.3 |
| 02 | 2 | 4000 | 0.00091536 | 0.000941866 | 3/2/1 | 1 [0]: x^4-1 | no | no | no | no | inconclusive | 17.8 |
| 03 | 0 | 4000 | 0.00100047 | 0.0010696 | 3/2/1 | 1 [0]: x^4-1 | no | no | no | no | fail | 17.6 |
| 03 | 1 | 4000 | 0.00102302 | 0.00100005 | 4/3/1 | 2 [0,1]: x^4-1, x^4-1 | yes | no | no | no | fail | 18 |
| 03 | 2 | 4000 | 0.00093354 | 0.00098425 | 3/2/1 | 1 [0]: x^4-1 | no | no | no | no | inconclusive | 18.8 |
| 04 | 0 | 4000 | 0.00101186 | 0.00105831 | 3/2/1 | 1 [0]: x^4-1 | no | no | no | no | fail | 16.4 |
| 04 | 1 | 4000 | 0.00110985 | 0.0011162 | 5/4/2 | 2 [0,1]: x^4-1, x^4-1 | yes | yes | yes | yes | fail | 16.3 |
| 04 | 2 | 4000 | 0.00105053 | 0.00106048 | 5/4/1 | 2 [0,1]: x^4-1, x^4-1 | yes | no | no | no | inconclusive | 18.4 |
| 05 | 0 | 4000 | 0.00142933 | 0.00127673 | 5/6/2 | 2 [0,1]: x^4-1, x^4-1 | yes | yes | yes | yes | fail | 16.3 |
| 05 | 1 | 4000 | 0.00117932 | 0.00115838 | 4/3/2 | 2 [0,1]: x-1, x^4-1 | yes | yes | no | no | fail | 17.6 |
| 05 | 2 | 4000 | 0.00108417 | 0.00102856 | 4/3/1 | 1 [0]: x^4-1 | no | no | no | no | inconclusive | 17.8 |

## Run profile and timing

The five training-data seeds are 2158, 4792, 3174, 688, 5727. Each training dataset contains 1000 initial conditions followed for 30 iterations, producing 30,000 transition pairs. The shared fixed holdout uses seed 9999 and 200 initial conditions, producing 6000 transition pairs.

The model maps 3 state dimensions to a 2-dimensional latent space, with encoder widths `[64, 32]`, latent-map widths `[32, 32]`, and decoder widths `[32, 64]`. Training is full batch for a fixed 4,000 epochs with Adam at learning rate `0.003` and objective `MSE(D(E(x)), x) + MSE(D(G(E(x))), y)`. There is no early stopping or validation-based checkpoint selection; checkpoints are the final epoch and the holdout is used only for post-training evaluation.

Mean final-epoch training total is `0.0010421` (range `0.000913063-0.00142933`); mean fixed-holdout total is `0.00104663` (range `0.000932498-0.00127673`).

Training timing was not recorded in the source artifacts. CMGDB totals 262.92 minutes (4.38 hours). Mean CMGDB time is 17.5 minutes (range 16.3-18.8). CMGDB uses subdivision `25/28/29`, the `adaptive_precomputed` backend, encoded-training-pair bounds plus 1%, padding=true, and exact region-of-attraction computation disabled.

**Visualization note:** Morse-set panels apply a display-only minimum box side of 0.75% of each plotted axis span. The saved CMGDB boxes and topology are unchanged.

## Operationally incomplete or invalid cells

None.

## Derived artifacts

- `summary.pdf` - six-page visual report; green frames indicate periodic bistability.
- `cells.csv` - strict flat inventory with all four topology flags.
- `cells.json` - strict per-cell records and provenance.
- `aggregate_summary.json` - strict topology, tolerance, loss, and timing aggregates.
