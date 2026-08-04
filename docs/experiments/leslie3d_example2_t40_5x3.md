# Leslie3D Example2 (Patrick reconstruction): T=40, 5x3 sweep

Date: 2026-08-03 to 2026-08-04  
Status: complete; strict analyzer verified all 15 cells  
Scope: code-local exploratory experiment; nothing was written under `../paper/`.

## Question and success criterion

This sweep tests whether keeping the reconstructed Patrick Example2 data count
fixed while lengthening every trajectory from `T=20` to `T=40` improves
recovery of the precise direct Leslie-map attractor indices.

The primary success criterion is deliberately exact and graph-based: a cell
succeeds only when its learned Morse graph has exactly two sinks and both sink
indices are `(x^4-1, 0, 0)`. This is the two-dimensional latent counterpart of
the direct three-dimensional computation's two stable period-four sets, whose
indices are both `(x^4-1, 0, 0, 0)`. Having two period-four sinks plus any
additional sink is not counted as success.

## Fixed design

| item | setting |
|---|---|
| independent training datasets | five, training-data seeds `1,2,3,4,5` |
| model seeds per dataset | `0,1,2` |
| total trained models | 15 |
| validation design | one shared holdout, seed `9999` |
| initial conditions per dataset | 8,000 train and 2,000 validation |
| trajectory length | `T=40`, with `T0=0` discarded steps |
| transition pairs per dataset | 320,000 train and 80,000 validation |
| total transitions across five saved dataset trees | 1,600,000 train and 400,000 validation |
| sampling | independent uniform training draws; identical validation CSV in all five trees |
| sampling domain | `[0,220] x [0,154] x [0,108]` |

Each data seed is trained three times from independent model initializations.
The five training CSVs must have distinct SHA-256 hashes, while all five
validation CSVs must have one common hash and the common seed `9999`; the
strict analyzer treats either violation as a failed sweep design.

## Leslie map and network parameters

The physical map used for every generated transition is

```text
f(x1,x2,x3) = (
  (28.9*x1 + 29.8*x2 + 22.0*x3) * exp(-0.1*(x1+x2+x3)),
  0.7*x1,
  0.7*x2
)
```

Thus `theta=(28.9,29.8,22.0)`, the two survival parameters are `(0.7,0.7)`,
and the density-decay coefficient is `0.1`. The smaller direct-computation box
`[0,110] x [0,77] x [0,54]` is not the neural-network sampling domain.

The ambient/latent dimensions are `3/2`. Encoder, latent map, and decoder each
have two width-64 hidden layers with ReLU activations; encoder and latent-map
outputs use tanh, and decoder output uses sigmoid. Training uses Adam at
learning rate `0.001`, batch size `1024`, at most `1000` epochs, early-stopping
patience `100`, LR patience `10`, loss weights `(100,10,20)`, and gradient
clipping at `1.0`.

## CMGDB and precomputation

Every cell uses the canonical latent subdivision ladder `25/28/29`, subdivision
limit `10000`, one-percent encoded-data bound expansion, and padding. The box
map is explicitly pinned to `adaptive_precomputed`; this is not an implicit
`auto` resolution.

For latent dimension two and `subdiv_max=29`, the lookup uses
`M=ceil(29/2)=15`, hence a `32769 x 32769` corner lattice containing exactly
`1,073,807,361` neural-map evaluations. The configured table-point cap is
`1.2 billion`. CMGDB runs serially, one trained cell at a time, with allocation
guards `CMGDB_MAPGRAPH_MAX_VERTICES=40000000` and
`CMGDB_MAPGRAPH_MAX_EDGES=1200000000`.

## Training and diagnosis

All 15 independently initialized models completed and have distinct checkpoint
SHA-256 hashes. All 15 diagnoses report `ok`: none flags encoder collapse or
latent-map overcontraction. Four cells used the full 1,000-epoch ceiling; the
other eleven stopped under the common patience rule.

| dataset | model seed | epochs | best epoch | train minutes | best validation total |
|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 1,000 | 978 | 48.4948 | 0.02501040 |
| 1 | 1 | 183 | 82 | 9.0036 | 0.02397619 |
| 1 | 2 | 332 | 231 | 18.8494 | 0.02689020 |
| 2 | 0 | 1,000 | 999 | 43.3058 | 0.01996158 |
| 2 | 1 | 268 | 167 | 12.5505 | 0.01770122 |
| 2 | 2 | 460 | 359 | 24.0492 | 0.02012770 |
| 3 | 0 | 182 | 81 | 9.5470 | 0.03382398 |
| 3 | 1 | 1,000 | 993 | 45.6506 | 0.01933345 |
| 3 | 2 | 1,000 | 994 | 46.1708 | 0.02246956 |
| 4 | 0 | 207 | 106 | 8.7421 | 0.02855981 |
| 4 | 1 | 348 | 247 | 17.2271 | 0.02016605 |
| 4 | 2 | 435 | 334 | 20.5150 | 0.01702748 |
| 5 | 0 | 332 | 231 | 16.2217 | 0.02319120 |
| 5 | 1 | 261 | 160 | 12.6752 | 0.01889482 |
| 5 | 2 | 358 | 257 | 15.0448 | 0.01889452 |

Across all cells, the mean best validation total is `0.02240188`, with range
`[0.01702748, 0.03382398]`. Active training sums to 348.0476 minutes; the
mean is 23.2032 minutes per model. The dataset-level validation means range
from `0.01926350` (dataset 2) to `0.02529226` (dataset 1), while the largest
single-cell value is dataset 3 / seed 0. Thus the independent data draw has a
visible effect even before comparing Morse decompositions.

## Results

The exact target was recovered in **1 of 15 cells (6.7%)**. The sole success
is dataset 5 / model seed 0. It has seven Morse nodes and exactly two graph
sinks, and both sink indices are `(x^4-1, 0, 0)`.

| dataset | model | nodes/edges | graph-sink indices | exact | tolerance | CMGDB min |
|---:|---:|---:|---|:---:|---|---:|
| 1 | 0 | 10/10 | `x^4-1`; `x^2-1` | no | fail/fail | 16.4216 |
| 1 | 1 | 6/5 | `x^8-1`; `x^2-1` | no | fail/fail | 16.4862 |
| 1 | 2 | 11/11 | `x^4-1`; `x^2-1` | no | fail/fail | 16.3366 |
| 2 | 0 | 4/3 | `x^2-1`; `x^4-1` | no | fail/fail | 19.1644 |
| 2 | 1 | 8/8 | `x^8-1`; `x^2-1` | no | inconclusive/inconclusive | 18.1105 |
| 2 | 2 | 6/5 | `x^4-1`; trivial | no | fail/fail | 20.0886 |
| 3 | 0 | 4/3 | `x^4-1`; `x-1` | no | inconclusive/fail | 16.7186 |
| 3 | 1 | 7/7 | `x^4-1`; `x^2-1`; `x^4-1` | no | fail/inconclusive/fail | 16.7190 |
| 3 | 2 | 4/3 | trivial; `x^4-1` | no | fail/fail | 17.8284 |
| 4 | 0 | 5/4 | `x^2-1`; `x^4-1` | no | fail/inconclusive | 16.4331 |
| 4 | 1 | 6/5 | `x^4-1`; `x^2-1` | no | fail/inconclusive | 31.3075* |
| 4 | 2 | 8/9 | `x^4-1`; `x^2-1` | no | fail/inconclusive | 26.5170 |
| 5 | 0 | 7/6 | `x^4-1`; `x^4-1` | **yes** | fail/fail | 17.1177 |
| 5 | 1 | 6/5 | `x^4-1`; `x^2-1` | no | inconclusive/inconclusive | 17.0800 |
| 5 | 2 | 7/6 | `x^4-1`; trivial | no | fail/fail | 18.2426 |

`trivial` means `(0,0,0)`. A tolerance result is inconclusive when that
minimal set contains no sampled encoded points. The asterisked dataset 4 /
seed 1 duration includes an approximately 14-minute deliberate `SIGSTOP`
while an unrelated CMGDB job finished, so its 31.3075-minute value is not a
clean compute-time measurement. Dataset 4 / seed 2 also crossed a system-sleep
interval; the process remained intact and its internal timer recorded 26.5170
minutes rather than the much larger operating-system elapsed time.

Fourteen cells have exactly two graph sinks. Eleven have exactly two graph
sinks with both sinks of attractor type. The normalized sink-index outcomes
are seven `x^4-1 + x^2-1`, three `x^4-1 + trivial`, two
`x^8-1 + x^2-1`, one `x^4-1 + x-1`, one three-sink
`x^4-1 + x^4-1 + x^2-1`, and one exact `x^4-1 + x^4-1`.
All 15 full topology signatures are distinct. Dataset 5 / seed 1 additionally
has a nonminimal attractor-type `x^4-1` node, which is retained verbatim and
flagged by the metrics consistency diagnostic.

Tolerance is a separate diagnostic and is not part of the exact graph-based
criterion. Eight cells have evaluable tolerance results for every minimal set;
none passes, and their 16 evaluated minimal sets all fail. The other seven
cells have at least one zero-sample, hence inconclusive, minimal-set check.

### Comparison with the saved `T=20` sweeps

| sweep | train pairs/dataset | optimization | subdivision | mean best val total | exactly two attractor-type sinks | exact target |
|---|---:|---|---|---:|---:|---:|
| saved `T=20` minibatch `strict` | 160,000 | batch 1,024 | 24/25/29 | 0.04367535 | 11/15 | 2/15 (13.3%) |
| this `T=40` minibatch sweep | 320,000 | batch 1,024 | 25/28/29 | 0.02240188 | 11/15 | 1/15 (6.7%) |
| saved `T=20` full-batch sweep | 160,000 | full batch | 24/25/29 | 0.18302903 | 8/15 | 4/15 (26.7%) |

Doubling trajectory length sharply lowers the held-out validation objective, but
does **not** improve recovery of the precise two-period-four Conley result in
this 15-cell sample. The exact rate decreases from `2/15` to `1/15`, while the
count with exactly two attractor-type graph sinks stays `11/15`. This is not a
controlled topology-only comparison: the old sweeps used subdivision
`24/25/29`, whereas this sweep uses the canonical `25/28/29`. It therefore
supports the limited conclusion that more steps along the same number of
trajectories are not, by themselves, sufficient evidence of a solution. The
separate `T=25`, `N=50,000` sweep tests increased trajectory count/data volume
instead.

### Verification and timing

The strict analyzer verifies 15 complete datasets/cells, five distinct
training-CSV hashes, one shared validation CSV/hash and seed `9999`, and the
requested `8,000/2,000`, `T=40` design. A separate checkpoint-hash audit finds
15 distinct model files.
Every CMGDB log independently records `adaptive_precomputed`, subdivision
`25/28/29`, and the same `1,073,807,361`-point lattice. Raw recorded CMGDB
time is 284.5718 minutes total, with mean 18.9715 minutes and range
16.3366--31.3075 minutes; the pause caveat above prevents interpreting that
aggregate as a clean performance benchmark.

Artifact QA found 15 nonempty checkpoints, 15 raw Morse graphs, 15 raw
Morse-set files, 15 metrics files, and 15 CMGDB parameter logs. All 75 PDFs
validate and all 105 PNGs are nonempty. The exact-success graph and overlay
were inspected visually. The focused runner/analyzer suite passes all 13 tests,
and Ruff reports no findings. No file under `../paper/` was modified.

## Saved artifacts

- data: `data/leslie3d_example2_seedsweep_t40/dataset_{1..5}/`
- per-cell models and analyses:
  `output/leslie3d_example2_seedsweep_t40/dataset_{1..5}/seed_{0..2}/`
- raw graph and Morse boxes in each cell: `MG/morse_graph` and `MG/morse_sets`
- strict aggregate outputs: `output/leslie3d_example2_seedsweep_t40/analysis/`
- analyzer: `scripts/analyze_leslie3d_example2_t40_sweep.py`
- consolidated sweep inventory:
  `output/leslie3d_example2_seedsweep_t40/sweep_summary.json`
