# Leslie3D Example 2 (Patrick): 4x-data retrain

Date: 2026-08-03  
Status: complete (`data`, `scale`, `train`, `diagnose`, `morse`, `render`, and `metrics`)  
Scope: code-local exploratory experiment; nothing was written under `../paper/`.

## Result in one sentence

Quadrupling Patrick's reconstructed paper-run data counts improves every
held-out loss relative to the later seed-2 fresh retrain and produces two
minimal sets with the target period-four index `(x^4-1, 0, 0)`, but it also
produces a third, period-three minimal set: more data helps index recovery but
does **not** recover either Patrick's archived five-node/two-sink graph or the
precise direct map's two-sink structure.

## Isolation and exact settings

The experiment config is
`src/latentdynamics/configs/leslie3d_example2_large_data_4x.yaml`. A direct
diff against `leslie3d_example2.yaml` has only four value changes:

- `n_samples_train: 8000 -> 32000`
- `n_samples_val: 2000 -> 8000`
- `data_dir: data/leslie3d_example2_large_data_4x`
- `output_dir: output/leslie3d_example2_large_data_4x`

Everything else is fixed. In particular:

| group | exact settings |
|---|---|
| Leslie map | `theta=(28.9, 29.8, 22.0)`, survival `(0.7, 0.7)` |
| dimensions | ambient `3`, latent `2` |
| encoder | widths `[64,64]`, ReLU, tanh output |
| latent map | widths `[64,64]`, ReLU, tanh output |
| decoder | widths `[64,64]`, ReLU, sigmoid output |
| optimizer | Adam, learning rate `0.001`, batch size `1024` |
| training | at most `1000` epochs, patience `100`, LR patience `10`, factor `0.1`, threshold `0.001`, minimum LR `0` |
| losses | weights `(100,10,20)`, gradient clipping at `1.0` |
| sampling | uniform, 20 transitions per initial condition, no skipped steps, train seed `42`, validation seed `9999` |
| scaling | min-max, fit on train inputs and images |
| network seed | `2` |
| CMGDB | subdivision `25/28/29`, limit `10000`, 1% bound padding, padding enabled, auto box-map backend, table-point cap `1.2B` |

For this adaptive subdivision ladder, `box_map_backend: auto` resolves to the
`adaptive_precomputed` backend. It evaluates the latent map once on the finest
corner lattice, `(2^15+1)^2 = 1,073,807,361` points, in automatically sized
batches and then serves CMGDB from the precomputed table. The 1.2B
`max_table_points` setting permits that table. The MapGraph vertex/edge guards
described below are separate native-cache allocation limits.

The run used Python 3.13.14, PyTorch 2.11.0, CMGDB 1.3.3, MPS, and repository
revision `0898b9420787cb20a8657a3f76cacd89e166a2bb`. The raw YAML SHA-256 is
`d506dcb28ee72e642242e7ba2b3a92e518dcc5297569ea9aac627ceb05ff75a5`.

Patrick's raw paper-run CSVs and training script were not archived. This is
therefore a 4x-data *fresh retraining attempt of the packaged reconstruction of
Patrick's second paper configuration*, not a continuation of his checkpoint.

## Data

| split | initial conditions | transitions | CSV lines | size | SHA-256 |
|---|---:|---:|---:|---:|---|
| train | 32,000 | 640,000 | 640,001 | 42 MiB | `e87cf49530a4b5052e169b766f0e04e4eedbf8aaf55094763c290a15e24bf51e` |
| validation | 8,000 | 160,000 | 160,001 | 10 MiB | `4afb625c7a092b0af8bc3eb9d18d84df16e25bf7c33f6e273765d5fd20b6a5e0` |

The distinct scaler is
`output/leslie3d_example2_large_data_4x/scalers/train/scaler.gz` (SHA-256
`43c47abd3e9d81f3d22755a1dfead6c3165029d3040fa3ed9895ae4479937dfc`).
Its observed minima are `(0,0,0)` and maxima are
`(219.99862132,153.99903493,107.99430381)`.

## Commands and timing

Run from `code/`. The initial all-stage command was:

```bash
/usr/bin/time -p ../.venv/bin/python pipeline.py \
  --config leslie3d_example2_large_data_4x \
  --stages data,scale,train,diagnose,morse,render,metrics \
  --device mps \
  --expected-cells 1
```

It completed data generation, scaling, training, and diagnosis, then CMGDB's
new allocation guard stopped before graph construction because an initial
level-25 cache requires 33,554,432 vertices while the default guard permits
16,777,216. The saved checkpoint was retained. The remaining stages were run
without retraining, using allocation guards (not dynamical parameters):

```bash
CMGDB_MAPGRAPH_MAX_EDGES=1200000000 \
CMGDB_MAPGRAPH_MAX_VERTICES=40000000 \
/usr/bin/time -p ../.venv/bin/python pipeline.py \
  --config leslie3d_example2_large_data_4x \
  --stages morse,render,metrics \
  --device mps \
  --expected-cells 1
```

The retry re-derived exactly the same printed fresh-data bounds as the first
attempt. For a clean rerun, the two guard variables can be placed on the
all-stage command from the beginning.

| work | duration |
|---|---:|
| training | 864.06 s (14.401 min) |
| CMGDB | 962.604 s (16.0434 min) |
| successful `morse,render,metrics` command | 984.41 s |
| both executed commands, including the guarded first CMGDB attempt | 1,897.10 s (31 min 37.1 s) |

## Training and diagnosis

Early stopping selected zero-based epoch 74 and stopped after 175 epochs. The
saved model SHA-256 is
`269a4cc42ea7ec0d33d28c3cc881c77a0f609bcff2b5dcf45b35ec1925c0addd`.

Patrick's archived losses are not directly comparable because the original
training inputs are unavailable. The appropriate controlled loss comparison
is the later seed-2 fresh retrain, which uses the same packaged settings with
8,000/2,000 initial conditions:

| selected-checkpoint validation loss | later 8k/2k seed-2 retrain | new 32k/8k retrain | change |
|---|---:|---:|---:|
| total | 0.0510421985 | 0.0353687080 | -30.71% |
| reconstruction | 0.0004566789 | 0.0003157968 | -30.85% |
| prediction | 0.0003530722 | 0.0002666858 | -24.47% |
| semiconjugacy | 0.0000921790 | 0.0000561082 | -39.13% |

`diagnose.json` reports `ok`: no encoder collapse and no latent-map
overcontraction. The latent contraction ratio is `0.8841448455` (the later
8k/2k fresh retrain has `0.7435266`).

## CMGDB result

The encoded-data domain, including configured padding, is

```text
[-0.3274068534374237, -0.42167341709136963]
    to
[ 0.5607303977012634,  1.0140669345855713 ]
```

CMGDB returns eight Morse sets and 71,068 saved boxes. Minimal nodes are
`{0,1,5}`. The transitive-reduction edges are

```text
2 -> 1
3 -> 2
4 -> 3
4 -> 0
6 -> 4
6 -> 5
7 -> 6
```

Per-node indices, box counts, and latent bounding extents are:

| node | minimal | Conley index | boxes | z1 extent | z2 extent |
|---:|:---:|---|---:|---|---|
| 0 | yes | `(x^4-1, 0, 0)` | 62,659 | `[0.094925, 0.249199]` | `[-0.315991, -0.148441]` |
| 1 | yes | `(x^3-1, 0, 0)` | 362 | `[0.173905, 0.185831]` | `[-0.221087, -0.206715]` |
| 2 | no | `(0, x^3-1, 0)` | 960 | `[0.178025, 0.185072]` | `[-0.215040, -0.206891]` |
| 3 | no | `(0, 0, x-1)` | 366 | `[0.180030, 0.181386]` | `[-0.211448, -0.209607]` |
| 4 | no | `(0, x^4-1, 0)` | 2,200 | `[0.143278, 0.207568]` | `[-0.261923, -0.185596]` |
| 5 | yes | `(x^4-1, 0, 0)` | 1,287 | `[0.009493, 0.382333]` | `[-0.398714, -0.065280]` |
| 6 | no | `(0, x^4-1, 0)` | 3,219 | `[0.057033, 0.304491]` | `[-0.362961, -0.110147]` |
| 7 | no | `(0, x-1, 0)` | 15 | `[0.114439, 0.114819]` | `[-0.415627, -0.415189]` |

The graph is internally consistent according to the pipeline check: all
attractor-type indices are minimal and no index is trivial. That structural
check is distinct from the tolerance test below.

## Sampled tolerance metrics

All three learned minimal sets fail the sampled tolerance comparison and are
flagged as spurious by the metric:

| node | index | boxes | tau-bar | sampled points | max residual | residual / tau |
|---:|---|---:|---:|---:|---:|---:|
| 0 | `(x^4-1,0,0)` | 62,659 | 5.46912e-5 | 81 | 0.0117548 | 214.93x |
| 1 | `(x^3-1,0,0)` | 362 | 6.50579e-5 | 3 | 0.0193568 | 297.53x |
| 5 | `(x^4-1,0,0)` | 1,287 | 5.42450e-5 | 90 | 0.00478460 | 88.20x |

These finite failures do not certify semiconjugacy or invariant sets for the
original map. Node 1 has only three semiconjugacy samples, so its numerical
ratio is especially sparse, but it still fails the implemented check.

## Scientific comparison

### Primary comparison: Patrick's archived second paper run

Patrick's preserved graph has five nodes, exactly two minimal nodes, and both
minimal indices are `(x^4-1,0,0)`. Its edges are
`2->0, 2->1, 3->1, 4->3`; per-node box counts are
`{0:6581, 1:26257, 2:14010, 3:15266, 4:61}`. Its nonminimal indices are
`2:(0,x^4-1,0)`, `3:(0,x^4-1,0)`, and `4:(0,0,x-1)`.

The 4x retrain is therefore **not a reproduction of Patrick's graph**. It does
recover two period-four sinks, but has eight rather than five nodes, three
rather than two sinks, an additional period-three attractor/repeller branch,
and a different order relation. Latent extents and raw box counts should not be
compared geometrically across these networks because each encoder defines a
different coordinate system.

### Precise direct-map computation

The precise direct-map audit has six recurrent sets and exactly two sinks;
both sink indices are `(x^4-1,0,0,0)`, corresponding to the 2D latent target
`(x^4-1,0,0)`. Its exact level-33 saved-set reachability reduction is
`2->1, 3->0, 3->1, 4->2, 5->3, 5->4`.

Here “precise” means exact Conley strings for the six saved recurrent sets and
exact reachability among those sets in the replicated floating-point,
corner-sampled level-33 graph; it is not an interval proof for the continuous
Leslie map.

Thus the 4x retrain succeeds only at the narrow criterion that two of its
minimal sets have the target period-four index. It fails the stronger and
scientifically necessary exact-attractor criterion because it also has the
period-three minimal node 1. Its nonminimal indices and graph are not a
nodewise recovery of the direct computation either.

### Auxiliary comparison: later seed-2 fresh retrain

The later 8k/2k fresh retrain has three nodes, edges `2->0,2->1`, and two
minimal indices `(x^2-1,0,0)` and `(x-1,0,0)`; box counts are
`{0:465,1:168,2:123}`. Relative to that run, more data substantially improves
held-out fit and restores two period-four sink indices, but replaces the
two-sink error with a three-sink error. Data volume alone is therefore not a
sufficient fix at seed 2.

## Artifacts and verification

Primary artifacts:

- config: `src/latentdynamics/configs/leslie3d_example2_large_data_4x.yaml`
- data: `data/leslie3d_example2_large_data_4x/`
- output: `output/leslie3d_example2_large_data_4x/`
- checkpoint: `output/leslie3d_example2_large_data_4x/seed_2/models/autoencoder.pt`
- graph: `output/leslie3d_example2_large_data_4x/seed_2/MG/morse_graph`
- Morse boxes: `output/leslie3d_example2_large_data_4x/seed_2/MG/morse_sets`
- metrics: `output/leslie3d_example2_large_data_4x/seed_2/metrics.json`
- normalized run provenance: `output/leslie3d_example2_large_data_4x/seed_2/run_manifest.json`

Key output SHA-256 values:

| artifact | SHA-256 |
|---|---|
| checkpoint | `269a4cc42ea7ec0d33d28c3cc881c77a0f609bcff2b5dcf45b35ec1925c0addd` |
| raw DOT graph | `38f4686ba45d7851368fdeb25d3ae96ac5f744e5e2ab3140df59b8d65bd8dcea` |
| Morse-box CSV | `0b87953123eccdaaea0f6b2061485fcb83566d48027b4482120fa49456329ce4` |
| metrics JSON | `fc6fe3b9dd67a5be55dea20f4c8cde9fde9d9825b3a6ac3b2d88cb4c2e11df17` |

Post-run QA verified 16 required nonempty artifacts, parsed all 71,068 finite
Morse-box rows with labels exactly `0..7`, confirmed minimal labels `{0,1,5}`
and the internal graph-consistency flag, opened every PNG successfully, and
confirmed all five rendered PDFs are valid one-page documents. The graph and
overlay PNGs were also inspected visually. Rendering used the pipeline's
approximate uniform-grid region-of-attraction fallback because exact CMGDB-grid
RoA output was not requested (`compute_roa: false`).
