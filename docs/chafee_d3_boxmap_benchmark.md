# Bounded D3 CMGDB BoxMap benchmark

The benchmark compares two ways to provide the same padded neural rectangle
map to CMGDB:

1. **On demand:** evaluate all eight corners of each three-dimensional
   rectangle in batched neural forward passes during CMGDB.
2. **Precomputed:** evaluate each unique uniform-grid corner once, then use
   vectorized array lookup during CMGDB.

The canonical `latent_3d/seed_0` checkpoint is the primary model because it is
the only completed D3 checkpoint in the current study archive. Additional D3
run roots can be supplied explicitly; the harness verifies their sidecar
dimension and input hashes.

## Pilot ladder

| Total subdivision | Cells per axis | Total cells | Unique precomputed corners |
|---:|---:|---:|---:|
| 12 | 16 | 4,096 | 4,913 |
| 15 | 32 | 32,768 | 35,937 |
| 18 | 64 | 262,144 | 274,625 |

Subdivision 24 (256 cells per axis, 16,777,216 total cells) is
**extrapolation-only**. The script contains a non-configurable level-18 hard
cap. The existing level-24 precomputed result is read only as context and is
never rerun.

The default protocol uses one full level-12 warmup per backend and three
measured repeats at each pilot level. Backend order alternates by repeat.
Every graph gets a fresh process, so native peak memory and failures are
isolated.

## Instrumentation

Each trial records:

- model load, neural warmup, precompute, CMGDB, callback, neural-forward, and
  total times;
- sampled whole-process peak RSS plus worker `ru_maxrss`;
- scalar/batch callback calls, rectangles, neural-forward calls, and evaluated
  corner points;
- cells, cached edges, Morse nodes/edges, minimal nodes, Morse-set sizes, and a
  graph fingerprint.

Primary comparisons keep lookup-only CMGDB time separate from precompute time.
That supports both a one-use end-to-end comparison and a reuse/amortization
interpretation.

## Stop rules

- 60 seconds and 3 GiB RSS per worker by default.
- 30 million cached edges, 750,000 callback rectangles, and 8 million neural
  points by default.
- Any worker failure or resource stop prevents higher-level pilots for that
  checkpoint.
- Any on-demand/precomputed graph-fingerprint disagreement prevents
  escalation.
- The script cannot dispatch a subdivision above 18.

Level-24 time, edge count, neural work, and memory are projected in two ways:
constant cost per cell from the largest completed pilot and a power-law fit to
pilot medians. These are capacity heuristics, not measurements or confidence
intervals.

Running the command without `--execute` prints the complete plan and performs
no CMGDB computation:

```bash
.venv/bin/python scripts/benchmark_chafee_d3_boxmap.py
```

An explicit bounded run writes only to a fresh benchmark directory:

```bash
.venv/bin/python scripts/benchmark_chafee_d3_boxmap.py --execute
```
