# Leslie3D Example 2: T=25, N=50,000 seed sweep

Date: 2026-08-03  
Status: running (`data`, `scale`, training, and diagnosis complete; CMGDB pending)  
Scope: code-local exploratory experiment; nothing was written under `../paper/`.

## Question and design

This experiment repeats the five-dataset by three-network-seed robustness
sweep for Patrick's Leslie3D Example 2 while holding the trajectory length at
`T=25` and increasing the total number of sampled initial conditions to
`N=50,000` per dataset. Here `N` follows the paper table's initial-condition
notation, not the number of one-step transition pairs. The packaged 80/20
split is preserved:

- 40,000 independently sampled training initial conditions for each of five
  data seeds (`1,2,3,4,5`);
- one common 10,000-initial-condition validation set with seed `9999`;
- three independent network initializations (`0,1,2`) per training dataset,
  for 15 trained models in total;
- 25 retained transitions per initial condition and no discarded steps.

Thus each dataset tree contains 1,000,000 training and 250,000 validation
transition pairs. Across the five trees this is 5,000,000 independently
sampled training pairs plus five references to the same 250,000-pair holdout
CSV. Validation is deliberately shared so held-out losses are directly
comparable across both seed axes.

## Leslie map and fixed settings

The experiment uses

```text
f(x1,x2,x3) = (
    (28.9*x1 + 29.8*x2 + 22.0*x3) * exp(-0.1*(x1+x2+x3)),
    0.7*x1,
    0.7*x2
)
```

Training initial conditions are uniform on
`[0,220] x [0,154] x [0,108]`. The smaller
`[0,110] x [0,77] x [0,54]` domain belongs only to the separate direct-map
ground-truth computation and is not used for neural-network training here.

All network, optimizer, loss, scaling, and CMGDB settings are inherited from
`src/latentdynamics/configs/leslie3d_example2.yaml`. In particular, the latent
dimension is 2; encoder, latent map, and decoder each have two width-64 hidden
layers; Adam uses learning rate `0.001` and batch size `1024`; the loss weights
are `(100,10,20)`; training has at most 1,000 epochs with patience 100; and the
CMGDB subdivision ladder is `25/28/29` with limit 10,000 and 1% inferred-bound
padding.

The box-map backend is pinned to `adaptive_precomputed`. In two latent
dimensions the finest corner lattice contains
`(2^(29-14)+1)^2 = 32,769^2 = 1,073,807,361` points. The raw Morse sets, DOT
Morse graph, rendered graph/sets, checkpoint, diagnosis, metrics, and
provenance manifest are retained separately for every cell below
`output/leslie3d_example2_seedsweep_t25_n50000/dataset_D/seed_S/`.

## Verified input data

| dataset | train seed | train rows | train CSV SHA-256 | validation seed | validation rows | validation CSV SHA-256 |
|---:|---:|---:|---|---:|---:|---|
| 1 | 1 | 1,000,000 | `9350bd5e1276792471d507e9454bff94c1eb8b5360c2d6f0f8a4a4e9940becd2` | 9999 | 250,000 | `f55553a78bffa5edec2200dcd676b2efb81e2a443b6f4d072795a3976caefbd5` |
| 2 | 2 | 1,000,000 | `648175a04c5527e4965ed9ddcb418b1a885d1c55a72cf7c56e8e695d06bbfde2` | 9999 | 250,000 | same |
| 3 | 3 | 1,000,000 | `8c75ed72a8af12c5a423cc1b97237b682c56f29f4ed51237f69c0803832c72fd` | 9999 | 250,000 | same |
| 4 | 4 | 1,000,000 | `c5042cc993b812384fab81cc258b6be9d32b1bde6a84e495188ef6806cf9c0c2` | 9999 | 250,000 | same |
| 5 | 5 | 1,000,000 | `359ca1390d132b967767c9423a3e8172a7942cbb93e331d13d2f3501e317c61e` | 9999 | 250,000 | same |

The five training hashes are distinct and the validation hash is identical,
as required by the design. CSV line counts are one larger than the row counts
shown because each file has one header line.

## Reproduction commands

Run from `code/`. Data generation and scaling were performed with:

```bash
PYTHONPATH=src .venv/bin/python scripts/retrain_seed_sweep.py \
  --example leslie3d_example2 \
  --trajectory-length 25 \
  --total-initial-conditions 50000 \
  --box-map-backend adaptive_precomputed \
  --tag t25_n50000 \
  --stages data,scale
```

Training and diagnosis use the same overrides plus:

```bash
--device mps --stages train,diagnose --skip-completed
```

CMGDB is run serially, one cell at a time, with the native allocation guards
`CMGDB_MAPGRAPH_MAX_VERTICES=40000000` and
`CMGDB_MAPGRAPH_MAX_EDGES=1200000000`. Rendering and metrics follow without
retraining. Strict aggregation is performed with:

```bash
PYTHONPATH=src .venv/bin/python \
  scripts/analyze_leslie3d_example2_t40_sweep.py \
  --sweep-root output/leslie3d_example2_seedsweep_t25_n50000 \
  --data-root data/leslie3d_example2_seedsweep_t25_n50000 \
  --expected-t 25 \
  --expected-train-initial-conditions 40000 \
  --expected-validation-initial-conditions 10000
```

## Results

### Training and diagnosis

All 15 independently initialized models completed and all 15 diagnostics
returned `ok`. The checkpoint hashes are pairwise distinct. Epoch numbers in
the table are zero-based, as stored in `training_summary.json`.

| dataset | model seed | best epoch | epochs run | training min | selected validation total | latent contraction ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 23 | 124 | 16.09 | 0.0516392 | 0.496252 |
| 1 | 1 | 50 | 151 | 19.56 | 0.0253105 | 0.884354 |
| 1 | 2 | 126 | 227 | 29.45 | 0.0220619 | 0.869444 |
| 2 | 0 | 60 | 161 | 20.81 | 0.0363817 | 0.583356 |
| 2 | 1 | 121 | 222 | 28.99 | 0.0203358 | 0.945502 |
| 2 | 2 | 97 | 198 | 25.41 | 0.0249734 | 1.068957 |
| 3 | 0 | 16 | 117 | 14.40 | 0.0476901 | 0.586539 |
| 3 | 1 | 100 | 201 | 24.81 | 0.0217640 | 0.692614 |
| 3 | 2 | 41 | 142 | 17.58 | 0.0366683 | 1.068616 |
| 4 | 0 | 39 | 140 | 17.71 | 0.0297277 | 0.854415 |
| 4 | 1 | 171 | 272 | 34.97 | 0.0180095 | 1.195032 |
| 4 | 2 | 128 | 229 | 29.63 | 0.0204962 | 1.023171 |
| 5 | 0 | 40 | 141 | 18.81 | 0.0387514 | 0.687133 |
| 5 | 1 | 97 | 198 | 26.02 | 0.0235091 | 0.859698 |
| 5 | 2 | 59 | 160 | 20.66 | 0.0314123 | 0.925553 |

The selected validation-total loss has mean `0.0299154`, population standard
deviation `0.0099755`, and range `0.0180095--0.0516392`. Total recorded
training time is `20,693.77` seconds (`5.748` serial compute-hours), with a
mean of `22.99` minutes per cell and range `14.40--34.97` minutes.

### Morse graphs and Conley indices

Pending completion of all 15 serialized CMGDB cells. The primary success
criterion is exactly two graph sinks, with both sink Conley-index tuples equal
to `(x^4-1,0,0)`, matching the two period-four attractors in the precise
direct-map computation after dropping its unused fourth homological slot.
