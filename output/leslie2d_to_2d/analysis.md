# 2D->2D Leslie: CMGDB structure and complexity analysis

Generated from `configs/leslie2d_to_2d.yaml` on macOS-26.5-arm64-arm-64bit-Mach-O
with 14 CPUs, torch 2.11.0, device cpu.

**Scope:** the pipeline's standard regime is 2D throughout (analytic system, encoder, latent map, decoder all act on 2D state). The only higher-D case in the project is the coral model. Optimization recommendations are organized accordingly: Section 4 covers the 2D pipeline (the common case); Section 5 covers high-D extrapolations (relevant only for the coral model).

## 1. Setup

Ambient system: `LeslieContraction(th1=20.0, th2=20.0, survival_p1=0.7, lower_bounds=[0,0], upper_bounds=[90,70])`
(Patrick's 2D-base parameters from `archive/patrick/2D_leslie_base_computation.py`).

Network: 2-layer 64-wide MLPs for encoder, latent map, decoder — all 2->2.
Loss weights `[100, 10, 20]` (recon, AE second-step, latent dynamics).
Training: early-stopped at epoch 331/1000 on test plateau.
Final training loss: `L_total = 3.22e-04`, `L_dyn = 3.34e-06`.

Ambient bounds (analytic CMGDB):
  - lower: `[0.0, 0.0]`
  - upper: `[90.0, 70.0]`

Latent bounds (encoded data, +epsilon padding):
  - lower: `[-0.8645, -0.6873]`
  - upper: `[0.5411, 0.689]`

**Canonical CMGDB methodology going forward:** **uniform grid** (`subdiv_init = subdiv_min = subdiv_max`) with **padding off**. This matches `notebooks/Example_Leslie_model.ipynb`, where bistability for `th=20` first appears at uniform subdiv = 17.

## 2. Uniform-grid subdivision sweep

Side-by-side analytic vs learned 2->2 latent map, identical uniform-grid CMGDB parameters, padding=False. *Minimal* nodes = Morse nodes with no outgoing edges (attractors); minimal >= 2 means bistability.

| subdiv | analytic time | analytic nodes/edges/min | latent time | latent nodes/edges/min | latent / analytic |
|-------:|--------------:|-------------------------:|------------:|-----------------------:|------------------:|
|     10 |        0.04s  |                  2/1/1   |      0.16s  |                1/0/1   |             4.35x |
|     12 |        0.15s  |                  4/3/1   |      0.63s  |                1/0/1   |             4.13x |
|     14 |        0.55s  |                  3/2/1   |      2.49s  |                2/1/1   |             4.49x |
|     16 |        2.15s  |                  3/2/1   |      9.77s  |                4/3/1   |             4.55x |
| **17** |    **4.20s**  |              **8/7/2**   |  **19.28s** |            **6/5/2**   |             4.59x |
|     18 |        8.47s  |                  7/6/2   |     51.09s  |                5/4/2   |             6.03x |
|     19 |       24.94s  |                  7/6/2   |     82.82s  |                6/5/2   |             3.32x |
|     20 |       32.40s  |                  8/7/2   |    194.45s  |                5/4/2   |             6.00x |

**Headline finding — bistability:** both the analytic 2D Leslie and the learned 2->2 latent map first detect bistability at uniform subdiv = **17**, matching the notebook. With uniform + no padding, the latent map is **not** behind the analytic in detecting bistability.

**Coarser transient structure:** at smax >= 17 the latent has typically 5-6 nodes vs analytic 7-8. The missing nodes are trivial-Conley-index `(0,0,0)` transient layers between the rank-1 source `(0, x^3-1, 0)` and the period-3 attractor. The recurrent core (two attractors + rank-1 source) is preserved.

**Cost:** latent run is 3-6x slower than analytic at matched subdivisions; dominated by per-box NN evaluation cost in `CMGDB.BoxMap`. Latent total-box counts are consistently *lower* than analytic (fewer Morse sets to refine).

See `scaling.png` for log-scale wall-clock and box-count curves.

### Canonical Morse-graph diff at uniform smax=17 (bistability threshold)

Analytic (8 nodes, 11090 boxes):

| node | Conley polynomial | role |
|-----:|-------------------|------|
| 0 | `(x^3-1, 0, 0)` | period-3 attractor (minimal) |
| 1 | `(x-1, 0, 0)`   | fixed-point attractor (minimal) |
| 2 | `(0, 0, 0)`     | trivial transient, flows into 1 |
| 3 | `(0, 0, 0)`     | trivial transient, flows into 2 |
| 4 | `(0, x^3-1, 0)` | rank-1 source, flows into 0 and 3 |
| 5 | `(0, 0, 0)`     | trivial transient, flows into 4 |
| 6 | `(0, 0, 0)`     | trivial transient, flows into 5 |
| 7 | `(0, 0, 0)`     | trivial transient, flows into 0 |

Learned latent (6 nodes, 6117 boxes):

| node | Conley polynomial | role |
|-----:|-------------------|------|
| 0 | `(x-1, 0, 0)`   | fixed-point attractor (minimal) |
| 1 | `(x^3-1, 0, 0)` | period-3 attractor (minimal) |
| 2 | `(0, 0, 0)`     | trivial transient, flows into 1 |
| 3 | `(0, 0, 0)`     | trivial transient, flows into 2 |
| 4 | `(0, 0, 0)`     | trivial transient, flows into 3 |
| 5 | `(0, x^3-1, 0)` | rank-1 source, flows into 0 and 4 |

The latent preserves both attractors, the rank-1 source, and the qualitative partial order. It has 3 trivial-index transient nodes between source and period-3 attractor; analytic has 5. "Training-loss-small implies bistability-detection-equivalent at the same subdivision threshold" holds for this 2D->2D matched-dimension experiment under uniform + no padding.

## 3. BoxMap implementation comparison

From `scripts/optimize_box_map.py`, run at adaptive smax=20 (legacy methodology). Relative speedups across implementations carry over to uniform.

| implementation | per-box (us) | CMGDB total (s) | Morse nodes | boxes | speedup vs default |
|----------------|-------------:|----------------:|------------:|------:|-------------------:|
| `analytic_default` | 12.92 | 5.49 | 4 | 46632 | 1.00x |
| `analytic_batched` | 6.00 | 2.96 | 4 | 46632 | 1.85x |
| `latent_default` | 67.46 | 13.41 | 2 | 25767 | 1.00x |
| `latent_batched` | 17.37 | 4.06 | 2 | 25767 | 3.30x |
| `latent_numpy` | 10.42 | 2.74 | 2 | 25767 | 4.89x |
| `latent_centered` | 22.92 | 79.93 | 1 | 123232 | 0.17x |

`latent_numpy` (Linear weights extracted as NumPy, single matmul per box on the `2^d=4` corners) is **5x faster** than the current `latent_default` PyTorch-scalar path, and **faster than the analytic Python step** for the trained network. Identical Morse output. `latent_centered` (single center eval + operator-norm `L=32` padding) is *slower* than `latent_default` and produces a coarser graph — operator-norm `L` is too loose; see [[lipschitz-bound-loose-for-trained-MLP]] memory.

## 4. Optimization levers for the 2D pipeline (the common case)

In rough order of payoff:

### 4.1 Pre-evaluate the whole grid up front (uniform-only, biggest lever)

Under uniform grid, the box partition is fixed up-front. Each corner is shared between up to `2^d` neighboring boxes. We can:

1. Build the level-`k` product grid of unique corners — `(2^(k/d)+1)^d` points.
2. Evaluate the latent map on all of them in **one batched forward pass**.
3. Build a precomputed dict: `box_index -> outer_cover_Rect`.
4. Wrap `box_map(rect)` to map the incoming Rect to its grid index and return the precomputed Rect.

This eliminates all NN evaluations from CMGDB's inner loop. The only per-call cost during CMGDB is a Rect-to-index conversion and a dict lookup.

**Expected speedup at uniform smax=20, 2D:** the current `latent_default` measures 194s, dominated by NN per-call overhead × ~4M scalar calls. A single batched forward on `~1M` unique corners completes in ~1-2s on CPU; CMGDB graph-construction adds 5-10s. **Plausible total: 5-15s. 15-40x speedup.** Bigger at higher smax.

**Caveats:**
- Pure uniform-grid optimization — adaptive mode kills it.
- Memory: `~16MB` corner buffer at `k=20, d=2`; `~256MB` at `k=24`. Chunk for `k > 24`.
- Rect-to-index lookup is exact since box corners are at known half-step lattice positions.

Implementation: ~100 lines in `analysis/morse.py`. Detect uniform mode from `(subdiv_init == subdiv_min == subdiv_max)` and dispatch.

### 4.2 Train with Jacobian regularization on the recurrent region

Per-box `mode='corners'` outer-cover error scales like `|sin θ_J| * σ_max(J) * box_diameter` where `θ_J` is the Jacobian's rotation off the coordinate axes. The trained NN's Jacobian is typically anisotropic and rotated relative to the box axes; this is what costs you extra subdivisions vs the analytic map (whose Jacobian at fixed points is diagonal).

Adding a penalty `λ * ||J_φ(x)||_F` at sampled latent points during training directly reduces the `σ_max(J)` factor. Unlike weight decay, it targets the operator the BoxMap sees. Dimension-independent; bigger payoff than activation choice.

### 4.3 Replace `make_box_map` with `latent_numpy`

Measured 5x in 2D. Strict implementation speedup, identical Morse output. Worth doing even if (4.1) is implemented (the warm-up corner evaluation in 4.1 should also use NumPy).

### 4.4 IBP / CROWN-IBP outer covers

Replace `mode='corners'` with layer-by-layer interval bound propagation through the MLP. Cost is `O(layers * width^2)` per box, **independent of `d`**, and gives a rigorously tight outer cover. In 2D the `2^d=4` corner count is already cheap, so IBP's main 2D value is letting you use `padding=False` *with a rigorous bound* (rather than relying on no-padding as a numerical estimate). This is the relevant lever if you want a defensible "training-loss-bounded implies Morse-graph-equivalent" theorem rather than the empirical observation we have now.

### 4.5 Activation / initialization choices (second-order)

- **ReLU vs tanh/GELU**: ReLU is exact within each activation-pattern cell but non-rigorous in boxes that straddle kinks. At smax >= 17 in 2D, kink-straddling boxes are a vanishing fraction (`O(2^(-k/d))`), so the effect is small. Smooth activations (tanh, GELU) avoid kink boxes entirely; their corners-only error is `O(diameter)` first-order plus `O(diameter^2)` curvature. Probably 1-2 fewer subdivisions needed at coarse `k`; negligible at fine `k`.
- **Orthogonal init / spectral-norm constraints**: keep the Jacobian closer to a rotation, reducing the anisotropy factor in the error model. Lower-cost than (4.2) but less targeted.

## 5. High-D extrapolation (relevant only for the coral model)

For ambient dimension `d >> 2` (i.e. the coral model, not the standard 2D pipeline), the `mode='corners'` `2^d` factor becomes a structural problem:
- `d=10` → `2^10 = 1024` corner evaluations per box.
- Patrick's archived 10D Leslie run reports `1073 min ~ 18 hours` at adaptive smax=28.

The 2D-pipeline optimizations (4.1, 4.2, 4.3) still apply, plus high-D specific:

- **`mode='random', num_pts=128`** instead of `corners`. For `d=10`, that's `1024 -> 128`, ~8x speedup at the cost of a probabilistic outer approximation.
- **Vectorized BoxMap with `latent_numpy`** is much higher-impact in high-D (the PyTorch per-call overhead is amortized over `2^d` corners — 100-200x speedup plausible at `d=10`).
- **Batched `box_map(rects)` API on the CMGDB side** (requires CMGDB patch) compounds with `latent_numpy`.

## 6. Artifacts

- Canonical run (legacy adaptive pipeline.yaml subdivisions): `output/leslie2d_to_2d/seed_0/MG/{morse_graph,morse_sets,morse_graph.pdf,morse_sets.pdf}`
- **Uniform-grid sweep (current canonical methodology):** `output/leslie2d_to_2d_uniform_nopad/{profile_results.json,morse/...}`
- Legacy adaptive sweep: `output/leslie2d_to_2d_profile/{profile_results.json,morse/...}`
- BoxMap-variant comparison: `output/leslie2d_to_2d_optimize/{optimize_results.json,morse/...}`
- Plots: `output/leslie2d_to_2d/{scaling.png,optimization.png}`
- Scripts: `scripts/profile_cmgdb_2d.py`, `scripts/optimize_box_map.py`, `scripts/write_2d_cmgdb_analysis.py`
