# Leslie3D Example 2: Patrick checkpoint 4x-data fine-tune

Date: 2026-08-03  
Status: complete (three fine-tunes, numerical recurrent screen, and full-resolution CMGDB on seed 2)  
Scope: isolated exploratory experiment; Patrick's checkpoint and scaler are read-only inputs

## Result in one sentence

Warm-starting Patrick's model on four times as many initial conditions cuts
held-out total loss by 53.7--55.2% and semiconjugacy loss by 85.6--86.8%, but
all three continuations change the recurrent dynamics: the best-fit seed has
one period-four and one period-eight attracting cycle, so this is a fit
improvement but **not** a recovery of the direct Leslie map's topology.

## Question and controlled design

This experiment asks whether Patrick's archived five-node Leslie3D Example 2
network can be improved by continuing optimization on more uniformly sampled
trajectories. It is a **weight warm start**, not a training resume: all encoder,
latent-map, and decoder weights are loaded, while Adam and
`ReduceLROnPlateau` start with fresh state.

The config is
`src/latentdynamics/configs/leslie3d_example2_patrick_finetune_4x.yaml`.
It reuses the existing controlled 4x dataset from the earlier from-scratch
comparison:

| split | initial conditions | iterations | one-step pairs | seed |
|---|---:|---:|---:|---:|
| train | 32,000 | 20 | 640,000 | 42 |
| validation | 8,000 | 20 | 160,000 | 9999 |

The raw data live under `data/leslie3d_example2_large_data_4x/`. The data stage
validates their metadata against the configured Leslie map and keeps them
unchanged. Reusing the same data makes the warm-start result directly
comparable to the completed 4x from-scratch seed-2 run; initialization is the
controlled difference.

## Frozen source artifacts

Patrick's original raw training CSVs were not archived. His archived scaler
*was* preserved, and the fine-tune must use it because the checkpoint weights
expect precisely those input coordinates. Refitting a nearly identical scaler
on the 4x data would change the coordinate system and confound weight
continuation with an input transformation.

| source | SHA-256 |
|---|---|
| `models/encoder.pt` | `e581773b1ab0dfdb1002ffc1542331b71398b4c7cb37e323c653f47c4fb67255` |
| `models/dynamics.pt` | `b062ae69cd855f3ff304a46a3532b45048f9628f5990032df400831821d92d60` |
| `models/decoder.pt` | `855a1eee3bfa6f57935cd58b9241725c70eca04ef8c1aadee267b04fbff0b57f` |
| `data/scalers/scaler` | `bb908b946d259fd6aa6a716cc003f789631e21bc7c9aa0a6a64c09ac629aa5e1` |

All sources are under `replay_sources/leslie3d_example2/`. The config sets
`paths.scaler_read_only: true`; both the pipeline and the scaler stage fail
closed if `scale` is requested, even with `--force-overwrite`. The training
stage also rejects a warm-start source that resolves to its own output model
directory. Every new training summary records the source checkpoint hashes
and explicitly records that optimizer and scheduler state were not restored.

## Fine-tuning settings

The architecture, loss weights, batch size, and Leslie parameters match
Patrick's reconstructed experiment. The continuation changes the learning
rate, data volume, epoch cap, patience, and stochastic minibatch ordering:

| setting | value |
|---|---|
| warm-start source | `replay_sources/leslie3d_example2/models` |
| learning rate | `1e-4` (one tenth of from-scratch setting) |
| batch size | 1,024 |
| maximum epochs | 300 |
| early-stop patience | 50 |
| LR patience / factor | 10 / 0.1 |
| loss weights | `(100, 10, 20)` |
| gradient clipping | 1.0 |
| shuffle / stochastic seeds | 0, 1, 2 |
| output root | `output/leslie3d_example2_patrick_finetune_4x/` |

All three cells start from byte-identical Patrick weights. The seed controls
minibatch order and subsequent stochastic operations; it is not a random
weight initialization seed in this experiment.

Before the first optimizer update, each cell evaluates the loaded checkpoint
on the complete 160,000-pair validation split. The component losses are saved
as `training_summary.json["initial_val"]`. This provides the correct within-cell
baseline for deciding whether continued training improves fit.

The pre-fit checkpoint is also eligible for best-validation restoration. If
no optimizer update beats it, the saved model remains byte-equivalent in
weights to the loaded warm start and the summary records
`best_epoch: -1`, `best_source: warm_start_initial`, and the baseline loss
breakdown in `selected_val`. If an update wins, `best_source` is
`training_epoch` and `best_epoch` uses the usual zero-based epoch index.

## Commands

Run from `code/`. Validate/reuse the dataset once:

```bash
../.venv/bin/python pipeline.py \
  --config leslie3d_example2_patrick_finetune_4x \
  --stages data \
  --cell-index 0 \
  --expected-cells 3 \
  --device cpu
```

Then run the independent training cells. On separate accelerators or hosts,
these commands can run concurrently:

```bash
../.venv/bin/python pipeline.py \
  --config leslie3d_example2_patrick_finetune_4x \
  --stages train,diagnose \
  --cell-index 0 \
  --expected-cells 3 \
  --device mps

../.venv/bin/python pipeline.py \
  --config leslie3d_example2_patrick_finetune_4x \
  --stages train,diagnose \
  --cell-index 1 \
  --expected-cells 3 \
  --device mps

../.venv/bin/python pipeline.py \
  --config leslie3d_example2_patrick_finetune_4x \
  --stages train,diagnose \
  --cell-index 2 \
  --expected-cells 3 \
  --device mps
```

On one Apple GPU, run the three cells sequentially. Concurrent MPS training
processes contend for the same accelerator and do not provide three-way
hardware parallelism.

After comparing validation improvements and diagnostics, run CMGDB only for
the promising cell or cells. Replace `K` with `0`, `1`, or `2`:

```bash
CMGDB_MAPGRAPH_MAX_EDGES=1200000000 \
CMGDB_MAPGRAPH_MAX_VERTICES=40000000 \
../.venv/bin/python pipeline.py \
  --config leslie3d_example2_patrick_finetune_4x \
  --stages morse,render,metrics \
  --cell-index K \
  --expected-cells 3 \
  --device mps
```

Do not add `scale` to any command for this experiment.

## Resource estimate and sequencing

The earlier 4x from-scratch run processed the same 640,000-pair training set
for 175 epochs in 14.4 minutes on MPS. Linear extrapolation gives an upper
estimate of about 25 minutes per cell if all 300 epochs run; early stopping may
finish sooner. The dataset is already present, so no additional 52 MiB data
generation is required. Each model checkpoint is small (about 70 KiB in the
legacy source; the unified output is of the same order).

The published `25/28/29` adaptive CMGDB computation is the expensive phase.
Its finest 2-D corner lattice contains 1,073,807,361 points; a two-coordinate
float64 lookup table alone is about 16 GiB before graph and temporary storage.
The previous 4x run took about 16 minutes for CMGDB with raised allocation
guards. Multiple full-resolution CMGDB jobs should not run concurrently on a
single workstation. Train/diagnose first, then compute topology selectively.

## Decision criteria

Validation loss alone is not a topology selector. A useful outcome must be
judged against Patrick's loaded baseline, the 4x from-scratch run, and the
direct-map numerical baseline:

1. The best held-out total and semiconjugacy losses should improve relative to
   that cell's `initial_val` values.
2. CMGDB should retain exactly two minimal period-four sinks with index
   `(x^4-1,0,0)` and introduce no extra sink.
3. The nonminimal roles should move toward the direct-map indices: the
   period-two saddle toward `(0,x^2+1,0)`, the positive fixed saddle toward
   `(0,x+1,0)`, and the period-four saddle remain `(0,x^4-1,0)`.
4. The graph order should be role-aligned before comparing integer node IDs.
5. Sampled semiconjugacy residuals should be reported beside tolerance; a
   smaller ordinary validation loss does not establish the paper's tolerance
   hypothesis.

The zero-update Patrick checkpoint remains the control. If all continuations
worsen the Conley-Morse structure, that is evidence that uniform-data
fine-tuning does not address the locally underconstrained saddle dynamics,
even if it reduces average held-out loss.

## Completed preflight checks

- A CPU, read-only pass over all 160,000 held-out pairs (no optimizer update)
  gives Patrick's loaded checkpoint the following expected baseline on this
  dataset/scaler combination: total `0.0669690123`, reconstruction
  `0.0005478170`, prediction `0.0006725218`, and semiconjugacy
  `0.0002731046`. Each executed cell will recompute and persist its own values;
  small backend-dependent floating-point differences are possible.
- The config validates and expands to exactly three isolated output cells.
- The existing 4x train/validation metadata match the configured Leslie map,
  sample counts, iteration count, and sampling seeds.
- Patrick's three-file legacy checkpoint loads into the configured unified
  13,575-parameter architecture.
- Source checkpoint and scaler hashes match the values above.
- A protected-scaler write attempt fails before opening the scaler.
- A warm-start smoke test loads Patrick's model, records pre-fit validation,
  saves only under a temporary output, and leaves every source hash unchanged.
- Focused config, pipeline, warm-start, and existing experiment tests pass.

## Executed training results

The three cells were run concurrently on CPU so an unrelated MPS sweep could
continue.  Every cell selected an optimizer-updated checkpoint rather than the
registered Patrick baseline.

| shuffle seed | best epoch | epochs run | minutes | selected total | change from Patrick | selected semiconjugacy | change from Patrick |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 187 | 238 | 13.1040 | `0.0310400321` | -53.65% | `3.931331e-5` | -85.61% |
| 1 | 187 | 238 | 13.1422 | `0.0307035963` | -54.15% | `3.876723e-5` | -85.80% |
| 2 | 299 | 300 | 15.9802 | `0.0299876725` | -55.22% | `3.604101e-5` | -86.80% |

All three `diagnose.json` files report `ok`: no encoder collapse and no
latent-map over-contraction.  Seed 2 also improves on the earlier 4x
from-scratch run by 15.21% in total loss and 35.77% in semiconjugacy loss.
Its best epoch is the final allowed epoch, so additional optimization might
reduce ordinary validation loss further; the recurrent checks below show why
loss alone is not a reason to continue it.

The source checkpoint and scaler SHA-256 values remained exactly equal to the
frozen values above after all three jobs.

## Numerical recurrent screen

The non-rigorous screening report is
`output/leslie3d_example2_patrick_finetune_4x/topology_screen_p8.json`.  It
recomputes all 160,000 holdout losses, iterates 4,096 encoded validation
initial conditions, and searches local periodic roots.  Patrick's unmodified
checkpoint sends every probed orbit to one of two supported period-four
cycles.  The continuations instead show:

| model | supported forward-orbit basins | unclassified after 600 steps | numerical saddle evidence |
|---|---|---:|---|
| Patrick baseline | period 4: 66.02%, period 4: 33.62% | 0% | period-1 repeller; period-4 saddles |
| seed 0 | period 8: 59.94% | 40.06% | period-1 and period-2 saddles |
| seed 1 | period 8: 54.00%, period 4: 1.39% | 44.60% | period-1 and period-2 saddles |
| seed 2 | period 8: 57.81%, period 4: 42.19% | 0% | period-1 and period-2 saddles |

A separate return-map check iterated each putative period-eight orbit for
5,000 steps in both float32 and float64.  The float64 eight-step return errors
are at machine precision while the four-step errors remain `0.01681`,
`0.01176`, and `0.00963` for seeds 0, 1, and 2.  The respective eigenvalue
moduli of `D(G^8)` are `(0.04585,0.04585)`, `(0.000177,0.39926)`, and
`(0.16385,0.77972)`.  These are genuine numerically attracting period-eight
cycles, not float32 roundoff or slowly converging period-four cycles.

Fine-tuning therefore improves the local period-1/period-2 saddle picture but
period-doubles one of Patrick's desired sinks.  Seed 2 is the least-bad
continuation because it retains one period-four basin and classifies every
probed orbit; it was selected for the full computation.

## Full-resolution CMGDB result for seed 2

The unchanged published subdivision ladder `25/28/29` was run with the normal
allocation guards.  CMGDB took 20.7958 minutes and returned four Morse sets:

| node | minimal | Conley index | boxes | numerical objects located inside its saved boxes |
|---:|:---:|---|---:|---|
| 0 | yes | `(x^4-1,0,0)` | 44,325 | the attracting period-eight cycle (all eight phases) |
| 1 | yes | `(x^4-1,0,0)` | 93,991 | the attracting period-four cycle |
| 2 | no | `(0,x^3+x^2+x+1,0)` | 14,113 | period-1 saddle, period-2 saddle, and one period-4 saddle root |
| 3 | no | `(0,x^4-1,0)` | 11,596 | a period-four saddle root |

The transitive-reduction edges are

```text
3 -> 0
3 -> 2
2 -> 1
```

Both minimal CMGDB labels are superficially the desired `(x^4-1,0,0)`, but
node 0 is not evidence for a period-four learned sink: direct inclusion tests
place every phase of the stable period-eight orbit in node 0's saved boxes.
The most plausible interpretation is that the outer approximation connects
the nearby phase pairs into four index components.  This distinction is why
the point-dynamics screen and saved-box localization must accompany the
printed Conley polynomial.

The graph is internally consistent, but the sampled lift/tolerance condition
still fails:

| node | tau-bar | max sampled residual | residual / tau | samples |
|---:|---:|---:|---:|---:|
| 0 | `3.39879e-5` | `0.00405680` | 119.36x | 486 |
| 1 | `3.41925e-5` | `0.0395221` | 1,155.87x | 60 |

Thus the fine-tune reduces average semiconjugacy error and the sampled failure
ratio for one attractor, but it does not satisfy the theorem's condition.

Primary completed artifacts:

- selected checkpoint: `output/leslie3d_example2_patrick_finetune_4x/seed_2/models/autoencoder.pt`
- screen: `output/leslie3d_example2_patrick_finetune_4x/topology_screen_p8.json`
- graph: `output/leslie3d_example2_patrick_finetune_4x/seed_2/MG/morse_graph`
- saved boxes: `output/leslie3d_example2_patrick_finetune_4x/seed_2/MG/morse_sets`
- metrics: `output/leslie3d_example2_patrick_finetune_4x/seed_2/metrics.json`

SHA-256: checkpoint `7921b925...fe5c`, graph `225622fb...03fd`, saved
boxes `aa3e558f...e2df9`, and metrics `85581b0a...03f`.

## Interpretation and next experiment

This run rules out "not enough uniform training data" as a sufficient
explanation.  More data plus ordinary loss fine-tuning makes the average fit
much better, yet crosses a period-doubling bifurcation and merges several
saddle roles in the combinatorial graph.  The useful next experiment is a
topology-aware continuation: a lower learning rate with milestone
checkpoints, recurrent-cycle screening during training, and targeted samples
near the direct period-1/period-2 saddles and the two period-four basins.  That
would let model selection stop before the period-four multiplier crosses the
unit circle, rather than choosing solely by global validation MSE.
