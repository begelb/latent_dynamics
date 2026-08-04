# Leslie3D guarded chart refinement

The accepted invariant-aware v2 run froze the encoder and decoder so its
latent rectangle and invariant landmarks stayed directly comparable with the
archived Example-2 chart. That was useful for attribution, but it is not a
claim that the inherited chart is optimal.

`scripts/train_leslie3d_alternating_chart.py` is an isolated continuation that
tests the user's stronger idea without silently changing the accepted run.
It verifies the exact primary-v2 checkpoint hash, then performs two phases:

1. **Chart surgery.** Freeze the latent map and train only the final encoder
   affine layer and first decoder affine layer. The objective combines
   symmetric reconstruction, decoded one-step prediction, frozen-map
   semiconjugacy, reference-chart trust, separation margins between distinct
   recurrent objects, a deterministic local-secant anti-fold bank, and latent
   inverse consistency. Held-out gates reject reconstruction/prediction
   regressions, excessive chart drift, lost recurrent separation, or collapsed
   local secants. The training objective retains the accepted run's 4x saddle
   tube, 3x origin-fan, and 8x audited-transition sample weights.
2. **Map repair.** Freeze the selected chart, recompute every training and
   validation latent, all 16 invariant phases, their normalization scales, and
   every multi-step trajectory target. Exact recurrent-anchor projection owns
   the latent map's final affine layer, so that layer is excluded from Adam;
   the optimizer updates only the map interior. This avoids asking Adam to
   learn directions that projection immediately removes. The first stage has
   no rollout gradient and repairs anchor closure, characteristic polynomials,
   and periodic stability roles. Rollout training starts only after those
   strict topology gates pass and at least 250 topology epochs have elapsed.

The chart phase selected epoch 488. Relative to the frozen source chart, its
held-out reconstruction and decoded-prediction ratios are `1.00172` and
`1.00121`, its frozen-map semiconjugacy ratio is `1.04979`, and the encoder
drift RMSE is `8.89e-4`. The minimum named-object separation ratio is
`1.00201`. The tightest useful margins improve modestly: `S2`--`p_star` from
`0.0344503` to `0.0348055`, `P1`--`S4` from `0.0382622` to `0.0386807`, and
`P1`--`S2` from `0.0418626` to `0.0428275`. Thus the guarded encoder update
does make the finite-resolution chart slightly easier, but it is a 1--2.3%
separation gain rather than a qualitative unfolding.

The completed 4,000-epoch safety run was rejected. Its independently selected
best epoch was 3059: periodic role violation was zero, but maximum
characteristic-polynomial relative error stalled at `0.128834` and the
untruncated held-out rollout loss was `0.0219528`. Thus rollout training never
activated. The atomic final-state bundle is
`models/map_training_state.pt`, SHA-256
`ab9ccaf72b0e9a414c767d9eeec295fd60f8efaaaf85734de4bed1b3411085af`;
it records the last epoch-3999 model, the epoch-3059 historical best, Adam and
scheduler states, the still-inactive rollout stage, history, and RNG state.

The opt-in `--continue-map-training-state` path addresses one identifiable
optimization restriction without pretending to restart the run. Hidden map
layers retain the restored Adam moments. The final affine layer is kept out of
Adam and receives a separate no-momentum update. At each epoch, with anchor
feature design `F` and final affine matrix `Theta = [W.T; b]`, an SVD builds
`null(F)` and projects every objective-term output gradient into it. The
explicit update therefore satisfies `F delta = 0` to float64 precision. The
code logs `F delta` every epoch, then re-applies the exact affine projection
after both the output update and hidden Adam step, because changing hidden
features changes `F`. This gives the output layer useful tangent directions
without stale Adam moments or sacrificing exact anchor closure.

Continuation restores the **last** model from the state bundle, not the
epoch-3059 candidate: pairing that best model with epoch-3999 Adam moments
would not be a valid optimizer continuation. The original bundle is opened
read-only and hash-checked again after training. All new logs, checkpoints,
and summaries go into a new `continuations/map_state_e..._nullspace/`
directory; an existing destination is rejected. Before the first step, the
loader also requires a full-row-rank anchor design, a finite feasible boundary
within `1e-3` preactivation residual (the real boundary is `6.67e-9`), and the
strict normalized-anchor gate. It matches train/validation, metadata,
manifest, scaler, replay weights, and the relevant training configuration to
the completed source attempt. These fingerprints are embedded directly in
format-v2 continuation states. A recursive continuation inherits a saved
fixed output-layer learning rate unless the CLI explicitly overrides it.

The completed 4,000-epoch constrained continuation was also rejected. It
resumed the last epoch-3999 state and ran absolute epochs 4000--7999 in
266.34 seconds. Its selected epoch was 6809, with replay total `0.0358600`,
held-out validation ratio `0.994251`, maximum normalized anchor error
`1.20e-6`, and zero role-margin violation. However, the maximum
characteristic-polynomial relative error was `0.149037` (the fixed gate is
`0.05`) and the exact held-out rollout loss was `0.0219452` (the fixed
absolute gate is `0.00697`). Topology recovery therefore never occurred and
rollout-gradient training never activated. The maximum actually installed
anchor-nullspace residual over all steps was `2.35e-8`.

The rejected candidate is preserved only for diagnosis at
`continuations/map_state_e4000_to_e8000_nullspace/models/alternating_candidate.pt`,
SHA-256
`fa3894fd845f013b17d042ad029f59046ef1affeb9890e0f323473e60ef6c095`.
No `autoencoder.pt` was promoted, and no new cycle census or CMGDB computation
was run for this rejected state. The result isolates the scientific issue:
the encoder can improve recurrent-set separation without degrading physical
accuracy, but neither interior-only repair nor anchor-preserving output
tangent directions found a compatible map satisfying the local-spectrum and
long-horizon gates.

The original map attempt failed at epoch 58 because the weighted 63/64-step
rollout gradient reached `7.112e24`; its elements were finite, but the standard
float32 norm overflowed. The origin-positive-cone fan dominated that direction.
The repaired optimizer uses float64 norm accumulation, independently caps each
weighted objective-term gradient, then applies the existing global norm cap.
It rejects genuinely non-finite elements rather than replacing them.

Rollout optimization now uses a conservative curriculum: horizons 1--4 for
the first 250 rollout-stage epochs, then horizons through 16. Reverse-mode
segments are truncated every eight map applications; this changes the
gradient surrogate but leaves every reported forward prediction and loss
unchanged. Horizons 31/32, 63/64, and the complete held-out inventory through
319/320 are evaluation-only. Promotion still requires the exact, untruncated
held-out 320-step loss to remain within 1.5x the post-chart, pre-projection
reference map **and** below the absolute source-audit ceiling `0.00697`.
The absolute gate is essential: the refined chart's pre-projection rollout is
`0.044879`, and exact projection lowers it to `0.023231`, so a ratio-only gate
would reward a candidate for comparison against a degraded chart baseline.

Physical one-step losses are also compared directly with the accepted primary
v2 source. Its held-out scaled reconstruction and prediction MSEs are
`0.000241393` and `0.000274302`; the projected refined-chart ratios are
`1.00192` and `0.98838`, within the existing `1.02`/`1.05` limits. Thus chart
surgery cannot trade away source-chart physical accuracy to satisfy its own
latent baseline.

The preserved chart passes a deliberately loose *recoverability* screen after
exact projection: normalized anchor error `1.09e-6`, maximum characteristic
relative error `0.7131`, role-margin violation `0.05214`, held-out replay ratio
`0.9925`, and held-out rollout ratio `0.5176`. This only authorizes a repair
attempt. It does not satisfy the strict characteristic/role promotion gates.
Every evaluation interval writes model, optimizer, scheduler, best-candidate,
history, and RNG state; the combined training-state bundle is installed by an
atomic file replacement at a valid post-projection boundary.

The downstream CMGDB rectangle must be derived again from the selected chart;
the fixed bounds used by the earlier subdivision audit are stale after any
encoder update. The dedicated config therefore leaves bounds unset and uses
the audited subdivision maximum 30 in an isolated output directory.

Read-only validation (loads the real data/checkpoint, rebuilds all caches, and
writes nothing):

```bash
.venv/bin/python scripts/train_leslie3d_alternating_chart.py --validate-only
```

The completed chart phase is preserved at the exact checkpoint SHA-256
`7f528f001f66652689aa9b3b31b8c46909084f2c82a31de739a59877679b24fc`.
There is no valid epoch-57 optimizer checkpoint, so the repaired path is a
fresh map epoch-zero attempt, not an epoch-59 continuation. The completed
map-only safety run used:

```bash
.venv/bin/python scripts/train_leslie3d_alternating_chart.py \
  --resume-chart-refined \
  --device cpu \
  --map-epochs 4000 \
  --map-learning-rate 5e-8 \
  --rollout-learning-rate 1e-8 \
  --rollout-weight 0.001 \
  --rollout-absolute-limit 0.00697 \
  --rollout-min-topology-epochs 250 \
  --rollout-short-epochs 250 \
  --rollout-medium-max-horizon 16 \
  --rollout-backprop-steps 8 \
  --spectral-start-epoch 100 \
  --spectral-ramp-epochs 2000 \
  --per-term-gradient-clip-norm 1.0
```

The completed constrained continuation from that run used (the default
output-layer rate followed the restored hidden-Adam rate and later scheduler
changes):

```bash
.venv/bin/python scripts/train_leslie3d_alternating_chart.py \
  --continue-map-training-state \
  --device cpu \
  --continuation-epochs 4000 \
  --expected-training-state-sha256 \
  ab9ccaf72b0e9a414c767d9eeec295fd60f8efaaaf85734de4bed1b3411085af
```

An explicit fixed no-momentum rate can instead be supplied with
`--output-layer-learning-rate`. This is a new optimizer direction beginning
at epoch 4000, so it is an exact *state* continuation but not a claim that the
original algorithm itself ran uninterrupted.

A one-epoch real-data smoke run in a temporary directory found design
rank/nullity `16/49`. The tangent-gradient norm was `0.985956`; the requested
and installed output-step norms were `4.92978e-8` and `4.92607e-8`. The ideal
float64 `max |F delta|` was `4.67e-22`, the actually installed float32 value
was `8.21e-9`, and exact projection left a maximum preactivation residual of
`8.80e-9`. Hidden Adam advanced from step 4000 to 4001, the scheduler advanced
from evaluation 400 to 401, and the source state hash remained unchanged.
This verifies the mechanism; one epoch is not evidence of convergence.

The scaler was serialized with scikit-learn 1.7.1 and the current environment
loads it with 1.8.0, which emits a persistence compatibility warning. The
numerical smoke test succeeded, but this remains a provenance caveat until the
environment is pinned or the scaler is regenerated and audited.

This remains a numerical experiment. A finite secant bank is not an
injectivity proof, a continuous 3-to-2 map cannot embed an open 3-D
neighborhood, and periodic-orbit supervision does not certify an index pair.
The first implementation also does not yet turn the independently found extra
period-1/2/4 roots into moving hard negatives; a fresh root census and another
map-repair round remain required before a new Morse computation is meaningful.
