# Patrick fine-tune checkpoint screen

`screen_leslie3d_patrick_finetunes.py` is a fast selection step for the three
warm-start repeats. It is intended to decide which saved checkpoint should get
the expensive CMGDB computation. It does not construct a multivalued map,
isolating neighborhoods, a Morse graph, or Conley indices.

## Run after all three training cells finish

From `code/`:

```bash
../.venv/bin/python scripts/screen_leslie3d_patrick_finetunes.py \
  --include-baseline \
  --output output/leslie3d_example2_patrick_finetune_4x/topology_screen.json
```

The command refuses to overwrite an existing report. Omit `--output` to print
the JSON to standard output. It also hashes every config, data, scaler,
checkpoint, and training-summary input before and after analysis, so a training
cell that is still writing cannot silently enter the comparison.

## Fixed comparison contract

Every checkpoint is evaluated on all 160,000 transitions in the 4x validation
CSV (8,000 independently sampled initial conditions and 20 transitions each).
The screen transforms those points with Patrick's archived, read-only scaler;
it never fits or writes a scaler. The validation metadata must match the config
parameter vector, trajectory length, sample count, and validation seed before
the checkpoints are loaded.

For completed fine-tune cells, the report preserves `initial_val` and treats
`selected_val` in `training_summary.json` as the authoritative loss of the
saved checkpoint. This matters when no epoch improves on Patrick's initial
weights: `best_epoch` is then `-1`, and per-epoch history is not the selected
model. The screen independently recomputes the checkpoint's weighted loss terms
over the complete holdout. Both calculations are sample-weighted and should
agree to floating-point accumulation error; a material discrepancy indicates a
checkpoint, data, scaler, or loss-definition mismatch.

## Numerical recurrent-structure probes

The forward-orbit probe encodes 4,096 fixed-validation initial conditions,
iterates the latent map for 600 burn-in steps, tests tail closure at periods
1 through 8, and clusters phase-invariant cycle signatures. A cluster needs at
least 0.5% of all probed initial conditions to count in the ranking. This is
evidence for attracting recurrent behavior, not a complete inventory.

The root probe searches a 13-by-13 grid, augmented by the forward-orbit cycle
representatives, for zeros of `G^p(z)-z` at return periods 1, 2, and 4. It
deduplicates roots by primitive cycle and reports eigenvalues of `D(G^p)`.
Those eigenvalues support numerical labels such as attractor, saddle, or
repeller. The bounded multi-start search can miss roots, and these local labels
do not prove isolation or determine a Conley index.

## Ranking rule

Candidates are ordered lexicographically:

1. agreement with the direct map's two attracting period-4 cycles, with no
   additional supported attracting periods;
2. local-root evidence for two period-4 attractors, a period-2 saddle, and a
   period-1 saddle;
3. fewer forward orbits without a numerically closed tail; and
4. lower total loss on the fixed holdout.

This deliberately prevents a small validation-loss improvement from outranking
an obviously worse recurrent pattern. The ranking is still only a checkpoint
selection heuristic. The selected checkpoint must be passed through the
unchanged CMGDB/Conley pipeline before making a topological claim.
