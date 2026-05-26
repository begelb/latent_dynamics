# fig_coral_data_scaling

Paper Fig. 1.469: how the framework's quality scales with training-set size on the coral system. Sweeps `train_N` over `N ∈ {100, 200, 500, 1000, 2000, 5000}` with 30 seeds each.

## Paper figures

- `paper/figures/morse_metric_plot.pdf`
- `paper/figures/histogram_train_500.pdf`

## Source of paper run

Brittany. Preserved partially under `code/replay_sources/coral/train_<N>/seed_*/`. The active replay mirror has a usable subset: `train_100` seeds `0, 1, 10, 11, 12, 13, 15, 17, 18, 19` and all `train_2000` seeds. The `train_200`, `train_500`, `train_1000`, and `train_5000` cells currently have zero-byte checkpoints and/or Morse graph artifacts.

- training script:    `archive/brittany/main_scripts/train.py`
- CMGDB script:       `archive/brittany/main_scripts/morse_graph.py`
- experiment driver:  `archive/brittany/coral_experiment_scripts/run_coral_experiments.sh`
- metric script:      `archive/brittany/coral_experiment_scripts/compute_morse_metric.py`
- plot script:        `archive/brittany/coral_experiment_scripts/plot_morse_metrics.py`

## Status

**partial read-only replay**. `configs/coral_data_scaling.yaml` points at Brittany's preserved tree and should not be used for full recomputation. The current mirror can replay the usable `train_100` subset and all `train_2000` seeds; the other sizes require source-artifact recovery or a deliberate fresh recompute in a writable copy.

## Reproduction commands

Replay one known-good cell (`train_2000`, seed 0):

```bash
python pipeline.py --config configs/coral_data_scaling.yaml --stages render,metrics --cell-index 120 --expected-cells 180
```

Replay all currently usable cells by selecting only the non-empty source
artifact cells. A full unfiltered run will fail when it reaches zero-byte
artifact cells. Fresh recomputation requires a writable YAML copy with
`paths.output_dir` outside `replay_sources/` and `paths.read_only: false`.

## Expected scientific output

Paper plot shows the rate at which 30 seeds successfully resolve all three Morse sets (`a₀`, `a₁`, `r`) increases with `N`. At `N=100, 200` the sweep is unreliable; by `N=2000-5000` essentially all seeds succeed.

Aggregate: a `data_scaling_success_rates.json` summarising per-N and per-fixed-point success rates is written by the `metrics` stage.

## Hyperparameter audit

Identical to `coral_basic.md` except `data.n_samples_train: [100, 200, 500, 1000, 2000, 5000]` and `data.sampling_method: sobol`. The CMGDB settings now match Brittany's archived values: `subdiv_init=8`, `subdiv_min=8`, `subdiv_max=12`.

## Verification

After replaying a usable cell:

```bash
python pipeline.py --config configs/coral_data_scaling.yaml --stages metrics --cell-index 120 --expected-cells 180
# Aggregate metrics.json (one per cell) should let us reproduce the
# paper's success-rate curve: per-N count of seeds whose three encoded
# fixed points (a0, a1, r) fall in three distinct Morse sets.
```

The complete success-rate curve remains incomplete until the zero-byte
`train_200`, `train_500`, `train_1000`, and `train_5000` source artifacts are
recovered or recomputed.
