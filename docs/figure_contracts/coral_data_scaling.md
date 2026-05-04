# fig_coral_data_scaling

Paper Fig. 1.469: how the framework's quality scales with training-set size on the coral system. Sweeps `train_N` over `N ∈ {100, 200, 500, 1000, 2000, 5000}` with 30 seeds each.

## Paper figures

- `paper/figures/morse_metric_plot.pdf`
- `paper/figures/histogram_train_500.pdf`

## Source of paper run

Brittany. Preserved partially under `code/output/coral/train_<N>/seed_*/`. Same incomplete-upload state as `coral_basic`: per-seed checkpoints and `MG/morse_graph` files are 0 bytes. Some `MG/morse_sets` CSV files are non-empty.

- training script:    `archive/brittany/main_scripts/train.py`
- CMGDB script:       `archive/brittany/main_scripts/morse_graph.py`
- experiment driver:  `archive/brittany/coral_experiment_scripts/run_coral_experiments.sh`
- metric script:      `archive/brittany/coral_experiment_scripts/compute_morse_metric.py`
- plot script:        `archive/brittany/coral_experiment_scripts/plot_morse_metrics.py`

## Status

**blocked-by-empty-checkpoints** for replay; **scratch-only** for fresh runs.

## Reproduction commands

180-cell sweep on AMAREL (6 sizes × 30 seeds):

```bash
python pipeline.py --config configs/scratch/coral_data_scaling.yaml --stages data,scale --skip-completed
CONFIG=configs/scratch/coral_data_scaling.yaml STAGES=train,diagnose,morse EXPECTED_CELLS=180 \
  sbatch --array=0-179 --export=ALL,CONFIG,STAGES,EXPECTED_CELLS \
  slurm/pipeline_array.sbatch

# After the array completes:
python pipeline.py --config configs/scratch/coral_data_scaling.yaml --stages render,metrics --skip-completed
```

Local one-cell smoke check (no full sweep):

```bash
python pipeline.py --config configs/scratch/coral_data_scaling.yaml --stages all --cell-index 0 --expected-cells 180
```

## Expected scientific output

Paper plot shows the rate at which 30 seeds successfully resolve all three Morse sets (`a₀`, `a₁`, `r`) increases with `N`. At `N=100, 200` the sweep is unreliable; by `N=2000-5000` essentially all seeds succeed.

Aggregate: a `data_scaling_success_rates.json` summarising per-N and per-fixed-point success rates is written by the `metrics` stage.

## Hyperparameter audit

Identical to `coral_basic.md` except `data.n_samples_train: [100, 200, 500, 1000, 2000, 5000]` and `data.sampling_method: sobol`.

Drift: same as `coral_basic` — `cmgdb.subdiv_max` defaults to 10 in our YAML vs 12 in the archive `mg_params_log.txt` files.

## Verification

After full sweep:

```bash
python pipeline.py --config configs/scratch/coral_data_scaling.yaml --stages metrics
# Aggregate metrics.json (one per cell) should let us reproduce the
# paper's success-rate curve: per-N count of seeds whose three encoded
# fixed points (a0, a1, r) fall in three distinct Morse sets.
```

If the success rate at the smaller `N` values disagrees with the paper, investigate the diagnose-stage outputs first (`frac_unconverged`) before re-tuning training; the paper's success rates implicitly include training failures, not just CMGDB-config drift.
