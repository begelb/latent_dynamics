# fig_coral_adaptive

Paper Fig. 1.528: how adaptive sampling on top of a 500-point base improves Morse-set resolution as the augmenting-sample count `M` grows. Sweeps `M ∈ {100, 200, 300, 400, 500}` with 30 seeds each.

## Paper figures

- `paper/figures/morse_metric_plot_adaptive.pdf`
- `paper/figures/histogram_train_500_300_adaptive.pdf`
- `paper/figures/morse_sets_1D_after_adaptive_sampling.pdf`

## Source of paper run

Brittany. Preserved partially under `code/output/coral/train_500_<M>_adaptive/seed_*/`. Sizes `M ∈ {100, 200, 300}` have non-empty `MG/morse_sets` CSVs across most seeds; sizes `400, 500` are essentially empty.

- training script:    `archive/brittany/main_scripts/train.py`
- adaptive sampler:   present in `archive/brittany/main_scripts/make_data.py` and the experiment shell scripts
- CMGDB script:       `archive/brittany/main_scripts/morse_graph.py`
- experiment driver:  `archive/brittany/coral_experiment_scripts/run_adaptive_experiments.sh`
- adaptive datasets:  `code/data/coral/train_500_<M>_adaptive.csv` (precomputed; the `data` stage does not re-derive these)

## Status

**partial replay-ready** for `M ∈ {100, 200, 300}`; **blocked-by-missing-data-or-checkpoints** for `M ∈ {400, 500}`.

`configs/coral_adaptive.yaml` is `read_only: true` and points at the preserved Brittany tree. `configs/scratch/coral_adaptive.yaml` provides a fresh-run path that writes to `output/scratch/coral_adaptive/`.

## Reproduction commands

Replay the working sizes:

```bash
python pipeline.py --config configs/coral_adaptive.yaml --stages render,metrics
# Will fail on the 400 and 500 sizes (empty artifacts) but succeed on 100, 200, 300.
```

Full fresh sweep on AMAREL (5 sizes × 30 seeds = 150 cells):

```bash
python pipeline.py --config configs/scratch/coral_adaptive.yaml --stages scale --skip-completed
CONFIG=configs/scratch/coral_adaptive.yaml STAGES=train,diagnose,morse EXPECTED_CELLS=150 \
  sbatch --array=0-149 --export=ALL,CONFIG,STAGES,EXPECTED_CELLS \
  slurm/pipeline_array.sbatch
```

Note: adaptive sampling assumes `data/coral/train_500_<M>_adaptive.csv` already exists. The `data` stage does **not** regenerate adaptive datasets (would require running the legacy adaptive workflow). If those CSVs are missing, regenerate with `archive/brittany/coral_experiment_scripts/run_adaptive_experiments.sh` or use only the M values whose CSV is on disk.

## Expected scientific output

Paper plot shows that adaptive augmentation drives the success rate above the static-data curve (Fig. 1.469) at smaller total budgets. The 1D Morse-set figure for an adaptive run shows tighter bracketing of the three encoded fixed points compared to a non-adaptive run with the same total samples.

## Hyperparameter audit

Identical to `coral_basic.md`, plus:

| param                       | archive value           | YAML value                                       |
|-----------------------------|-------------------------|--------------------------------------------------|
| data.sampling_method        | adaptive                | adaptive                                         |
| data.n_samples_train        | 500 (base)              | 500                                              |
| data.train_files            | 5 explicit files        | `train_500_{100,200,300,400,500}_adaptive`       |

The training, CMGDB, and per-fixed-point membership behavior all derive from `coral_basic.md`.

## Verification

```bash
# Replay the three working sizes:
python pipeline.py --config configs/coral_adaptive.yaml --stages render,metrics

# Inspect the per-size metrics.json; expect labels for a0, a1, r to be in
# three distinct Morse sets at M >= 200. At M=100, a small fraction of seeds
# may still fail to bracket all three.
```
