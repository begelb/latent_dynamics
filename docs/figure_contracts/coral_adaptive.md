# fig_coral_adaptive

Paper Fig. 1.528: how adaptive sampling on top of a 500-point base improves Morse-set resolution as the augmenting-sample count `M` grows. Sweeps `M ∈ {100, 200, 300, 400, 500}` with 30 seeds each.

## Paper figures

- `paper/figures/morse_metric_plot_adaptive.pdf`
- `paper/figures/histogram_train_500_300_adaptive.pdf`
- `paper/figures/morse_sets_1D_after_adaptive_sampling.pdf`

## Source of paper run

Brittany. Preserved partially under `code/replay_sources/coral/train_500_<M>_adaptive/seed_*/`. Sizes `M ∈ {100, 200, 300}` have non-empty checkpoints and Morse graph artifacts for all seeds; sizes `400, 500` currently have zero-byte checkpoints and/or Morse graph artifacts.

- training script:    `archive/brittany/main_scripts/train.py`
- adaptive sampler:   present in `archive/brittany/main_scripts/make_data.py` and the experiment shell scripts
- CMGDB script:       `archive/brittany/main_scripts/morse_graph.py`
- experiment driver:  `archive/brittany/coral_experiment_scripts/run_adaptive_experiments.sh`
- adaptive datasets:  `code/replay_sources/coral/data/train_500_<M>_adaptive.csv` (precomputed; the `data` stage does not re-derive these)

## Status

**partial read-only replay** for `M ∈ {100, 200, 300}`; **blocked-by-empty-checkpoints** for `M ∈ {400, 500}`.

`configs/coral_adaptive.yaml` is `read_only: true` and points at Brittany's preserved tree. Fresh recomputation requires a writable YAML copy with `paths.output_dir` outside `replay_sources/` and `paths.read_only: false`.

## Reproduction commands

Replay one known-good cell (`M=300`, seed 0):

```bash
python pipeline.py --config configs/coral_adaptive.yaml --stages render,metrics --cell-index 60 --expected-cells 150
```

An unfiltered replay will process M = 100, 200, and 300 first, then fail when
it reaches the zero-byte M = 400 source artifacts. Select only usable cell
indices when generating partial metrics.

Note: adaptive sampling assumes `replay_sources/coral/data/train_500_<M>_adaptive.csv` already exists. The `data` stage does **not** regenerate adaptive datasets (would require running the legacy adaptive workflow). If those CSVs are missing, regenerate with `archive/brittany/coral_experiment_scripts/run_adaptive_experiments.sh` or use only the M values whose CSV is on disk.

## Expected scientific output

Paper plot shows that adaptive augmentation drives the success rate above the static-data curve (Fig. 1.469) at smaller total budgets. The 1D Morse-set figure for an adaptive run shows tighter bracketing of the three encoded fixed points compared to a non-adaptive run with the same total samples.

## Hyperparameter audit

Identical to `coral_basic.md`, plus:

| param                       | archive value           | YAML value                                       |
|-----------------------------|-------------------------|--------------------------------------------------|
| data.sampling_method        | adaptive                | adaptive                                         |
| data.n_samples_train        | 500 (base)              | 500                                              |
| data.train_files            | 5 explicit files        | `train_500_{100,200,300,400,500}_adaptive`       |

The training, CMGDB, and per-fixed-point membership behavior all derive from
`coral_basic.md`. The CMGDB settings now match Brittany's archived values:
`subdiv_init=8`, `subdiv_min=8`, `subdiv_max=12`.

## Verification

```bash
# Replay a known-good M=300 cell:
python pipeline.py --config configs/coral_adaptive.yaml --stages render,metrics --cell-index 60 --expected-cells 150

# Inspect the per-size metrics.json; expect labels for a0, a1, r to be in
# three distinct Morse sets at M >= 200. At M=100, a small fraction of seeds
# may still fail to bracket all three.
```
