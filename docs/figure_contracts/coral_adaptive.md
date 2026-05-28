# fig_coral_adaptive

Paper Fig. 1.528: how adaptive sampling on top of a 500-point base improves Morse-set resolution as the augmenting-sample count `M` grows. Sweeps `M ∈ {100, 200, 300, 400, 500}` with 30 seeds each.

## Paper figures

- `paper/figures/morse_metric_plot_adaptive.pdf`
- `paper/figures/histogram_train_500_300_adaptive.pdf`
- `paper/figures/morse_sets_1D_after_adaptive_sampling.pdf`

## Source of paper run

The preserved replay tree under `code/replay_sources/coral/train_500_<M>_adaptive/seed_*/`. Sizes `M ∈ {100, 200, 300}` have non-empty checkpoints and Morse graph artifacts for all seeds; sizes `400, 500` are fresh-reproducible from a writable config copy.

- training script:    `archive/brittany/main_scripts/train.py`
- adaptive sampler:   present in `archive/brittany/main_scripts/make_data.py` and the experiment shell scripts
- CMGDB script:       `archive/brittany/main_scripts/morse_graph.py`
- experiment driver:  `archive/brittany/coral_experiment_scripts/run_adaptive_experiments.sh`
- adaptive datasets:  `code/replay_sources/coral/data/train_500_<M>_adaptive.csv` (precomputed; the `data` stage does not re-derive these)

## Status

**Read-only replay** for `M ∈ {100, 200, 300}`; `M ∈ {400, 500}` are fresh-reproducible from a writable config copy.

`configs/coral_adaptive.yaml` is `read_only: true` and points at the preserved replay tree. Fresh recomputation requires a writable YAML copy with `paths.output_dir` outside `replay_sources/` and `paths.read_only: false`.

## Reproduction commands

Replay one known-good cell (`M=300`, seed 0):

```bash
python pipeline.py --config configs/coral_adaptive.yaml --stages render,metrics --cell-index 60 --expected-cells 150
```

An unfiltered replay processes M = 100, 200, and 300 from the preserved tree;
M = 400 and 500 are fresh-reproducible from a writable config copy. Select the
replayable cell indices when generating replay metrics.

Note: adaptive sampling assumes `replay_sources/coral/data/train_500_<M>_adaptive.csv` already exists. The `data` stage does **not** regenerate adaptive datasets (this uses the adaptive sampling workflow). To regenerate them, run `archive/brittany/coral_experiment_scripts/run_adaptive_experiments.sh`, or use the M values whose CSV is on disk.

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
`coral_basic.md`. The CMGDB settings match the archived values:
`subdiv_init=8`, `subdiv_min=8`, `subdiv_max=12`.

## Verification

```bash
# Replay a known-good M=300 cell:
python pipeline.py --config configs/coral_adaptive.yaml --stages render,metrics --cell-index 60 --expected-cells 150

# Inspect the per-size metrics.json; expect labels for a0, a1, r to be in
# three distinct Morse sets at M >= 200. At M=100, a small fraction of seeds
# may still fail to bracket all three.
```
