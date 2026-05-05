# Paper-vs-output figure parity (snapshot 2026-05-03)

Side-by-side comparison of `paper/figures/` and the rendered Morse graphs in
`code/output/<experiment>/seed_0/MG/`. Conducted after the modular-pipeline
refactor and before Patrick's `Leslie10D` and non-spurious `Leslie3D` paper
artifacts were restored under `archive/patrick/`.

This is a historical diagnostic snapshot, not the current source-of-truth for
paper provenance. Use `docs/PAPER_REPRODUCTION.md` and the files under
`docs/figure_contracts/` for the archive-backed contracts. The mismatch rows
below are still useful as warnings: fresh retraining can converge in loss while
missing the paper Morse graph, so paper claims must be checked through saved
artifacts, CMGDB output, and validation metrics.

## Comparison

| Experiment           | Paper file                                          | Paper Morse sets | Our output (seed_0)                                                | Match |
|----------------------|-----------------------------------------------------|------------------|--------------------------------------------------------------------|-------|
| leslie3d_spurious    | `morse_graph_new_colors.pdf` + `latent_trajectory.png` | 6 (chain `5→3→1`, `5→4`, `2→{0,1}`, `3→1`)                                 | 6, identical Hasse edges                                            | exact |
| leslie3d_success     | `Leslie_3D_10KData_Mgraph.png`                      | 6 (chain `5→4→3→2→{0,1}`, indices `(0,0,0), (0,x+1,0), (0,x²+1,0), (0,x⁴-1,0), (x⁴-1,0,0)×2`) | 1 trivial sink `(x-1,0,0)`                                          | far off |
| 10D Embedded Leslie (`leslie_contraction`) | `LeslieContraction_10D_10kData_Mgraph.png` | 4 (`3→{2,1,0}`, indices `(0,x³-1,0), (0,0,x-1), (x-1,x-1,0), (x³-1,0,0)`) | 1 trivial sink `(0,0,0)`                                            | far off |
| chafee_infante       | `ci_morse_graph.pdf`                                | 7 (rich Hasse: 2 attractors `(x-1,0,0)`, 3 saddles `(0,x-1,0)`, 2 sources `(0,0,x-1)`) | 1 trivial sink `(x-1,0,0)`                                          | far off |
| coral_basic          | `coral_morse_graph.pdf`                             | 3 (`2→{0,1}`, indices `(0,x-1), (x-1,0), (x-1,0)`) | not directly comparable - all `output/coral/train_500/seed_*/MG/morse_graph` files are 0 bytes (incomplete uploads); the only non-empty coral artifacts live under `train_500_300_adaptive` and show 4 Morse sets including a spurious `(0,0)` node | gap |

This snapshot compares the package retrains, not the later-restored Patrick
archives. The source map now distinguishes Brittany's spurious Leslie 3D run
from Patrick's non-spurious Leslie 3D run:

- `leslie3d_spurious` reuses the legacy 3-file checkpoint at
  `output/Leslie_3D/spurious_attractor_ex/models/{encoder,dynamics,decoder}.pt`
  via `latentdynamics.training.load_legacy_checkpoint`. It reproduces the
  paper figure exactly.
- `leslie_contraction` and `leslie3d_success` have preserved Patrick paper
  artifacts under `archive/patrick/Leslie10D/` and
  `archive/patrick/Leslie3D/`; the `code/output/` rows below are fresh retrains,
  not the original paper sources.
- The three "far off" entries were retrained from scratch in the previous
  session with the then-current package defaults
  (`configs/_shared/defaults.yaml`: `lr=0.001, batch_size=1024, epochs=1000,
  patience=100, loss_mode=weighted`) plus per-experiment `loss_weights`.
  Training converges to very low loss (`leslie_contraction: 3.13e-2`,
  `leslie3d_success: 6.39e-3`, `chafee_infante: 3.39e-3`) but the latent map
  collapses to a single attracting fixed point.

## Diagnosis

Low recon loss + degenerate latent dynamics means the autoencoder is
satisfying the reconstruction term by mapping inputs to a near-constant
latent and reconstructing from a memorised mean. The dynamics term
(`loss_dyn`) is small in absolute terms but the latent map is identity-on-a-
point, which is trivially satisfied. CMGDB on this latent map then finds
exactly one Morse set.

Three concrete suspects, in order of probability:

1. **Per-experiment hyperparams missing.** Brittany's original
   `legacy/config_yaml/{coral,Leslie_3D_larger_domain_tail_only}.yaml` only
   covers two systems; `leslie3d_success`, `leslie_contraction`, and
   `chafee_infante` were never tuned in the new YAML schema. Likely the
   paper-faithful runs used larger `loss_weights[2]` (the dynamics term),
   higher `learning_rate`, larger `epochs`, or smaller `batch_size`.
2. **Patience too tight.** Default `patience=100` early-stops Chafee at
   epoch 440 of the requested 4000. The latent dynamics term may need a long
   warmup before it can shape the latent space against the
   already-optimal recon term.
3. **`encoder_out_activation: tanh` (default)** with the recon term
   weighted 10x the dynamics term lets the encoder produce near-constant
   output trivially.

## Next session - punch list

Ranked by how likely each is to actually fix the regression:

1. **Re-tune `loss_weights` per experiment.** Try `[1, 1, 10]` or
   `[1, 1, 100]` for the three failing experiments to force the
   semiconjugacy term to dominate. Verify with a single seed at half the
   epoch budget before committing.
2. **Bump `patience` to 500-1000** and `epochs` to 5000+ for chafee_infante
   specifically.
3. **Check `encoder_out_activation`.** `none` may be the right choice for
   chafee_infante (continuous flow with structurally rich phase space);
   `sigmoid` or `relu` for the Leslie systems.
4. **Inspect Brittany's original training scripts** in
   `legacy/main_scripts/train.py` and `legacy/Leslie_analysis_scripts/` for
   the exact hyperparams used in the paper runs. The new YAMLs were
   reverse-engineered, not copied.
5. **Re-upload coral_basic and coral_data_scaling artifacts.** Most coral
   `seed_*/MG/morse_graph` files in `output/coral/` are 0 bytes (599 of 993
   `.pt` files are partial uploads). Either re-sync from the cluster that
   originally produced them or retrain.
6. **Render `coral_basic` once it has artifacts** and compare to
   `coral_morse_graph.pdf`. Most likely matches once the data is there since
   coral was Brittany's main tuned config.

## What is working

- Modular pipeline: `data → scale → train → morse → render → metrics`,
  every stage loadable from disk via the on-disk artifacts of any prior
  stage. Verified with `--stages morse` and `--stages render` standalone.
- `morse` stage saves only the DOT (`MG/morse_graph`) and CSV
  (`MG/morse_sets`); the figure renders are deferred to the `render`
  stage. No matplotlib pop-ups (matplotlib `Agg` backend forced in
  `latentdynamics/viz/__init__.py`).
- MPS training fixed: the divergence we saw earlier was from
  `non_blocking=True` in `Trainer._run_epoch`, which races on CPU→MPS
  transfers. With `non_blocking=False`, MPS trains identically to CPU.
  Default device priority is now MPS > CUDA > CPU.
- `legacy/src/` keeps Brittany's old class definitions on the import path
  so the 3-file pickled checkpoints (`encoder.pt`, `dynamics.pt`,
  `decoder.pt`) are still loadable through
  `latentdynamics.training.load_legacy_checkpoint`. This is why
  `leslie3d_spurious` matches the paper exactly.
- 119 / 119 tests pass, including 3 `test_reproducibility.py` slow tests
  on CPU.

## Files of interest for next session

- `code/configs/{leslie_contraction,leslie3d_success,chafee_infante}.yaml` - tune these.
- `code/src/latentdynamics/training/trainer.py` - `_run_epoch` and `fit`.
- `code/legacy/main_scripts/train.py` - reference for original hyperparams.
- `code/legacy/Leslie_analysis_scripts/` - paper-figure scripts; may
  contain the exact CMGDB bounds and subdivision settings used.
- `code/output/<expt>/seed_0/logs/history.json` - per-epoch loss history;
  inspect to see when (if ever) `loss_dyn` starts shaping training.

## 2026-05-04 structural follow-up

Implemented before another long retrain:

- `make_data` is now non-destructive. Existing `*.csv` +
  `*_metadata.json` pairs are treated as source artifacts and are not
  overwritten; partial pairs raise instead of silently regenerating.
- Adaptive coral data is represented as precomputed input. The `data` stage
  validates the saved adaptive train files and `test.csv` rather than calling
  an unimplemented sampler.
- The neural architecture config is component-wise: encoder, latent map, and
  decoder can each set `hidden_shapes`, activation, and terminal activation.
  This captures Marcio's Chafee-Infante architecture
  `[64,32] / [32,32] / [32,64]`.
- Training now exposes scheduler factor/threshold/min-lr and optional
  gradient clipping in YAML. `data.scaling: none` adds a raw-coordinate
  identity scaler for archived workflows that did not MinMax-scale data.
- CMGDB configs can fix latent bounds and choose `padding`; Chafee-Infante now
  records the archived `[-3, -2] x [3, 2]`, `padding=false`, and
  `subdiv_init/min/max = 10/14/28` settings.

The parity status of the already saved retrained outputs above is unchanged
until those experiments are rerun from the updated configs. No saved data or
output artifacts were deleted or edited by these structural changes.
