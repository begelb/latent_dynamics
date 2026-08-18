# fig_coral_basic

Paper figure `fig:coral_latent_dynamics`: 13D coral population system, 1D latent,
bistable extinction-versus-recovery dynamics.

## Paper figures

By published file name:

- `coral_morse_graph.pdf` (panel a)
- `morse_sets_1D.pdf` (panel b: custom 1D segment plot with encoded fixed points)

## Source of paper run

The featured paper model is specifically
`replay_sources/coral/train_500/seed_16/`, as recorded in the replay-tree
README and hard-coded in the author's archived 1D figure generator. The basic
`train_500` sweep is fresh-reproducible from a writable config copy, but a new
training run is not provenance for this selected paper asset.

- parameter log: `replay_sources/coral/train_500/seed_16/mg_params_log.txt`
- checkpoint: `replay_sources/coral/train_500/seed_16/models/`
- raw CMGDB artifacts: `replay_sources/coral/train_500/seed_16/MG/{morse_graph,morse_sets}`
- paper-data metadata: `replay_sources/coral/data/coral/train_500_metadata.json`
- scaler: `replay_sources/coral/data/scalers/train_500/scaler.gz`

### Panel (a) renderer (open item)

The published `coral_morse_graph.pdf` renders exactly the shipped seed-16
Morse-graph DOT (node labels, indices, colors, and edges verified
identical), but the PDF itself was rendered by an author with an
unidentified toolchain (TimesNewRoman fonts; it does not
match any Graphviz/Cairo render on record). The pipeline replay re-renders
the identical DOT with the package styling; it will not byte-match the
published PDF.

### Panel (b) generator

`scripts/render_coral_morse_sets_1d.py` generates the published 1D band plot
from the seed-16 `morse_sets`, the encoder, and the train_500 scaler: three
Morse-set bands on the latent interval with the encoded fixed points
overlaid (`E(a0)` star, `E(a1)` triangle, `E(r)` square; no legend). The
script is faithful to the author's archived plotting script; the 13D fixed
points `a0`, `a1`, `r` are the constants asserted in
`src/latentdynamics/systems/coral.py` (`RedCoralModel.FIXED_POINTS`). How
those constants were originally derived (root-finding run or external
reference) is not recorded. The pipeline render produces the band plot
without the fixed-point overlay. Byte-exact reproduction of the published
PDF depends on the matplotlib build (font embedding); the content is fully
determined by the saved artifacts.

## Status

**Read-only artifact replay; basic `train_500` is fresh-reproducible from a
writable config copy.** `src/latentdynamics/configs/coral_basic.yaml` points at
the preserved replay tree. Its `data.sampling_method` is `sobol`, matching the
archived paper-data metadata (scrambled Sobol, scramble seed 42); an earlier
copy of the config carried a stale `uniform` value and must not be used as
data-generation provenance for the paper run.

## Reproduction commands

Replay the selected paper cell from the preserved tree:

```bash
python pipeline.py --config coral_basic --stages render,metrics \
  --cell-index 16 --expected-cells 30
```

Regenerate the published panel (b):

```bash
python scripts/render_coral_morse_sets_1d.py
```

Fresh retraining requires copying the YAML to a writable output location and
setting `paths.read_only: false`; do not write into `replay_sources/coral/`.

## Expected scientific output

The seed-16 graph has three Morse sets in 1D: two attractors at the encoded
`E(a₀)` (extinction) and `E(a₁)` (healthy steady state), separated by one
repeller node. Its edges are `2 -> 0` and `2 -> 1`, with indices `(x-1, 0)`,
`(x-1, 0)`, and `(0, x-1)`.

For the featured seed, the exact latent interval is
`[-1.0138611376285553, 0.4264032423496246]`. CMGDB uses
`subdiv_init=8`, `subdiv_min=8`, `subdiv_max=12`, `subdiv_limit=10000`, and
padding. The archived generator first expands the encoded-data interval by
`0.01 * width`; CMGDB padding is a separate setting.

In panel (b), `E(a0)` lies inside band `|π⁻¹(0)|` and `E(a1)` inside
`|π⁻¹(1)|`, while `E(r)` (≈ -0.27) lies outside all three plotted bands; the
replay metrics record labels `a0=0`, `a1=1`, and `r` unassigned.

## Hyperparameter audit

| param                       | archive value           | YAML value             | source | notes |
|-----------------------------|-------------------------|------------------------|--------|-------|
| arch.num_layers             | 3                       | 3                      | replay config | ✓ |
| arch.hidden_shape           | 64                      | 64                     | replay config | ✓ |
| arch.high_dims              | 13                      | 13                     | replay config | ✓ |
| arch.low_dims               | 1                       | 1                      | replay config | ✓ |
| arch.encoder_out_activation | tanh                    | tanh                   | replay config | ✓ |
| arch.latent_out_activation  | tanh                    | tanh                   | replay config | ✓ |
| arch.decoder_out_activation | sigmoid                 | sigmoid                | replay config | ✓ |
| training.loss_weights       | [10, 10, 1]             | [10, 10, 1]            | replay config | ✓ |
| training.epochs             | 1000                    | 1000                   | replay config | ✓ |
| training.learning_rate      | 0.001                   | 0.001                  | replay config | ✓ |
| training.batch_size         | 1024                    | 1024                   | replay config | ✓ |
| training.patience           | 100                     | 100                    | replay config | ✓ |
| data.n_samples_train        | 500                     | [500]                  | archived metadata | ✓ |
| data.n_samples_val          | 10000                   | 10000                  | replay config | not independently recorded in train metadata |
| data.n_iterations           | 20                      | 20                     | archived metadata | ✓ |
| data.sampling_method        | scrambled Sobol         | sobol                  | archived metadata | corrected from a stale `uniform` value |
| data.train_seed             | 42                      | 42                     | archived metadata | Sobol scramble seed |
| data.initial-condition box  | `[0]*13` -> `[1300,1150,750,520,270,120,35,20,7,5,5,2,2]` | not encoded in YAML | archived metadata | exact paper-data domain; matches `RedCoralModel` defaults |
| cmgdb.subdiv_init           | 8                       | 8                      | seed-16 parameter log | ✓ |
| cmgdb.subdiv_min             | 8                       | 8                      | seed-16 parameter log | ✓ |
| cmgdb.subdiv_max            | 12                      | 12                     | seed-16 parameter log | ✓ |
| cmgdb.subdiv_limit          | 10000                   | 10000                  | seed-16 parameter log | ✓ |
| cmgdb.lower_bounds          | `[-1.0138611376285553]` | inferred               | seed-16 parameter log | featured model |
| cmgdb.upper_bounds          | `[0.4264032423496246]`  | inferred               | seed-16 parameter log | featured model |
| cmgdb.bounds_epsilon_frac   | 0.01                    | 0.01                   | archived generator | applied before CMGDB padding |
| cmgdb.padding               | true                    | true                   | replay config | ✓ |

`tab:coral_data` (demographic parameters b_i, s_i) is typeset by hand; every
value matches the defaults in `src/latentdynamics/systems/coral.py` exactly.

## Residual/tolerance table rows

The `Red coral` rows of `tab:sampled_residual_tolerance` are computed on the
seed-16 model's two minimal-node blocks; see
[`sampled_residual_tolerance.md`](sampled_residual_tolerance.md).

## Verification

```bash
python pipeline.py --config coral_basic --stages render,metrics \
  --cell-index 16 --expected-cells 30
# metrics.json should assign distinct labels to E(a0) and E(a1) and leave
# E(r) unassigned; morse_graph_consistency: n_morse_sets=3, consistent=true.
```

The preserved coral replay tree carries raw graphs, raw Morse sets, parameter
logs, and checkpoints for 30 seeds at six sample sizes; only
`train_500/seed_16` is required by the current manuscript. Rendered PDFs are
not present in every cell and may be regenerated from the saved raw
artifacts.
