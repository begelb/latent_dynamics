# Replay notebooks

Each notebook re-renders one paper example's figures from **saved artifacts** —
no training and no CMGDB recompute. They are thin wrappers over
`latentdynamics.replay.load_experiment`, so the cells stay short and readable;
add your own analysis on top as needed.

## Running

From the repository root (`code/`), using the prebuilt venv at the project root:

```bash
# interactive
../.venv/bin/jupyter lab

# or execute one headless (writes outputs back into the notebook)
../.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/leslie_contraction.ipynb
```

They also run as-is in VS Code / Jupyter when the project venv is selected.
Rendered figures are written to `notebooks/rendered/<experiment>/`.

## The notebooks

| Notebook | Paper | Reproduces | Notes |
|---|---|---|---|
| `leslie_contraction.ipynb` | §5.2 — 10D embedded Leslie | bistable Morse graph + Morse sets | Patrick's archived paper run |
| `leslie3d.ipynb` | §5.3 — 3D Leslie | spurious vs. correct regime; latent-trajectory overlay | the spurious orbit never settles — the failure the example exposes |
| `chafee_infante.ipynb` | §5.4 — Chafee–Infante PDE | multistable Morse graph + Morse sets | a fresh local retrain, not pinned to Marcio's exact paper run |
| `coral.ipynb` | §5.5 — red coral | Morse graph + 1D Morse sets + §5.5.1 success metric; §5.5.2 population histograms | the Morse-graph cell uses a local `train_500` retrain (the preserved `train_500` checkpoints are 0-byte) |

See [`../docs/REPRODUCTION_GAPS.md`](../docs/REPRODUCTION_GAPS.md) for the full
per-figure status (what is replay-ready, partial, or blocked, and how to close
each gap).

## Exploring other runs

`load_experiment` accepts any config name; sweep configs (coral) also take a
`train_file` and `seed`:

```python
from latentdynamics.replay import load_experiment, available_experiments

available_experiments()                       # list config names
exp = load_experiment("coral_data_scaling", train_file="train_2000", seed=3)
exp.show_morse_graph()
exp.show_morse_sets()
exp.diagnostics()
```

`ReplayExperiment` also exposes `encode()` / `advance()` (latent dynamics),
`show_latent_trajectory()`, and `render_morse()` (returns figure paths without
displaying). Pass `output_dir=...` to load a run from a tree other than the
config's default.
