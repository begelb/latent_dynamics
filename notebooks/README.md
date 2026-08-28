# Paper notebooks

One notebook per paper example, a CMGDB primer, and the direct Leslie
baselines. Each notebook is a thin driver over the `latentdynamics` package
and runs end-to-end on a fresh Google Colab kernel: the install cell installs
CMGDB from PyPI, clones this repository, and installs the package, so the
released checkpoints and configs are available.

| Notebook | Paper section | System | Colab |
|---|---|---|---|
| [00_cmgdb_intro.ipynb](00_cmgdb_intro.ipynb) | -- | CMGDB primer (planar Leslie, no autoencoder) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/begelb/latent_dynamics/blob/paper/notebooks/00_cmgdb_intro.ipynb) |
| [01_leslie_baselines.ipynb](01_leslie_baselines.ipynb) | 4.1, 4.2 | Direct 2-D and 3-D Leslie references | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/begelb/latent_dynamics/blob/paper/notebooks/01_leslie_baselines.ipynb) |
| [02_leslie_2d_contraction.ipynb](02_leslie_2d_contraction.ipynb) | 4.1 | 2-D Leslie + contraction (10-D embedding) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/begelb/latent_dynamics/blob/paper/notebooks/02_leslie_2d_contraction.ipynb) |
| [03_leslie3d_example1.ipynb](03_leslie3d_example1.ipynb) | 4.2 | 3-D Leslie -- spurious attractor | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/begelb/latent_dynamics/blob/paper/notebooks/03_leslie3d_example1.ipynb) |
| [04_chafee_infante.ipynb](04_chafee_infante.ipynb) | 4.4 | Chafee-Infante PDE (64-D) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/begelb/latent_dynamics/blob/paper/notebooks/04_chafee_infante.ipynb) |
| [05_coral.ipynb](05_coral.ipynb) | 4.3 | Red coral (13-D) population | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/begelb/latent_dynamics/blob/paper/notebooks/05_coral.ipynb) |

## The three modes

Notebooks 02-05 run in sections, following the order of the method: the
system, the autoencoder, the data, training, CMGDB, figures. Each section
defines its parameters as plain variables seeded with the paper's values;
whatever you change is checked against the paper's configuration and any
difference is reported.

| `MODE` | what it does | typical cost |
|--------|--------------|--------------|
| `"quick"` | recompute the Morse graph of the *saved* model on the coarse `QUICK_SUBDIV` grid | seconds |
| `"morse"` | recompute the Morse graph of the *saved* model at your `SUBDIV` (paper value by default) | minutes |
| `"retrain"` | train a fresh model, then compute its Morse graph at your `SUBDIV` | minutes-hours (GPU recommended) |

The saved networks that `quick` and `morse` recompute from ship *inside the
repository* (`artifacts/reference_models/`, a few hundred kilobytes of
weights, scalers, and recorded latent bounds), so both modes work from a bare
clone -- Colab included -- with no artifact download. The released replay
bundles carry only the heavyweight saved artifacts (Morse sets, figures,
data pairs).

- `QUICK_SUBDIV` / `SUBDIV` -- CMGDB subdivision triples `(init, min, max)`.
  **Coarse grids can merge nearby recurrent sets and change the Morse
  graph**, so `quick` runs are previews, not paper-quality results.
- `SEED`, and for coral `TRAIN_FILE`, select which run to load or train.
- The Morse graph and Morse sets are drawn directly with CMGDB's own
  plotting (`PlotMorseGraph`, `PlotMorseSets`, `PlotMorseSets1D`), one
  figure per cell, from the live objects of the recompute.
- Chafee-Infante adds `COMPUTE_ROA`: exact basins from the map graph, at the
  cost of a second pass over the phase space.

The baselines notebook (01) has no modes: its two direct computations take a
subdivision triple each, defaulting to quick previews with the published
reference values noted beside them.

## Where output goes

Nothing a notebook does ever touches the preserved paper trees
(`replay_sources/`, `paper_figures/`):

- `quick`/`morse` recomputes write to
  `output/notebooks/<experiment>/morse_<i>-<m>-<x>/`.
- `retrain` writes a self-contained run under
  `output/notebooks/<experiment>/retrain_<timestamp>/`.
- Rendered figures land beside the artifacts they were drawn from.
