# Paper notebooks

One notebook per paper example, plus a CMGDB primer. Each notebook is a thin
driver over the `latentdynamics` package and runs end-to-end on a fresh Google
Colab kernel. The install cell pulls a prebuilt wheel of the
[CMGDB fork](https://github.com/bernardorivas/CMGDB) — which is not on PyPI —
followed by the package and its dependencies, so nothing is compiled.

| Notebook | Paper § | System | Colab |
|---|---|---|---|
| [00_cmgdb_intro.ipynb](00_cmgdb_intro.ipynb) | — | CMGDB primer (planar Leslie, no autoencoder) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/begelb/latent_dynamics/blob/paper/notebooks/00_cmgdb_intro.ipynb) |
| [01_leslie_2d_contraction.ipynb](01_leslie_2d_contraction.ipynb) | 5.1 | 2-D Leslie + contraction (10-D embedding) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/begelb/latent_dynamics/blob/paper/notebooks/01_leslie_2d_contraction.ipynb) |
| [02_leslie3d_example1.ipynb](02_leslie3d_example1.ipynb) | 5.2.1 | 3-D Leslie — spurious attractor | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/begelb/latent_dynamics/blob/paper/notebooks/02_leslie3d_example1.ipynb) |
| [03_leslie3d_example2.ipynb](03_leslie3d_example2.ipynb) | 5.2.2 | 3-D Leslie — bistability | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/begelb/latent_dynamics/blob/paper/notebooks/03_leslie3d_example2.ipynb) |
| [04_chafee_infante.ipynb](04_chafee_infante.ipynb) | 5.3 | Chafee-Infante PDE (64-D) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/begelb/latent_dynamics/blob/paper/notebooks/04_chafee_infante.ipynb) |
| [05_coral.ipynb](05_coral.ipynb) | 5.4 | Red coral (13-D) population | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/begelb/latent_dynamics/blob/paper/notebooks/05_coral.ipynb) |

## The three modes

Notebooks 01–05 run in sections, following the order of the method: the system,
the autoencoder, the data, training, CMGDB, figures. Each section defines its
own parameters as plain variables, seeded with the paper's values, so exploring
means editing a number where it is explained rather than assembling an override
dict. Whatever you change is checked against the paper's configuration, and any
difference is reported.

The **parameters cell** near the top holds only what spans sections:

| `MODE` | what it does | typical cost |
|--------|--------------|--------------|
| `"replay"` | re-render the paper's saved Morse graph and Morse sets | seconds |
| `"morse"` | recompute the Morse graph of the *saved* model at your `SUBDIV` | seconds–minutes |
| `"retrain"` | train a fresh model, then compute its Morse graph at your `SUBDIV` | minutes–hours (GPU recommended) |

- `SUBDIV` — the CMGDB subdivision triple `(init, min, max)`, used by both
  recomputing modes. A toy value like `(10, 14, 20)` is a fast qualitative
  preview; the paper value for the example is noted in a comment. **Coarse
  grids can merge nearby recurrent sets and change the Morse graph**, so they
  are previews, not paper-quality results.
- `BOX_SCALE` — how much to inflate Morse-set boxes so tiny attractor sets stay
  visible at paper figure size: `"auto"`, a float, or a `{label: factor}` dict.
  Drawing only; it never changes what was computed.
- `SEED`, and for coral `TRAIN_FILE`, select which run to load or train.

Chafee-Infante adds `COMPUTE_ROA` in its regions-of-attraction section: exact
basins come from the map graph CMGDB returns, at the cost of a second pass over
the phase space.

## Where output goes

Nothing a notebook does ever touches the preserved paper trees
(`replay_sources/`, `paper_figures/`):

- `morse` recomputes write to `output/notebooks/<experiment>/morse_<i>-<m>-<x>/`.
- `retrain` writes a self-contained run under
  `output/notebooks/<experiment>/retrain_<timestamp>/`.
- Re-rendered replay figures land in `notebooks/rendered/<experiment>/`.

## A caveat on Chafee-Infante retraining

Fresh Chafee-Infante retrains currently overfit and can fail the two-attractor
ground truth. The **replay artifacts are the paper reference**; `retrain` mode
is there for experimentation, not verification.

## Local execution

The notebooks run locally against the project venv as well as on Colab:

```bash
jupyter nbconvert --to notebook --execute notebooks/0*.ipynb
```

With the default parameters (`MODE = "replay"`) this re-renders every figure
without training or recomputing CMGDB. The install cell runs only on Colab, so
a local run keeps whatever `latentdynamics` and CMGDB the venv already has —
an editable checkout is not replaced by the released wheel.
