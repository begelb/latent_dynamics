# Paper Notebooks

This directory contains Jupyter notebooks that replicate the paper's computational results. All notebooks install the `latentdynamics` package from GitHub and run end-to-end on a fresh Google Colab kernel.

## Figure Notebooks

These notebooks replay saved latent-dynamics computations to generate paper figures — minimal compute, suitable for free Colab.

| Notebook | Paper Section | System | What It Shows | Colab |
|---|---|---|---|---|
| [01_Leslie_2D.ipynb](figures/01_Leslie_2D.ipynb) | §5.1 | Leslie 2-gen + contraction | 2D latent bistable Morse graph | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/begelb/latent_dynamics/blob/master/code/notebooks/figures/01_Leslie_2D.ipynb) |
| [02_Leslie_3D.ipynb](figures/02_Leslie_3D.ipynb) | §5.2 | Leslie 3D embedded | 3D latent bistable Morse graph | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/begelb/latent_dynamics/blob/master/code/notebooks/figures/02_Leslie_3D.ipynb) |
| [03_Chafee_Infante.ipynb](figures/03_Chafee_Infante.ipynb) | §5.4 | Chafee-Infante PDE (64D) | 2D latent bistable Morse graph | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/begelb/latent_dynamics/blob/master/code/notebooks/figures/03_Chafee_Infante.ipynb) |
| [04_Red_Coral.ipynb](figures/04_Red_Coral.ipynb) | §5.5 | Red Coral (13D) population | 1D latent bistable Morse graph; data scaling experiment | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/begelb/latent_dynamics/blob/master/code/notebooks/figures/04_Red_Coral.ipynb) |

## Training Notebooks

These notebooks run the latent-dynamics training pipeline end-to-end — expensive compute, **requires GPU runtime** on Colab. See the GPU caveat at the top of each notebook.

| Notebook | Paper Section | System | What It Does |
|---|---|---|---|
| [Leslie_2D.ipynb](training/Leslie_2D.ipynb) | §5.1 | Leslie 2-gen + contraction | Train the autoencoder + latent map from scratch |
| [Leslie_3D_ex1.ipynb](training/Leslie_3D_ex1.ipynb) | §5.2 | Leslie 3D (example 1) | Train the autoencoder + latent map from scratch |
| [Leslie_3D_ex2.ipynb](training/Leslie_3D_ex2.ipynb) | §5.2 | Leslie 3D (example 2) | Train the autoencoder + latent map from scratch |
| [Chafee_Infante.ipynb](training/Chafee_Infante.ipynb) | §5.4 | Chafee-Infante PDE (64D) | Train the autoencoder + latent map from scratch |

## Viewing Rendered HTML

The `figures/` notebooks can be viewed as static HTML locally:

```bash
cd code/notebooks
jupyter nbconvert --to html figures/*.ipynb --output-dir rendered/
open rendered/01_Leslie_2D.html
```

However, **Colab is the canonical viewing and execution environment** for this code. Use the Colab badges in the table above to open each notebook directly in the browser.

## Data and Artifact Retrieval

Each figure notebook automatically fetches its required data and pre-trained artifacts from the GitHub Release `v0.1.0-data` on first run. The artifacts are cached locally so re-runs are fast.

Training notebooks write their outputs to a timestamped directory (via the config's `run_root` setting).

## Notebook Structure

Each notebook follows this pattern:

1. **Install cell** — `!pip install git+https://github.com/begelb/latent_dynamics.git`
2. **Title & metadata** — Section number, system name, brief description
3. **Setup** — Load config, fetch artifacts, initialize
4. **Analysis & visualization** — Replays results, plots figures

All imports are from the installed `latentdynamics` package. No local paths or `sys.path` hacks.
