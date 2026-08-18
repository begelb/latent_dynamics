# Paper notebooks

The notebook set is deliberately small: one CMGDB primer and one replay tour
for each of the four application families in the manuscript. The application
notebooks explain the scientific comparison, load checksummed release
artifacts, display the saved panels, and print only a few headline invariants.
They do not duplicate the training or CMGDB pipelines.

| Notebook | Paper example | Colab |
|---|---|---|
| [00_cmgdb_intro.ipynb](00_cmgdb_intro.ipynb) | Short CMGDB primer using a planar Leslie map | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/begelb/latent_dynamics/blob/paper/notebooks/00_cmgdb_intro.ipynb) |
| [01_leslie_2d_contraction.ipynb](01_leslie_2d_contraction.ipynb) | Extended 2-D Leslie model embedded in 10 dimensions | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/begelb/latent_dynamics/blob/paper/notebooks/01_leslie_2d_contraction.ipynb) |
| [02_leslie3d_example1.ipynb](02_leslie3d_example1.ipynb) | 3-D Leslie: direct, fine latent, coarsened, and lower-resolution views | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/begelb/latent_dynamics/blob/paper/notebooks/02_leslie3d_example1.ipynb) |
| [03_coral.ipynb](03_coral.ipynb) | 13-D red-coral model with a 1-D latent model | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/begelb/latent_dynamics/blob/paper/notebooks/03_coral.ipynb) |
| [04_chafee_infante.ipynb](04_chafee_infante.ipynb) | Chafee–Infante latent dimensions 1, 2, and 3, coarsening, RoA, and statistics | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/begelb/latent_dynamics/blob/paper/notebooks/04_chafee_infante.ipynb) |

## Colab behavior

On Colab, the setup cell clones the public repository and installs it in
editable mode. Keeping the checkout is intentional: the artifact fetcher needs
the repository's `artifacts/manifest.json`. It then installs the prebuilt
CMGDB `v1.3.3+fork.3` wheel and, on first load, downloads only the checksummed
bundles used by that notebook.

The badges assume `github.com/begelb/latent_dynamics`, branch `paper`. They
become usable once the replay bundles are published; until then the manifest
reports `release_url_base: PENDING` and the fetcher explains where to place
manually obtained bundles.

## Scope

- The application notebooks are replay-only and finish in seconds once their
  bundles are cached.
- Full CMGDB recomputation and retraining commands remain in
  [REPRODUCING.md](../REPRODUCING.md).
- The Chafee–Infante bifurcation diagram is a static manuscript asset. Its
  generator and source data were not preserved, so the companion code does not
  claim to reproduce it.
- Sampled residual and tolerance calculations are numerical evidence, not a
  mathematical certification over an entire domain.

## Local execution

After installing the project and placing or fetching the replay bundles:

```bash
jupyter nbconvert --to notebook --execute notebooks/0*.ipynb
```

The Colab setup cells are guarded, so they do not replace a local environment.
