# Latent Dynamics

Companion code for

> **Characterizing High-dimensional Dynamics by Combinatorial-Topological
> Methods on a Latent Space**
> P. Bailon, M. Gameiro, B. Gelb, W. Kalies, M. Kramar, K. Mischaikow,
> B. Rivas, E. Vieira.
> DOI / arXiv: *in preparation*.

## Description

We take trajectory data from a dynamical system `f` on a high-dimensional space `X` and train a standard autoencoder with encoder `E`, decoder `D`, and latent map `g` so that `g ∘ E ≈ E ∘ f` on the data. We apply [CMGDB](https://github.com/marciogameiro/CMGDB) to the *latent* map `g` to compute a Conley–Morse graph: a finite poset of combinatorial Morse sets with Conley indices. This describes attractors, repellers, and connecting orbits at a chosen grid resolution. The paper's main theorem gives a quantitative condition—a residual bound against a tolerance—under which attracting blocks of the latent combinatorial model lift to attracting blocks and hence attractors of the original system `f`. The code computes latent Morse graphs, their coarsenings, regions of attraction, and *sampled* estimates of the residual and tolerance quantities. Sampled estimates are not certified bounds. Nothing this repository produces is a computer-assisted proof.

## Examples

| Family | System | Latent dim | Section |
|---|---|---|---|
| `leslie_2gen_contraction` | 2-D overcompensatory Leslie model embedded in 10-D | 2 | Extended two-dimensional Leslie model |
| `leslie3d_example1` | 3-D overcompensatory Leslie model (reference computation, fine latent model, Morse-graph coarsening, coarser grid) | 2 | Three-dimensional Leslie model |
| `coral` | 13-D Mediterranean red-coral population model | 1 | Red coral population model |
| `chafee_infante` | Chafee–Infante PDE, 64-mode spectral discretization, λ=28 | 1, 2, 3 | Chafee–Infante |

## Installation

You need Python 3.11.4–3.13 and [Graphviz](https://graphviz.org) (`dot`) to draw Morse graphs.

```sh
git clone <repository-url>
cd <repository>
pip install -e .
```

(`uv sync` also works.)

CMGDB `>=1.5.0` is installed from PyPI (prebuilt wheels for macOS, Linux, and Windows).

## Artifacts

Trained models, datasets, and saved computations are distributed as checksummed bundles separate from this repository (see `artifacts/manifest.json`). When you fetch them, the system checks the SHA-256 of every bundle before extracting and treats extracted inputs as read-only:

```python
from latentdynamics.replay import fetch_artifacts
fetch_artifacts("coral")          # -> replay_sources/coral/
```

Until the release bundles are published, the fetcher tells you where to place bundles you obtain manually. All outputs from calculations go into `output/` (or a directory you specify with `--output`), never into `replay_sources/`.

**Artifact security model.** Bundle integrity depends on the SHA-256 manifest committed in this repository. Bundles are checked before extraction. Tar members are strictly filtered—no absolute paths, traversal, links, or unexpected empty files. Extracted inputs are made read-only. Some bundle members are Python pickles (scikit-learn scalers saved with joblib, one legacy plot-data `.pkl`). They are only loaded from the checksum-verified `replay_sources/` tree. Model checkpoints are plain `state_dict` tensors loaded with `torch.load(weights_only=True)`. Loading legacy pickled modules needs an explicit environment opt-in and is not used by any shipped workflow. If you place bundles manually or point tools at files outside the repository, you are responsible for trusting those files.

## Quick test

```sh
python pipeline.py --config coral_basic --stages render,metrics \
    --cell-index 16 --expected-cells 30
```

This renders the red-coral Morse graph and Morse sets from the saved computation in the artifact bundle and checks the metrics (three Morse nodes; two minimal attractor nodes with fixed-point Conley index) in a few seconds.

## Reproduction

1. **Replay** (seconds–minutes): Re-render figures and re-check invariants from the saved computations in the artifact bundles. This is deterministic up to font and renderer differences in PDFs.
2. **Recompute** (minutes–hours): Rerun CMGDB and the analysis pipeline from the shipped *trained models*. Morse graphs, Conley indices, coarsenings, regions of attraction, and residual/tolerance estimates are recomputed from scratch. Results match the published invariants exactly (the box maps are deterministic). Sampled residual searches match to sampling precision.
3. **Retrain** (minutes–hours): Retrain autoencoders from the saved or regenerated datasets. Training was not seeded end-to-end in the original runs. Retraining reproduces the qualitative story but **not** necessarily the exact published models.

See `REPRODUCING.md` for one documented entry point per paper computation with runtime tiers, and `docs/manuscript_code_matrix.md` for the full figure/table-to-code matrix with expected invariants.

## Repository layout

```
src/latentdynamics/    package: systems, sampling, models, training, config,
                       analysis (incl. Morse coarsening, Conley indices,
                       regions of attraction, sampled residual/tolerance),
                       viz, replay, cli
scripts/               one driver per paper computation (see REPRODUCING.md)
notebooks/             reader-facing replay notebooks (Colab-ready)
tests/                 focused unit tests (`pytest -m "not slow"`)
artifacts/             bundle manifest + frozen reference results (the
                       published numbers, checksummed)
docs/                  figure contracts + the manuscript-to-code matrix
```

## Citing

> P. Bailon, M. Gameiro, B. Gelb, W. Kalies, M. Kramar, K. Mischaikow,
> B. Rivas, E. Vieira.
> *Characterizing High-dimensional Dynamics by Combinatorial-Topological
> Methods on a Latent Space.*

Also cite CMGDB: https://github.com/marciogameiro/CMGDB.

## License

The companion software and author-contributed replay artifacts are released under the MIT License; see `LICENSE`.

### Attribution

- CMGDB `1.5.0` (https://github.com/marciogameiro/CMGDB) computes the Morse graphs, Conley indices, and reachability queries.
- Legacy application code was adapted from the MIT-licensed [MORALS](https://github.com/Ewerton-Vieira/MORALS) project. Its license is preserved in `licenses/MORALS-LICENSE`.
- The replay files contain author-contributed model checkpoints, datasets, and saved computations, released under the repository's MIT License.
