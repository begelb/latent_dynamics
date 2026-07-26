# fig_chafee_infante

Paper §1.256: Chafee-Infante PDE in spectral coordinates (64 Fourier modes), super-critical regime `α = 28`. Demonstrates the framework on a stiff PDE with multiple equilibria and saddle structure.

## Paper figures

- `paper/figures/ci_morse_graph.pdf`
- `paper/figures/ci_morse_sets.pdf`
- `paper/figures/ci_bif_diagram.pdf`

## Source of paper run

Marcio's reference Chafee-Infante computation is preserved under
`archive/marcio/scripts/`: `ci_model_weights.pth`, `train_data.csv`,
`autoencoder_model.py`, and `compute_dynamics.py`. The exact paper PDFs are
also present there.

- data: `archive/marcio/scripts/train_data.csv`
- model: `archive/marcio/scripts/ci_model_weights.pth`
- CMGDB: data-derived latent bounds with 10% per-axis expansion,
  `subdiv_init=14`, `subdiv_min=16`, `subdiv_max=22`, and `padding=True`

## Status

**Marcio's original computation is the production source.** The package replay
at `code/replay_sources/chafee_infante/replay/` uses a converted copy of his
weights but different CMGDB settings (`10/14/28`, fixed bounds
`[-3,-2]x[3,2]`, no padding). It is retained only as a reference computation
and must not be used to generate the manuscript's collapsed Morse set.

Earlier package retrains in `code/output/chafee_infante/` are exploratory and
should not be treated as the paper source.

## Reproduction commands

Inspect the current converted-data config without recomputing CMGDB:

```bash
python pipeline.py --config chafee_infante --stages diagnose --max-seeds 1
```

Reconstruct Marcio's original cell graph and compute the connection-complete
coarse Morse representation:

```bash
python scripts/coarsen_chafee_infante.py
```

For comparison only, the later package computation can be selected explicitly:

```bash
python scripts/coarsen_chafee_infante.py \
  --computation reference --with-connections
```

## Expected scientific output

Paper figure shows seven Morse sets: two attractors `(x-1, 0, 0)`, three saddles `(0, x-1, 0)`, two repellers `(0, 0, x-1)`. The Hasse diagram is non-trivial.

## Coarse Morse representation

For the manuscript's bistability-level view, the five nonminimal nodes can be
collapsed to a single fiber while the two attractors remain separate. Following
the manuscript notation, the merged node is named `M(1)`: it represents the
nine unstable equilibria and the connecting orbits between them at the level of
the target coarse Morse representation.

```bash
python scripts/coarsen_chafee_infante.py
```

The resulting projection is `0 -> 0`, `1 -> 1`, and `{2,3,4,5,6} -> 2`;
the quotient Hasse edges are `2 -> 0` and `2 -> 1`. The script verifies that
the induced quotient is acyclic, recomputes the directed CMGDB cell graph from
Marcio's raw checkpoint and training data, and draws the merged set with the
cell-graph connections between the fine components in its fiber. His graph has
19,373 cells. The five fine sets contribute 1,515 recurrent cells and path
completion adds 2,702 connection cells, giving 4,217 cells in `M(1)` with no
overlap with the two singleton attracting sets.

The quotient has `M(1)` pointing to both attracting nodes, corresponding to
`M(0-)` and `M(0+)`. It intentionally assigns no Conley index to the merged
node: that index would have to be recomputed from an index pair for the union.

Connection completion is always enabled for the default Marcio computation.

### Exploratory basin overlay

To overlay the coarse Morse representation on Marcio's original uniform-grid
regions of attraction:

```bash
python scripts/plot_chafee_coarse_morse_roa_overlay.py
```

This reproduces his separate `16/16/16` uniform basin computation on the same
data-derived bounds, draws the two uniquely assigned basins translucently, and
then overlays the connection-complete adaptive `M(1)` and the two attracting
Morse sets. Outputs are
`paper_figures/coarsened/chafee_infante/morse_roa_overlay.{pdf,png}`.

When the live CMGDB `MapGraph` and Morse graph are available (during the Morse
computation, rather than from the saved DOT/CSV alone), use
`compute_connection_complete_morse_sets(map_graph, morse_graph, projection)`.
For each quotient fiber this computes the intersection of the forward-reachable
downset and reverse-reachable upset of its fine Morse cells. Thus it adds
exactly the cell-graph vertices on paths between Morse components mapped to the
same coarse node, while excluding connections from `M(1)` to `M(0-)` or
`M(0+)`. `write_connection_complete_morse_sets(...)` writes those augmented
cell sets in the CSV format used by the plotting code. The saved `morse_graph`
and `morse_sets` files do not retain the transient cell-graph edges, so this
path completion cannot be reconstructed from those two artifacts after the
fact.

## Hyperparameter audit

| param                       | archive value           | YAML value             | source line                                                | notes |
|-----------------------------|-------------------------|------------------------|------------------------------------------------------------|-------|
| arch.encoder hidden_shapes  | [64, 32]                | [64, 32]               | src/latentdynamics/configs/chafee_infante.yaml                                | ✓     |
| arch.latent_map hidden_shapes | [32, 32]              | [32, 32]               |                                                            | ✓     |
| arch.decoder hidden_shapes  | [32, 64]                | [32, 64]               |                                                            | ✓     |
| arch.activation             | tanh                    | tanh                   | autoencoder_model.py:82,84,90,92,98,100                    | ✓     |
| arch.encoder_out_activation | none                    | none                   | autoencoder_model.py:85 (Linear, no terminal)              | ✓     |
| arch.latent_out_activation  | none                    | none                   | autoencoder_model.py:93                                    | ✓     |
| arch.decoder_out_activation | none                    | none                   | autoencoder_model.py:101                                   | ✓     |
| arch.high_dims              | 64                      | 64                     |                                                            | ✓     |
| arch.low_dims               | 2                       | 2                      |                                                            | ✓     |
| training.epochs             | 4000                    | 4000                   | autoencoder_model.py:73                                    | ✓     |
| training.learning_rate      | 0.003                   | 0.003                  | autoencoder_model.py:72                                    | ✓     |
| training.scheduler_factor   | 0.5                     | 0.5                    | autoencoder_model.py:113                                   | ✓     |
| training.scheduler_min_lr   | 1e-6                    | 1e-6                   | autoencoder_model.py:115                                   | ✓     |
| training.loss_weights       | recon + pred only       | [1, 1, 0]              | train_model.py:32,37 (no semiconjugacy term)               | ✓     |
| data.scaling                | none                    | none                   | no scaler in archive                                       | ✓     |
| data.n_samples_train        | 1000                    | 1000                   | autoencoder_model.py:13                                    | ✓     |
| data.n_iterations           | 30                      | 30                     | autoencoder_model.py:14 (time_steps)                       | ✓     |
| data.tau                    | 0.1                     | 0.1                    | autoencoder_model.py:12                                    | ✓     |
| data.sampling_method        | uniform (seed 7206)     | uniform                | autoencoder_model.py:40,55                                 | ✓     |
| cmgdb.subdiv_init           | 14                      | 10                     | compute_dynamics.py:58                                     | production uses archive |
| cmgdb.subdiv_min            | 16                      | 14                     | compute_dynamics.py:56                                     | production uses archive |
| cmgdb.subdiv_max            | 22                      | 28                     | compute_dynamics.py:57                                     | production uses archive |
| cmgdb.lower_bounds          | encoded-data min - 10%  | [-3, -2]               | compute_dynamics.py:31-40,60                               | production uses archive |
| cmgdb.upper_bounds          | encoded-data max + 10%  | [3, 2]                 | compute_dynamics.py:31-40,61                               | production uses archive |
| cmgdb.padding               | true                    | false                  | compute_dynamics.py:53-54                                  | production uses archive |

The architecture and training-side values match. The packaged YAML's CMGDB
values belong to the later reference computation and deliberately do not
replace Marcio's archived settings for the production figure.

## Verification

```bash
python scripts/coarsen_chafee_infante.py
# Expected:
#   7 fine Morse nodes on 19,373 map cells
#   M(1) = 1,515 recurrent + 2,702 connection cells
#   no coarse-set overlaps
```
