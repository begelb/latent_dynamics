# CMGDB_utils Reference

A project-internal reference for `CMGDB_utils` — Marcio Gameiro's auxiliary
Python toolkit built on top of CMGDB. Based on direct reading of
`archive/CMGDB_utils` (commit `4bce196`, master branch of
`github.com/marciogameiro/CMGDB_utils`, declared version 1.1.1).

This is **not** installed in the project's `.venv` and currently is not on
the dispatched-job dependency path. The clone sits in `archive/` as a
reference; if we adopt any of its primitives, we can either copy-vendor the
specific module or do a proper editable install (see §3 for the dep
constraints).

Cross-references are given as `file:line` against
`archive/CMGDB_utils/src/CMGDB_utils/`.

## 1. What CMGDB_utils is

A pure-Python package combining two distinct things in one namespace:

1. **An alternative Morse-graph engine** — `Model`, `CubicalGrid`,
   `ComputeMorseGraph`, `ComputeConleyMorseGraph`. This is a complete
   re-implementation of the CMGDB pipeline that runs on a **fixed
   uniform cubical grid** (no hierarchical subdivision) and uses
   `DSGRN.MorseDecomposition` for the SCC step plus
   `CMGDB.ComputeConleyIndex` (the standalone pybind, not the full
   `ComputeConleyMorseGraph`) for indices.
2. **Auxiliary tools that consume a CMGDB Morse graph** —
   `NonTrivialCMGraph`, `LatticeAttractors`, `AdjacencyMatrix`,
   `MarkovContraction`, `compute_morse_graph_from_mvm`, and a family of
   plot/save helpers. These take a `CMGDB.MorseGraph` (or the alternative
   engine's tuple-form output) and produce derived structures.

We are interested almost entirely in **(2)**. The alternative engine in
**(1)** does not match our methodology — we already have a working
adaptive-or-uniform CMGDB pipeline with the `box_map_backend` plumbing
(see `cmgdb_reference.md`), and it doesn't need a second engine.

## 2. Layout

19 modules, all under `src/CMGDB_utils/`. Grouped by role:

### Alternative pipeline (do not need)

| File | Role |
| --- | --- |
| `Model.py:4-13` | `Model(lower_bounds, upper_bounds, grid_size, F, periodic=None, map_type='BoxMap', padding=False)` — input bundle for the alternative engine. **Different signature from `CMGDB.Model`.** |
| `CubicalGrid.py:8-85` | `CubicalGrid(lower_bounds, upper_bounds, grid_size)` — uniform fixed-size grid; methods `min_vertex(idx)`, `max_vertex(idx)`, `grid_cover(box, padding)`. Indexing via `np.ravel_multi_index` (column-major). |
| `ComputeMorseGraph.py:8-97` | `compute_multivalued_map(grid, model)`, `ComputeMorseGraph(model)`, `ComputeConleyMorseGraph(model, acyclic_check=True)`. Builds a `DSGRN.Digraph`, runs `DSGRN.MorseDecomposition`, optionally calls `CMGDB.ComputeConleyIndex` per Morse set. |

### Box-map primitives (no extra deps)

| File | Role |
| --- | --- |
| `BoxMap.py:7-92` | `CornerPoints`, `CenterPoint`, `SamplePoints`, `BoxMap`, `BoxMapSample`, `MultiBoxMap`. The `BoxMap` here is **not** the same as `CMGDB.BoxMap`: this one has no `padding=True` parameter at the top level, and `BoxMapSample` returns a `(f_box, X)` tuple so the caller can recover the sample points. `MultiBoxMap` returns a *list* of small boxes (one per evaluation point) rather than a single bounding box — used by the alternative engine's `map_type='MultiBoxMap'`. |
| `BoxMapData.py:9` | `BoxMapData` class — data-driven box map from `(X, Y)` point clouds. Same role as CMGDB's `BoxMapData`. |

### Filters and analysis on CMGDB Morse graphs

| File | Role |
| --- | --- |
| `NonTrivialCMGraph.py:9-98` | `NonTrivialCMGraph(morse_graph)`, `NonTrivialCMGraphPyChomP(morse_graph)`, `graph_from_dotfile(dot_fname)`. **Removes trivial-Conley-index nodes** (those whose annotation is `(0, 0, ..., 0)`) and transitively-reduces the resulting graph. See §5 — this is the most directly useful piece for us. |
| `LatticeAttractors.py:8-170` | `transitive_closure`, `morse_graph_attractors`, `lattice_attractors`, `lattice_repellers`. Computes the **lattice of attractors** of a Morse graph as a `pychomp.DirectedAcyclicGraph`. |
| `AdjacencyMatrix.py:14-227` | `weighted_adjacency_matrix(cubical_complex, model)`, `morse_graph_adjacency_matrix(model)`, `attractor_eigenvalues(...)`, `eigenvectos_min_attractor(...)`, `plot_eigenvalues(...)`. Eigenvalue/eigenvector analysis of Markov-like adjacency matrices derived from the multi-valued map. |
| `MarkovContraction.py:4-86` | `average_rows_inplace`, `add_cols_inplace`, `contract_markov_matrix`, `morse_set_self_weights`. Markov-matrix contraction by Morse set membership. |
| `compute_morse_graph_from_mvm.py:7-181` | `morse_graph_from_edges`, `morse_graph_from_mvm`, `lattice_attractors_from_mvm`, `attractors_from_mvm`, `repellers_from_mvm`, etc. Reconstruct a Morse graph or its derivatives from a **saved edge list** (multi-valued map) without re-running the dynamics. |

### Plotting (two parallel families)

| File | Role |
| --- | --- |
| `PlotMorseGraph.py:9-...` | `PlotMorseGraph(mg_data, ...)` — renders the alternative engine's `(morse_graph, morse_decomp, vertex_mapping)` tuple. |
| `PlotMorseGraph_new.py:11-...` | `PlotMorseGraph_new(morse_graph, ...)` — renders a CMGDB `MorseGraph` object directly. |
| `PlotMorseSets.py:10-...` | `PlotMorseSets(mg_data, cubical_complex, ...)` — scatter of cubes from the alternative engine. |
| `PlotMorseSets_new.py:11-...` | `PlotMorseSets_new(morse_sets, ...)` — scatter from raw morse-set data (rectangles + labels). |
| `PlotGraph.py:7-...` | `PlotGraph(graph, ...)` — generic DAG renderer (works on `DirectedAcyclicGraph`). |
| `PlotLatticeAttractors.py:9-...` | `PlotLatticeAttractors(lattice_att, ...)` — wrapper around `PlotGraph` for attractor lattices. |

The `_new` variants take CMGDB's native objects; the un-suffixed variants
take the alternative engine's tuple data. If we use anything from this
package the `_new` plotters are the ones to look at — but our project
already has its own renderers in `src/latentdynamics/viz/morse_plots.py`
that handle DOT/CSV input with the paper palette.

### Utilities

| File | Role |
| --- | --- |
| `DirectedAcyclicGraph.py:6` | `DirectedAcyclicGraph` — clean pure-Python DAG with vertex/edge labels, `descendants()`, `transitive_reduction()`, `transitive_closure()`. **Self-contained**, no external imports. |
| `SaveMorseSets.py:8,28` | `SaveMorseSets(morse_graph_data, cubical_complex, fname)` and `LoadMorseSetFile(fname)`. Different CSV schema from `CMGDB.SaveMorseSets`; works with the alternative engine. |

## 3. Install

Declared dependencies (`pyproject.toml:14-22`):

```
CMGDB >= 1.3.1
DSGRN >= 1.9.0
graphviz >= 0.20
matplotlib >= 3.6.0
numpy >= 1.23.0
scipy >= 1.11.4
```

Plus `pychomp` (imported by `LatticeAttractors.py` and the PyChomp variants
of `NonTrivialCMGraph`) and `pydot` (imported by `NonTrivialCMGraph.py`).
These are **not** declared in `pyproject.toml` even though they are
required at import time.

Our current venv has CMGDB (the editable archive install). The rest
(`DSGRN`, `pychomp`, `pydot`) are **not** installed. Practical
consequences:

- `import CMGDB_utils` will **fail** because `__init__.py:11` imports
  `ComputeMorseGraph` which `import DSGRN` at the top.
- Even if you bypass the package import and load specific modules
  directly, `LatticeAttractors`, `NonTrivialCMGraph`, and
  `compute_morse_graph_from_mvm` require `pychomp` / `pydot` / `DSGRN`.
- Modules that work without extra deps (only stdlib + numpy + CMGDB):
  `BoxMap.py`, `BoxMapData.py`, `DirectedAcyclicGraph.py`, `CubicalGrid.py`
  (no imports of DSGRN/pychomp), and `MarkovContraction.py` (uses only
  numpy and `collections`).

If we want to use `NonTrivialCMGraphPyChomP` against our CMGDB Morse
graphs, the minimum install is:

```
pip install pychomp pydot
```

Plus a local editable install of `CMGDB_utils`:

```
pip install -e archive/CMGDB_utils --no-deps
```

`--no-deps` because the declared `DSGRN>=1.9.0` is more invasive than we
need and DSGRN has its own non-trivial build (Boost, etc.). Without
`--no-deps`, pip would try to install DSGRN and fail unless the system
has its build prerequisites.

**Alternatively, vendor what we need.** `NonTrivialCMGraphPyChomP` is ~35
lines plus `DirectedAcyclicGraph` (~100 lines, no deps) plus
`transitive_closure` / `transitive_reduction` (which `DirectedAcyclicGraph`
provides). Copy those two files into `code/src/latentdynamics/` and
sidestep DSGRN/pychomp entirely. Recommended unless we end up needing
multiple CMGDB_utils features.

## 4. Public API quick reference

The names below are reachable from `CMGDB_utils.<name>` (assuming
imports succeed):

### Alternative engine

- `Model(lower_bounds, upper_bounds, grid_size, F, periodic=None, map_type, padding)` — `Model.py:4`. `map_type` is one of `'BoxMap'/'B'`, `'MultiBoxMap'/'M'`, `'GraphMap'/'G'`; with `'GraphMap'`, `F` is interpreted as a precomputed adjacency dict rather than a callable.
- `CubicalGrid(lower_bounds, upper_bounds, grid_size)` — `CubicalGrid.py:8`. Methods of interest: `dimension()`, `size()`, `min_vertex(idx)`, `max_vertex(idx)`, `grid_cover(box, padding=False)`.
- `ComputeMorseGraph(model)` — `ComputeMorseGraph.py:40`. Returns `(morse_graph_data, cubical_complex)` where `morse_graph_data == (DAG, DSGRN.MorseDecomposition, vertex_mapping)`.
- `ComputeConleyMorseGraph(model, acyclic_check=True)` — `ComputeMorseGraph.py:63`. Same return shape; per-vertex annotation is a string of Conley index polynomials via `CMGDB.ComputeConleyIndex`.

### Box maps

- `BoxMap(f, box, mode='corners', num_pts=10)` — `BoxMap.py:31`. Same axis-wise min/max contract as `CMGDB.BoxMap`, but with no padding parameter. The default mode `'corners'` enumerates `2^d` corner points.
- `BoxMapSample(f, box, mode='random', num_pts=100)` — `BoxMap.py:36`. Returns `(f_box, X)` so the caller can see which random samples were used. With `mode='random'` the second element is the sample list; otherwise empty.
- `MultiBoxMap(f, box, box_size=None, mode='corners', num_pts=10)` — `BoxMap.py:62`. Returns a **list of small boxes**, one centered on each sample point, with side length `box_size` (defaults to half the input box size for `corners`/`random`, equal for `center`). Intended to be paired with `map_type='MultiBoxMap'`, which then unions per-cell covers across the list.
- `BoxMapData(X, Y, ...)` — `BoxMapData.py:9`. Data-driven box map.

### Filters and analysis (work on CMGDB MorseGraphs)

- `NonTrivialCMGraphPyChomP(morse_graph)` — `NonTrivialCMGraph.py:44`. **The working variant.** Returns a `CMGDB_utils.DirectedAcyclicGraph` containing only vertices whose annotation is **not** the trivial Conley index `(0, 0, ..., 0)`, with edges between them coming from the transitive closure of the original graph, then transitively reduced.
- `NonTrivialCMGraph(morse_graph)` — `NonTrivialCMGraph.py:9`. **Bugged**: line 11 references `DirectedAcyclicGraph()` without importing it. Will raise `NameError` on first call. Use the PyChomP variant; see §7.
- `graph_from_dotfile(dot_fname)` — `NonTrivialCMGraph.py:80`. Parses a CMGDB-style DOT file into a `DirectedAcyclicGraph` keyed by integer vertex names with annotation labels.
- `transitive_closure(morse_graph)`, `morse_graph_attractors(morse_graph)`, `lattice_attractors(morse_graph)`, `lattice_repellers(morse_graph)` — `LatticeAttractors.py:8-170`. All return `pychomp.DirectedAcyclicGraph` objects.
- `weighted_adjacency_matrix(cc, model)`, `morse_graph_adjacency_matrix(model, acyclic_check=True)`, `attractor_eigenvalues(W, mg_data, latt_att, att_vert, num_evals=100)`, `eigenvectos_min_attractor(...)`, `plot_eigenvalues(...)` — `AdjacencyMatrix.py:14-227`. Uses the alternative engine's tuple data.
- `compute_morse_graph_from_mvm` module — multiple `*_from_mvm` functions to recover Morse graph / attractor lattice from a saved multi-valued-map edge list.

### Building blocks

- `DirectedAcyclicGraph` — `DirectedAcyclicGraph.py:6`. Methods: `add_vertex(v, label='')`, `add_edge(u, v, label='')`, `remove_edge`, `vertex_label`, `vertices()`, `edges()`, `adjacencies(v)`, `descendants(v)`, `transitive_closure()`, `transitive_reduction()`.

## 5. Useful pieces for this project

### `NonTrivialCMGraphPyChomP` — directly relevant

Our memory entry `latent-morse-coarser-than-analytic` records the finding
that under uniform-grid + no-padding CMGDB on Patrick's 2D Leslie, the
trained latent map reaches bistability at the same subdivision threshold
(smax=17) as the analytic ambient map, but its **full Morse graph is
still coarser by ~2 trivial-Conley-index transient nodes**. The
straightforward way to compare the *non-trivial* parts is to filter both
graphs through `NonTrivialCMGraphPyChomP` first.

Concrete usage:

```python
# Assuming pychomp + pydot installed and CMGDB_utils editable-installed
import CMGDB
import CMGDB_utils

morse_graph, _ = CMGDB.ComputeConleyMorseGraph(model)
nt_graph = CMGDB_utils.NonTrivialCMGraphPyChomP(morse_graph)
# nt_graph is a CMGDB_utils.DirectedAcyclicGraph with the trivial-index
# nodes removed and transitively reduced edges.
```

For our pipeline this would slot into `src/latentdynamics/analysis/`
alongside `morse_metrics.py`. The cleanest integration is the
"vendor" route in §3 — copy `DirectedAcyclicGraph.py` and a stripped
version of `NonTrivialCMGraph.py` (removing the pychomp+pydot
imports we don't need and using our own DAG class) into the
project.

### `lattice_attractors` — interesting, less load-bearing

The lattice-of-attractors view is the natural follow-up to a Morse graph
that has multiple coexisting attractors: it tells you which subsets of
attractors can be jointly invariant. For Patrick's 2D Leslie with both
fixed-point and period-3 attractors, this lattice would have 4 elements
(empty, {fp}, {p3}, {fp, p3}) — not a deep finding but a useful canonical
structure to display. Requires pychomp.

### Box-map primitives — duplicates of CMGDB, don't use

The `BoxMap.py` here partly duplicates `CMGDB.ComputeBoxMap`. The only
non-redundant variant is `MultiBoxMap`, which is paired tightly with the
alternative engine's `map_type='MultiBoxMap'` and has no analogue in
CMGDB.

## 6. Tradeoffs vs the main CMGDB pipeline

The alternative engine in CMGDB_utils is a **different methodology**, not
a feature you'd want to layer on top of CMGDB:

| | CMGDB (our pipeline) | CMGDB_utils alternative engine |
| --- | --- | --- |
| Grid | Adaptive binary subdivision tree, or fixed-depth uniform when `init==min==max` | Fixed uniform cubical grid only |
| SCC engine | C++ Tarjan in `GraphTheory.hpp` | `DSGRN.MorseDecomposition` (Python) |
| Conley index | C++ chomp via the full `ComputeConleyMorseGraph` | Python loop calling `CMGDB.ComputeConleyIndex` per Morse set |
| Output | `CMGDB.MorseGraph` (pybind class with `morse_set_boxes()`, etc.) | `(DirectedAcyclicGraph, DSGRN.MorseDecomposition, vertex_mapping)` tuple |
| Performance | C++ critical path | Python loops; expected substantially slower at moderate `grid_size` |
| Refinement strategy | Adaptive: `subdiv_init`, `subdiv_min`, `subdiv_max`, `subdiv_limit` | None — grid is fixed at construction |

If we wanted the same uniform-grid methodology we already use, the
alternative engine doesn't add anything functional; if anything it would
be slower because the inner SCC + adjacency loops are pure Python.

## 7. Gotchas

- **Top-level package import fails without DSGRN.** `__init__.py:11`
  unconditionally imports `ComputeMorseGraph`, whose top-level `import
  DSGRN` raises if DSGRN is not installed. So you cannot
  `import CMGDB_utils` until DSGRN is available — or you bypass
  `__init__.py` and import the specific submodule directly.
- **`NonTrivialCMGraph` (non-PyChomP) is bugged.** `NonTrivialCMGraph.py:11`
  references `DirectedAcyclicGraph()` without an import statement. The
  symbol is exposed via the package's `__init__.py` (`from
  CMGDB_utils.DirectedAcyclicGraph import *`), but inside
  `NonTrivialCMGraph.py` itself there is no `import`, so calling
  `NonTrivialCMGraph(...)` raises `NameError`. The `PyChomP` variant on
  line 44 correctly uses `CMGDB_utils.DirectedAcyclicGraph()`. Until
  upstream fixes this, prefer `NonTrivialCMGraphPyChomP`.
- **`SaveMorseSets` in this package writes a different CSV from
  `CMGDB.SaveMorseSets`.** Schema is anchored on the alternative engine's
  cubical-complex indices, not on box coordinates. Be careful not to mix
  the two.
- **No installation in our `.venv`.** Anything in this doc that says
  "you would call X" assumes a future install. Today, the code in
  `archive/CMGDB_utils` is read-only reference.

## 8. Recommended path forward

Three options if we decide we want any of these capabilities:

1. **Vendor just `NonTrivialCMGraphPyChomP` + `DirectedAcyclicGraph`** into
   `code/src/latentdynamics/analysis/`. Strip the pychomp/pydot imports
   (the algorithm uses only `morse_graph.vertices()`,
   `morse_graph.adjacencies(v)`, `morse_graph.annotations(v)`,
   and our own DAG class for the output). ~150 lines, zero new deps.
   Closes the trivial-index transient question raised in
   [[latent-morse-coarser-than-analytic]].
2. **Editable install with `--no-deps` + `pip install pychomp pydot`**.
   Gives us the full `NonTrivialCMGraphPyChomP` + `lattice_attractors`
   surface. Costs: two new deps, an `archive/CMGDB_utils` editable
   pinned in the venv, and a possibly-broken `import CMGDB_utils` at
   the package level until DSGRN is also available. Cluster dispatch
   would need the same.
3. **Leave it as reference only.** Read it, ignore it operationally.

Option 1 is the most aligned with how this project handles small,
load-bearing utilities. The vendor cost is low and we avoid coupling
the dispatched pipeline to DSGRN/pychomp.
