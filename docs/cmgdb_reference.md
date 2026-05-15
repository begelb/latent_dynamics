# CMGDB Reference

A project-internal reference for the CMGDB library as we use it. Based on
direct reading of `archive/CMGDB` (commit `857ec8b`, tag v1.3.2 of
`github.com/marciogameiro/CMGDB`) — the same source the
project depends on via the `CMGDB==1.3.2` PyPI wheel pin in
`code/pyproject.toml`.

Cross-references are given as `file:line` against `archive/CMGDB/src/CMGDB/`.

## 1. What CMGDB computes

CMGDB ("Conley-Morse Graph Database") takes a **discrete dynamical system**
on a rectangular phase space — concretely, a multi-valued combinatorial
analogue of an iterated continuous map `f: R^d -> R^d` — and produces:

- A **Morse decomposition** of the recurrent set: a finite partition of the
  recurrent dynamics into disjoint "Morse sets" (each a maximal path-strongly
  connected component of the combinatorial map graph), realised as collections
  of axis-aligned boxes.
- A **Morse graph**: the partial order on Morse sets induced by reachability
  in the map graph. Nodes that are sinks (no outgoing edges) correspond to
  attractors; sources (no incoming edges) to repellers.
- For each Morse set, its **Conley index**: a homotopy-type invariant of the
  isolated invariant set, reported here as a polynomial in `x` describing the
  graded module of relative homologies.

The user provides:
- a `BoxMap` callable `g(rect) -> rect`, sending a rectangle (typically by
  the `2^d`-corner outer cover, see §6) to an outer rectangular envelope of
  its forward image,
- axis-aligned phase-space bounds,
- four subdivision-control integers (see §5).

All of the analysis is **combinatorial after the BoxMap call**: CMGDB never
queries the underlying continuous map `f` again — it operates on the multi-
valued map induced by `g` on a hierarchically subdivided grid.

## 2. Layout

`archive/CMGDB/src/CMGDB/` has two halves:

### Python wrapper

A thin Python layer that re-exports the pybind11 bindings and adds utilities.
Every name listed here is reachable as `CMGDB.<name>`.

| File | Purpose |
| --- | --- |
| `__init__.py:1-9` | Re-exports `_cmgdb` (pybind module) and all Python helpers via `from ... import *`. |
| `ComputeBoxMap.py:30-50` | Pure-Python `BoxMap(f, rect, mode, padding, num_pts)`. The user's main entry point for evaluating their continuous `f` as an outer cover. See §6. |
| `BoxMapData.py:1-94` | `BoxMapData(X, Y, ...)` — data-driven point-cloud box map; **not used in this project**. |
| `SaveMorseData.py:5-12` | `SaveMorseSets(morse_graph, fname)` — writes the per-Morse-set boxes to CSV. |
| `PlotMorseGraph.py:11-100` | `PlotMorseGraph(morse_graph, ...)` — returns a `graphviz.Source` ready to render. |
| `PlotMorseSets.py:11-31` | `PlotMorseSets(morse_sets, ...)` — matplotlib scatter of the Morse boxes. |
| `LoadMorseSetFile.py:3-9` | `LoadMorseSetFile(fname)` — parses a Morse sets CSV back. |

In this project's `src/latentdynamics/analysis/morse.py`, we **do not** use
`PlotMorseGraph` / `PlotMorseSets` directly; we re-render from the on-disk DOT
and CSV via our own `render_morse_from_files` (see
`src/latentdynamics/viz/morse_plots.py`) so the figures share the
paper-wide palette.

### C++ core (`src/CMGDB/_cmgdb/`)

| Directory / file | Purpose |
| --- | --- |
| `CMGDB.cpp` | Single pybind11 translation unit. All `m.def(...)` registrations live here. |
| `include/database/` | Core algorithm: subdivision, grids, maps, the `Compute_Morse_Graph` template. |
| `include/chomp/` | Cubical homology subsystem — computes the Conley index from a relative chain complex. |

Within `include/database/`, the headers we care about (alphabetic):

| File | Role |
| --- | --- |
| `Compute_Morse_Graph.hpp` | The main algorithm. The `MorseDecomposition` hierarchical tree and its construction. |
| `Grid.h` | Abstract grid interface; pybind binding factory `GridBinding` (Grid.h:163). |
| `Map.h` | Abstract map interface (`operator()` on a `Geo` rectangle). |
| `MapGraph.h` | Multi-valued directed graph induced by a map on a grid; `MapGraphBinding` at line 148. |
| `Model.h` | Top-level user configuration: bounds, subdivision parameters, and the map. `ModelBinding` at line 575. Also defines `PHASE_GRID = PointerGrid` (line 19). |
| `ModelMap.h` / `ModelMapF.h` | Two concrete `Map` implementations. `ModelMap` is a **hardcoded** 2D Leslie-style map (ModelMap.h:37-38); `ModelMapF` wraps a user-supplied `std::function`. We always use `ModelMapF` indirectly via the Python `box_map` callable. |
| `MorseGraph.h` | Output data structure: DAG of Morse sets with attached Conley indices. `MorseGraphBinding` at line 460. |
| `PointerGrid.h` / `PointerTree.h` | Tree-based grid used in phase space (`PHASE_GRID`). |
| `RectGeo.h` | Axis-aligned rectangle geometry: `lower_bounds`, `upper_bounds`. |
| `TreeGrid.h` | Abstract parent of `PointerGrid` / `SuccinctGrid` / `CompressedTreeGrid`. Defines `cover()` over the binary subdivision tree. |
| `UniformGrid.h` | Regular Cartesian grid used for **parameter space** (`PARAMETER_GRID`), not the phase space partition. The name is potentially misleading: it has **nothing to do with the uniform-grid CMGDB methodology** we use; that mode is configured at the algorithm level via `subdiv_init == subdiv_min == subdiv_max`. |
| `ChompMap.h` | Adapter that lets `chomp::ConleyIndex` consume a `database::Map`. |

Headers we generally do not touch: `Atlas.h`/`AtlasGeo.h` (legacy chart-based
parameter space), `EdgeGrid.h` (unused in our pipeline), `SuccinctGrid.h` /
`CompressedTreeGrid.h` (read-only / compressed grid variants used for
checkpointing).

Within `include/chomp/`, the only files our path actually exercises are
`ConleyIndex.h`, `RelativeMapHomology.h`, `SparseMatrix.h`, and
`FrobeniusNormalForm.h` (via `conleyIndexString.h` in the database dir).

## 3. Build

`setup.py` uses the standard `CMakeExtension` pattern: setuptools delegates
to CMake, which calls `pybind11_add_module(_cmgdb src/CMGDB/_cmgdb/CMGDB.cpp)`
(`CMakeLists.txt:70`) and produces a single C-extension
`src/CMGDB/_cmgdb.cpython-*.so` placed inside the Python package. External
deps that CMake searches for: pybind11 (CONFIG), Boost (chrono, thread,
serialization), and `sdsl` (succinct data structures). GMP is referenced for
the homology integer arithmetic.

We have CMGDB installed editable from `archive/CMGDB` in the local venv so
that local source edits take effect on the next `import`. The dispatched
path (cluster, CI, coauthor checkouts) gets the wheel via the pin in
`code/pyproject.toml`. See the `cmgdb-install-split` memory for the
keep-the-pin-don't-clobber rules.

## 4. Public Python API

`CMGDB.__init__` exposes everything from `_cmgdb` plus the Python helpers in
one flat namespace. The names you will see in this project's code:

### Configuration

- `CMGDB.Model(subdiv_min, subdiv_max, subdiv_init, subdiv_limit, lower_bounds, upper_bounds, box_map)` — pybind constructor (one of nine overloads, `Model.h:575-613`). Wraps `box_map` as a `ModelMapF` so CMGDB can invoke the user callable from C++.

### Box-map utilities

- `CMGDB.BoxMap(f, rect, mode='corners', padding=False, num_pts=10)` — `ComputeBoxMap.py:30-50`. Despite the name, this is a Python helper, **not** a class. It evaluates a point-valued `f` at sample points inside `rect` and returns the bounding rectangle. The convention everywhere in our codebase is to wrap it as `box_map(rect): return CMGDB.BoxMap(g, rect, padding=...)`. Modes:
  - `'corners'` (default): evaluate at all `2^d` corners (`ComputeBoxMap.py:32`).
  - `'center'`: single center evaluation; **forces** `padding=True` (line 35).
  - `'random'`: uniform `num_pts` random samples (line 37-41).
  - Unknown mode silently returns `[]` (line 42-43).

  Padding semantics (line 47-48): when `padding=True`, the output rectangle is
  inflated axis-wise by the input rectangle's side length on each side. This
  is the same convention emulated by our `numpy` and `uniform_precomputed`
  backends in `src/latentdynamics/analysis/morse.py`.

### Compute

- `CMGDB.ComputeConleyMorseGraph(model) -> (MorseGraph, MapGraph)` — `CMGDB.cpp:50-84`. Main entry point. Builds the Morse decomposition **and** computes a Conley index for every vertex.
- `CMGDB.ComputeMorseGraph(model) -> (MorseGraph, MapGraph)` — `CMGDB.cpp:86-104`. Same algorithm but skips Conley-index computation.
- `CMGDB.ComputeConleyIndex(X_cubes, A_cubes, sizes, periodic, F, acyclic_check)` — `CMGDB.cpp:36-48`. Standalone combinatorial Conley index computation for a precomputed index pair `(X, A)`. We don't use this directly.

### Output

- `morse_graph.num_vertices()`, `morse_graph.vertices()`, `morse_graph.adjacencies(v)`, `morse_graph.edges()`, `morse_graph.morse_set(v)`, `morse_graph.morse_set_boxes(v)`, `morse_graph.annotations(v)` — `MorseGraph.h:460-475`. `morse_set_boxes(v)` returns a `list[list[float]]` of `2d`-vectors (lower bounds followed by upper bounds); `annotations(v)` returns the Conley index polynomial strings.
- `CMGDB.SaveMorseSets(morse_graph, fname)` — writes one row per box per vertex (`SaveMorseData.py:5-12`).
- `CMGDB.PlotMorseGraph(morse_graph, ...)` — returns a `graphviz.Source` you can render to PDF/SVG.

### Pybind classes

`m.def(...)` and `<class>Binding(m)` registrations live in `CMGDB.cpp:253-266` and the per-header bindings:

| Python | C++ source |
| --- | --- |
| `CMGDB.Model` | `ModelBinding` (`Model.h:575-613`) |
| `CMGDB.Grid` | `GridBinding` (`Grid.h:163`) |
| `CMGDB.MapGraph` | `MapGraphBinding` (`MapGraph.h:148`) |
| `CMGDB.MorseGraph` | `MorseGraphBinding` (`MorseGraph.h:460`) |

## 5. The algorithm

Trace of `CMGDB.ComputeConleyMorseGraph(model)`:

1. **Pybind hop** through `CMGDB.cpp:50` into the C++ free function.
2. **Algorithm entry**: `Compute_Morse_Graph(&morsegraph, phase_space, map, init, min, max, limit)` at `CMGDB.cpp:61`. The template lives in `Compute_Morse_Graph.hpp:355-414`.
3. **Pre-subdivision** (`Compute_Morse_Graph.hpp:361`): subdivide the phase-space grid `init` times before anything else. After this, the working depth is shifted so `Min` and `Max` parameters passed to the recursive code are `min - init` and `max - init`.
4. **Construct root MorseDecomposition** wrapping the (now pre-subdivided) phase space at depth 0 (`Compute_Morse_Graph.hpp:380`).
5. **Hierarchical refinement** via `ConstructMorseDecomposition` (`Compute_Morse_Graph.hpp:177-229`). This is a priority-queue-driven loop:
   - Pop the largest-size pending node.
   - If `depth > Min AND size > Limit`: stop refining this node (`Compute_Morse_Graph.hpp:201-205`).
   - Call `node.decompose(f)` (`Compute_Morse_Graph.hpp:128-138`), which builds the `MapGraph` on the node's subgrid (`MapGraph.h:33-35`), runs Tarjan-style SCC + reachability on it (`GraphTheory.hpp`), and stores the resulting list of `decomposition_` subgrids and the inter-component reachability matrix on the node.
   - If the decomposition is empty, mark the node `spurious` (`Compute_Morse_Graph.hpp:210-213`).
   - If `depth < Max`, call `node.spawn()` (`Compute_Morse_Graph.hpp:139-149`) to create one child per Morse set, each holding the **subdivided** subgrid of that component (`Compute_Morse_Graph.hpp:221` — `child->grid()->subdivide()`).
6. **Assemble the output graph** via `ConstructMorseGraph` (`Compute_Morse_Graph.hpp:248-350`). This is a postorder traversal of the decomposition tree. Vertices in the final `MorseGraph` are created **at depth = Min**; below `Min`, intermediate decompositions are still computed but their components are collapsed into vertices added at the `Min` cutoff. Inter-vertex edges come from the reachability relations stored on `MorseDecomposition` nodes during step 5.
7. **Conley index per vertex** (`CMGDB.cpp:72-78`). For each vertex `v`, take the union of boxes `S = morsegraph.grid(v)`, ask the chomp library to compute the Conley index of the isolated invariant set whose index pair is `(X, A) = (S ∪ F(S), F(S) \ S)`. Implemented in `chomp/ConleyIndex.h:42-150`. The resulting graded homology, with the induced action of the chain map of `F`, is then summarised as a polynomial via `conleyIndexString` (the Frobenius-normal-form path, `SparseMatrix.h:341`).
8. **MapGraph** (`CMGDB.cpp:81`). The auxiliary `MapGraph` returned alongside the `MorseGraph` is a flat representation of the multi-valued map at the **final** (deepest) grid. Not needed for the standard reproduction pipeline, but useful for diagnostics.

The user's `box_map` callable is invoked from C++ at one place: `Map.h:10` —
each `MapGraph::adjacencies(v)` call computes
`grid_->cover((*f_)(grid_->geometry(v)))`, where `(*f_)` resolves to
`ModelMapF::operator()` (`ModelMapF.h:45-72`), which calls the Python
callable through pybind11 (releasing the GIL transparently). `grid_->cover`
walks the binary subdivision tree to enumerate every grid cell that
intersects the returned image rectangle.

### Subdivision parameters

| Parameter | Meaning |
| --- | --- |
| `subdiv_init` | The phase-space grid is subdivided this many times *before* hierarchical refinement starts. Sets a coarse floor — useful so the first SCC pass already has enough resolution to detect coarse structure. |
| `subdiv_min` | Minimum total subdivision depth at which a Morse-decomposition node is allowed to materialise as a vertex in the output graph. Depths between `init` and `min` exist only as scratch; their decompositions get collapsed into `min`-depth vertices during postorder traversal. |
| `subdiv_max` | The hard ceiling on subdivision depth for any branch. No node above this depth gets refined further. |
| `subdiv_limit` | A budget: a node deeper than `min` whose grid has more than this many cells is left un-refined. Prevents one runaway component from eating the budget. |

**Uniform vs adaptive mode.**

- Setting `subdiv_init == subdiv_min == subdiv_max == k` gives a fixed-depth
  uniform partition: every leaf of the decomposition tree sits at depth `k`,
  no early stopping. This is the **canonical methodology** we adopted (see
  the `latent-morse-coarser-than-analytic` memory). It also enables the
  whole-grid pre-evaluation optimisation in our `uniform_precomputed`
  backend.
- Setting them apart (e.g. `init=16, min=23, max=24`) gives the legacy
  adaptive mode — coarse first, refine only where needed.

The CMGDB internal representation of "uniform" is just the special case
where the binary subdivision tree happens to be a complete tree of fixed
depth `k`. There is no separate code path.

## 6. BoxMap details

The user-supplied `box_map(rect) -> rect` is called once per box per
`MapGraph::adjacencies` query. The conventional implementation calls
`CMGDB.BoxMap(g, rect, mode='corners')` where `g(x): R^d -> R^d` is the
underlying continuous map. With `mode='corners'`:

```python
# ComputeBoxMap.py:32-50, condensed
X = CornerPoints(rect)           # 2^d corner points
Y = [g(x) for x in X]            # 2^d evaluations of g
Y_l = [min over Y]               # axis-wise lower envelope
Y_u = [max over Y]               # axis-wise upper envelope
if padding:
    side = upper - lower
    Y_l -= side; Y_u += side
return Y_l + Y_u                 # flat 2d-length list
```

The `2^d` factor is exact (one evaluation per corner), and the inner loop
is a Python `for` loop over `X`. **This is the loop that dominated our 2D
runs** at ~17μs/torch-call × ~4M boxes = ~70s at subdiv 20. Our
`numpy` and `uniform_precomputed` backends in
`src/latentdynamics/analysis/morse.py` short-circuit this loop:

- `numpy` runs the matmul chain on all `2^d` corners in one shot per box.
- `uniform_precomputed` pre-evaluates the entire fixed-depth corner grid
  once and replaces the per-box `box_map` call with a dict-style lookup.

In both cases we emulate the `mode='corners'` + `padding=True` semantics
exactly (corners → axis-wise min/max → optional inflation by `box_size`).

## 7. Output formats

### `morse_sets` (CSV)

Written by `SaveMorseSets` (`SaveMorseData.py:5-12`). Schema (one row per
box):

```
l_1, l_2, ..., l_d,  u_1, u_2, ..., u_d,  vertex_id
```

- Columns `0..d-1` are the box's lower-corner coordinates.
- Columns `d..2d-1` are the upper corners.
- Column `2d` is the integer vertex id in the Morse graph (0-indexed).

### `morse_graph` (graphviz DOT)

Written by `MorseGraph::save` (`MorseGraph.h:430+` — invoked from inside
`CMGDB.PlotMorseGraph` or via `computeMorseGraph`'s output-file path). Each
vertex carries:

- `label="v : (a_1, a_2, ...)"`: the vertex id followed by its annotations,
  which for `ComputeConleyMorseGraph` are the Conley index polynomial
  strings (one per homology dimension) produced by `conleyIndexString`.
- `shape=ellipse`, `style=filled, fillcolor=#<hex>`: color comes from a
  Python-side palette assigned by `PlotMorseGraph`.

Edges `u -> v` represent reachability: there exists a connecting orbit
from a point in Morse set `u` to one in Morse set `v` in the multi-valued
combinatorial map. Sinks (out-degree 0) are attractors; sources (in-degree
0) are repellers.

### `SingleCMG_statistics.txt`

`CMGDB.cpp:129-139` writes this file in the **current working directory**
when the legacy `MorseGraphIntvalMap` / `MorseGraphMap` entry points are
used. The newer `ComputeConleyMorseGraph` path does not write it. Contents:
grid size, computation time, internal/external memory peaks (gated by
`MEMORYBOOKKEEPING`). We never call the legacy entry points; if you see
this file appear, something other than the standard pipeline ran.

## 8. Preprocessor switches

The defaults in `CMGDB.cpp:11-12`:

```cpp
// #define CMG_VERBOSE       <-- commented out; verbose disabled by default
#define MEMORYBOOKKEEPING    <-- bookkeeping always enabled
```

| Switch | Effect when defined |
| --- | --- |
| `CMG_VERBOSE` | All progress / status prints fire. In stock upstream this is **off**, but several `std::cout` lines were left unwrapped (the legacy "Total Time..." stub, the `nodes_processed % 1000` heartbeat, the `Compute_Morse_Graph` initialisation messages, the `ConleyIndex: calling RelativeMapHomology` print, and the `Dimension d: ...` matrix dump in `RelativeMapHomology.h:438`). In our local archive we added the missing `#ifdef CMG_VERBOSE` wrappers around those sites; a stock 1.3.2 wheel will still print them. |
| `MEMORYBOOKKEEPING` | Tracks peak grid + SCC memory in globals (`Compute_Morse_Graph.hpp:25-27`). With the verbose flag on, the summary block at the end of `Compute_Morse_Graph` is also printed. The bookkeeping itself has negligible cost; the prints are now wrapped in `#ifdef CMG_VERBOSE` locally. |
| `NO_REACHABILITY` (`GraphTheory.hpp:41` and `Compute_Morse_Graph.hpp:305, 325`) | Skips the reachability pass. SCC vertices still get added but with no edges, so the Morse graph collapses to a vertex set. Debug-only. |
| `DO_CONLEY_INDEX` (`Compute_Morse_Graph.hpp:29`) | Included for historical reasons; not on the live code path. Conley index in our pipeline is computed in `CMGDB.cpp:72-78` and not gated by this macro. |
| `CONLEYINDEXCUTOFF` (`Compute_Morse_Graph.hpp:430`, `ConleyIndex.h:232+`) | Truncates homology computation past a given dimension. Off by default. |

To re-enable upstream verbose output locally, build with
`-DCMG_VERBOSE`. Easiest is to add it in `CMakeLists.txt` to the
`pybind11_add_module` target's compile definitions.

## 9. Notes on the source

Things to be aware of when reading or patching CMGDB:

- **The pybind layer is a single TU.** Every binding lives in
  `CMGDB.cpp`; the per-class `<Class>Binding(py::module &m)` functions are
  collected from each header. There is no separate `_cmgdb` directory tree
  of bindings — adding a new exposed method means editing the header where
  the C++ class is defined.
- **`ModelMap` is dead weight for our purposes.** It's a built-in 2D Leslie
  map; the C++ side has no notion of "the user's Python function" except
  through `ModelMapF`, which wraps a `std::function`. Every `Model(...)`
  constructor that takes a Python callable routes through `ModelMapF`.
- **`Grid::cover(geo)` walks the binary subdivision tree.** It is the only
  place the algorithm pays for "which cells does this image rectangle hit?".
  Cost is `O(d * (number of cells touched))`. For the uniform-depth case it
  reduces to an axis-wise floor/ceil lookup; in adaptive mode it descends
  the tree.
- **The `MEMORYBOOKKEEPING` accumulators are process-globals.** Long-lived
  Python processes that call `ComputeConleyMorseGraph` repeatedly will see
  monotonically growing reported "max" values across runs. Not a leak, just
  global state.
- **MPI / stored-graph code in `MapGraph.h`** (`#ifdef CMDB_STORE_GRAPH`,
  lines 83-119) shells out to a script `./COMPUTEGRAPHSCRIPT`. Dead code in
  our build.
- **Boost serialization is mandatory.** Even though we never save grids, the
  build pulls in `boost::serialization` because `SuccinctGrid` /
  `PointerGrid` use `BOOST_CLASS_EXPORT_IMPLEMENT` in `CMGDB.cpp:32, 34`.

## 10. How this project uses CMGDB

For reference, our `compute_morse_graph` (`src/latentdynamics/analysis/
morse.py`) does the following:

1. Build a `box_map(rect)` callable from a torch `latent_map`. Three
   backends are supported via `cmgdb_cfg.box_map_backend`:
   - `pytorch`: wraps `CMGDB.BoxMap(g, rect, padding=padding)` where `g`
     is a torch forward-pass closure.
   - `numpy`: extracts MLP weights once, does a NumPy matmul on the
     `2^d`-corner stack per box. Eliminates PyTorch overhead from the
     CMGDB hot path.
   - `uniform_precomputed`: requires `subdiv_init == subdiv_min ==
     subdiv_max == k` with `k % d == 0`. Evaluates the entire level-`k`
     corner grid in one batched torch forward, then makes `box_map(rect)`
     a centroid-keyed lookup into the precomputed table. No NN
     evaluations in the CMGDB inner loop.
2. Construct `CMGDB.Model(min, max, init, limit, lower, upper, box_map)`.
3. Call `CMGDB.ComputeConleyMorseGraph(model)` and write the two output
   artefacts via our `save_morse_graph_artifacts` helper (DOT + CSV).
4. Re-render to PDF / PNG via `render_morse_from_files` (deferred to the
   `render` stage), using the project palette.

The legacy `subdiv_init < subdiv_min < subdiv_max` adaptive mode is still
supported for reproduction of older figures, but the canonical methodology
is uniform with `padding=False` and `box_map_backend=uniform_precomputed`
(see the `latent-morse-coarser-than-analytic` and `cmgdb-boxmap-bottleneck`
memories).
