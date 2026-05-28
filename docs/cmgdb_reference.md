# CMGDB reference

Reference for the Conley-Morse Graph Database (CMGDB) Python bindings used by this project. Compiled against the installed version at `.venv/lib/python3.13/site-packages/CMGDB/`, cross-checked against the C++ source at `archive/CMGDB/src/CMGDB/_cmgdb/include/database/`.

Use this doc as the single source of truth for the CMGDB API; do not re-derive semantics from experimentation.

## 1. Package layout

```
CMGDB/
  __init__.py            # flat re-export of everything below
  _cmgdb.*.so            # pybind11 C++ core (Model, Grid, MorseGraph, MapGraph,
                         # ComputeConleyMorseGraph, ComputeMorseGraph,
                         # ComputeConleyIndex, MorseGraphMap, MorseGraphIntvalMap)
  BoxMapData.py          # CMGDB.BoxMapData: data-driven box map
  ComputeBoxMap.py       # CMGDB.BoxMap, CornerPoints, CenterPoint, SamplePoints
  SaveMorseData.py       # CMGDB.SaveMorseSets
  LoadMorseSetFile.py    # CMGDB.LoadMorseSetFile
  PlotMorseGraph.py      # CMGDB.PlotMorseGraph
  PlotMorseSets.py       # CMGDB.PlotMorseSets, CMGDB.PlotBoxesScatter
```

Everything is accessible as `CMGDB.<name>`.

## 2. Subdivision parameters (verified against C++ source)

`Compute_Morse_Graph.hpp` line 359:
```cpp
for ( int i = 0; i < (int)Init; ++ i ) phase_space -> subdivide ();
Compute_Morse_Graph ( MG, phase_space, f, Min - Init, Max - Init, Limit );
```

Each call to `phase_space->subdivide()` performs **one bisection** of every leaf in the grid tree (`CompressedTree::subdivide` in `CompressedTree.h:37`), alternating across coordinate axes. So after `k` subdivisions the grid has **`2^k` total cells**.

| param | meaning |
|---|---|
| `subdiv_init` | number of up-front bisections before the Morse algorithm starts. Yields `2^subdiv_init` cells. |
| `subdiv_min` | minimum depth of the Morse decomposition tree. Effective extra-refinement budget is `Min - Init`. |
| `subdiv_max` | maximum depth of the Morse decomposition tree. Effective extra-refinement budget is `Max - Init`. |
| `subdiv_limit` | hard cap on the size of a single Morse-decomposition node, applied only after `depth > Min` (see `Compute_Morse_Graph.hpp:200`). |

**Uniform grid** is the special case `subdiv_init == subdiv_min == subdiv_max`: the `Min - Init = Max - Init = 0` extra refinement budget means no further subdivision happens — the whole computation runs on a fixed `2^k` grid. This is the canonical methodology for this project.

**Adaptive grid** is `subdiv_init < subdiv_min ≤ subdiv_max`. The algorithm refines only cells that haven't resolved into a stable SCC, up to depth `max`, halting earlier on any Morse-set node whose box count exceeds `limit`.

**Common conversions (2D):**

| smax | total cells | per-axis (approx square) |
|---:|---:|---:|
| 10 | 2^10 = 1024 | 32 × 32 |
| 17 | 2^17 ≈ 131k | 362 × 362 |
| 20 | 2^20 ≈ 1M | 1024 × 1024 |
| 24 | 2^24 ≈ 16M | 4096 × 4096 |

## 3. `CMGDB.Model`

Configuration container holding phase-space bounds, subdivision parameters, and the box map. 11 constructor overloads; the ones this project uses:

```python
# Most common (Patrick's pattern):
model = CMGDB.Model(subdiv_min, subdiv_max, subdiv_init, subdiv_limit,
                    lower_bounds, upper_bounds, box_map)

# Minimal (no init/limit; defaults Init=0, Limit=10000):
model = CMGDB.Model(subdiv_min, subdiv_max, lower_bounds, upper_bounds, box_map)
```

`box_map` is a callable `Rect -> Rect`:
- input: list of length `2d`, `[l_1,...,l_d, u_1,...,u_d]`.
- output: list of length `2d` enclosing `f(B)` for the input box `B`.

Useful methods on the instance:
- `phase_dim()`, `param_dim()`
- `phase_lower_bounds()`, `phase_upper_bounds()`, `phase_periodic()`
- `phase_subdiv_init/min/max/limit()`
- `phaseSpace() -> Grid`

## 4. Box maps

Three ways to supply the `box_map` callable to `Model`:

### 4.1 `CMGDB.BoxMap(f, rect, mode, padding, num_pts)`

Stateless utility — evaluates a point function `f: R^d -> R^d` on a set of sample points inside `rect` and returns the bounding box of their images.

```python
def f(x): return [x[0]**2, x[1]]

def box_map(rect):
    return CMGDB.BoxMap(f, rect, mode='corners', padding=False)
```

- `mode='corners'`: evaluate at all `2^d` corners. Default; deterministic.
- `mode='center'`: evaluate at the centre point only. *Forces `padding=True`*.
- `mode='random'`: sample `num_pts` uniform points inside `rect`. Probabilistic, no rigorous outer-cover guarantee.
- `padding=True`: adds one box-width of slop to each output bound. Use when the cover is known to be loose (e.g. centre mode) or as a rigorous Lipschitz cushion.

`f` is called once per sample point; the whole `BoxMap` call is `O(2^d)` for corners mode.

### 4.2 `CMGDB.BoxMapData(X, Y, ...)`

Data-driven box map: pre-computed `(X, Y)` arrays where `Y_i = f(X_i)`. For each query `rect`, it filters `X` to points inside `rect` and returns the bbox of the corresponding `Y` values.

```python
box_map = CMGDB.BoxMapData(
    X, Y,
    map_empty='interp',     # 'interp' | 'outside' | 'terminate'
    lower_bounds=lower,     # required if map_empty='outside'
    upper_bounds=upper,
    domain_padding=True,    # enlarge query rect by (u-l) per dim before filtering
    padding=False,          # add box-width slop to output bounds
)
```

- `map_empty='interp'`: if no `X` falls in `rect`, double `rect` size until non-empty (the *image* of an expanded rect, conservatively).
- `map_empty='outside'`: if no `X` in `rect`, return a box deliberately outside the domain (forces CMGDB to treat as escaping).
- `map_empty='terminate'`: raise.

**Linear scan**: `BoxMapData.map_points` does `np.all((X >= l) & (X <= u), axis=1)` per call. There is a `TODO: Use a KD-Tree or a grid` in the source. For a regular lattice, the right specialization is a direct index conversion — see Section 8.2 below.

### 4.3 Custom callable

Anything matching the `Rect -> Rect` signature works. The project's standard wrapper is `analysis/morse.make_box_map(latent_map)` which evaluates a PyTorch latent map at the corners and returns the corner-bbox.

## 5. Computation entry points

| function | computes | returns |
|---|---|---|
| `ComputeConleyMorseGraph(model)` | Morse decomposition + Conley index per Morse set | `(MorseGraph, MapGraph)` |
| `ComputeMorseGraph(model)` | Morse decomposition only (faster, no Conley index) | `(MorseGraph, MapGraph)` |
| `MorseGraphMap(min, max, lower, upper, output_fname, F)` | Morse decomposition from box map (convenience; no explicit Model) | `MorseGraph` |
| `MorseGraphIntvalMap(min, max, lower, upper, params, output_fname)` | Morse decomposition of a 1D interval map at fixed parameters | `MorseGraph` |
| `ComputeConleyIndex(X_cubes, A_cubes, sizes, periodic, F, acyclic_check)` | Low-level Conley index for a combinatorial pair | `list[str]` |

`ComputeConleyMorseGraph(model)` internally calls `Compute_Morse_Graph(...)` from `Compute_Morse_Graph.hpp`. The box map is called **once per grid cell, sequentially, from C++**. No batching API is exposed.

## 6. `CMGDB.MorseGraph`

Represents the computed Morse decomposition: vertices = Morse sets, edges = reachability.

| method | returns | notes |
|---|---|---|
| `num_vertices()` | int | |
| `vertices()` | `list[int]` | usually `[0, 1, ..., n-1]` |
| `edges()` | `list[(int,int)]` | **transitively reduced** |
| `edges_unreduced()` | `list[(int,int)]` | all edges |
| `adjacencies(v)` | `list[int]` | successors of `v` in reduced graph |
| `adjacencies_unreduced(v)` | `list[int]` | successors in full graph |
| `annotations(v)` | `list[str]` | Conley index strings if computed by `ComputeConleyMorseGraph` |
| `morse_set(v)` | `list[int]` | **grid cell indices** of cells in Morse set `v` |
| `morse_set_boxes(v)` | `list[list[float]]` | **boxes** `[l_1,...,l_d, u_1,...,u_d]` for Morse set `v` |
| `phase_space_box(index)` | `list[float]` | box for a grid cell given its integer index |

Note: `morse_set(v)` returns integer indices into the grid; `phase_space_box(index)` is the inverse map. For a **uniform** grid, the cell indices have a known mathematical mapping to lattice coordinates — useful for direct indexing without dict lookup (Section 8.2).

## 7. I/O and plotting

```python
CMGDB.SaveMorseSets(morse_graph, "morse_sets.csv")
                              # CSV rows: [l_1,...,l_d, u_1,...,u_d, node_id]
loaded = CMGDB.LoadMorseSetFile("morse_sets.csv")   # -> list[list[float]]

# Morse graph rendered via graphviz:
g = CMGDB.PlotMorseGraph(morse_graph)
g.render("morse_graph", format="pdf", cleanup=True)

# Or use morse_graph.graphviz() directly to get a DOT string.

# Morse sets scatter (2D projection):
CMGDB.PlotMorseSets(morse_graph, fig_fname="morse_sets.pdf")
# Accepts either a MorseGraph object, a CSV filename, or a list of boxes.
```

## 8. Common workflows in this project

### 8.1 Analytic system + corners-only

```python
from latentdynamics.systems import build_system
system = build_system("leslie_contraction", params={"th1": 20, "th2": 20, ...})

def box_map(rect):
    return CMGDB.BoxMap(system.step, rect, mode="corners", padding=False)

model = CMGDB.Model(smax, smax, smax, 10000,
                    [0.0, 0.0], [90.0, 70.0], box_map)
morse_graph, _ = CMGDB.ComputeConleyMorseGraph(model)
```

### 8.2 Learned latent map: pre-evaluate the whole uniform grid (proposed optimization)

Under `init = min = max = k`, the grid is fixed and the lattice points are at known half-step positions. Pre-compute the latent map on all unique lattice points in one batched forward pass, then construct a custom `box_map` that does pure index arithmetic.

```python
# Pseudocode for the optimization sketched in output/leslie2d_to_2d/analysis.md §4.1.
import numpy as np

dx = (upper - lower) / 2**(k/d)            # uniform cell width per axis (approx)
n_per_axis = 2**(k/d) + 1                  # number of lattice points per axis
grid_pts = np.array(np.meshgrid(*[
    np.linspace(lower[i], upper[i], n_per_axis) for i in range(d)
])).reshape(d, -1).T                       # (n_corners^d, d)

with torch.no_grad():
    image_pts = latent_map(torch.from_numpy(grid_pts)).numpy()
                                            # one batched forward pass

# Two options:
# (a) Hand (grid_pts, image_pts) to CMGDB.BoxMapData and patch the linear scan:
box_map = CMGDB.BoxMapData(grid_pts, image_pts, map_empty="outside",
                           lower_bounds=lower, upper_bounds=upper)

# (b) Custom callable doing O(1) lattice indexing per box:
def box_map(rect):
    # convert rect bounds -> lattice indices -> 2^d corner slice of image_pts
    # return bounding box of those 2^d image points
    ...

model = CMGDB.Model(k, k, k, 10000, lower, upper, box_map)
morse_graph, _ = CMGDB.ComputeConleyMorseGraph(model)
```

(b) is strictly faster than (a) because `BoxMapData.map_points` is an `O(|X|)` linear scan per call. Either eliminates NN evaluations from CMGDB's inner loop.

### 8.3 Inspecting Morse output

```python
for v in morse_graph.vertices():
    poly = morse_graph.annotations(v)  # e.g. ['(x^3-1, 0, 0)'] for period-3 attractor
    boxes = morse_graph.morse_set_boxes(v)
    targets = morse_graph.adjacencies(v)
    print(f"node {v}: index={poly}, boxes={len(boxes)}, -> {targets}")
```

`adjacencies(v) == []` means `v` is an **attractor** (no outgoing edges in the reduced graph). The number of attractors gives `morse_minimal_nodes` in the profile script.

## 9. Things to watch out for

- **`mode='center'` forces `padding=True`** — see `ComputeBoxMap.py:36`. You cannot have a center-only outer cover without enlargement.
- **`padding=True` semantics** — for `BoxMap`, padding is `box_width` per axis (i.e. `u - l` per axis added to both `y_l` and `y_u`). For `BoxMapData`, the same: padding is the input rect's width.
- **`domain_padding` is different from `padding`** — `domain_padding` enlarges the *query* rect by `(u-l)` before filtering `X`; `padding` enlarges the *output* image rect.
- **`box_map` must return a rectangle** — empty or inverted output causes silent failure.
- **`MorseGraph.edges()` is transitively reduced**; use `edges_unreduced()` if you need the full reachability.
- **Conley index strings** appear only after `ComputeConleyMorseGraph` (not `ComputeMorseGraph` or `MorseGraphMap`), and only when the index is non-trivial.
- **Subdivision in the C++ tree alternates axes** — so smax = 2m+1 (odd) in 2D yields an anisotropic grid (one axis bisected one more time than the other).

## 10. Source-level entry points (when in doubt, read these)

| question | C++ file:line |
|---|---|
| What does `subdivide()` do? | `CompressedTree.h:37` |
| How are `init/min/max` consumed? | `Compute_Morse_Graph.hpp:351-389` |
| When does `limit` halt subdivision? | `Compute_Morse_Graph.hpp:200-203` |
| How is the box map called? | `ModelMapF.h` (`operator()`) |
| What does `morse_set_boxes` extract? | `MorseGraph.h` |

## 11. Project usage references

| location | what it shows |
|---|---|
| `notebooks/Example_Leslie_model.ipynb` | uniform grid + `padding=False` reaches bistability at smax=17 for Leslie th=20 |
| `archive/patrick/2D_leslie_base_computation.py` | adaptive analytic baseline (`init=16, min=23, max=24`) |
| `archive/patrick/Leslie3D/` | adaptive learned-map runs (the 18-hour 10D-style computation) |
| `archive/marcio/scripts/` | Chafee-Infante PDE pipeline (64-mode spectral, 2D latent, tanh) |
| `src/latentdynamics/analysis/morse.py` | this project's `make_box_map` wrapper |
| `scripts/profile_cmgdb_2d.py` | uniform-vs-adaptive sweep driver |
| `scripts/optimize_box_map.py` | benchmark of corners-vs-numpy-vs-center BoxMap variants |
| `output/leslie2d_to_2d/analysis.md` | the canonical 2D analysis report |
