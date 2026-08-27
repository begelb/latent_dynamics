# CMGDB scaling on the Chafee--Infante d=3 latent map

**Status: resolved 2026-08-25.** The mechanism was identified and fixed in
CMGDB 1.5.0. This project pins `cmgdb>=1.5.0`, and the d=3 study runs on the
released package.

## The problem (historical)

Against CMGDB 1.4.0 the d=3 adaptive stage (subdivisions `21, 24, 33`) was
killed after 1 h 57 m without finishing, and a uniform-18
`ComputeConleyMorseGraph` exceeded 600 s. Cost grew faster than linearly in
the cell count.

## The mechanism

Separating the phases (a Morse-only run against the same persisted box map)
showed graph *construction* was never the problem: uniform-18 construction
took 4.4 s on 1.4.0. The blowup was the Conley phase:
`FiberComplex::preboundary` short-circuited into a full-complex Smith normal
form, roughly O(m^3) per fiber, and Chafee-style maps -- padded corner images
of a learned latent map -- produce large fibers, unlike the Leslie and Henon
maps of CMGDB's standard benchmarks, which is why the term had gone unnoticed.
Two further issues at scale: the multi-gigabyte CSR edge array (1.1e9 edges at
uniform s24) grew by repeated doubling with a ~3x transient memory peak, and
the returned `map_graph` was never cached, so post-processing walks
re-evaluated the map per adjacency query.

## The fix (CMGDB 1.5.0)

- Conley fiber preboundaries use the Morse-reduction path, with a runtime
  boundary-validation check.
- The edge array is reserved up front from a per-chunk projection
  (automatic; tunable via the `reserve_edges` / `reserve_min_edges` kwargs).
- `cache_map_graph=True` returns an eagerly cached `map_graph`, and
  `map_graph.build_cache()` upgrades a lazy one later.

## Measured after the fix

Same persisted box map, same bounds; identical Morse graphs, Morse sets, and
Conley indices in every row.

| computation                       | CMGDB 1.4.0        | CMGDB 1.5.0           |
|-----------------------------------|--------------------|-----------------------|
| uniform 16 (Conley)               | 29.4 s             | 8.1 s                 |
| uniform 18 (Conley)               | > 600 s            | 42.2 s                |
| adaptive 18/20/22 (Conley)        | > 300 s            | 104.6 s               |
| adaptive 21/24/33 (full stage)    | killed at 1 h 57 m | 77.7 s                |
| uniform s24 (Morse only)          | -- (memory-bound)  | 298 s, 9.7 GB, untuned |

The five computations are kept as a permanent regression suite in the CMGDB
repository (`benchmarks/benchmark.py --chafee`), driven by the 12 KB
latent-map weights exported from this study.
