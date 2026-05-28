"""Combinatorial-topological analysis on top of CMGDB."""

from .cmgdb_roa import (
    EXACT_ROA_FILENAME,
    CellROA,
    collapse_roa_to_lca,
    compute_and_save_exact_roa,
    compute_exact_roa,
    load_exact_roa,
    save_exact_roa,
)
from .morse import (
    LatentBounds,
    compute_morse_graph,
    infer_latent_bounds,
    make_box_map,
    make_box_map_numpy,
    make_box_map_uniform_precomputed,
)
from .morse_graph_parser import MorseGraph
from .morse_metrics import (
    check_unique_membership,
    find_morse_label_1d,
    find_seed_subdirs,
    get_minimal_labels,
)
from .tolerance import (
    Box,
    Edge,
    MorseSet,
    compute_max_semiconjugacy_error,
    compute_min_boundary_separation,
    distance_point_to_boundary,
    is_in_range,
    orthogonal_distance,
)

__all__ = [
    "Box",
    "CellROA",
    "EXACT_ROA_FILENAME",
    "Edge",
    "LatentBounds",
    "MorseGraph",
    "MorseSet",
    "check_unique_membership",
    "collapse_roa_to_lca",
    "compute_and_save_exact_roa",
    "compute_exact_roa",
    "compute_max_semiconjugacy_error",
    "compute_min_boundary_separation",
    "compute_morse_graph",
    "distance_point_to_boundary",
    "find_morse_label_1d",
    "find_seed_subdirs",
    "get_minimal_labels",
    "infer_latent_bounds",
    "is_in_range",
    "load_exact_roa",
    "make_box_map",
    "make_box_map_numpy",
    "make_box_map_uniform_precomputed",
    "orthogonal_distance",
    "save_exact_roa",
]
