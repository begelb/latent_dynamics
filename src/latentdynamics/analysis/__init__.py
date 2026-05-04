"""Combinatorial-topological analysis on top of CMGDB."""

from .morse import LatentBounds, compute_morse_graph, infer_latent_bounds, make_box_map
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
    "Edge",
    "LatentBounds",
    "MorseSet",
    "check_unique_membership",
    "compute_max_semiconjugacy_error",
    "compute_min_boundary_separation",
    "compute_morse_graph",
    "distance_point_to_boundary",
    "find_morse_label_1d",
    "find_seed_subdirs",
    "get_minimal_labels",
    "infer_latent_bounds",
    "is_in_range",
    "make_box_map",
    "orthogonal_distance",
]
