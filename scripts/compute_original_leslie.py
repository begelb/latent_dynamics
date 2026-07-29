"""Compute CMGDB Morse data directly from the analytic Leslie maps.

The maintained default is the two-generation subsystem underlying the paper's
10D contracting example.  The 2D map is not a separately typed formula: it is
the first-two-coordinate projection of the configured 10D map on the invariant
plane where the eight contracting coordinates vanish.  This construction keeps
the parameters and phase-space bounds tied to the model that generated the
latent computation. ``--system 3d`` is an exploratory extension; its default
resolution is a preview and may not separate nearby attractors.

Run from ``code/``::

    python scripts/compute_original_leslie.py
    python scripts/compute_original_leslie.py --system 3d --subdiv 20 22 24
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path

import CMGDB
import numpy as np
from numpy.typing import NDArray

from latentdynamics.analysis.morse_metrics import get_minimal_labels
from latentdynamics.config import load_config
from latentdynamics.systems import LeslieContraction, build_system
from latentdynamics.viz import PALETTE, render_morse_from_files, save_morse_graph_artifacts

CODE_ROOT = Path(__file__).resolve().parents[1]
LOCAL_CMGDB_ROOT = (CODE_ROOT.parent / "archive" / "CMGDB").resolve()
LESLIE_2D_CORRESPONDENCE_PALETTE = (
    PALETTE[1],  # direct 0: periodic attractor -> latent periodic-attractor color
    PALETTE[2],  # direct 1: (0, x^3+1, 0) -> latent node 2
    PALETTE[0],  # direct 2: invariant circle -> latent invariant-circle color
    PALETTE[3],  # direct 3: (0, x^3-1, 0) -> latent node 3
    PALETTE[5],  # direct 4: trivial index, with no latent counterpart
    PALETTE[4],  # direct 5: (0, 0, x-1) -> latent node 4
)


def require_local_cmgdb() -> Path:
    """Fail before a long run unless CMGDB resolves to the maintained local fork."""
    module_path = Path(CMGDB.__file__).resolve()
    if LOCAL_CMGDB_ROOT not in module_path.parents:
        raise RuntimeError(
            "CMGDB must be imported from the maintained local checkout at "
            f"{LOCAL_CMGDB_ROOT}; imported {module_path}. Run `uv sync --all-extras` "
            "from code/ and use code/.venv/bin/python."
        )
    return module_path


def make_adaptive_precomputed_box_map(
    map_function: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    lower: list[float],
    upper: list[float],
    *,
    subdiv_max: int,
    padding: bool,
    max_table_points: int,
) -> Callable[[list[float]], list[float]]:
    """Precompute a NumPy map on every finest-grid corner used by CMGDB.

    Recent CMGDB versions expose this as ``make_precomputed_box_map``.  The
    local implementation keeps this paper computation runnable with the
    declared CMGDB 1.3.2 dependency as well.
    """
    if hasattr(CMGDB, "make_precomputed_box_map"):
        return CMGDB.make_precomputed_box_map(
            map_function,
            lower,
            upper,
            subdiv_max=subdiv_max,
            mode="adaptive",
            padding=padding,
            max_table_points=max_table_points,
        )

    lower_array = np.asarray(lower, dtype=np.float64)
    upper_array = np.asarray(upper, dtype=np.float64)
    dimension = len(lower_array)
    max_axis_depth = (subdiv_max + dimension - 1) // dimension
    boxes_per_axis = 2**max_axis_depth
    corners_per_axis = boxes_per_axis + 1
    table_points = corners_per_axis**dimension
    if table_points > max_table_points:
        raise ValueError(
            f"adaptive precomputed table needs {table_points} corners, exceeding "
            f"the limit {max_table_points}"
        )

    shape = (corners_per_axis,) * dimension
    step = (upper_array - lower_array) / boxes_per_axis
    chunk_size = min(4 * 1024 * 1024, table_points)
    outputs: NDArray[np.float64] | None = None
    output_dimension = -1
    for start in range(0, table_points, chunk_size):
        end = min(start + chunk_size, table_points)
        flat_indices = np.arange(start, end, dtype=np.int64)
        grid_indices = np.stack(np.unravel_index(flat_indices, shape), axis=-1)
        points = lower_array + grid_indices * step
        values = np.asarray(map_function(points), dtype=np.float64)
        if outputs is None:
            output_dimension = int(values.shape[1])
            outputs = np.empty((table_points, output_dimension), dtype=np.float64)
        outputs[start:end] = values

    if outputs is None:
        raise RuntimeError("the corner table unexpectedly contains no points")
    output_grid = outputs.reshape((*shape, output_dimension))
    combinations = np.asarray(list(itertools.product(range(2), repeat=dimension)), dtype=np.int64)
    axis_indices = np.arange(dimension, dtype=np.int64)

    def box_map(rect: list[float]) -> list[float]:
        rectangle = np.asarray(rect, dtype=np.float64)
        lower_index = np.round((rectangle[:dimension] - lower_array) / step).astype(np.int64)
        upper_index = np.round((rectangle[dimension:] - lower_array) / step).astype(np.int64)
        np.clip(lower_index, 0, boxes_per_axis, out=lower_index)
        np.clip(upper_index, 0, boxes_per_axis, out=upper_index)
        endpoints = np.stack([lower_index, upper_index], axis=0)
        corner_indices = endpoints[combinations, axis_indices]
        corners = output_grid[tuple(corner_indices.T)].reshape(2**dimension, output_dimension)
        image_lower = corners.min(axis=0)
        image_upper = corners.max(axis=0)
        if padding:
            box_size = rectangle[dimension:] - rectangle[:dimension]
            image_lower -= box_size
            image_upper += box_size
        return np.concatenate([image_lower, image_upper]).tolist()

    return box_map


def configured_leslie_2d() -> tuple[
    Callable[[NDArray[np.float64]], NDArray[np.float64]],
    LeslieContraction,
]:
    """Return the exact 2D restriction of the maintained 10D Leslie map."""
    config = load_config("leslie_2gen_contraction")
    ambient = build_system(config.system.name, config.system.params)
    if not isinstance(ambient, LeslieContraction) or ambient.dim != 10:
        raise TypeError("leslie_2gen_contraction must configure a 10D LeslieContraction")

    def projected_head(points: NDArray[np.float64]) -> NDArray[np.float64]:
        array = np.asarray(points, dtype=np.float64)
        single = array.ndim == 1
        batch = np.atleast_2d(array)
        if batch.shape[1] != 2:
            raise ValueError(f"expected 2D Leslie points; received shape {array.shape}")
        embedded = np.zeros((batch.shape[0], ambient.dim), dtype=np.float64)
        embedded[:, :2] = batch
        result = ambient.step(embedded)[:, :2]
        return result[0] if single else result

    return projected_head, ambient


def leslie_3d(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """The paper's analytic 3-generation map at theta=(28.9, 29.8, 22.0)."""
    array = np.asarray(points, dtype=np.float64)
    single = array.ndim == 1
    batch = np.atleast_2d(array)
    x0, x1, x2 = batch[:, 0], batch[:, 1], batch[:, 2]
    decay = np.exp(-0.1 * (x0 + x1 + x2))
    result = np.column_stack(((28.9 * x0 + 29.8 * x1 + 22.0 * x2) * decay, 0.7 * x0, 0.7 * x1))
    return result[0] if single else result


def _defaults(
    system: str,
) -> tuple[
    Callable,
    list[float],
    list[float],
    tuple[int, int, int],
    int,
    LeslieContraction | None,
]:
    if system == "2d":
        map_function, ambient = configured_leslie_2d()
        return (
            map_function,
            ambient.lower_bounds[:2].tolist(),
            ambient.upper_bounds[:2].tolist(),
            (27, 29, 30),
            1_500_000_000,
            ambient,
        )
    return (
        leslie_3d,
        [0.0, 0.0, 0.0],
        [220.0, 154.0, 108.0],
        (20, 22, 24),
        1_200_000_000,
        None,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=("2d", "3d"), default="2d")
    parser.add_argument("--subdiv", type=int, nargs=3, metavar=("INIT", "MIN", "MAX"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--subdiv-limit", type=int, default=10_000)
    parser.add_argument(
        "--box-map-backend",
        choices=("adaptive_precomputed", "on_demand"),
        default="adaptive_precomputed",
        help=(
            "Use a finest-grid corner table or evaluate CMGDB.BoxMap as cells are "
            "requested. The on-demand backend is required when the finest table "
            "would be prohibitively large."
        ),
    )
    args = parser.parse_args()

    cmgdb_module_path = require_local_cmgdb()
    print(f"using local CMGDB: {cmgdb_module_path}", flush=True)

    map_function, lower, upper, default_subdiv, max_table_points, ambient = _defaults(args.system)
    subdiv_init, subdiv_min, subdiv_max = tuple(args.subdiv or default_subdiv)
    output = args.output or (
        CODE_ROOT
        / "output"
        / "original_leslie"
        / (
            f"leslie_2d_exact_restriction_s{subdiv_init}_{subdiv_min}_{subdiv_max}"
            if args.system == "2d"
            else f"leslie_3d_s{subdiv_init}_{subdiv_min}_{subdiv_max}"
        )
    )
    morse_dir = output / "MG"
    morse_dir.mkdir(parents=True, exist_ok=True)

    setup_start = time.perf_counter()
    if args.box_map_backend == "adaptive_precomputed":
        print(
            f"precomputing exact Leslie box map on {lower} -> {upper} at subdiv_max={subdiv_max}",
            flush=True,
        )
        box_map = make_adaptive_precomputed_box_map(
            map_function,
            lower,
            upper,
            subdiv_max=subdiv_max,
            padding=True,
            max_table_points=max_table_points,
        )
    else:
        print(
            f"using on-demand exact Leslie box map on {lower} -> {upper}",
            flush=True,
        )

        def box_map(rect: list[float]) -> list[float]:
            return CMGDB.BoxMap(map_function, rect, padding=True)

    setup_seconds = time.perf_counter() - setup_start

    print(f"box-map setup finished in {setup_seconds / 60:.2f} minutes", flush=True)
    print("computing Conley--Morse graph", flush=True)
    compute_start = time.perf_counter()
    model = CMGDB.Model(
        subdiv_min,
        subdiv_max,
        subdiv_init,
        args.subdiv_limit,
        lower,
        upper,
        box_map,
    )
    morse_graph, _ = CMGDB.ComputeConleyMorseGraph(model)
    compute_seconds = time.perf_counter() - compute_start

    dot_path, _ = save_morse_graph_artifacts(morse_graph, morse_dir)
    render_morse_from_files(
        morse_dir,
        bounds_lower=lower,
        bounds_upper=upper,
        palette=LESLIE_2D_CORRESPONDENCE_PALETTE if args.system == "2d" else PALETTE,
        out_dir=output,
        labels_2d=("$x_1$", "$x_2$"),
        min_box_side_frac=0.0025,
    )
    minimal = sorted(get_minimal_labels(dot_path))
    manifest = {
        "system": args.system,
        "map": (
            {
                "implementation": (
                    "projection of latentdynamics.systems.LeslieContraction.step "
                    "onto zero-based coordinates [0,1] with coordinates [2,...,9] "
                    "set to zero"
                ),
                "source_config": "leslie_2gen_contraction.yaml",
                "ambient_dimension": ambient.dim,
                "ambient_parameters": ambient.params,
                "ambient_bounds": {
                    "lower": ambient.lower_bounds.tolist(),
                    "upper": ambient.upper_bounds.tolist(),
                },
                "invariant_plane": "x[2]=...=x[9]=0 (zero-based indexing)",
                "parameter_match_by_construction": True,
            }
            if ambient is not None
            else {
                "implementation": "analytic LeslieModel3D formula",
                "parameters": {
                    "theta": [28.9, 29.8, 22.0],
                    "survival": [0.7, 0.7],
                },
            }
        ),
        "bounds": {"lower": lower, "upper": upper},
        "subdivision": {
            "init": subdiv_init,
            "min": subdiv_min,
            "max": subdiv_max,
            "limit": args.subdiv_limit,
        },
        "padding": True,
        "rendering": {
            "min_box_side_frac": 0.0025,
            "display_only": True,
            "node_colors": (
                {
                    str(node): color
                    for node, color in enumerate(LESLIE_2D_CORRESPONDENCE_PALETTE)
                }
                if args.system == "2d"
                else "default"
            ),
            "color_correspondence": (
                "attractors matched by dynamical role; nonminimal nodes matched by "
                "equal Conley index; unmatched trivial-index direct node 4 shown "
                "in canonical teal"
                if args.system == "2d"
                else None
            ),
        },
        "box_map_backend": args.box_map_backend,
        "cmgdb": {
            "distribution_version": version("CMGDB"),
            "source": str(LOCAL_CMGDB_ROOT.relative_to(CODE_ROOT.parent)),
            "module_path": str(cmgdb_module_path),
        },
        "setup_seconds": round(setup_seconds, 3),
        "compute_seconds": round(compute_seconds, 3),
        "morse_nodes": morse_graph.num_vertices(),
        "minimal_nodes": minimal,
        "minimal_node_count": len(minimal),
        "three_dimensional_preview": args.system == "3d",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print(f"artifacts: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
