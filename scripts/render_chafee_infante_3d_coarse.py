"""Render the connection-complete three-dimensional Chafee--Infante quotient.

The coarse Morse boxes form an adaptive dyadic grid.  For the three-dimensional
view, each adaptive box is expanded only in an integer terminal-grid index
space, and the exact exposed terminal faces of the union are cached.  This
preserves the computed union without a convex hull, interpolation, point
sampling, or geometric inflation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from numpy.typing import NDArray

from latentdynamics.viz import (
    plot_morse_set_projections_from_csv,
    render_morse_graph_from_dot,
)
from latentdynamics.viz.morse_plots import (
    _cubical_edgecolors,
    _subtly_shaded_cubical_facecolors,
)
from latentdynamics.viz.style import (
    apply_paper_style,
    save_figure,
    save_latent_figure,
)

CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = (
    CODE_ROOT
    / "output"
    / "chafee_latent_dimension_study"
    / "latent_3d"
    / "seed_0"
)
DEFAULT_SOURCE = DEFAULT_RUN / "MG_adaptive_coarse_marcio"
DEFAULT_OUTPUT = DEFAULT_SOURCE / "render_identity"

PALETTE = ("#FFB000", "#DC267F", "#7F7F7F")
LEGEND_LABELS = {
    0: "$M(0^-)$",
    1: "$M(0^+)$",
    2: "$M(1)$",
}
LABEL_DRAW_ORDER = (2, 0, 1)
SURFACE_CACHE_SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_bounds(path: Path) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    lower = np.asarray(payload["lower"], dtype=np.float64)
    upper = np.asarray(payload["upper"], dtype=np.float64)
    if lower.shape != (3,) or upper.shape != (3,) or not np.all(lower < upper):
        raise ValueError(f"invalid 3-D bounds in {path}")
    return lower, upper


def _load_boxes(
    path: Path,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
]:
    data = np.loadtxt(path, delimiter=",", ndmin=2, dtype=np.float64)
    if data.shape[1] != 7:
        raise ValueError(f"{path} has shape {data.shape}; expected (n, 7)")
    raw_labels = data[:, 6]
    labels = np.rint(raw_labels).astype(np.int64)
    if not np.allclose(raw_labels, labels) or set(np.unique(labels)) != {0, 1, 2}:
        raise ValueError("coarse Morse-set labels must be exactly {0, 1, 2}")
    widths = data[:, 3:6] - data[:, :3]
    if np.any(widths <= 0.0):
        raise ValueError("every coarse Morse box must have positive widths")
    terminal_widths = widths.min(axis=0)
    scales = np.rint(widths / terminal_widths).astype(np.int64)
    if not np.allclose(
        widths,
        scales * terminal_widths,
        rtol=1e-11,
        atol=1e-13,
    ):
        raise ValueError(
            "coarse Morse boxes do not have integer terminal-grid scales"
        )
    return data, labels, scales, terminal_widths


def _aligned_lower_indices(
    data: NDArray[np.float64],
    bounds_lower: NDArray[np.float64],
    terminal_widths: NDArray[np.float64],
) -> NDArray[np.int64]:
    indices = np.rint(
        (data[:, :3] - bounds_lower) / terminal_widths
    ).astype(np.int64)
    reconstructed = bounds_lower + indices * terminal_widths
    if not np.allclose(data[:, :3], reconstructed, rtol=1e-11, atol=1e-12):
        raise ValueError("coarse Morse boxes are not aligned to the terminal grid")
    return indices


def _grid_shape(
    bounds_lower: NDArray[np.float64],
    bounds_upper: NDArray[np.float64],
    terminal_widths: NDArray[np.float64],
) -> NDArray[np.int64]:
    shape = np.rint(
        (bounds_upper - bounds_lower) / terminal_widths
    ).astype(np.int64)
    if not np.allclose(
        bounds_lower + shape * terminal_widths,
        bounds_upper,
        rtol=1e-11,
        atol=1e-12,
    ):
        raise ValueError("CMGDB bounds do not align to the terminal grid")
    if np.any(shape <= 0):
        raise ValueError(f"invalid terminal grid shape {shape.tolist()}")
    return shape


def _expand_terminal_voxel_keys(
    lower_indices: NDArray[np.int64],
    scales: NDArray[np.int64],
    labels: NDArray[np.int64],
    shape: NDArray[np.int64],
) -> tuple[NDArray[np.uint64], dict[str, Any]]:
    scale_rows, scale_counts = np.unique(scales, axis=0, return_counts=True)
    terminal_counts = scale_counts * np.prod(scale_rows, axis=1)
    total = int(terminal_counts.sum())
    volume = int(np.prod(shape))
    keys = np.empty(total, dtype=np.uint64)
    offset = 0

    for scale in scale_rows:
        mask = np.all(scales == scale, axis=1)
        rows = np.flatnonzero(mask)
        sx, sy, sz = (int(value) for value in scale)
        ox, oy, oz = np.meshgrid(
            np.arange(sx, dtype=np.int64),
            np.arange(sy, dtype=np.int64),
            np.arange(sz, dtype=np.int64),
            indexing="ij",
        )
        local_offsets = (
            (ox.reshape(-1) * int(shape[1]) + oy.reshape(-1))
            * int(shape[2])
            + oz.reshape(-1)
        )
        base = (
            (lower_indices[rows, 0] * int(shape[1]) + lower_indices[rows, 1])
            * int(shape[2])
            + lower_indices[rows, 2]
        )
        geometry = (base[:, None] + local_offsets[None, :]).reshape(-1)
        expanded_labels = np.repeat(labels[rows], local_offsets.size)
        count = geometry.size
        keys[offset : offset + count] = (
            expanded_labels.astype(np.uint64) * np.uint64(volume)
            + geometry.astype(np.uint64)
        )
        offset += count

    if offset != total:
        raise RuntimeError(f"terminal expansion wrote {offset} keys, expected {total}")
    keys.sort()
    if keys.size > 1 and np.any(keys[1:] == keys[:-1]):
        raise ValueError("same-label adaptive boxes overlap on terminal voxels")

    geometry_by_label: dict[int, NDArray[np.uint64]] = {}
    label_terminal_counts: dict[str, int] = {}
    for label in (0, 1, 2):
        start = int(np.searchsorted(keys, np.uint64(label * volume), side="left"))
        stop = int(
            np.searchsorted(keys, np.uint64((label + 1) * volume), side="left")
        )
        geometry = keys[start:stop] - np.uint64(label * volume)
        geometry_by_label[label] = geometry
        label_terminal_counts[str(label)] = int(geometry.size)

    overlap_counts: dict[str, int] = {}
    for left, right in ((0, 1), (0, 2), (1, 2)):
        small = geometry_by_label[left]
        large = geometry_by_label[right]
        locations = np.searchsorted(large, small)
        valid = locations < large.size
        overlap = int(
            np.count_nonzero(
                valid
                & (
                    large[
                        np.minimum(locations, max(large.size - 1, 0))
                    ]
                    == small
                )
            )
        ) if large.size else 0
        overlap_counts[f"{left},{right}"] = overlap
    if any(overlap_counts.values()):
        raise ValueError(
            f"coarse labels overlap on terminal voxels: {overlap_counts}"
        )

    diagnostics = {
        "adaptive_box_count": int(labels.size),
        "terminal_voxel_count": total,
        "terminal_voxel_counts_by_label": label_terminal_counts,
        "terminal_voxel_overlap_counts": overlap_counts,
        "adaptive_scale_distribution": [
            {
                "scale": scale.tolist(),
                "adaptive_boxes": int(count),
                "terminal_voxels": int(terminal_count),
            }
            for scale, count, terminal_count in zip(
                scale_rows,
                scale_counts,
                terminal_counts,
                strict=True,
            )
        ],
    }
    return keys, diagnostics


def _geometry_by_label(
    keys: NDArray[np.uint64],
    shape: NDArray[np.int64],
) -> dict[int, NDArray[np.uint64]]:
    volume = int(np.prod(shape))
    result: dict[int, NDArray[np.uint64]] = {}
    for label in (0, 1, 2):
        start = int(np.searchsorted(keys, np.uint64(label * volume), side="left"))
        stop = int(
            np.searchsorted(keys, np.uint64((label + 1) * volume), side="left")
        )
        result[label] = keys[start:stop] - np.uint64(label * volume)
    return result


def _decode_geometry(
    geometry: NDArray[np.uint64],
    shape: NDArray[np.int64],
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    yz = np.uint64(int(shape[1]) * int(shape[2]))
    nz = np.uint64(int(shape[2]))
    x = (geometry // yz).astype(np.int64)
    remainder = geometry % yz
    y = (remainder // nz).astype(np.int64)
    z = (remainder % nz).astype(np.int64)
    return x, y, z


def _neighbor_is_occupied(
    geometry: NDArray[np.uint64],
    neighbor: NDArray[np.uint64],
    valid: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    occupied = np.zeros(geometry.size, dtype=bool)
    if not np.any(valid):
        return occupied
    query = neighbor[valid]
    locations = np.searchsorted(geometry, query)
    found = locations < geometry.size
    safe = np.minimum(locations, max(geometry.size - 1, 0))
    found &= geometry[safe] == query
    occupied[np.flatnonzero(valid)] = found
    return occupied


def _encode_face(
    axis: int,
    plane: NDArray[np.int64],
    u: NDArray[np.int64],
    v: NDArray[np.int64],
    base: int,
) -> NDArray[np.uint64]:
    return (
        (
            (
                np.uint64(axis) * np.uint64(base)
                + plane.astype(np.uint64)
            )
            * np.uint64(base)
            + u.astype(np.uint64)
        )
        * np.uint64(base)
        + v.astype(np.uint64)
    )


def _exposed_face_codes_for_label(
    geometry: NDArray[np.uint64],
    shape: NDArray[np.int64],
    face_base: int,
) -> tuple[NDArray[np.uint64], NDArray[np.int8]]:
    x, y, z = _decode_geometry(geometry, shape)
    strides = (
        np.uint64(int(shape[1]) * int(shape[2])),
        np.uint64(int(shape[2])),
        np.uint64(1),
    )
    coordinates = (x, y, z)
    codes: list[NDArray[np.uint64]] = []
    sides: list[NDArray[np.int8]] = []

    for axis in range(3):
        coordinate = coordinates[axis]
        if axis == 0:
            first, second = y, z
        elif axis == 1:
            first, second = x, z
        else:
            first, second = x, y

        valid_minus = coordinate > 0
        minus_neighbor = geometry - strides[axis]
        occupied_minus = _neighbor_is_occupied(
            geometry,
            minus_neighbor,
            valid_minus,
        )
        exposed_minus = ~occupied_minus
        codes.append(
            _encode_face(
                axis,
                coordinate[exposed_minus],
                first[exposed_minus],
                second[exposed_minus],
                face_base,
            )
        )
        sides.append(np.full(np.count_nonzero(exposed_minus), -1, dtype=np.int8))

        valid_plus = coordinate < int(shape[axis]) - 1
        plus_neighbor = geometry + strides[axis]
        occupied_plus = _neighbor_is_occupied(
            geometry,
            plus_neighbor,
            valid_plus,
        )
        exposed_plus = ~occupied_plus
        codes.append(
            _encode_face(
                axis,
                coordinate[exposed_plus] + 1,
                first[exposed_plus],
                second[exposed_plus],
                face_base,
            )
        )
        sides.append(np.full(np.count_nonzero(exposed_plus), 1, dtype=np.int8))

    face_codes = np.concatenate(codes)
    face_sides = np.concatenate(sides)
    if np.unique(face_codes).size != face_codes.size:
        raise ValueError("one coarse label generated duplicate exposed faces")
    return face_codes, face_sides


def _resolve_cross_label_interfaces(
    codes_by_label: dict[int, NDArray[np.uint64]],
    sides_by_label: dict[int, NDArray[np.int8]],
) -> tuple[NDArray[np.uint64], NDArray[np.int8], NDArray[np.int8], dict[str, int]]:
    adjacency_counts: dict[str, int] = {}
    for left, right in ((0, 1), (0, 2), (1, 2)):
        adjacency_counts[f"{left},{right}"] = int(
            np.intersect1d(
                codes_by_label[left],
                codes_by_label[right],
                assume_unique=True,
            ).size
        )

    # At a shared interface draw one deterministic face.  Attractor faces have
    # priority over M(1), and node 0 has priority only if the two attractors
    # touch (which is not expected for this computation).
    retained_codes: list[NDArray[np.uint64]] = []
    retained_sides: list[NDArray[np.int8]] = []
    retained_labels: list[NDArray[np.int8]] = []
    higher_priority = np.empty(0, dtype=np.uint64)
    for label in (0, 1, 2):
        codes = codes_by_label[label]
        sides = sides_by_label[label]
        keep = (
            np.ones(codes.size, dtype=bool)
            if higher_priority.size == 0
            else ~np.isin(codes, higher_priority, assume_unique=False)
        )
        retained_codes.append(codes[keep])
        retained_sides.append(sides[keep])
        retained_labels.append(
            np.full(np.count_nonzero(keep), label, dtype=np.int8)
        )
        higher_priority = np.unique(
            np.concatenate((higher_priority, codes[keep]))
        )

    return (
        np.concatenate(retained_codes),
        np.concatenate(retained_sides),
        np.concatenate(retained_labels),
        adjacency_counts,
    )


def _projection_overlap_diagnostics(
    geometry_by_label: dict[int, NDArray[np.uint64]],
    shape: NDArray[np.int64],
) -> dict[str, Any]:
    decoded = {
        label: _decode_geometry(geometry, shape)
        for label, geometry in geometry_by_label.items()
    }
    result: dict[str, Any] = {}
    for i, j in ((0, 1), (0, 2), (1, 2)):
        footprints: dict[int, NDArray[np.int64]] = {}
        for label, coordinates in decoded.items():
            footprints[label] = np.unique(
                coordinates[i] * int(shape[j]) + coordinates[j]
            )
        pairwise = {}
        for left, right in ((0, 1), (0, 2), (1, 2)):
            pairwise[f"{left},{right}"] = int(
                np.intersect1d(
                    footprints[left],
                    footprints[right],
                    assume_unique=True,
                ).size
            )
        triple = np.intersect1d(
            np.intersect1d(
                footprints[0],
                footprints[1],
                assume_unique=True,
            ),
            footprints[2],
            assume_unique=True,
        )
        result[f"z{i + 1},z{j + 1}"] = {
            "footprint_counts_by_label": {
                str(label): int(footprint.size)
                for label, footprint in footprints.items()
            },
            "pairwise_overlap_counts": pairwise,
            "triple_overlap_count": int(triple.size),
        }
    return result


def _build_surface_cache(
    source: Path,
    cache_dir: Path,
    bounds_lower: NDArray[np.float64],
    bounds_upper: NDArray[np.float64],
) -> dict[str, Any]:
    data, labels, scales, terminal_widths = _load_boxes(source)
    lower_indices = _aligned_lower_indices(
        data,
        bounds_lower,
        terminal_widths,
    )
    shape = _grid_shape(bounds_lower, bounds_upper, terminal_widths)
    if np.any(lower_indices < 0) or np.any(lower_indices + scales > shape):
        raise ValueError("coarse Morse boxes extend beyond the CMGDB bounds")

    keys, expansion = _expand_terminal_voxel_keys(
        lower_indices,
        scales,
        labels,
        shape,
    )
    by_label = _geometry_by_label(keys, shape)
    face_base = int(max(shape) + 1)
    codes_by_label: dict[int, NDArray[np.uint64]] = {}
    sides_by_label: dict[int, NDArray[np.int8]] = {}
    for label, geometry in by_label.items():
        codes, sides = _exposed_face_codes_for_label(
            geometry,
            shape,
            face_base,
        )
        codes_by_label[label] = codes
        sides_by_label[label] = sides

    face_codes, face_sides, face_labels, adjacency = (
        _resolve_cross_label_interfaces(codes_by_label, sides_by_label)
    )
    projection_diagnostics = _projection_overlap_diagnostics(by_label, shape)

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "surface_face_codes.npy", face_codes)
    np.save(cache_dir / "surface_face_sides.npy", face_sides)
    np.save(cache_dir / "surface_face_labels.npy", face_labels)
    diagnostics = {
        "schema_version": SURFACE_CACHE_SCHEMA_VERSION,
        "source": {
            "path": str(source),
            "sha256": _sha256(source),
        },
        "bounds_lower": bounds_lower.tolist(),
        "bounds_upper": bounds_upper.tolist(),
        "terminal_widths": terminal_widths.tolist(),
        "terminal_grid_shape": shape.tolist(),
        "face_encoding_base": face_base,
        **expansion,
        "same_label_exposed_face_counts": {
            str(label): int(codes.size)
            for label, codes in codes_by_label.items()
        },
        "cross_label_face_adjacency_counts": adjacency,
        "rendered_face_counts_by_label": {
            str(label): int(np.count_nonzero(face_labels == label))
            for label in (0, 1, 2)
        },
        "rendered_face_count": int(face_codes.size),
        "cross_label_interface_rule": (
            "one deterministic face; node 0 then node 1 then node 2 priority"
        ),
        "projection_terminal_footprints": projection_diagnostics,
    }
    cache_files = (
        "surface_face_codes.npy",
        "surface_face_sides.npy",
        "surface_face_labels.npy",
    )
    diagnostics["cache_files"] = {
        name: {
            "path": str(cache_dir / name),
            "sha256": _sha256(cache_dir / name),
        }
        for name in cache_files
    }
    (cache_dir / "surface_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return diagnostics


def _surface_cache_matches(
    diagnostics: dict[str, Any],
    *,
    source: Path,
    required: tuple[Path, ...],
    bounds_lower: NDArray[np.float64],
    bounds_upper: NDArray[np.float64],
) -> bool:
    if diagnostics.get("schema_version") != SURFACE_CACHE_SCHEMA_VERSION:
        return False
    if diagnostics.get("source", {}).get("sha256") != _sha256(source):
        return False

    try:
        cached_lower = np.asarray(diagnostics["bounds_lower"], dtype=np.float64)
        cached_upper = np.asarray(diagnostics["bounds_upper"], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return False
    if not (
        np.array_equal(cached_lower, bounds_lower)
        and np.array_equal(cached_upper, bounds_upper)
    ):
        return False

    cache_files = diagnostics.get("cache_files")
    if not isinstance(cache_files, dict):
        return False
    for path in required:
        metadata = cache_files.get(path.name)
        if not isinstance(metadata, dict):
            return False
        expected_sha256 = metadata.get("sha256")
        if (
            not isinstance(expected_sha256, str)
            or expected_sha256 != _sha256(path)
        ):
            return False
    return True


def _load_or_build_surface_cache(
    source: Path,
    cache_dir: Path,
    bounds_lower: NDArray[np.float64],
    bounds_upper: NDArray[np.float64],
) -> dict[str, Any]:
    diagnostics_path = cache_dir / "surface_diagnostics.json"
    required = (
        cache_dir / "surface_face_codes.npy",
        cache_dir / "surface_face_sides.npy",
        cache_dir / "surface_face_labels.npy",
    )
    if diagnostics_path.is_file() and all(path.is_file() for path in required):
        try:
            diagnostics = json.loads(
                diagnostics_path.read_text(encoding="utf-8")
            )
        except (OSError, TypeError, ValueError):
            diagnostics = None
        if isinstance(diagnostics, dict) and _surface_cache_matches(
            diagnostics,
            source=source,
            required=required,
            bounds_lower=bounds_lower,
            bounds_upper=bounds_upper,
        ):
            return diagnostics
    return _build_surface_cache(
        source,
        cache_dir,
        bounds_lower,
        bounds_upper,
    )


def _decode_faces(
    codes: NDArray[np.uint64],
    sides: NDArray[np.int8],
    *,
    encoding_base: int,
    origin: NDArray[np.float64],
    widths: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    base = np.uint64(encoding_base)
    v = (codes % base).astype(np.int64)
    remainder = codes // base
    u = (remainder % base).astype(np.int64)
    remainder //= base
    plane = (remainder % base).astype(np.int64)
    axis = (remainder // base).astype(np.int64)
    if not np.all(np.isin(axis, (0, 1, 2))):
        raise ValueError("cached face code contains an invalid axis")

    faces = np.empty((codes.size, 4, 3), dtype=np.float64)
    for current_axis in (0, 1, 2):
        for side in (-1, 1):
            mask = (axis == current_axis) & (sides == side)
            if not np.any(mask):
                continue
            p = plane[mask]
            first = u[mask]
            second = v[mask]
            if current_axis == 0:
                x = origin[0] + p * widths[0]
                y0 = origin[1] + first * widths[1]
                y1 = y0 + widths[1]
                z0 = origin[2] + second * widths[2]
                z1 = z0 + widths[2]
                if side == -1:
                    faces[mask] = np.stack(
                        (
                            np.column_stack((x, y0, z0)),
                            np.column_stack((x, y1, z0)),
                            np.column_stack((x, y1, z1)),
                            np.column_stack((x, y0, z1)),
                        ),
                        axis=1,
                    )
                else:
                    faces[mask] = np.stack(
                        (
                            np.column_stack((x, y0, z0)),
                            np.column_stack((x, y0, z1)),
                            np.column_stack((x, y1, z1)),
                            np.column_stack((x, y1, z0)),
                        ),
                        axis=1,
                    )
            elif current_axis == 1:
                y = origin[1] + p * widths[1]
                x0 = origin[0] + first * widths[0]
                x1 = x0 + widths[0]
                z0 = origin[2] + second * widths[2]
                z1 = z0 + widths[2]
                if side == -1:
                    faces[mask] = np.stack(
                        (
                            np.column_stack((x0, y, z0)),
                            np.column_stack((x0, y, z1)),
                            np.column_stack((x1, y, z1)),
                            np.column_stack((x1, y, z0)),
                        ),
                        axis=1,
                    )
                else:
                    faces[mask] = np.stack(
                        (
                            np.column_stack((x0, y, z0)),
                            np.column_stack((x1, y, z0)),
                            np.column_stack((x1, y, z1)),
                            np.column_stack((x0, y, z1)),
                        ),
                        axis=1,
                    )
            else:
                z = origin[2] + p * widths[2]
                x0 = origin[0] + first * widths[0]
                x1 = x0 + widths[0]
                y0 = origin[1] + second * widths[1]
                y1 = y0 + widths[1]
                if side == -1:
                    faces[mask] = np.stack(
                        (
                            np.column_stack((x0, y0, z)),
                            np.column_stack((x1, y0, z)),
                            np.column_stack((x1, y1, z)),
                            np.column_stack((x0, y1, z)),
                        ),
                        axis=1,
                    )
                else:
                    faces[mask] = np.stack(
                        (
                            np.column_stack((x0, y0, z)),
                            np.column_stack((x0, y1, z)),
                            np.column_stack((x1, y1, z)),
                            np.column_stack((x1, y0, z)),
                        ),
                        axis=1,
                    )
    return faces, axis


def _configure_cubical_axes(
    ax: Any,
    *,
    show_ticks: bool = False,
    show_axis_labels: bool = False,
) -> None:
    """Apply the minimal paper frame used by every coarse cubical view."""

    ax.set_xlabel("$z_1$" if show_axis_labels else "", labelpad=5)
    ax.set_ylabel("$z_2$" if show_axis_labels else "", labelpad=5)
    ax.set_zlabel("")
    if show_axis_labels:
        ax.text2D(
            0.95 if show_ticks else 0.90,
            0.60 if show_ticks else 0.55,
            "$z_3$",
            transform=ax.transAxes,
            rotation=90,
            rotation_mode="anchor",
            ha="center",
            va="center",
            clip_on=False,
        )
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        if not show_ticks:
            axis.set_ticks([])
        axis.pane.set_visible(False)
        axis.line.set_color((0.12, 0.12, 0.12, 0.72))
    ax.grid(False)


def _render_cubical(
    cache_dir: Path,
    diagnostics: dict[str, Any],
    output: Path,
    *,
    show_legend: bool,
    elev: float,
    azim: float,
    show_ticks: bool = False,
    show_axis_labels: bool = False,
    formats: tuple[str, ...] = ("pdf", "png"),
    dpi: int = 300,
) -> list[Path]:
    codes = np.load(cache_dir / "surface_face_codes.npy", mmap_mode="r")
    sides = np.load(cache_dir / "surface_face_sides.npy", mmap_mode="r")
    labels = np.load(cache_dir / "surface_face_labels.npy", mmap_mode="r")
    origin = np.asarray(diagnostics["bounds_lower"], dtype=np.float64)
    widths = np.asarray(diagnostics["terminal_widths"], dtype=np.float64)
    faces, _ = _decode_faces(
        codes,
        sides,
        encoding_base=int(diagnostics["face_encoding_base"]),
        origin=origin,
        widths=widths,
    )

    apply_paper_style()
    facecolors = _subtly_shaded_cubical_facecolors(
        faces,
        labels,
        PALETTE,
        light_azdeg=300.0,
        light_altdeg=55.0,
        strength=0.32,
        highlight_strength=0.12,
    )
    attractor_faces = labels < 2
    facecolors[attractor_faces] = _subtly_shaded_cubical_facecolors(
        faces[attractor_faces],
        labels[attractor_faces],
        PALETTE,
        light_azdeg=300.0,
        light_altdeg=55.0,
        strength=0.16,
        highlight_strength=0.12,
    )
    edgecolors = _cubical_edgecolors(
        labels,
        PALETTE,
        alpha=0.10,
        light_edges_on_dark_faces=True,
    )
    edgecolors[attractor_faces, 3] = 0.32
    fig = plt.figure(figsize=(6.14, 5.25), layout="constrained")
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(
        Poly3DCollection(
            faces,
            facecolors=facecolors,
            edgecolors=edgecolors,
            linewidths=0.035,
            alpha=0.99,
            rasterized=True,
            zsort="average",
            shade=False,
        )
    )
    occupied_lower = faces.reshape(-1, 3).min(axis=0)
    occupied_upper = faces.reshape(-1, 3).max(axis=0)
    span = occupied_upper - occupied_lower
    margin = np.maximum(0.035 * span, widths)
    ax.set_xlim(
        float(occupied_lower[0] - margin[0]),
        float(occupied_upper[0] + margin[0]),
    )
    ax.set_ylim(
        float(occupied_lower[1] - margin[1]),
        float(occupied_upper[1] + margin[1]),
    )
    ax.set_zlim(
        float(occupied_lower[2] - margin[2]),
        float(occupied_upper[2] + margin[2]),
    )
    ax.set_box_aspect(span)
    ax.set_proj_type("ortho")
    ax.view_init(elev=elev, azim=azim)
    _configure_cubical_axes(
        ax,
        show_ticks=show_ticks,
        show_axis_labels=show_axis_labels,
    )
    if show_legend:
        handles = [
            mpatches.Patch(
                facecolor=PALETTE[label],
                edgecolor=(0.08, 0.08, 0.08, 0.25),
                label=LEGEND_LABELS[label],
            )
            for label in (0, 1, 2)
        ]
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.01),
            ncol=3,
            frameon=False,
            handlelength=1.0,
            columnspacing=0.9,
        )
    return save_figure(
        fig,
        output,
        formats=formats,
        close=True,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.03,
    )


def _render_projections(
    source: Path,
    output: Path,
    bounds_lower: NDArray[np.float64],
    bounds_upper: NDArray[np.float64],
) -> dict[str, list[str]]:
    rendered: dict[str, list[str]] = {}
    for pair in ((0, 1), (0, 2), (1, 2)):
        plots = plot_morse_set_projections_from_csv(
            source,
            pairs=(pair,),
            bounds_lower=bounds_lower,
            bounds_upper=bounds_upper,
            palette=PALETTE,
            paper_style=True,
            min_box_side_frac=0.0,
            label_draw_order=LABEL_DRAW_ORDER,
        )
        plot = plots[pair]
        paths = save_latent_figure(
            plot.fig,
            output / f"morse_sets_z{pair[0] + 1}_z{pair[1] + 1}",
            formats=("pdf", "png"),
            close=True,
        )
        rendered[f"z{pair[0] + 1},z{pair[1] + 1}"] = [
            str(path) for path in paths
        ]
    return rendered


def _render_overview(output: Path) -> list[Path]:
    images = (
        output / "morse_graph.png",
        output / "morse_sets_z1_z2.png",
        output / "morse_sets_z1_z3.png",
        output / "morse_sets_z2_z3.png",
    )
    if not all(path.is_file() for path in images):
        raise FileNotFoundError("cannot compose overview before component PNGs")
    apply_paper_style()
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 7.5), layout="constrained")
    for index, (ax, path) in enumerate(zip(axes.flat, images, strict=True)):
        ax.imshow(plt.imread(path))
        ax.set_axis_off()
        ax.text(
            0.5,
            -0.03,
            f"({chr(ord('a') + index)})",
            transform=ax.transAxes,
            ha="center",
            va="top",
        )
    return save_figure(
        fig,
        output / "coarse_morse_overview",
        formats=("pdf", "png"),
        close=True,
        dpi=240,
        bbox_inches="tight",
        pad_inches=0.03,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = _parser().parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    morse_sets = source / "morse_sets"
    morse_graph = source / "morse_graph"
    bounds_path = source.parent / "bounds.json"
    for path in (morse_sets, morse_graph, bounds_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    output.mkdir(parents=True, exist_ok=True)
    cache_dir = source / "surface_cache_identity"
    bounds_lower, bounds_upper = _load_bounds(bounds_path)

    diagnostics = _load_or_build_surface_cache(
        morse_sets,
        cache_dir,
        bounds_lower,
        bounds_upper,
    )
    graph_outputs = render_morse_graph_from_dot(
        morse_graph,
        output,
        basename="morse_graph",
        formats=("pdf", "png"),
    )
    projection_outputs = _render_projections(
        morse_sets,
        output,
        bounds_lower,
        bounds_upper,
    )
    cubical_outputs = _render_cubical(
        cache_dir,
        diagnostics,
        output / "morse_sets_cubical_3d",
        show_legend=True,
        elev=3.0,
        azim=-55.0,
    )
    cubical_no_legend_outputs = _render_cubical(
        cache_dir,
        diagnostics,
        output / "morse_sets_cubical_3d_no_legend",
        show_legend=False,
        elev=3.0,
        azim=-55.0,
    )
    cubical_elevated_outputs = _render_cubical(
        cache_dir,
        diagnostics,
        output / "morse_sets_cubical_3d_elevated",
        show_legend=True,
        elev=22.0,
        azim=-55.0,
    )
    overview_outputs = _render_overview(output)

    all_outputs = [
        *graph_outputs,
        *(
            Path(path)
            for paths in projection_outputs.values()
            for path in paths
        ),
        *cubical_outputs,
        *cubical_no_legend_outputs,
        *cubical_elevated_outputs,
        *overview_outputs,
    ]
    manifest = {
        "schema_version": 1,
        "source": {
            "directory": str(source),
            "morse_graph_sha256": _sha256(morse_graph),
            "morse_sets_sha256": _sha256(morse_sets),
        },
        "display": {
            "node_labels": {
                str(label): text
                for label, text in LEGEND_LABELS.items()
            },
            "palette": {
                str(label): PALETTE[label]
                for label in (0, 1, 2)
            },
            "projection_label_draw_order": list(LABEL_DRAW_ORDER),
            "projection_box_inflation": 0.0,
            "cubical_geometry": (
                "exact exposed terminal-grid faces of the adaptive box union"
            ),
            "cubical_cameras": {
                "primary": {"elev": 3.0, "azim": -55.0},
                "elevated": {"elev": 22.0, "azim": -55.0},
            },
            "cubical_ticks": False,
            "cubical_axis_labels": False,
        },
        "surface_diagnostics": diagnostics,
        "outputs": {
            path.name: {
                "path": str(path),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in all_outputs
        },
    }
    manifest_path = output / "render_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "rendered_face_count": diagnostics["rendered_face_count"],
                "cross_label_face_adjacency_counts": diagnostics[
                    "cross_label_face_adjacency_counts"
                ],
                "projection_terminal_footprints": diagnostics[
                    "projection_terminal_footprints"
                ],
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
