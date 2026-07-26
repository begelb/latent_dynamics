"""Compute local Conley indices from saved original-Leslie-3D Morse sets.

This postprocessor does not recompute the global Morse graph. It reconstructs
the uniform cubical map near each requested saved Morse set, verifies the
combinatorial index-pair conditions, and calls the low-level local-CMGDB
``ComputeConleyIndex`` binding.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT.parent
LOCAL_CMGDB_ROOT = (PROJECT_ROOT / "archive" / "CMGDB").resolve()
LOCAL_CMGDB_SRC = LOCAL_CMGDB_ROOT / "src"
if str(LOCAL_CMGDB_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_CMGDB_SRC))

import CMGDB  # noqa: E402

INT_PHASE_WIDTH = 1 << 60
TRUNCATION_ERROR = 1 << 10
EXPECTED_SYSTEM = "original 3D Leslie"
EXPECTED_BOX_MAP = "CMGDB.BoxMap(f, rect, padding=False)"
EXPECTED_THETA = [28.9, 29.8, 22.0]
EXPECTED_SURVIVAL = [0.7, 0.7]
ALIGNMENT_REL_TOL = 1e-11
ALIGNMENT_ABS_TOL = 1e-12
DOT_NODE_RE = re.compile(r'^\s*"?(\d+)"?\s*\[')
DOT_EDGE_RE = re.compile(r'^\s*"?(\d+)"?\s*->\s*"?(\d+)"?\s*;?\s*$')


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def git_state(repository: Path) -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status_lines = subprocess.run(
        ["git", "-C", str(repository), "status", "--short"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {
        "repository": str(repository),
        "revision": revision,
        "dirty": bool(status_lines),
        "status": status_lines,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_metadata(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(path),
    }


def file_signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def resolve_screen_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    candidates = [resolved, resolved / "screen"]
    for candidate in candidates:
        if (
            (candidate / "manifest.json").is_file()
            and (candidate / "morse_graph").is_file()
            and (candidate / "MG" / "morse_sets").is_file()
        ):
            return candidate
    raise FileNotFoundError(
        f"{resolved} is not a complete screen artifact directory or a run root "
        "containing screen/"
    )


def read_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    if manifest.get("system") != EXPECTED_SYSTEM:
        raise ValueError(
            f"expected system {EXPECTED_SYSTEM!r}, found {manifest.get('system')!r}"
        )
    if manifest.get("algorithm") != "ComputeMorseGraph":
        raise ValueError(
            "saved input must come from a graph-only ComputeMorseGraph screen run"
        )
    if manifest.get("box_map") != EXPECTED_BOX_MAP:
        raise ValueError(
            f"expected box map {EXPECTED_BOX_MAP!r}, found {manifest.get('box_map')!r}"
        )
    return manifest


def finite_vector(
    value: Any,
    *,
    name: str,
    length: int,
) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must be a list of length {length}")
    result = [float(entry) for entry in value]
    if not all(math.isfinite(entry) for entry in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


def manifest_parameters(
    manifest: dict[str, Any],
) -> tuple[list[float], list[float], list[float], list[float]]:
    bounds = manifest.get("bounds")
    if not isinstance(bounds, dict):
        raise ValueError("manifest bounds must be an object")
    lower = finite_vector(bounds.get("lower"), name="bounds.lower", length=3)
    upper = finite_vector(bounds.get("upper"), name="bounds.upper", length=3)
    if any(lo >= hi for lo, hi in zip(lower, upper, strict=True)):
        raise ValueError(f"invalid bounds: {lower} -> {upper}")
    theta = finite_vector(manifest.get("theta"), name="theta", length=3)
    survival = finite_vector(manifest.get("survival"), name="survival", length=2)
    if theta != EXPECTED_THETA:
        raise ValueError(f"expected theta={EXPECTED_THETA}, found {theta}")
    if survival != EXPECTED_SURVIVAL:
        raise ValueError(
            f"expected survival={EXPECTED_SURVIVAL}, found {survival}"
        )
    return lower, upper, theta, survival


def split_counts(depth: int, dimension: int) -> list[int]:
    quotient, remainder = divmod(depth, dimension)
    return [quotient + int(axis < remainder) for axis in range(dimension)]


def infer_depth(
    rectangle: list[float],
    lower: list[float],
    upper: list[float],
) -> int:
    dimension = len(lower)
    counts: list[int] = []
    for axis in range(dimension):
        root_width = upper[axis] - lower[axis]
        box_width = rectangle[dimension + axis] - rectangle[axis]
        if box_width <= 0.0:
            raise ValueError(f"box has nonpositive width on axis {axis}: {rectangle}")
        exponent = round(math.log2(root_width / box_width))
        if exponent < 0 or not math.isclose(
            box_width,
            root_width / (2**exponent),
            rel_tol=ALIGNMENT_REL_TOL,
            abs_tol=ALIGNMENT_ABS_TOL,
        ):
            raise ValueError(f"box is not dyadic on axis {axis}: {rectangle}")
        counts.append(exponent)
    if sum(counts) >= 64:
        raise ValueError(
            f"uniform grid has too many cubes for uint64 indexing: exponents={counts}"
        )
    for depth in range(dimension * max(counts) + 1):
        if split_counts(depth, dimension) == counts:
            return depth
    raise ValueError(
        f"box widths do not arise from cyclic bisection: exponents={counts}"
    )


def flatten(coordinates: tuple[int, ...], sizes: list[int]) -> int:
    index = 0
    stride = 1
    for coordinate, size in zip(coordinates, sizes, strict=True):
        index += coordinate * stride
        stride *= size
    return index


def unflatten(index: int, sizes: list[int]) -> tuple[int, ...]:
    coordinates = []
    for size in sizes:
        coordinates.append(index % size)
        index //= size
    if index:
        raise ValueError("cube index exceeds the uniform grid")
    return tuple(coordinates)


def rectangle_to_cube(
    rectangle: list[float],
    lower: list[float],
    upper: list[float],
    sizes: list[int],
) -> int:
    dimension = len(lower)
    coordinates: list[int] = []
    for axis in range(dimension):
        root_width = upper[axis] - lower[axis]
        scaled = (rectangle[axis] - lower[axis]) / root_width * sizes[axis]
        coordinate = round(scaled)
        if not 0 <= coordinate < sizes[axis]:
            raise ValueError(
                f"box lower corner is outside axis {axis}: {rectangle}"
            )
        expected_lower = lower[axis] + root_width * coordinate / sizes[axis]
        expected_upper = lower[axis] + root_width * (coordinate + 1) / sizes[axis]
        if not math.isclose(
            rectangle[axis],
            expected_lower,
            rel_tol=ALIGNMENT_REL_TOL,
            abs_tol=ALIGNMENT_ABS_TOL,
        ) or not math.isclose(
            rectangle[dimension + axis],
            expected_upper,
            rel_tol=ALIGNMENT_REL_TOL,
            abs_tol=ALIGNMENT_ABS_TOL,
        ):
            raise ValueError(
                f"box is not aligned with the inferred grid on axis {axis}: "
                f"{rectangle}"
            )
        coordinates.append(coordinate)
    return flatten(tuple(coordinates), sizes)


def cube_to_rectangle(
    index: int,
    lower: list[float],
    upper: list[float],
    sizes: list[int],
) -> list[float]:
    coordinates = unflatten(index, sizes)
    lows = [
        lower[axis]
        + (upper[axis] - lower[axis]) * coordinates[axis] / sizes[axis]
        for axis in range(len(lower))
    ]
    highs = [
        lower[axis]
        + (upper[axis] - lower[axis]) * (coordinates[axis] + 1) / sizes[axis]
        for axis in range(len(lower))
    ]
    return lows + highs


def uniform_cover(
    rectangle: list[float],
    lower: list[float],
    upper: list[float],
    sizes: list[int],
) -> list[int]:
    """Mirror TreeGrid's inclusive 60-bit integer intersection cover."""

    dimension = len(lower)
    coordinate_ranges: list[range] = []
    for axis in range(dimension):
        domain_lower = lower[axis]
        domain_upper = upper[axis]
        rectangle_lower = rectangle[axis]
        rectangle_upper = rectangle[dimension + axis]
        if rectangle_upper < domain_lower or rectangle_lower > domain_upper:
            return []

        width = domain_upper - domain_lower
        normalized_lower = max(
            0.0,
            min(1.0, (rectangle_lower - domain_lower) / width),
        )
        normalized_upper = max(
            0.0,
            min(1.0, (rectangle_upper - domain_lower) / width),
        )
        integer_lower = max(
            0,
            int(INT_PHASE_WIDTH * normalized_lower) - TRUNCATION_ERROR,
        )
        integer_upper = min(
            INT_PHASE_WIDTH,
            int(INT_PHASE_WIDTH * normalized_upper) + TRUNCATION_ERROR,
        )

        integer_box_width = INT_PHASE_WIDTH // sizes[axis]
        first = max(
            0,
            (integer_lower + integer_box_width - 1) // integer_box_width - 1,
        )
        last = min(sizes[axis] - 1, integer_upper // integer_box_width)
        if first > last:
            return []
        coordinate_ranges.append(range(first, last + 1))

    return [
        flatten(tuple(coordinates), sizes)
        for coordinates in itertools.product(*coordinate_ranges)
    ]


def expected_manifest_counts(manifest: dict[str, Any]) -> dict[int, int] | None:
    raw_counts = manifest.get("morse_boxes_per_node")
    if raw_counts is None:
        return None
    if not isinstance(raw_counts, dict):
        raise ValueError("manifest morse_boxes_per_node must be an object")
    counts: dict[int, int] = {}
    for raw_node, raw_count in raw_counts.items():
        node = int(raw_node)
        count = int(raw_count)
        if count < 1:
            raise ValueError(f"manifest gives nonpositive box count for node {node}")
        counts[node] = count
    return counts


def parse_dot(path: Path) -> tuple[list[int], list[list[int]]]:
    nodes: set[int] = set()
    edges: set[tuple[int, int]] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        edge_match = DOT_EDGE_RE.match(raw_line)
        if edge_match:
            source, target = map(int, edge_match.groups())
            nodes.update((source, target))
            edges.add((source, target))
            continue
        node_match = DOT_NODE_RE.match(raw_line)
        if node_match:
            nodes.add(int(node_match.group(1)))
    if not nodes:
        raise ValueError(f"no graph nodes found in {path}")
    return sorted(nodes), [list(edge) for edge in sorted(edges)]


def validate_saved_artifacts(
    graph_path: Path,
    manifest: dict[str, Any],
    observed_counts: dict[int, int],
    requested_nodes: list[int],
) -> dict[str, Any]:
    graph_nodes, graph_edges = parse_dot(graph_path)
    counted_nodes = sorted(observed_counts)
    if graph_nodes != counted_nodes:
        raise ValueError(
            f"DOT nodes {graph_nodes} do not match saved-set nodes {counted_nodes}"
        )
    missing = set(requested_nodes) - set(graph_nodes)
    if missing:
        raise ValueError(f"requested nodes absent from graph: {sorted(missing)}")

    manifest_node_count = manifest.get("morse_nodes")
    if manifest_node_count is not None and int(manifest_node_count) != len(graph_nodes):
        raise ValueError(
            f"manifest reports {manifest_node_count} nodes but DOT has "
            f"{len(graph_nodes)}"
        )
    raw_manifest_edges = manifest.get("edges")
    if raw_manifest_edges is not None:
        manifest_edges = sorted(
            [int(edge[0]), int(edge[1])] for edge in raw_manifest_edges
        )
        if manifest_edges != graph_edges:
            raise ValueError(
                f"manifest edges {manifest_edges} do not match DOT edges {graph_edges}"
            )

    graph_sources = {source for source, _ in graph_edges}
    graph_minimal = sorted(set(graph_nodes) - graph_sources)
    raw_manifest_minimal = manifest.get("minimal_nodes")
    if raw_manifest_minimal is not None:
        manifest_minimal = sorted(int(node) for node in raw_manifest_minimal)
        if manifest_minimal != graph_minimal:
            raise ValueError(
                f"manifest minimal nodes {manifest_minimal} do not match DOT "
                f"minimal nodes {graph_minimal}"
            )
    return {
        "nodes": graph_nodes,
        "edges": graph_edges,
        "minimal_nodes": graph_minimal,
    }


def load_requested_nodes(
    path: Path,
    requested_nodes: list[int],
    lower: list[float],
    upper: list[float],
    manifest: dict[str, Any],
) -> tuple[dict[int, dict[str, Any]], dict[int, int], str]:
    requested = set(requested_nodes)
    node_data: dict[int, dict[str, Any]] = {
        node: {"depth": None, "sizes": None, "cubes": set()} for node in requested
    }
    counts: Counter[int] = Counter()
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            digest.update(raw_line)
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                fields = stripped.decode("utf-8").split(",")
            except UnicodeDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid UTF-8") from error
            if len(fields) != 7:
                raise ValueError(
                    f"{path}:{line_number}: expected 7 comma-separated fields, "
                    f"found {len(fields)}"
                )
            try:
                raw_node = float(fields[-1])
            except ValueError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid Morse node {fields[-1]!r}"
                ) from error
            node = int(raw_node)
            if raw_node != node or node < 0:
                raise ValueError(
                    f"{path}:{line_number}: Morse node must be a nonnegative integer"
                )
            counts[node] += 1
            if node not in requested:
                continue

            try:
                rectangle = [float(field) for field in fields[:-1]]
            except ValueError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid box coordinates"
                ) from error
            if not all(math.isfinite(value) for value in rectangle):
                raise ValueError(f"{path}:{line_number}: nonfinite box coordinate")

            data = node_data[node]
            if data["depth"] is None:
                depth = infer_depth(rectangle, lower, upper)
                data["depth"] = depth
                data["sizes"] = [
                    2**count for count in split_counts(depth, len(lower))
                ]
            sizes = data["sizes"]
            if not isinstance(sizes, list):
                raise AssertionError("uniform sizes were not initialized")
            cube = rectangle_to_cube(rectangle, lower, upper, sizes)
            cubes = data["cubes"]
            if not isinstance(cubes, set):
                raise AssertionError("cube set was not initialized")
            if cube in cubes:
                raise ValueError(
                    f"{path}:{line_number}: duplicate cube for requested node {node}"
                )
            cubes.add(cube)

    missing = requested - counts.keys()
    if missing:
        raise ValueError(f"requested nodes absent from saved Morse sets: {sorted(missing)}")

    manifest_counts = expected_manifest_counts(manifest)
    observed_counts = dict(sorted(counts.items()))
    if manifest_counts is not None and observed_counts != dict(sorted(manifest_counts.items())):
        raise ValueError(
            "saved Morse-set counts do not match manifest: "
            f"observed={observed_counts}, manifest={manifest_counts}"
        )
    manifest_nodes = manifest.get("morse_nodes")
    if manifest_nodes is not None and int(manifest_nodes) != len(observed_counts):
        raise ValueError(
            f"manifest reports {manifest_nodes} nodes but saved file has "
            f"{len(observed_counts)}"
        )
    return node_data, observed_counts, digest.hexdigest()


class LeslieMap:
    def __init__(self, theta: list[float], survival: list[float]) -> None:
        self.theta = theta
        self.survival = survival

    def __call__(self, point: list[float]) -> list[float]:
        x0, x1, x2 = point
        return [
            (
                self.theta[0] * x0
                + self.theta[1] * x1
                + self.theta[2] * x2
            )
            * math.exp(-0.1 * (x0 + x1 + x2)),
            self.survival[0] * x0,
            self.survival[1] * x1,
        ]


def compute_node(
    node: int,
    data: dict[str, Any],
    lower: list[float],
    upper: list[float],
    leslie_map: LeslieMap,
    progress_every: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    depth = data["depth"]
    sizes = data["sizes"]
    S = data["cubes"]
    if not isinstance(depth, int) or not isinstance(sizes, list) or not isinstance(S, set):
        raise ValueError(f"node {node} was not loaded correctly")

    def image(cube: int) -> list[int]:
        rectangle = cube_to_rectangle(cube, lower, upper, sizes)
        mapped = CMGDB.BoxMap(leslie_map, rectangle, padding=False)
        return uniform_cover(mapped, lower, upper, sizes)

    image_started = time.perf_counter()
    image_set: set[int] = set()
    for position, cube in enumerate(S, start=1):
        image_set.update(image(cube))
        if position % progress_every == 0:
            print(
                f"node {node}: built F(S) for {position:,}/{len(S):,} cubes",
                flush=True,
            )
    image_seconds = time.perf_counter() - image_started

    s_missing_from_image = S - image_set
    s_subset_image = not s_missing_from_image
    X = image_set
    A = X - S
    condition_a_missing = image_set - X
    condition_a = not condition_a_missing
    a_subset_x = A <= X
    x_minus_a_equals_s = X - A == S

    local_map_started = time.perf_counter()
    local_map: dict[int, list[int]] = {}
    condition_b_violations = 0
    condition_b_examples: list[dict[str, Any]] = []
    map_edges = 0
    empty_restricted_images = 0
    for position, cube in enumerate(X, start=1):
        targets = image(cube)
        if cube in A:
            leaves_exit_set = sorted(
                target for target in targets if target in X and target not in A
            )
            if leaves_exit_set:
                condition_b_violations += 1
                if len(condition_b_examples) < 10:
                    condition_b_examples.append(
                        {
                            "cube": cube,
                            "targets_in_X_minus_A": leaves_exit_set[:10],
                        }
                    )
        restricted_targets = [target for target in targets if target in X]
        local_map[cube] = restricted_targets
        map_edges += len(restricted_targets)
        if not restricted_targets:
            empty_restricted_images += 1
        if position % progress_every == 0:
            print(
                f"node {node}: built F|X for {position:,}/{len(X):,} cubes",
                flush=True,
            )
    local_map_seconds = time.perf_counter() - local_map_started

    condition_b = condition_b_violations == 0
    pair_valid = (
        s_subset_image
        and condition_a
        and condition_b
        and a_subset_x
        and x_minus_a_equals_s
    )
    result: dict[str, Any] = {
        "node": node,
        "status": "ready_for_homology" if pair_valid else "invalid_index_pair",
        "depth": depth,
        "sizes": sizes,
        "morse_set_cubes": len(S),
        "image_cubes": len(image_set),
        "pair_cubes": len(X),
        "exit_cubes": len(A),
        "map_edges": map_edges,
        "empty_restricted_images": empty_restricted_images,
        "index_pair_checks": {
            "S_subset_F_S": s_subset_image,
            "S_subset_F_S_missing_count": len(s_missing_from_image),
            "A_subset_X": a_subset_x,
            "X_minus_A_equals_S": x_minus_a_equals_s,
            "F_X_minus_A_subset_X": condition_a,
            "F_X_minus_A_missing_count": len(condition_a_missing),
            "F_A_intersect_X_subset_A": condition_b,
            "F_A_intersect_X_subset_A_violation_count": condition_b_violations,
            "F_A_intersect_X_subset_A_examples": condition_b_examples,
        },
        "acyclic_fiber_check": True,
        "conley_index": None,
        "timings_seconds": {
            "build_F_S": round(image_seconds, 6),
            "build_local_map_and_check_pair": round(local_map_seconds, 6),
            "homology": None,
            "total": None,
        },
    }
    if not pair_valid:
        result["timings_seconds"]["total"] = round(
            time.perf_counter() - started, 6
        )
        return result

    homology_started = time.perf_counter()
    conley_index = list(
        CMGDB.ComputeConleyIndex(
            sorted(X),
            sorted(A),
            sizes,
            [False] * len(lower),
            local_map,
            True,
        )
    )
    homology_seconds = time.perf_counter() - homology_started
    if conley_index:
        if len(conley_index) == len(lower) + 1:
            result["status"] = "complete"
            result["conley_index"] = conley_index
        else:
            result["status"] = "unexpected_index_dimension"
            result["conley_index"] = conley_index
    else:
        result["status"] = "homology_or_acyclicity_failure"
    result["timings_seconds"]["homology"] = round(homology_seconds, 6)
    result["timings_seconds"]["total"] = round(
        time.perf_counter() - started, 6
    )
    return result


def node_text(result: dict[str, Any]) -> str:
    checks = result["index_pair_checks"]
    timings = result["timings_seconds"]
    lines = [
        f"Node: {result['node']}",
        f"Status: {result['status']}",
        f"Uniform cyclic depth: {result['depth']}",
        f"Uniform sizes: {result['sizes']}",
        f"Morse-set cubes: {result['morse_set_cubes']}",
        f"Index pair (X, A): ({result['pair_cubes']}, {result['exit_cubes']})",
        f"S subset F(S): {checks['S_subset_F_S']}",
        f"F(X-A) subset X: {checks['F_X_minus_A_subset_X']}",
        f"F(A) intersect X subset A: {checks['F_A_intersect_X_subset_A']}",
        (
            "F(A) intersect X subset A violations: "
            f"{checks['F_A_intersect_X_subset_A_violation_count']}"
        ),
        f"Acyclic-fiber check enabled: {result['acyclic_fiber_check']}",
        f"Conley index: {result['conley_index']}",
        f"Total seconds: {timings['total']}",
    ]
    return "\n".join(lines) + "\n"


def summary_text(summary: dict[str, Any]) -> str:
    lines = [
        f"Status: {summary['status']}",
        f"Screen artifacts: {summary['screen_artifact_dir']}",
        f"Requested nodes: {summary['requested_nodes']}",
        f"Completed nodes: {summary['completed_nodes']}",
        f"Failed nodes: {summary['failed_nodes']}",
        f"Local CMGDB revision: {summary['cmgdb']['revision']}",
        f"Local CMGDB dirty: {summary['cmgdb']['dirty']}",
        (
            "Local CMGDB revision matches screen: "
            f"{summary['cmgdb_revision_matches_screen']}"
        ),
    ]
    for result in summary["results"]:
        lines.append(
            f"Node {result['node']}: status={result['status']}, "
            f"index={result['conley_index']}"
        )
    return "\n".join(lines) + "\n"


def write_json(path: Path, value: dict[str, Any]) -> None:
    write_text(path, json.dumps(value, indent=2) + "\n")


def write_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "screen",
        type=Path,
        help="Screen artifact directory or run root containing screen/",
    )
    parser.add_argument("--node", type=int, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=100_000)
    args = parser.parse_args()

    if args.progress_every < 1:
        raise ValueError("--progress-every must be positive")
    if any(node < 0 for node in args.node):
        raise ValueError("requested node numbers must be nonnegative")
    requested_nodes = list(dict.fromkeys(args.node))
    if len(requested_nodes) != len(args.node):
        raise ValueError("each requested node may be specified only once")

    module_path = Path(CMGDB.__file__).resolve()
    if LOCAL_CMGDB_ROOT not in module_path.parents:
        raise RuntimeError(
            f"expected CMGDB below {LOCAL_CMGDB_ROOT}, imported {module_path}"
        )
    cmgdb_state = {
        "version": version("CMGDB"),
        "module_path": str(module_path),
        **git_state(LOCAL_CMGDB_ROOT),
    }

    screen_dir = resolve_screen_dir(args.screen)
    manifest_path = screen_dir / "manifest.json"
    graph_path = screen_dir / "morse_graph"
    morse_sets_path = screen_dir / "MG" / "morse_sets"
    manifest = read_manifest(manifest_path)
    lower, upper, theta, survival = manifest_parameters(manifest)

    morse_sets_signature_before = file_signature(morse_sets_path)
    node_data, observed_counts, morse_sets_sha256 = load_requested_nodes(
        morse_sets_path,
        requested_nodes,
        lower,
        upper,
        manifest,
    )
    morse_sets_signature_after = file_signature(morse_sets_path)
    if morse_sets_signature_after != morse_sets_signature_before:
        raise RuntimeError(f"{morse_sets_path} changed while it was being read")
    graph_summary = validate_saved_artifacts(
        graph_path,
        manifest,
        observed_counts,
        requested_nodes,
    )

    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = (CODE_ROOT / output_dir).resolve()
    else:
        output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    output_dir.mkdir(parents=True)

    screen_cmgdb = manifest.get("cmgdb")
    screen_revision = (
        screen_cmgdb.get("revision")
        if isinstance(screen_cmgdb, dict)
        else None
    )
    run_config = {
        "started_at": utc_now(),
        "script": str(Path(__file__).resolve()),
        "algorithm": "saved uniform Morse sets -> verified local index pair -> ComputeConleyIndex",
        "morse_graph_recomputed": False,
        "adaptive_phase_grid_recovered": False,
        "index_pair_grid": "uniform grid at each saved Morse-set depth",
        "screen_artifact_dir": str(screen_dir),
        "source_artifacts": {
            "manifest": file_metadata(manifest_path),
            "morse_graph": file_metadata(graph_path),
            "morse_sets": {
                "path": str(morse_sets_path),
                "size_bytes": morse_sets_signature_after[0],
                "mtime_ns": morse_sets_signature_after[1],
                "sha256": morse_sets_sha256,
            },
        },
        "saved_graph": graph_summary,
        "requested_nodes": requested_nodes,
        "observed_morse_boxes_per_node": {
            str(node): count for node, count in observed_counts.items()
        },
        "bounds": {"lower": lower, "upper": upper},
        "theta": theta,
        "survival": survival,
        "box_map": EXPECTED_BOX_MAP,
        "cmgdb": cmgdb_state,
        "screen_cmgdb": screen_cmgdb,
        "cmgdb_revision_matches_screen": (
            screen_revision == cmgdb_state["revision"]
            if screen_revision is not None
            else None
        ),
        "output_dir": str(output_dir),
    }
    write_json(output_dir / "run_config.json", run_config)
    print(json.dumps(run_config, indent=2), flush=True)

    leslie_map = LeslieMap(theta, survival)
    results: list[dict[str, Any]] = []
    for node in requested_nodes:
        print(f"node {node}: starting saved-set Conley computation", flush=True)
        try:
            result = compute_node(
                node,
                node_data[node],
                lower,
                upper,
                leslie_map,
                args.progress_every,
            )
        except Exception as error:
            result = {
                "node": node,
                "status": "error",
                "error_type": type(error).__name__,
                "error": str(error),
                "conley_index": None,
            }
        results.append(result)
        write_json(output_dir / f"node_{node}.json", result)
        if "index_pair_checks" in result:
            write_text(
                output_dir / f"node_{node}.txt",
                node_text(result),
            )
        else:
            write_text(
                output_dir / f"node_{node}.txt",
                f"Node: {node}\nStatus: error\nError: {result['error']}\n",
            )
        print(json.dumps(result, indent=2), flush=True)

    completed_nodes = [
        result["node"] for result in results if result["status"] == "complete"
    ]
    failed_nodes = [
        result["node"] for result in results if result["status"] != "complete"
    ]
    summary = {
        **run_config,
        "finished_at": utc_now(),
        "status": "complete" if not failed_nodes else "failed",
        "completed_nodes": completed_nodes,
        "failed_nodes": failed_nodes,
        "conley_indices": {
            str(result["node"]): result["conley_index"]
            for result in results
            if result["status"] == "complete"
        },
        "results": results,
    }
    write_json(output_dir / "summary.json", summary)
    write_text(
        output_dir / "summary.txt",
        summary_text(summary),
    )
    return 0 if not failed_nodes else 1


if __name__ == "__main__":
    raise SystemExit(main())
