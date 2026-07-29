"""Focused tests for the bounded D3 BoxMap benchmark harness."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_module():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "benchmark_chafee_d3_boxmap.py"
    )
    spec = importlib.util.spec_from_file_location(
        "benchmark_chafee_d3_boxmap",
        script,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BENCH = _load_module()


def _checkpoint(tmp_path: Path, name: str = "d3") -> object:
    root = tmp_path / name
    models = root / "models"
    models.mkdir(parents=True)
    (models / "autoencoder.pt").write_bytes(b"checkpoint")
    (models / "autoencoder.json").write_text(
        json.dumps({"version": 1, "arch": {"low_dims": 3}}),
        encoding="utf-8",
    )
    (root / "bounds.json").write_text(
        json.dumps(
            {
                "lower": [-1.0, -2.0, -3.0],
                "upper": [1.0, 2.0, 3.0],
            }
        ),
        encoding="utf-8",
    )
    return BENCH._checkpoint_from_root(name, root)


def test_plan_is_bounded_and_never_dispatches_level24(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)

    plan = BENCH.build_plan(
        [checkpoint],
        subdivisions=(12, 15, 18),
        repeats=3,
        warmups=1,
    )

    assert plan["target_extrapolation"]["subdivision"] == 24
    assert plan["target_extrapolation"]["launch_permitted"] is False
    assert max(plan["pilot_subdivisions"]) == 18
    assert all(
        trial["subdivision"] <= 18 for trial in plan["planned_trials"]
    )
    assert len(plan["planned_trials"]) == 20
    level18 = plan["pilot_sizes"][-1]
    assert level18["uniform_cells"] == 262_144
    assert level18["precomputed_unique_corner_points"] == 65**3


@pytest.mark.parametrize(
    ("subdivisions", "message"),
    [
        ((12, 21), "hard pilot cap"),
        ((12, 16), "divisible by 3"),
    ],
)
def test_protocol_rejects_unsafe_subdivisions(
    tmp_path: Path,
    subdivisions: tuple[int, ...],
    message: str,
) -> None:
    checkpoint = _checkpoint(tmp_path)
    with pytest.raises(ValueError, match=message):
        BENCH.build_plan([checkpoint], subdivisions=subdivisions)


def test_protocol_rejects_resource_ceiling_overrides(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    limits = BENCH.BenchmarkLimits(
        timeout_seconds=BENCH.HARD_MAX_TIMEOUT_SECONDS + 1
    )
    with pytest.raises(ValueError, match="timeout_seconds"):
        BENCH.build_plan([checkpoint], limits=limits)


def test_output_must_not_overlap_checkpoint_or_existing_directory(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path)
    with pytest.raises(ValueError, match="overlaps protected"):
        BENCH.assert_safe_output_root(checkpoint.run_root / "benchmark", [checkpoint])
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        BENCH.assert_safe_output_root(existing, [checkpoint])


def test_direct_box_map_vectorizes_corners_and_counts_work() -> None:
    def linear(points: np.ndarray) -> np.ndarray:
        return 2.0 * points

    evaluator = BENCH.CountingEvaluator(linear, max_points=1_000)
    box_map = BENCH.DirectNeuralBoxMap(
        evaluator,
        dimension=3,
        padding=False,
        max_rectangles=10,
        max_forward_points=8,
    )
    rects = np.asarray(
        [
            [0.0, 1.0, 2.0, 0.5, 1.5, 2.5],
            [-1.0, -2.0, -3.0, 1.0, 2.0, 3.0],
        ]
    )

    actual = np.asarray(box_map.batch(rects))

    np.testing.assert_allclose(
        actual,
        [
            [0.0, 2.0, 4.0, 1.0, 3.0, 5.0],
            [-2.0, -4.0, -6.0, 2.0, 4.0, 6.0],
        ],
    )
    stats = box_map.stats()
    assert stats["batch_calls"] == 1
    assert stats["scalar_calls"] == 0
    assert stats["rectangles"] == 2
    assert stats["neural_corner_points"] == 16
    assert stats["neural_forward_calls"] == 2


def test_direct_box_map_enforces_neural_budget() -> None:
    evaluator = BENCH.CountingEvaluator(lambda values: values, max_points=7)
    box_map = BENCH.DirectNeuralBoxMap(
        evaluator,
        dimension=3,
        padding=False,
    )
    with pytest.raises(BENCH.BenchmarkBudgetExceeded, match="neural point budget"):
        box_map([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])


def _synthetic_result(
    *,
    backend: str,
    subdivision: int,
    repeat: int,
    cmgdb_seconds: float,
    fingerprint: str = "same",
) -> dict[str, object]:
    cells = 2**subdivision
    return {
        "status": "complete",
        "trial": {
            "trial_id": f"{backend}-{subdivision}-{repeat}",
            "checkpoint_id": "cp",
            "backend": backend,
            "subdivision": subdivision,
            "repeat": repeat,
            "warmup": False,
        },
        "timings": {
            "cmgdb_seconds": cmgdb_seconds,
            "precompute_seconds": 0.1 if backend == "precomputed" else 0.0,
            "end_to_end_excluding_model_load_seconds": (
                cmgdb_seconds + (0.1 if backend == "precomputed" else 0.0)
            ),
        },
        "supervisor": {"sampled_peak_rss_bytes": cells * 100},
        "graph": {
            "cached_edges": cells * 10,
            "graph_fingerprint": fingerprint,
        },
        "callback": {
            "neural_corner_points": cells * 8 if backend == "ondemand" else cells,
            "neural_forward_calls": 2,
            "rectangles": cells,
        },
    }


def test_summary_parity_and_extrapolation_use_measured_medians() -> None:
    results = []
    for subdivision, base_time in ((12, 1.0), (15, 8.0), (18, 64.0)):
        for backend in BENCH.BACKENDS:
            for repeat, offset in enumerate((-0.1, 0.0, 0.1)):
                results.append(
                    _synthetic_result(
                        backend=backend,
                        subdivision=subdivision,
                        repeat=repeat,
                        cmgdb_seconds=base_time + offset,
                    )
                )

    summary = BENCH.summarize_trials(results)
    extrapolation = BENCH.extrapolate_target(summary)

    assert all(
        parity["status"] == "verified" for parity in summary["graph_parity"]
    )
    level18 = next(
        group
        for group in summary["groups"]
        if group["backend"] == "ondemand" and group["subdivision"] == 18
    )
    assert level18["metrics"]["cmgdb_seconds"]["median"] == pytest.approx(64.0)
    projection = next(
        item
        for item in extrapolation["projections"]
        if item["backend"] == "ondemand"
    )
    assert projection["metrics"]["cmgdb_seconds"][
        "constant_per_cell_from_highest_pilot"
    ] == pytest.approx(64.0 * 64.0)
    assert projection["target_subdivision"] == 24
    assert extrapolation["target_dispatch_permitted_by_harness"] is False


def test_graph_mismatch_is_detected() -> None:
    results = [
        _synthetic_result(
            backend="ondemand",
            subdivision=12,
            repeat=0,
            cmgdb_seconds=1.0,
            fingerprint="direct",
        ),
        _synthetic_result(
            backend="precomputed",
            subdivision=12,
            repeat=0,
            cmgdb_seconds=1.0,
            fingerprint="lookup",
        ),
    ]
    assert BENCH.graph_parity(results)["status"] == "mismatch"
