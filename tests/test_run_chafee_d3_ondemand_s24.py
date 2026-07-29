from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

SCRIPT = Path(__file__).parents[1] / "scripts" / "run_chafee_d3_ondemand_s24.py"
SPEC = importlib.util.spec_from_file_location("run_chafee_d3_ondemand_s24", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


def _fake_inputs(tmp_path: Path, *, matrix_complete: bool = True) -> Any:
    training_root = tmp_path / "training"
    run_root = training_root / "runs" / RUNNER.TARGET_RUN_ID
    attempt_root = run_root / "attempts" / "attempt_001"
    return RUNNER.TargetInputs(
        training_root=training_root,
        run_root=run_root,
        attempt_root=attempt_root,
        checkpoint=attempt_root / "models" / "autoencoder.pt",
        checkpoint_sidecar=attempt_root / "models" / "autoencoder.json",
        train_data=tmp_path / "train_data.csv",
        plan_sha256="a" * 64,
        source_records={"checkpoint": {}, "train_data": {}},
        matrix_status={
            "required_runs": 15,
            "completed_runs": 15 if matrix_complete else 14,
            "complete": matrix_complete,
            "completed_run_ids": [],
            "incomplete_run_ids": [] if matrix_complete else ["last_run"],
        },
    )


def test_plan_is_exact_s24_on_demand_only(tmp_path: Path) -> None:
    inputs = _fake_inputs(tmp_path)
    output = tmp_path / "production-output"
    plan = RUNNER.build_plan(
        inputs,
        output_root=output,
        device="mps",
        max_edges=1_200_000_000,
        max_forward_points=800_000,
        rss_sample_seconds=0.1,
    )

    graph = plan["computation"]
    assert (
        graph["subdiv_init"],
        graph["subdiv_min"],
        graph["subdiv_max"],
    ) == (24, 24, 24)
    assert graph["expected_cells"] == 2**24
    assert graph["cells_per_axis"] == 256
    assert graph["CMGDB_MAPGRAPH_MAX_VERTICES"] == 2**24
    assert graph["CMGDB_MAPGRAPH_MAX_EDGES"] == 1_200_000_000
    assert graph["CMGDB_MAPGRAPH_RESERVE_EDGES"] == 1_200_000_000
    assert graph["CMGDB_MAPGRAPH_RESERVE_MIN_VERTICES"] == 2**24
    assert graph["backend"] == "batched_on_demand_neural"
    assert graph["precomputed"] is False
    assert plan["hard_postconditions"]["scalar_callback_calls_equal_zero"]
    assert RUNNER.EXPECTED_CALLBACK_RECTANGLES == 33_554_432
    assert RUNNER.EXPECTED_NEURAL_CORNER_POINTS == 268_435_456
    assert RUNNER.EXPECTED_BATCH_CALLS == 336
    assert (
        plan["hard_postconditions"][
            "callback_rectangles_equal_two_full_cell_passes"
        ]
        == 33_554_432
    )


def test_on_demand_box_map_batches_all_d3_corners() -> None:
    calls: list[np.ndarray] = []

    def identity(points: np.ndarray) -> np.ndarray:
        calls.append(points.copy())
        return points

    box_map = RUNNER.OnDemandNeuralBoxMap(
        identity,
        max_forward_points=8,
        padding=True,
    )
    result = np.asarray(
        box_map.batch(
            [
                [0.0, 0.0, 0.0, 1.0, 2.0, 3.0],
                [-2.0, -1.0, 4.0, -1.0, 1.0, 5.0],
            ]
        )
    )

    np.testing.assert_allclose(
        result,
        [
            [-1.0, -2.0, -3.0, 2.0, 4.0, 6.0],
            [-3.0, -3.0, 3.0, 0.0, 3.0, 6.0],
        ],
    )
    assert len(calls) == 2
    assert all(call.shape == (8, 3) for call in calls)
    assert box_map.stats()["batch_calls"] == 1
    assert box_map.stats()["scalar_calls"] == 0
    assert box_map.stats()["rectangles"] == 2


def test_on_demand_box_map_records_scalar_fallback() -> None:
    box_map = RUNNER.OnDemandNeuralBoxMap(
        lambda points: points,
        max_forward_points=8,
    )
    box_map([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    assert box_map.stats()["scalar_calls"] == 1


def test_compute_cpu_bounds_matches_ten_percent_rule() -> None:
    class RecordingEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.rows_seen: list[int] = []

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            self.rows_seen.append(int(values.shape[0]))
            return values[:, :3]

    encoder = RecordingEncoder()
    x = np.zeros((2, 64), dtype=np.float64)
    y = np.zeros((2, 64), dtype=np.float64)
    x[0, :3] = [0.0, 1.0, 2.0]
    x[1, :3] = [1.0, 2.0, 3.0]
    y[0, :3] = [2.0, 3.0, 4.0]
    y[1, :3] = [1.5, 2.5, 3.5]

    lower, upper = RUNNER.compute_cpu_bounds(encoder, x, y)

    np.testing.assert_allclose(lower, [-0.2, 0.8, 1.8])
    np.testing.assert_allclose(upper, [2.2, 3.2, 4.2])
    assert encoder.rows_seen == [4]


def test_graph_summary_uses_unreduced_edges_without_reducing() -> None:
    class FakeMorseGraph:
        def vertices(self) -> list[int]:
            return [0, 1, 2]

        def edges(self) -> list[tuple[int, int]]:
            raise AssertionError("native transitive reduction must not run")

        def edges_unreduced(self) -> list[tuple[int, int]]:
            return [(1, 0), (2, 0), (2, 1)]

        def morse_set(self, node: int) -> list[int]:
            return list(range(node + 1))

    class FakeMapGraph:
        def num_vertices(self) -> int:
            return 2**24

        def num_cached_edges(self) -> int:
            return 123

    summary = RUNNER._graph_summary(FakeMorseGraph(), FakeMapGraph())

    assert summary["minimal_nodes"] == [0]
    assert summary["morse_unreduced_edge_count"] == 3
    assert summary["morse_set_sizes"] == {"0": 1, "1": 2, "2": 3}


def test_output_must_be_fresh_and_disjoint(tmp_path: Path) -> None:
    inputs = _fake_inputs(tmp_path)
    safe = tmp_path / "analysis" / "fresh"
    assert RUNNER.assert_safe_fresh_output(safe, inputs) == safe.resolve()

    safe.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="overwrite"):
        RUNNER.assert_safe_fresh_output(safe, inputs)

    with pytest.raises(ValueError, match="overlaps"):
        RUNNER.assert_safe_fresh_output(
            inputs.attempt_root / "analysis",
            inputs,
        )


def test_execute_is_gated_on_all_fifteen_training_runs(tmp_path: Path) -> None:
    inputs = _fake_inputs(tmp_path, matrix_complete=False)
    output = tmp_path / "never-created"

    with pytest.raises(RuntimeError, match="all 15"):
        RUNNER.run_exact_s24(
            inputs=inputs,
            output_root=output,
            device_name="cpu",
            max_edges=100,
            max_forward_points=8,
            rss_sample_seconds=0.1,
        )
    assert not output.exists()
