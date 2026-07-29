from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"
SCRIPT = SCRIPTS / "run_chafee_d3_ondemand_5x3_controller.py"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "run_chafee_d3_ondemand_5x3_controller",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
CONTROLLER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CONTROLLER
SPEC.loader.exec_module(CONTROLLER)


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _dataset_seed(run_id: str) -> tuple[int, int]:
    match = CONTROLLER.worker.RUN_ID.fullmatch(run_id)
    assert match is not None
    return int(match.group(1)), int(match.group(2))


def _completed_summary(
    run_id: str,
    *,
    plan_sha256: str,
) -> dict[str, Any]:
    dataset, seed = _dataset_seed(run_id)
    return {
        "schema_version": 1,
        "status": "complete",
        "run_id": run_id,
        "dataset": dataset,
        "training_seed": seed,
        "analysis_plan_sha256": plan_sha256,
        "sources": {"checkpoint": {"sha256": f"checkpoint-{run_id}"}},
        "graph": {
            "cached_edges": 1_000_000 + dataset * 10 + seed,
            "morse_nodes": 5 + seed,
            "minimal_nodes": [0, 1],
        },
        "callback": {
            "scalar_calls": 0,
            "batch_calls": 336,
            "rectangles": 33_554_432,
            "neural_corner_points": 268_435_456,
        },
        "timings": {"cmgdb_seconds": 10.0, "total_seconds": 12.0},
        "memory": {"sampled_peak_rss_bytes": 123_456},
        "statistics": {
            "total_trajectories": 10_000,
            "excluded_zero_trajectories": 2_138,
            "conditioned_trajectories": 7_862,
            "counts": {
                "outside_both_basins": 1_000,
                "misclassified_in_negative_basin": 10,
                "misclassified_in_positive_basin": 12,
                "correctly_classified_in_negative_basin": 3_400,
                "correctly_classified_in_positive_basin": 3_440,
            },
            "percentages": {
                "outside_both_basins": 100_000 / 7_862,
                "misclassified_in_negative_basin": 1_000 / 7_862,
                "misclassified_in_positive_basin": 1_200 / 7_862,
                "correctly_classified_in_negative_basin": 340_000 / 7_862,
                "correctly_classified_in_positive_basin": 344_000 / 7_862,
            },
        },
    }


def _install_terminal(
    output_root: Path,
    run_id: str,
    *,
    plan_sha256: str,
    status: str = "completed",
) -> CONTROLLER.TerminalOutcome:
    dataset, seed = _dataset_seed(run_id)
    run_root = output_root / "runs" / run_id
    attempt = run_root / "attempts" / "attempt_001"
    if status == "completed":
        result = _completed_summary(run_id, plan_sha256=plan_sha256)
        result_path = _write_json(attempt / "summary.json", result)
        record_key = "summary"
        marker_name = "completed.json"
    else:
        result = {
            "schema_version": 1,
            "status": "completed_invalid",
            "run_id": run_id,
            "dataset": dataset,
            "training_seed": seed,
            "attempt": 1,
            "analysis_plan_sha256": plan_sha256,
            "reason_type": "ScientificInvalidError",
            "reason": "uniform graph has three attractors",
            "elapsed_seconds": 9.0,
        }
        result_path = _write_json(attempt / "completed_invalid.json", result)
        record_key = "result"
        marker_name = "completed_invalid.json"
    marker = {
        "schema_version": 1,
        "status": status,
        "run_id": run_id,
        "dataset": dataset,
        "training_seed": seed,
        "attempt": 1,
        "analysis_plan_sha256": plan_sha256,
        record_key: CONTROLLER.base._file_record(result_path),
    }
    _write_json(run_root / marker_name, marker)
    outcome = CONTROLLER.read_terminal_outcome(
        output_root,
        run_id,
        expected_plan_sha256=plan_sha256,
    )
    assert outcome is not None
    return outcome


def _analysis_plan(
    output_root: Path,
    worker_path: Path,
    *,
    device: str = "mps",
) -> str:
    plan = {
        "schema_version": 1,
        "computation": {
            "backend": "batched_on_demand_neural",
            "precomputed": False,
            "subdivisions": [24, 24, 24],
            "padding": True,
            "expected_cells": 2**24,
            "expected_callback_rectangles": 33_554_432,
            "expected_neural_corner_points": 268_435_456,
            "max_edges": 1_200_000_000,
            "max_forward_points": 800_000,
            "device": device,
            "trajectory_and_root_encoding_device": "cpu",
            "rss_sample_seconds": 0.1,
        },
        "concurrency": {"maximum_processes": 2},
        "common_sources": {
            "worker": CONTROLLER.base._file_record(worker_path),
        },
    }
    digest = CONTROLLER.base._payload_sha256(plan)
    _write_json(
        output_root / "analysis_plan.json",
        {
            "schema_version": 1,
            "plan_sha256": digest,
            "plan": plan,
        },
    )
    return digest


def test_terminal_reader_accepts_completed_and_scientific_invalid(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    complete = _install_terminal(
        output,
        CONTROLLER.RUN_IDS[0],
        plan_sha256="plan",
    )
    invalid = _install_terminal(
        output,
        CONTROLLER.RUN_IDS[1],
        plan_sha256="plan",
        status="completed_invalid",
    )

    assert complete.status == "completed"
    assert complete.result["status"] == "complete"
    assert invalid.status == "completed_invalid"
    assert invalid.result["reason_type"] == "ScientificInvalidError"


def test_terminal_reader_rejects_conflicting_markers(tmp_path: Path) -> None:
    output = tmp_path / "output"
    run_id = CONTROLLER.RUN_IDS[0]
    _install_terminal(output, run_id, plan_sha256="plan")
    run_root = output / "runs" / run_id
    _write_json(run_root / "completed_invalid.json", {"status": "completed_invalid"})

    with pytest.raises(ValueError, match="conflicting"):
        CONTROLLER.read_terminal_outcome(output, run_id)


def test_collect_skips_terminal_runs(tmp_path: Path) -> None:
    output = tmp_path / "output"
    _install_terminal(output, CONTROLLER.RUN_IDS[0], plan_sha256="plan")
    _install_terminal(
        output,
        CONTROLLER.RUN_IDS[1],
        plan_sha256="plan",
        status="completed_invalid",
    )

    outcomes, pending = CONTROLLER.collect_terminal_outcomes(
        output,
        expected_plan_sha256="plan",
    )

    assert set(outcomes) == set(CONTROLLER.RUN_IDS[:2])
    assert pending == list(CONTROLLER.RUN_IDS[2:])


def test_plan_validation_detects_worker_hash_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    frozen_worker = tmp_path / "worker.py"
    frozen_worker.write_text("print('frozen')\n")
    monkeypatch.setattr(CONTROLLER, "WORKER_PATH", frozen_worker.resolve())
    digest = _analysis_plan(output, frozen_worker)

    _, observed = CONTROLLER.validate_analysis_plan(
        output,
        expected_device="mps",
        max_edges=1_200_000_000,
        max_forward_points=800_000,
        rss_sample_seconds=0.1,
    )
    assert observed == digest

    frozen_worker.write_text("print('changed')\n")
    with pytest.raises(ValueError, match="SHA256"):
        CONTROLLER.validate_analysis_plan(
            output,
            expected_device="mps",
            max_edges=1_200_000_000,
            max_forward_points=800_000,
            rss_sample_seconds=0.1,
        )


def test_worker_command_is_an_isolated_exact_worker(tmp_path: Path) -> None:
    command = CONTROLLER.worker_command(
        CONTROLLER.RUN_IDS[0],
        output_root=tmp_path,
        device="mps",
        max_edges=1_200_000_000,
        max_forward_points=800_000,
        rss_sample_seconds=0.1,
    )

    assert command[:3] == [
        sys.executable,
        "-u",
        str(CONTROLLER.WORKER_PATH),
    ]
    assert command.count("--run-id") == 1
    assert "--device" in command and "mps" in command
    assert "1200000000" in command
    assert "800000" in command


def test_controller_caps_two_processes_and_retains_failed_logs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_ids = tuple(CONTROLLER.RUN_IDS[:4])
    output = tmp_path / "output"
    output.mkdir()
    plan_sha256 = _analysis_plan(output, CONTROLLER.WORKER_PATH)
    _install_terminal(output, run_ids[0], plan_sha256=plan_sha256)
    monkeypatch.setattr(CONTROLLER, "RUN_IDS", run_ids)
    monkeypatch.setattr(CONTROLLER, "active_external_workers", lambda: [])
    monkeypatch.setattr(
        CONTROLLER,
        "ensure_and_validate_analysis_plan",
        lambda *_args, **_kwargs: ({}, plan_sha256),
    )
    monkeypatch.setattr(
        CONTROLLER,
        "validate_analysis_plan",
        lambda *_args, **_kwargs: ({}, plan_sha256),
    )

    active = 0
    maximum_active = 0
    launched: list[str] = []
    failing = run_ids[2]

    class FakeProcess:
        next_pid = 10_000

        def __init__(self, run_id: str) -> None:
            nonlocal active, maximum_active
            self.run_id = run_id
            self.pid = FakeProcess.next_pid
            FakeProcess.next_pid += 1
            self.finished = False
            active += 1
            maximum_active = max(maximum_active, active)

        def poll(self) -> int | None:
            nonlocal active
            if self.finished:
                return 1 if self.run_id == failing else 0
            self.finished = True
            active -= 1
            if self.run_id != failing:
                status = (
                    "completed_invalid"
                    if self.run_id == run_ids[3]
                    else "completed"
                )
                _install_terminal(
                    output,
                    self.run_id,
                    plan_sha256=plan_sha256,
                    status=status,
                )
            return 1 if self.run_id == failing else 0

        def terminate(self) -> None:
            self.finished = True

        def kill(self) -> None:
            self.finished = True

    def fake_launch(
        run_id: str,
        *,
        invocation_root: Path,
        **_kwargs: Any,
    ) -> CONTROLLER.RunningWorker:
        launched.append(run_id)
        stdout_path = invocation_root / f"{run_id}.stdout.log"
        stderr_path = invocation_root / f"{run_id}.stderr.log"
        stdout = stdout_path.open("x", encoding="utf-8")
        stderr = stderr_path.open("x", encoding="utf-8")
        stdout.write(f"stdout for {run_id}\n")
        stderr.write(f"stderr for {run_id}\n")
        stdout.flush()
        stderr.flush()
        return CONTROLLER.RunningWorker(
            run_id=run_id,
            process=FakeProcess(run_id),
            command=["fake-worker", run_id],
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            stdout_file=stdout,
            stderr_file=stderr,
            started_at_utc="now",
            started_monotonic=0.0,
        )

    returncode = CONTROLLER.execute_controller(
        output_root=output,
        concurrency=2,
        device="mps",
        max_edges=1_200_000_000,
        max_forward_points=800_000,
        rss_sample_seconds=0.1,
        poll_seconds=0.001,
        launch_worker=fake_launch,
    )

    assert returncode == 1
    assert launched == list(run_ids[1:])
    assert maximum_active == 2
    invocation = output / "controller_runs" / "invocation_0001"
    failed_result = json.loads(
        (invocation / f"{failing}.process_result.json").read_text()
    )
    assert failed_result["returncode"] == 1
    assert failed_result["terminal_status"] is None
    assert (invocation / f"{failing}.stdout.log").is_file()
    assert (invocation / f"{failing}.stderr.log").is_file()
    assert not (output / "runs" / failing / "completed.json").exists()


def test_aggregate_writes_json_and_csv_for_all_terminal_outcomes(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    outcomes = {
        run_id: _install_terminal(
            output,
            run_id,
            plan_sha256="plan",
            status=(
                "completed_invalid"
                if run_id == CONTROLLER.RUN_IDS[-1]
                else "completed"
            ),
        )
        for run_id in CONTROLLER.RUN_IDS
    }

    json_path, csv_path = CONTROLLER.aggregate_terminal_outcomes(
        output,
        outcomes,
        plan_sha256="plan",
    )

    payload = json.loads(json_path.read_text())
    assert payload["run_counts"] == {
        "terminal": 15,
        "completed": 14,
        "completed_invalid": 1,
    }
    assert len(payload["rows"]) == 15
    assert payload["combined_correct_percentage_descriptive"]["n"] == 14
    assert len(csv_path.read_text().splitlines()) == 16


def test_aggregate_refuses_partial_matrix(tmp_path: Path) -> None:
    output = tmp_path / "output"
    outcome = _install_terminal(
        output,
        CONTROLLER.RUN_IDS[0],
        plan_sha256="plan",
    )
    with pytest.raises(ValueError, match="all 15"):
        CONTROLLER.aggregate_terminal_outcomes(
            output,
            {outcome.run_id: outcome},
            plan_sha256="plan",
        )
