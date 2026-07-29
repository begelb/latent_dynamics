"""Resume and aggregate the matched D3 uniform-s24 on-demand matrix.

The controller never performs CMGDB work in its own process.  It validates the
frozen worker and shared analysis plan, launches at most two isolated worker
subprocesses, preserves controller-side stdout/stderr logs, and skips run
directories that already contain a validated ``completed.json`` or
``completed_invalid.json`` marker.

Use this only after any manually launched worker pair has exited.  Without
``--execute`` the controller is read-only and prints the current matrix state.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import psutil

import run_chafee_d3_ondemand_5x3_worker as worker

base = worker.base

CODE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
WORKER_PATH = Path(worker.__file__).resolve()
DEFAULT_OUTPUT_ROOT = worker.DEFAULT_OUTPUT_ROOT
MAX_CONCURRENCY = 2
RUN_IDS = tuple(
    f"dataset_{dataset:02d}_seed_{seed:02d}_lr3e3_e4000"
    for dataset in range(1, 6)
    for seed in range(3)
)
TERMINAL_MARKERS = ("completed.json", "completed_invalid.json")


@dataclass(frozen=True)
class TerminalOutcome:
    run_id: str
    status: str
    marker_path: Path
    marker: dict[str, Any]
    result_path: Path
    result: dict[str, Any]


@dataclass
class RunningWorker:
    run_id: str
    process: subprocess.Popen[Any]
    command: list[str]
    stdout_path: Path
    stderr_path: Path
    stdout_file: TextIO
    stderr_file: TextIO
    started_at_utc: str
    started_monotonic: float

    def close_logs(self) -> None:
        self.stdout_file.close()
        self.stderr_file.close()


def _read_json(path: Path) -> dict[str, Any]:
    return base._read_json(path)


def _record_path(
    record: Mapping[str, Any],
    *,
    relative_to: Path,
) -> Path:
    raw = Path(str(record.get("path", "")))
    path = raw if raw.is_absolute() else relative_to / raw
    return path.resolve()


def _verify_file_record(
    record: Mapping[str, Any],
    *,
    relative_to: Path,
) -> Path:
    path = _record_path(record, relative_to=relative_to)
    observed = base._file_record(
        path,
        expected_sha256=str(record.get("sha256", "")),
    )
    if observed["size_bytes"] != int(record.get("size_bytes", -1)):
        raise ValueError(f"size mismatch for recorded artifact {path}")
    return path


def _run_root(output_root: Path, run_id: str) -> Path:
    return output_root.resolve() / "runs" / run_id


def read_terminal_outcome(
    output_root: Path,
    run_id: str,
    *,
    expected_plan_sha256: str | None = None,
) -> TerminalOutcome | None:
    """Read and validate one run-level terminal marker, if present."""

    run_root = _run_root(output_root, run_id)
    present = [run_root / name for name in TERMINAL_MARKERS if (run_root / name).exists()]
    if not present:
        return None
    if len(present) != 1:
        raise ValueError(f"{run_id} has conflicting terminal markers: {present}")
    marker_path = present[0]
    marker = _read_json(marker_path)
    expected_status = (
        "completed" if marker_path.name == "completed.json" else "completed_invalid"
    )
    if (
        marker.get("status") != expected_status
        or marker.get("run_id") != run_id
    ):
        raise ValueError(f"malformed terminal marker {marker_path}")
    if (
        expected_plan_sha256 is not None
        and marker.get("analysis_plan_sha256") != expected_plan_sha256
    ):
        raise ValueError(f"{run_id} terminal marker uses a different analysis plan")
    record_key = "summary" if expected_status == "completed" else "result"
    record = marker.get(record_key)
    if not isinstance(record, dict):
        raise ValueError(f"{marker_path} has no {record_key} artifact record")
    result_path = _verify_file_record(record, relative_to=run_root)
    result = _read_json(result_path)
    expected_result_status = (
        "complete" if expected_status == "completed" else "completed_invalid"
    )
    if (
        result.get("status") != expected_result_status
        or result.get("run_id") != run_id
    ):
        raise ValueError(f"terminal result disagrees with {marker_path}")
    if (
        expected_plan_sha256 is not None
        and result.get("analysis_plan_sha256") != expected_plan_sha256
    ):
        raise ValueError(f"{run_id} result uses a different analysis plan")
    return TerminalOutcome(
        run_id=run_id,
        status=expected_status,
        marker_path=marker_path,
        marker=marker,
        result_path=result_path,
        result=result,
    )


def collect_terminal_outcomes(
    output_root: Path,
    *,
    expected_plan_sha256: str | None = None,
) -> tuple[dict[str, TerminalOutcome], list[str]]:
    outcomes: dict[str, TerminalOutcome] = {}
    pending: list[str] = []
    for run_id in RUN_IDS:
        outcome = read_terminal_outcome(
            output_root,
            run_id,
            expected_plan_sha256=expected_plan_sha256,
        )
        if outcome is None:
            pending.append(run_id)
        else:
            outcomes[run_id] = outcome
    return outcomes, pending


def active_external_workers() -> list[dict[str, Any]]:
    """Return live matched-D3 worker processes not launched by this controller."""

    matches: list[dict[str, Any]] = []
    worker_name = WORKER_PATH.name
    for process in psutil.process_iter(("pid", "status", "cmdline")):
        try:
            status = process.info.get("status")
            command = [str(value) for value in (process.info.get("cmdline") or [])]
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
        if status == psutil.STATUS_ZOMBIE:
            continue
        if any(
            value == str(WORKER_PATH) or Path(value).name == worker_name
            for value in command
        ):
            matches.append({"pid": int(process.info["pid"]), "command": command})
    return matches


def validate_analysis_plan(
    output_root: Path,
    *,
    expected_device: str,
    max_edges: int,
    max_forward_points: int,
    rss_sample_seconds: float,
) -> tuple[dict[str, Any], str]:
    """Validate the frozen plan body, computation, and current worker hash."""

    plan_path = output_root.resolve() / "analysis_plan.json"
    envelope = _read_json(plan_path)
    plan = envelope.get("plan")
    plan_sha256 = str(envelope.get("plan_sha256", ""))
    if not isinstance(plan, dict) or base._payload_sha256(plan) != plan_sha256:
        raise ValueError(f"invalid analysis plan envelope: {plan_path}")
    computation = plan.get("computation")
    if not isinstance(computation, dict):
        raise ValueError("analysis plan has no computation record")
    expected = {
        "backend": "batched_on_demand_neural",
        "precomputed": False,
        "subdivisions": [24, 24, 24],
        "padding": True,
        "expected_cells": base.EXPECTED_CELLS,
        "expected_callback_rectangles": base.EXPECTED_CALLBACK_RECTANGLES,
        "expected_neural_corner_points": base.EXPECTED_NEURAL_CORNER_POINTS,
        "max_edges": max_edges,
        "max_forward_points": max_forward_points,
        "device": expected_device,
        "trajectory_and_root_encoding_device": "cpu",
        "rss_sample_seconds": rss_sample_seconds,
    }
    for key, value in expected.items():
        if computation.get(key) != value:
            raise ValueError(
                f"analysis plan computation {key!r} differs: "
                f"{computation.get(key)!r} != {value!r}"
            )
    concurrency = plan.get("concurrency")
    if (
        not isinstance(concurrency, dict)
        or int(concurrency.get("maximum_processes", -1)) != MAX_CONCURRENCY
    ):
        raise ValueError("analysis plan does not freeze maximum concurrency at 2")
    common_sources = plan.get("common_sources")
    if not isinstance(common_sources, dict):
        raise ValueError("analysis plan has no common source records")
    worker_record = common_sources.get("worker")
    if not isinstance(worker_record, dict):
        raise ValueError("analysis plan has no frozen worker record")
    worker_path = _verify_file_record(
        worker_record,
        relative_to=output_root.resolve(),
    )
    if worker_path != WORKER_PATH:
        raise ValueError(
            f"analysis plan worker {worker_path} is not controller worker {WORKER_PATH}"
        )
    return envelope, plan_sha256


def ensure_and_validate_analysis_plan(
    output_root: Path,
    *,
    device_name: str,
    max_edges: int,
    max_forward_points: int,
    rss_sample_seconds: float,
) -> tuple[dict[str, Any], str]:
    """Publish the worker's canonical plan if absent, then independently verify it."""

    inputs = worker._resolve_inputs(RUN_IDS[0])
    device = base._resolve_device(device_name)
    runtime = base._runtime_provenance(device)
    safe_root = worker._assert_safe_output_root(output_root, inputs)
    safe_root.mkdir(parents=True, exist_ok=True)
    worker._ensure_analysis_plan(
        safe_root,
        inputs=inputs,
        runtime=runtime,
        device=str(device),
        max_edges=max_edges,
        max_forward_points=max_forward_points,
        rss_sample_seconds=rss_sample_seconds,
    )
    return validate_analysis_plan(
        safe_root,
        expected_device=str(device),
        max_edges=max_edges,
        max_forward_points=max_forward_points,
        rss_sample_seconds=rss_sample_seconds,
    )


def _next_invocation_root(output_root: Path) -> Path:
    parent = output_root.resolve() / "controller_runs"
    parent.mkdir(parents=True, exist_ok=True)
    indices = []
    for path in parent.iterdir():
        if path.is_dir() and path.name.startswith("invocation_"):
            suffix = path.name.removeprefix("invocation_")
            if suffix.isdigit():
                indices.append(int(suffix))
    index = max(indices, default=0) + 1
    while True:
        path = parent / f"invocation_{index:04d}"
        try:
            path.mkdir(exist_ok=False)
            return path
        except FileExistsError:
            index += 1


@contextmanager
def _controller_lock(output_root: Path) -> Any:
    lock_path = output_root.resolve() / "controller.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"another matched-D3 controller holds {lock_path}"
            ) from error
        lock.seek(0)
        lock.truncate()
        lock.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "started_at_utc": base._utc_now(),
                    "controller": str(SCRIPT_PATH),
                }
            )
            + "\n"
        )
        lock.flush()
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def worker_command(
    run_id: str,
    *,
    output_root: Path,
    device: str,
    max_edges: int,
    max_forward_points: int,
    rss_sample_seconds: float,
) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(WORKER_PATH),
        "--run-id",
        run_id,
        "--output-root",
        str(output_root.resolve()),
        "--device",
        device,
        "--max-edges",
        str(max_edges),
        "--max-forward-points",
        str(max_forward_points),
        "--rss-sample-seconds",
        str(rss_sample_seconds),
    ]


def _launch_worker(
    run_id: str,
    *,
    invocation_root: Path,
    output_root: Path,
    device: str,
    max_edges: int,
    max_forward_points: int,
    rss_sample_seconds: float,
) -> RunningWorker:
    stdout_path = invocation_root / f"{run_id}.stdout.log"
    stderr_path = invocation_root / f"{run_id}.stderr.log"
    stdout_file = stdout_path.open("x", encoding="utf-8")
    stderr_file = stderr_path.open("x", encoding="utf-8")
    command = worker_command(
        run_id,
        output_root=output_root,
        device=device,
        max_edges=max_edges,
        max_forward_points=max_forward_points,
        rss_sample_seconds=rss_sample_seconds,
    )
    try:
        process = subprocess.Popen(
            command,
            cwd=CODE_ROOT,
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
        )
    except BaseException:
        stdout_file.close()
        stderr_file.close()
        raise
    return RunningWorker(
        run_id=run_id,
        process=process,
        command=command,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        stdout_file=stdout_file,
        stderr_file=stderr_file,
        started_at_utc=base._utc_now(),
        started_monotonic=time.monotonic(),
    )


def _process_result(
    running: RunningWorker,
    *,
    returncode: int,
    outcome: TerminalOutcome | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": running.run_id,
        "command": running.command,
        "pid": int(running.process.pid),
        "started_at_utc": running.started_at_utc,
        "finished_at_utc": base._utc_now(),
        "wall_seconds": time.monotonic() - running.started_monotonic,
        "returncode": int(returncode),
        "terminal_status": outcome.status if outcome is not None else None,
        "terminal_marker": (
            base._file_record(outcome.marker_path)
            if outcome is not None
            else None
        ),
        "stdout": base._file_record(running.stdout_path),
        "stderr": base._file_record(running.stderr_path),
    }


def _row_from_outcome(outcome: TerminalOutcome) -> dict[str, Any]:
    marker = outcome.marker
    result = outcome.result
    row: dict[str, Any] = {
        "run_id": outcome.run_id,
        "status": outcome.status,
        "dataset": int(marker["dataset"]),
        "training_seed": int(marker["training_seed"]),
        "attempt": int(marker["attempt"]),
        "analysis_plan_sha256": str(marker["analysis_plan_sha256"]),
        "result_path": str(outcome.result_path),
    }
    if outcome.status == "completed_invalid":
        row.update(
            {
                "reason_type": result.get("reason_type"),
                "reason": result.get("reason"),
                "elapsed_seconds": result.get("elapsed_seconds"),
            }
        )
        return row

    statistics_payload = result.get("statistics")
    graph = result.get("graph")
    callback = result.get("callback")
    timings = result.get("timings")
    memory = result.get("memory")
    if not all(
        isinstance(value, dict)
        for value in (statistics_payload, graph, callback, timings, memory)
    ):
        raise ValueError(f"{outcome.run_id} summary is missing aggregate fields")
    counts = statistics_payload.get("counts")
    percentages = statistics_payload.get("percentages")
    if not isinstance(counts, dict) or not isinstance(percentages, dict):
        raise ValueError(f"{outcome.run_id} summary has malformed statistics")
    combined_correct_count = int(
        counts["correctly_classified_in_negative_basin"]
    ) + int(counts["correctly_classified_in_positive_basin"])
    conditioned = int(statistics_payload["conditioned_trajectories"])
    row.update(
        {
            "checkpoint_sha256": result["sources"]["checkpoint"]["sha256"],
            "cached_edges": int(graph["cached_edges"]),
            "morse_nodes": int(graph["morse_nodes"]),
            "minimal_nodes": graph["minimal_nodes"],
            "scalar_calls": int(callback["scalar_calls"]),
            "batch_calls": int(callback["batch_calls"]),
            "rectangles": int(callback["rectangles"]),
            "neural_corner_points": int(callback["neural_corner_points"]),
            "cmgdb_seconds": float(timings["cmgdb_seconds"]),
            "total_seconds": float(timings["total_seconds"]),
            "sampled_peak_rss_bytes": memory.get("sampled_peak_rss_bytes"),
            "conditioned_trajectories": conditioned,
            **{name: int(value) for name, value in counts.items()},
            **{
                f"{name}_percentage": float(value)
                for name, value in percentages.items()
            },
            "combined_correct_count": combined_correct_count,
            "combined_correct_percentage": (
                100.0 * combined_correct_count / conditioned
            ),
        }
    )
    return row


def _descriptives(values: Sequence[float]) -> dict[str, float | int] | None:
    numeric = [float(value) for value in values]
    if not numeric:
        return None
    return {
        "n": len(numeric),
        "mean": statistics.fmean(numeric),
        "sample_standard_deviation": (
            statistics.stdev(numeric) if len(numeric) > 1 else 0.0
        ),
        "median": statistics.median(numeric),
        "minimum": min(numeric),
        "maximum": max(numeric),
    }


def aggregate_terminal_outcomes(
    output_root: Path,
    outcomes: Mapping[str, TerminalOutcome],
    *,
    plan_sha256: str,
) -> tuple[Path, Path]:
    """Write aggregate JSON and CSV only for the complete 15-run terminal matrix."""

    if set(outcomes) != set(RUN_IDS):
        missing = sorted(set(RUN_IDS) - set(outcomes))
        raise ValueError(f"cannot aggregate before all 15 terminal outcomes: {missing}")
    rows = [_row_from_outcome(outcomes[run_id]) for run_id in RUN_IDS]
    completed = [row for row in rows if row["status"] == "completed"]
    invalid = [row for row in rows if row["status"] == "completed_invalid"]
    payload = {
        "schema_version": 1,
        "status": "complete",
        "aggregated_at_utc": base._utc_now(),
        "analysis_plan_sha256": plan_sha256,
        "run_counts": {
            "terminal": len(rows),
            "completed": len(completed),
            "completed_invalid": len(invalid),
        },
        "combined_correct_percentage_descriptive": _descriptives(
            [float(row["combined_correct_percentage"]) for row in completed]
        ),
        "cmgdb_seconds_descriptive": _descriptives(
            [float(row["cmgdb_seconds"]) for row in completed]
        ),
        "rows": rows,
    }
    root = output_root.resolve()
    json_path = root / "aggregate_results.json"
    csv_path = root / "aggregate_results.csv"
    base._write_json_atomic(json_path, payload)

    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = csv_path.with_name(f".{csv_path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, sort_keys=True)
                        if isinstance(value, (dict, list))
                        else value
                    )
                    for key, value in row.items()
                }
            )
    temporary.replace(csv_path)
    return json_path, csv_path


def _write_controller_progress(
    output_root: Path,
    *,
    plan_sha256: str,
    invocation_root: Path,
    outcomes: Mapping[str, TerminalOutcome],
    pending: Sequence[str],
    process_failures: Sequence[Mapping[str, Any]],
) -> None:
    base._write_json_atomic(
        output_root.resolve() / "controller_progress.json",
        {
            "schema_version": 1,
            "updated_at_utc": base._utc_now(),
            "analysis_plan_sha256": plan_sha256,
            "invocation_root": str(invocation_root),
            "terminal_count": len(outcomes),
            "completed_count": sum(
                outcome.status == "completed" for outcome in outcomes.values()
            ),
            "completed_invalid_count": sum(
                outcome.status == "completed_invalid"
                for outcome in outcomes.values()
            ),
            "pending_run_ids": list(pending),
            "process_failures": list(process_failures),
        },
    )


def execute_controller(
    *,
    output_root: Path,
    concurrency: int,
    device: str,
    max_edges: int,
    max_forward_points: int,
    rss_sample_seconds: float,
    poll_seconds: float,
    launch_worker: Callable[..., RunningWorker] = _launch_worker,
) -> int:
    if concurrency < 1 or concurrency > MAX_CONCURRENCY:
        raise ValueError("concurrency must be 1 or 2")
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")

    with _controller_lock(output_root):
        external = active_external_workers()
        if external:
            raise RuntimeError(
                "matched-D3 workers are already active; start this controller "
                f"only after the manual first pair exits: {external}"
            )
        envelope, plan_sha256 = ensure_and_validate_analysis_plan(
            output_root,
            device_name=device,
            max_edges=max_edges,
            max_forward_points=max_forward_points,
            rss_sample_seconds=rss_sample_seconds,
        )
        del envelope
        outcomes, pending = collect_terminal_outcomes(
            output_root,
            expected_plan_sha256=plan_sha256,
        )
        if not pending:
            aggregate_terminal_outcomes(
                output_root,
                outcomes,
                plan_sha256=plan_sha256,
            )
            return 0

        invocation_root = _next_invocation_root(output_root)
        controller_manifest = {
            "schema_version": 1,
            "started_at_utc": base._utc_now(),
            "controller": base._file_record(SCRIPT_PATH),
            "worker": base._file_record(WORKER_PATH),
            "analysis_plan": base._file_record(
                output_root.resolve() / "analysis_plan.json"
            ),
            "analysis_plan_sha256": plan_sha256,
            "concurrency": concurrency,
            "initial_terminal_run_ids": sorted(outcomes),
            "scheduled_run_ids": list(pending),
        }
        base._write_json_exclusive(
            invocation_root / "invocation_manifest.json",
            controller_manifest,
        )

        queue = list(pending)
        running: dict[str, RunningWorker] = {}
        process_failures: list[dict[str, Any]] = []
        launched_worker_sha = base._sha256(WORKER_PATH)
        try:
            while queue or running:
                while queue and len(running) < concurrency:
                    # Freeze-check immediately before every process launch.
                    if base._sha256(WORKER_PATH) != launched_worker_sha:
                        raise RuntimeError("worker changed during controller invocation")
                    validate_analysis_plan(
                        output_root,
                        expected_device=device,
                        max_edges=max_edges,
                        max_forward_points=max_forward_points,
                        rss_sample_seconds=rss_sample_seconds,
                    )
                    run_id = queue.pop(0)
                    running[run_id] = launch_worker(
                        run_id,
                        invocation_root=invocation_root,
                        output_root=output_root,
                        device=device,
                        max_edges=max_edges,
                        max_forward_points=max_forward_points,
                        rss_sample_seconds=rss_sample_seconds,
                    )
                    print(f"controller: launched {run_id}", flush=True)

                finished: list[tuple[str, int]] = []
                for run_id, active in running.items():
                    returncode = active.process.poll()
                    if returncode is not None:
                        finished.append((run_id, int(returncode)))
                if not finished:
                    time.sleep(poll_seconds)
                    continue

                for run_id, returncode in finished:
                    active = running.pop(run_id)
                    active.close_logs()
                    outcome = read_terminal_outcome(
                        output_root,
                        run_id,
                        expected_plan_sha256=plan_sha256,
                    )
                    result = _process_result(
                        active,
                        returncode=returncode,
                        outcome=outcome,
                    )
                    base._write_json_exclusive(
                        invocation_root / f"{run_id}.process_result.json",
                        result,
                    )
                    if returncode != 0 or outcome is None:
                        process_failures.append(result)
                        print(
                            f"controller: {run_id} failed with return code "
                            f"{returncode}; evidence retained in {invocation_root}",
                            file=sys.stderr,
                            flush=True,
                        )
                    else:
                        outcomes[run_id] = outcome
                        print(
                            f"controller: {run_id} -> {outcome.status}",
                            flush=True,
                        )
                    _, still_pending = collect_terminal_outcomes(
                        output_root,
                        expected_plan_sha256=plan_sha256,
                    )
                    _write_controller_progress(
                        output_root,
                        plan_sha256=plan_sha256,
                        invocation_root=invocation_root,
                        outcomes=outcomes,
                        pending=still_pending,
                        process_failures=process_failures,
                    )
        except BaseException:
            interruption = {
                "schema_version": 1,
                "interrupted_at_utc": base._utc_now(),
                "error": traceback.format_exc(),
                "active_run_ids": sorted(running),
                "queued_run_ids": queue,
            }
            base._write_json_atomic(
                invocation_root / "controller_interruption.json",
                interruption,
            )
            for active in running.values():
                if active.process.poll() is None:
                    active.process.terminate()
            deadline = time.monotonic() + 5.0
            for active in running.values():
                while (
                    active.process.poll() is None
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.1)
                if active.process.poll() is None:
                    active.process.kill()
                active.close_logs()
            raise

        outcomes, pending_after = collect_terminal_outcomes(
            output_root,
            expected_plan_sha256=plan_sha256,
        )
        _write_controller_progress(
            output_root,
            plan_sha256=plan_sha256,
            invocation_root=invocation_root,
            outcomes=outcomes,
            pending=pending_after,
            process_failures=process_failures,
        )
        if len(outcomes) == len(RUN_IDS):
            aggregate_terminal_outcomes(
                output_root,
                outcomes,
                plan_sha256=plan_sha256,
            )
        base._write_json_atomic(
            invocation_root / "invocation_summary.json",
            {
                "schema_version": 1,
                "completed_at_utc": base._utc_now(),
                "terminal_count": len(outcomes),
                "pending_run_ids": pending_after,
                "process_failure_count": len(process_failures),
            },
        )
        return 0 if len(outcomes) == len(RUN_IDS) else 1


def plan_status(output_root: Path) -> dict[str, Any]:
    root = output_root.resolve()
    plan_path = root / "analysis_plan.json"
    plan_sha256: str | None = None
    plan_error: str | None = None
    if plan_path.is_file():
        try:
            envelope = _read_json(plan_path)
            plan = envelope.get("plan")
            candidate = str(envelope.get("plan_sha256", ""))
            if not isinstance(plan, dict) or base._payload_sha256(plan) != candidate:
                raise ValueError("plan envelope hash mismatch")
            plan_sha256 = candidate
        except Exception as error:
            plan_error = f"{type(error).__name__}: {error}"
    try:
        outcomes, pending = collect_terminal_outcomes(
            root,
            expected_plan_sha256=plan_sha256,
        )
        terminal_error = None
    except Exception as error:
        outcomes, pending = {}, list(RUN_IDS)
        terminal_error = f"{type(error).__name__}: {error}"
    return {
        "schema_version": 1,
        "output_root": str(root),
        "worker": base._file_record(WORKER_PATH),
        "analysis_plan_path": str(plan_path),
        "analysis_plan_sha256": plan_sha256,
        "analysis_plan_error": plan_error,
        "active_external_workers": active_external_workers(),
        "terminal_count": len(outcomes),
        "completed_run_ids": sorted(
            run_id
            for run_id, outcome in outcomes.items()
            if outcome.status == "completed"
        ),
        "completed_invalid_run_ids": sorted(
            run_id
            for run_id, outcome in outcomes.items()
            if outcome.status == "completed_invalid"
        ),
        "pending_run_ids": pending,
        "terminal_error": terminal_error,
        "production_launches_permitted": False,
        "note": (
            "Pass --execute only after active_external_workers is empty; "
            "the controller will then enforce a two-process maximum."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--concurrency", type=int, choices=(1, 2), default=2)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument("--max-edges", type=int, default=1_200_000_000)
    parser.add_argument("--max-forward-points", type=int, default=800_000)
    parser.add_argument("--rss-sample-seconds", type=float, default=0.1)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="launch pending workers; otherwise report read-only status",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.execute:
        print(json.dumps(plan_status(args.output_root), indent=2, sort_keys=True))
        return 0
    try:
        return execute_controller(
            output_root=args.output_root,
            concurrency=args.concurrency,
            device=args.device,
            max_edges=args.max_edges,
            max_forward_points=args.max_forward_points,
            rss_sample_seconds=args.rss_sample_seconds,
            poll_seconds=args.poll_seconds,
        )
    except Exception as error:
        traceback.print_exc()
        print(
            f"matched-D3 controller failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
