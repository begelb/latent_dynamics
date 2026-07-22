#!/usr/bin/env python3
"""External resource watchdog for the exact Leslie run (macOS)."""

from __future__ import annotations

import csv
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
PID_FILE = RUN_DIR / "run.pid"
CSV_PATH = RUN_DIR / "resource_usage.csv"
EVENT_PATH = RUN_DIR / "watchdog_event.txt"
RSS_ABORT_GIB = 30.0
FREE_PERCENT_ABORT = 8
DISK_FREE_ABORT_GIB = 20.0
POLL_SECONDS = 10


def wait_for_pid() -> int:
    for _ in range(120):
        if PID_FILE.exists():
            raw = PID_FILE.read_text(encoding="utf-8").strip()
            if raw:
                return int(raw)
        time.sleep(0.5)
    raise TimeoutError(f"timed out waiting for {PID_FILE}")


def process_sample(pid: int) -> tuple[float, float, str] | None:
    result = subprocess.run(
        ["ps", "-o", "rss=,%cpu=,etime=", "-p", str(pid)],
        check=False,
        capture_output=True,
        text=True,
    )
    line = result.stdout.strip()
    if result.returncode != 0 or not line:
        return None
    rss_kib, cpu_percent, elapsed = line.split(maxsplit=2)
    return float(rss_kib) / (1024 * 1024), float(cpu_percent), elapsed


def free_memory_percent() -> int:
    result = subprocess.run(
        ["memory_pressure", "-Q"], check=False, capture_output=True, text=True
    )
    for line in result.stdout.splitlines():
        marker = "System-wide memory free percentage:"
        if marker in line:
            return int(line.split(marker, 1)[1].strip().rstrip("%"))
    return -1


def abort(pid: int, reason: str) -> None:
    message = f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} ABORT {reason}\n"
    EVENT_PATH.write_text(message, encoding="utf-8")
    print(message, end="", flush=True)
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    for _ in range(20):
        time.sleep(0.5)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def main() -> int:
    pid = wait_for_pid()
    new_file = not CSV_PATH.exists()
    with CSV_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if new_file:
            writer.writerow(
                [
                    "unix_time",
                    "local_time",
                    "pid",
                    "elapsed",
                    "rss_gib",
                    "cpu_percent",
                    "system_free_percent",
                    "disk_free_gib",
                ]
            )
            handle.flush()
        while True:
            sample = process_sample(pid)
            if sample is None:
                print(f"process {pid} exited", flush=True)
                return 0
            rss_gib, cpu_percent, elapsed = sample
            free_percent = free_memory_percent()
            disk_free_gib = shutil.disk_usage(RUN_DIR).free / (1024**3)
            now = time.time()
            writer.writerow(
                [
                    f"{now:.3f}",
                    time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    pid,
                    elapsed,
                    f"{rss_gib:.6f}",
                    f"{cpu_percent:.1f}",
                    free_percent,
                    f"{disk_free_gib:.3f}",
                ]
            )
            handle.flush()
            print(
                f"pid={pid} elapsed={elapsed} rss={rss_gib:.3f}GiB "
                f"cpu={cpu_percent:.1f}% free={free_percent}% disk={disk_free_gib:.1f}GiB",
                flush=True,
            )
            if rss_gib >= RSS_ABORT_GIB:
                abort(pid, f"RSS {rss_gib:.3f} GiB >= {RSS_ABORT_GIB:.1f} GiB")
                return 2
            if 0 <= free_percent < FREE_PERCENT_ABORT:
                abort(pid, f"system free memory {free_percent}% < {FREE_PERCENT_ABORT}%")
                return 2
            if disk_free_gib < DISK_FREE_ABORT_GIB:
                abort(pid, f"disk free {disk_free_gib:.3f} GiB < {DISK_FREE_ABORT_GIB:.1f} GiB")
                return 2
            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
