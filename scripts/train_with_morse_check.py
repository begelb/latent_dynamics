"""Run the full pipeline for a config; retry training until the Morse graph
has at least N minimal nodes. After ``max_train_retries`` failed retraining
attempts (each with a rotated seed), regenerate the data and try one more
full cycle. Exit code reflects success or the failure mode.

Usage:
    python scripts/train_with_morse_check.py <config_name> [--min-minimal N]
        [--max-train-retries K]

The script expects to be run from the ``code/`` directory (relative paths in
the YAML configs).
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

CODE_ROOT = Path(__file__).resolve().parents[1]
# Do NOT call .resolve() here: the venv's python is a symlink chain that
# ultimately points at the system interpreter. Resolving it bypasses the
# virtualenv and the editable install of `latentdynamics` is invisible.
VENV_PY = CODE_ROOT.parent / ".venv" / "bin" / "python"


def count_minimal_nodes(dot_path: Path) -> int:
    """Return the number of nodes with no outgoing edges in a CMGDB DOT file."""
    text = dot_path.read_text()
    nodes: set[str] = set()
    sources: set[str] = set()
    for raw in text.splitlines():
        line = raw.strip()
        m = re.match(r'^"?(\d+)"?\s*\[', line)
        if m:
            nodes.add(m.group(1))
            continue
        m = re.match(r'^"?(\d+)"?\s*->\s*"?(\d+)"?', line)
        if m:
            sources.add(m.group(1))
    return len(nodes - sources)


def find_morse_graph(output_dir: Path) -> Path | None:
    matches = sorted(output_dir.rglob("MG/morse_graph"))
    return matches[0] if matches else None


def run_pipeline(config_rel: str, stages: str, force: bool) -> tuple[bool, str]:
    cmd = [
        str(VENV_PY), "pipeline.py",
        "--config", config_rel,
        "--stages", stages, "--quiet",
    ]
    if force:
        cmd.append("--force-overwrite")
    res = subprocess.run(cmd, cwd=CODE_ROOT, capture_output=True, text=True)
    tail = (res.stdout[-1500:] + "\n" + res.stderr[-1500:]).strip()
    return res.returncode == 0, tail


def set_seed(config_path: Path, seed: int) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg["seeds"] = [seed]
    with open(config_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def evaluate(output_dir: Path, min_minimal: int) -> tuple[int, Path | None]:
    mg = find_morse_graph(output_dir)
    if mg is None:
        return -1, None
    return count_minimal_nodes(mg), mg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_name")
    parser.add_argument("--min-minimal", type=int, default=2)
    parser.add_argument("--max-train-retries", type=int, default=10)
    args = parser.parse_args()

    config_path = CODE_ROOT / "configs" / f"{args.config_name}.yaml"
    config_rel = f"configs/{args.config_name}.yaml"
    if not config_path.exists():
        print(f"FAIL: config not found at {config_path}")
        return 1
    output_dir = CODE_ROOT / "output" / args.config_name
    backup_path = config_path.with_suffix(".yaml.bak")
    shutil.copy(config_path, backup_path)

    def cleanup() -> None:
        # Restore the original config so concurrent agents share a clean file.
        shutil.move(backup_path, config_path)

    try:
        t0 = time.time()
        print(f"[{args.config_name}] initial full pipeline")
        ok, tail = run_pipeline(config_rel, "all", force=True)
        if not ok:
            print(f"FAIL: initial pipeline failed\n{tail}")
            return 2

        for round_kind in ("train", "data"):
            attempts = args.max_train_retries if round_kind == "train" else 1
            for i in range(attempts + 1):  # +1 for the evaluation pass after the latest run
                n_min, mg = evaluate(output_dir, args.min_minimal)
                if mg is None:
                    print(f"FAIL: no morse_graph under {output_dir}")
                    return 3
                elapsed = int(time.time() - t0)
                print(
                    f"[{args.config_name}] round={round_kind} attempt={i} "
                    f"n_minimal={n_min} mg={mg.relative_to(CODE_ROOT)} elapsed={elapsed}s"
                )
                if n_min >= args.min_minimal:
                    print(
                        f"SUCCESS: {args.config_name} reached {n_min} minimal "
                        f"nodes (round={round_kind}, attempt={i})"
                    )
                    return 0
                if i >= attempts:
                    break
                if round_kind == "train":
                    new_seed = i + 1
                    print(f"[{args.config_name}] retraining with seed {new_seed}")
                    set_seed(config_path, new_seed)
                    ok, tail = run_pipeline(
                        config_rel, "scale,train,diagnose,morse,render,metrics", force=True
                    )
                else:
                    print(f"[{args.config_name}] regenerating data + retraining")
                    set_seed(config_path, args.max_train_retries + 1)
                    ok, tail = run_pipeline(config_rel, "all", force=True)
                if not ok:
                    print(f"FAIL: retry pipeline failed\n{tail}")
                    return 4

        print(
            f"FAIL: {args.config_name} did not reach {args.min_minimal} minimal nodes "
            f"after {args.max_train_retries} retrains and 1 data regen"
        )
        return 5
    finally:
        cleanup()


if __name__ == "__main__":
    sys.exit(main())
