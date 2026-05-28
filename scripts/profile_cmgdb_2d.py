"""Profile CMGDB on the 2D Leslie head: analytic vs learned 2D->2D latent map.

For each subdiv_max in a sweep, run CMGDB twice — once on the analytic
``LeslieContraction(th1=20, th2=20)`` step function and once on the learned
``LatentDynamicsAutoencoder.latent_map`` from a trained 2D->2D checkpoint —
recording wall-clock, Morse-graph structure, and total box count for each.

Also microbenchmarks the cost of evaluating each map at a single point and of
``CMGDB.BoxMap`` for one rectangle, since the BoxMap implementation evaluates
the dynamics at 2^d corner points sequentially via a Python list comprehension
(see ``CMGDB.ComputeBoxMap.BoxMap``). The 2^d corner-evaluation cost is the
main driver of the 10D scaling penalty observed in Patrick's archived runs.

Outputs:
  output/leslie2d_to_2d_profile/profile_results.json
  output/leslie2d_to_2d_profile/morse/<map>/<subdiv_max>/{morse_graph,morse_sets,...}

Example::

    python scripts/profile_cmgdb_2d.py \\
        --config configs/leslie2d_to_2d.yaml \\
        --subdiv-max 12 16 20 22 \\
        --device cpu
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import CMGDB
import numpy as np
import torch

from latentdynamics.analysis.morse import LatentBounds, infer_latent_bounds
from latentdynamics.config import load_config
from latentdynamics.sampling import load_scaler
from latentdynamics.systems import build_system
from latentdynamics.training import load_checkpoint
from latentdynamics.viz import save_morse_graph_artifacts

CODE_ROOT = Path(__file__).resolve().parents[1]


def _count_morse_nodes_edges(dot_path: Path) -> tuple[int, int, int]:
    """Parse a CMGDB DOT file: return (nodes, edges, minimal_nodes).

    A *minimal* node has no outgoing edges in the Morse partial order: those are
    the attractors. Used to detect bistability (>=2 minimal nodes), matching the
    convention in ``notebooks/Example_Leslie_model.ipynb``.
    """
    text = dot_path.read_text()
    nodes = 0
    edges = 0
    sources: set[str] = set()  # node ids that appear as edge sources
    declared: set[str] = set()  # node ids that have a label declaration
    for line in text.splitlines():
        stripped = line.strip().rstrip(";")
        if "->" in stripped:
            edges += 1
            src = stripped.split("->", 1)[0].strip()
            sources.add(src)
        elif "[label=" in stripped and "shape=" in stripped:
            nodes += 1
            nid = stripped.split("[", 1)[0].strip()
            declared.add(nid)
    minimal = len(declared - sources)
    return nodes, edges, minimal


def _count_morse_boxes(csv_path: Path) -> tuple[int, dict[int, int]]:
    """Count total boxes in a morse_sets CSV and per-Morse-set box counts."""
    counts: dict[int, int] = {}
    total = 0
    with csv_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            try:
                morse_id = int(parts[-1])
            except ValueError:
                continue
            counts[morse_id] = counts.get(morse_id, 0) + 1
            total += 1
    return total, counts


def _make_analytic_g(system) -> Callable[[Any], list[float]]:
    """The analytic step function; matches Patrick's 2D base computation."""

    def g(x: Any) -> list[float]:
        return list(system.step(np.asarray(x, dtype=np.float64)))

    return g


def _make_latent_g(
    autoencoder, device: torch.device
) -> Callable[[Any], list[float]]:
    """Wrap the learned latent map as a Python list->list callable for CMGDB."""
    latent_map = autoencoder.latent_map
    latent_map.eval()

    @torch.no_grad()
    def g(x: Any) -> list[float]:
        x_t = torch.as_tensor(x, dtype=torch.float32, device=device).reshape(1, -1)
        return latent_map(x_t)[0].cpu().numpy().tolist()

    return g


def _bench_single_call(g: Callable[[Any], list[float]], pt: np.ndarray, n: int = 1000) -> float:
    """Return median wall-clock seconds per single g(x) call."""
    pt_list = pt.tolist()
    # warmup
    for _ in range(5):
        g(pt_list)
    timings = []
    for _ in range(n):
        t0 = time.perf_counter()
        g(pt_list)
        timings.append(time.perf_counter() - t0)
    return float(np.median(timings))


def _bench_boxmap(g: Callable[[Any], list[float]], rect: list[float], n: int = 200) -> float:
    """Return median seconds per CMGDB.BoxMap(g, rect, padding=True) call."""
    for _ in range(5):
        CMGDB.BoxMap(g, rect, padding=True)
    timings = []
    for _ in range(n):
        t0 = time.perf_counter()
        CMGDB.BoxMap(g, rect, padding=True)
        timings.append(time.perf_counter() - t0)
    return float(np.median(timings))


def _run_morse(
    g: Callable[[Any], list[float]],
    bounds: LatentBounds,
    *,
    subdiv_init: int,
    subdiv_min: int,
    subdiv_max: int,
    subdiv_limit: int,
    padding: bool,
    out_dir: Path,
) -> dict[str, Any]:
    """Run CMGDB on g and save morse_graph/morse_sets; return statistics."""
    out_dir.mkdir(parents=True, exist_ok=True)

    def box_map(rect):
        return CMGDB.BoxMap(g, rect, padding=padding)

    model = CMGDB.Model(
        subdiv_min,
        subdiv_max,
        subdiv_init,
        subdiv_limit,
        bounds.lower.tolist(),
        bounds.upper.tolist(),
        box_map,
    )
    t0 = time.perf_counter()
    morse_graph, _ = CMGDB.ComputeConleyMorseGraph(model)
    duration_s = time.perf_counter() - t0

    dot_path, csv_path = save_morse_graph_artifacts(morse_graph, out_dir)
    n_nodes, n_edges, n_minimal = _count_morse_nodes_edges(dot_path)
    total_boxes, boxes_per_morse = _count_morse_boxes(csv_path)
    return {
        "duration_s": duration_s,
        "duration_min": duration_s / 60.0,
        "morse_nodes": n_nodes,
        "morse_edges": n_edges,
        "morse_minimal_nodes": n_minimal,
        "morse_total_boxes": total_boxes,
        "morse_boxes_per_set": boxes_per_morse,
        "dot": str(dot_path),
        "csv": str(csv_path),
    }


def _load_latent_bounds(cfg, train_file: str, autoencoder, device: torch.device) -> LatentBounds:
    """Reproduce morse_graph.py's bounds inference (encode all data)."""
    if cfg.cmgdb.lower_bounds is not None and cfg.cmgdb.upper_bounds is not None:
        return LatentBounds(
            lower=np.asarray(cfg.cmgdb.lower_bounds, dtype=np.float64),
            upper=np.asarray(cfg.cmgdb.upper_bounds, dtype=np.float64),
        )
    train = np.loadtxt(cfg.paths.data_dir / f"{train_file}.csv", delimiter=",", skiprows=1)
    test = np.loadtxt(cfg.paths.data_dir / "test.csv", delimiter=",", skiprows=1)
    scaler = load_scaler(cfg.paths.scaler_path(train_file))
    high = cfg.arch.high_dims
    pieces = [
        scaler.transform(train[:, :high]),
        scaler.transform(test[:, :high]),
        scaler.transform(train[:, high:]),
        scaler.transform(test[:, high:]),
    ]
    all_scaled = np.vstack(pieces)
    return infer_latent_bounds(
        autoencoder.encoder,
        all_scaled,
        epsilon_frac=cfg.cmgdb.bounds_epsilon_frac,
        device=device,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument(
        "--subdiv-max",
        type=int,
        nargs="+",
        default=[12, 14, 16, 18, 20, 22],
        help="space-separated list of subdiv_max values to sweep",
    )
    p.add_argument(
        "--subdiv-gap",
        type=int,
        default=1,
        help="subdiv_min = subdiv_max - subdiv-gap (default 1). Ignored if --uniform.",
    )
    p.add_argument(
        "--subdiv-init",
        type=int,
        default=None,
        help="adaptive: default subdiv_min - 7. Ignored if --uniform.",
    )
    p.add_argument("--subdiv-limit", type=int, default=10000)
    p.add_argument(
        "--uniform",
        action="store_true",
        help="force init=min=max for each subdiv value (uniform-grid mode, no adaptive refinement)",
    )
    p.add_argument(
        "--padding",
        choices=["true", "false"],
        default=None,
        help="override CMGDB.BoxMap padding (default: use config.cmgdb.padding)",
    )
    p.add_argument("--train-file", type=str, default="train")
    p.add_argument(
        "--maps",
        type=str,
        nargs="+",
        default=["analytic", "latent"],
        choices=["analytic", "latent"],
    )
    p.add_argument("--device", type=str, default="cpu", help="torch device for the latent map")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="defaults to <code>/output/<cfg.experiment>_profile",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = CODE_ROOT / "output" / f"{args.config.stem}_profile"
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_dir = cfg.paths.output_dir / "seed_0"
    autoencoder, _arch = load_checkpoint(seed_dir / "models")
    autoencoder.to(device)

    latent_bounds = _load_latent_bounds(cfg, args.train_file, autoencoder, device)

    # Build the analytic system & ambient bounds.
    system = build_system(cfg.system.name, cfg.system.params)
    ambient_bounds = LatentBounds(
        lower=np.asarray(system.lower_bounds, dtype=np.float64),
        upper=np.asarray(system.upper_bounds, dtype=np.float64),
    )

    # Microbenchmarks.
    micro: dict[str, Any] = {}
    if "analytic" in args.maps:
        g_a = _make_analytic_g(system)
        pt_a = (ambient_bounds.lower + ambient_bounds.upper) / 2
        rect_a = (
            ambient_bounds.lower.tolist()
            + ((ambient_bounds.lower + ambient_bounds.upper) / 2).tolist()
        )
        micro["analytic_single_call_s"] = _bench_single_call(g_a, pt_a)
        micro["analytic_boxmap_s"] = _bench_boxmap(g_a, rect_a)
    if "latent" in args.maps:
        g_l = _make_latent_g(autoencoder, device)
        pt_l = (latent_bounds.lower + latent_bounds.upper) / 2
        rect_l = (
            latent_bounds.lower.tolist()
            + ((latent_bounds.lower + latent_bounds.upper) / 2).tolist()
        )
        micro["latent_single_call_s"] = _bench_single_call(g_l, pt_l)
        micro["latent_boxmap_s"] = _bench_boxmap(g_l, rect_l)
    if "analytic" in args.maps and "latent" in args.maps:
        micro["latent_over_analytic_per_call"] = (
            micro["latent_single_call_s"] / micro["analytic_single_call_s"]
        )
        micro["latent_over_analytic_per_box"] = (
            micro["latent_boxmap_s"] / micro["analytic_boxmap_s"]
        )

    padding = cfg.cmgdb.padding if args.padding is None else (args.padding == "true")
    mode_label = "uniform" if args.uniform else "adaptive"

    # Sweep CMGDB.
    sweep_results: list[dict[str, Any]] = []
    for smax in args.subdiv_max:
        if args.uniform:
            smin = smax
            sinit = smax
        else:
            smin = max(1, smax - args.subdiv_gap)
            sinit = args.subdiv_init if args.subdiv_init is not None else max(1, smin - 7)
        print(f"--- [{mode_label}] subdiv_init={sinit} min={smin} max={smax} padding={padding} ---")
        row: dict[str, Any] = {
            "mode": mode_label,
            "padding": padding,
            "subdiv_init": sinit,
            "subdiv_min": smin,
            "subdiv_max": smax,
            "subdiv_limit": args.subdiv_limit,
        }
        if "analytic" in args.maps:
            print(f"  analytic ... ", end="", flush=True)
            out_a = out_dir / "morse" / "analytic" / f"smax_{smax}"
            row["analytic"] = _run_morse(
                _make_analytic_g(system),
                ambient_bounds,
                subdiv_init=sinit,
                subdiv_min=smin,
                subdiv_max=smax,
                subdiv_limit=args.subdiv_limit,
                padding=padding,
                out_dir=out_a,
            )
            print(
                f"{row['analytic']['duration_s']:.2f}s "
                f"nodes={row['analytic']['morse_nodes']} "
                f"edges={row['analytic']['morse_edges']} "
                f"min={row['analytic']['morse_minimal_nodes']} "
                f"boxes={row['analytic']['morse_total_boxes']}"
            )
        if "latent" in args.maps:
            print(f"  latent   ... ", end="", flush=True)
            out_l = out_dir / "morse" / "latent" / f"smax_{smax}"
            row["latent"] = _run_morse(
                _make_latent_g(autoencoder, device),
                latent_bounds,
                subdiv_init=sinit,
                subdiv_min=smin,
                subdiv_max=smax,
                subdiv_limit=args.subdiv_limit,
                padding=padding,
                out_dir=out_l,
            )
            print(
                f"{row['latent']['duration_s']:.2f}s "
                f"nodes={row['latent']['morse_nodes']} "
                f"edges={row['latent']['morse_edges']} "
                f"min={row['latent']['morse_minimal_nodes']} "
                f"boxes={row['latent']['morse_total_boxes']}"
            )
        sweep_results.append(row)

    summary = {
        "config": str(args.config),
        "device": str(device),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "torch_version": torch.__version__,
        "ambient_bounds": {
            "lower": ambient_bounds.lower.tolist(),
            "upper": ambient_bounds.upper.tolist(),
        },
        "latent_bounds": {
            "lower": latent_bounds.lower.tolist(),
            "upper": latent_bounds.upper.tolist(),
        },
        "microbench": micro,
        "sweep": sweep_results,
    }
    (out_dir / "profile_results.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_dir / 'profile_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
