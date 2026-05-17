"""Compare BoxMap implementations on a trained 2D->2D latent map.

Default ``CMGDB.BoxMap`` evaluates the dynamics at all 2^d corner points via a
Python list comprehension (sequential, scalar). For a neural-network ``f``, each
of those 2^d calls pays full per-tensor PyTorch overhead. We compare:

  ``latent_default``  scalar PyTorch (the current ``make_box_map`` path)
  ``latent_batched``  single batched torch.no_grad forward on all 2^d corners
  ``latent_numpy``    weights -> NumPy; one numpy matmul per box
  ``latent_centered`` one forward at center + Lipschitz-bound padding (1 eval)

For comparison we also include analytic flavors:

  ``analytic_default``  per-corner via Patrick's pattern
  ``analytic_batched``  vectorized ``system.step`` over all corners

Two kinds of measurement:
  1. Microbench: median per-call wall-clock on a single representative rect.
  2. Full CMGDB run at one ``subdiv_max`` — wall-clock and Morse node count, so
     we can confirm the optimizations preserve the Morse decomposition.

Outputs:
  output/leslie2d_to_2d_optimize/optimize_results.json

Example::

    python scripts/optimize_box_map.py \\
        --config configs/leslie2d_to_2d.yaml \\
        --subdiv-max 14 --device cpu
"""

from __future__ import annotations

import argparse
import itertools
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


# ---------- analytic variants ----------


def make_analytic_default(system):
    def g(x):
        return list(system.step(np.asarray(x, dtype=np.float64)))

    def box_map(rect):
        return CMGDB.BoxMap(g, rect, padding=True)

    return box_map


def make_analytic_batched(system):
    def box_map(rect):
        dim = len(rect) // 2
        list_intvals = [[rect[d], rect[d + dim]] for d in range(dim)]
        X = np.array(list(itertools.product(*list_intvals)), dtype=np.float64)
        Y = system.step(X)  # vectorised over leading axis (see DiscreteMap)
        pad = np.array([rect[d + dim] - rect[d] for d in range(dim)])
        Y_l = Y.min(axis=0) - pad
        Y_u = Y.max(axis=0) + pad
        return Y_l.tolist() + Y_u.tolist()

    return box_map


# ---------- latent variants ----------


def make_latent_default(autoencoder, device):
    latent_map = autoencoder.latent_map
    latent_map.eval()

    @torch.no_grad()
    def g(x):
        x_t = torch.as_tensor(x, dtype=torch.float32, device=device).reshape(1, -1)
        return latent_map(x_t)[0].cpu().numpy().tolist()

    def box_map(rect):
        return CMGDB.BoxMap(g, rect, padding=True)

    return box_map


def make_latent_batched(autoencoder, device):
    latent_map = autoencoder.latent_map
    latent_map.eval()

    def box_map(rect):
        dim = len(rect) // 2
        list_intvals = [[rect[d], rect[d + dim]] for d in range(dim)]
        X = np.array(list(itertools.product(*list_intvals)), dtype=np.float32)
        with torch.no_grad():
            X_t = torch.as_tensor(X, dtype=torch.float32, device=device)
            Y_t = latent_map(X_t)
            Y = Y_t.cpu().numpy()
        pad = np.array([rect[d + dim] - rect[d] for d in range(dim)])
        Y_l = Y.min(axis=0) - pad
        Y_u = Y.max(axis=0) + pad
        return Y_l.tolist() + Y_u.tolist()

    return box_map


def _extract_numpy_mlp(latent_map: torch.nn.Module):
    """Pull (W, b, activation) for each Linear in the latent MLP, as NumPy."""
    layers = []
    seq = latent_map.net  # nn.Sequential built by _build_mlp
    children = list(seq.children())
    i = 0
    while i < len(children):
        layer = children[i]
        if isinstance(layer, torch.nn.Linear):
            W = layer.weight.detach().cpu().numpy().astype(np.float64)
            b = layer.bias.detach().cpu().numpy().astype(np.float64)
            act = None
            if i + 1 < len(children):
                nxt = children[i + 1]
                if isinstance(nxt, torch.nn.ReLU):
                    act = "relu"
                elif isinstance(nxt, torch.nn.Tanh):
                    act = "tanh"
                elif isinstance(nxt, torch.nn.Sigmoid):
                    act = "sigmoid"
                elif isinstance(nxt, torch.nn.GELU):
                    act = "gelu"
                i += 1
            layers.append((W, b, act))
        i += 1
    return layers


def make_latent_numpy(autoencoder):
    layers = _extract_numpy_mlp(autoencoder.latent_map)

    def forward(X: np.ndarray) -> np.ndarray:
        # X: (n, dim_in) -> (n, dim_out)
        Y = X
        for W, b, act in layers:
            Y = Y @ W.T + b
            if act == "relu":
                Y = np.maximum(Y, 0.0)
            elif act == "tanh":
                Y = np.tanh(Y)
            elif act == "sigmoid":
                Y = 1.0 / (1.0 + np.exp(-Y))
            elif act == "gelu":
                Y = 0.5 * Y * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (Y + 0.044715 * Y**3)))
            # else: linear
        return Y

    def box_map(rect):
        dim = len(rect) // 2
        list_intvals = [[rect[d], rect[d + dim]] for d in range(dim)]
        X = np.array(list(itertools.product(*list_intvals)), dtype=np.float64)
        Y = forward(X)
        pad = np.array([rect[d + dim] - rect[d] for d in range(dim)])
        Y_l = Y.min(axis=0) - pad
        Y_u = Y.max(axis=0) + pad
        return Y_l.tolist() + Y_u.tolist()

    return box_map, forward


def lipschitz_upper_bound(latent_map: torch.nn.Module) -> float:
    """Conservative Lipschitz upper bound: product of operator norms of Linear
    weights times slope of intermediate ReLU/Tanh/Sigmoid (each <= 1)."""
    layers = _extract_numpy_mlp(latent_map)
    L = 1.0
    for W, _b, _act in layers:
        sv = np.linalg.svd(W, compute_uv=False)
        L *= float(sv[0])  # operator (spectral) norm
    return L


def make_latent_centered(autoencoder, device, L: float):
    """Center evaluation + Lipschitz padding -> 1 forward per box.

    Returns a rectangle that *outer-approximates* the image. May be looser
    than the 2^d corner cover, so the resulting Morse graph is a coarsening.
    """
    latent_map = autoencoder.latent_map
    latent_map.eval()

    @torch.no_grad()
    def g(x):
        x_t = torch.as_tensor(x, dtype=torch.float32, device=device).reshape(1, -1)
        return latent_map(x_t)[0].cpu().numpy()

    def box_map(rect):
        dim = len(rect) // 2
        rect_arr = np.asarray(rect, dtype=np.float64)
        center = (rect_arr[:dim] + rect_arr[dim:]) / 2.0
        half = (rect_arr[dim:] - rect_arr[:dim]) / 2.0
        radius = float(np.linalg.norm(half))  # Euclidean radius
        y = g(center)
        # Lipschitz envelope (L is global L2 Lipschitz constant of latent map).
        delta = L * radius
        size = rect_arr[dim:] - rect_arr[:dim]
        Y_l = y - delta - size
        Y_u = y + delta + size
        return Y_l.tolist() + Y_u.tolist()

    return box_map


# ---------- benchmark utilities ----------


def bench_box_map(box_map: Callable, rect: list[float], n: int = 200, warmup: int = 5) -> float:
    for _ in range(warmup):
        box_map(rect)
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        box_map(rect)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def run_cmgdb(box_map, bounds: LatentBounds, *, subdiv_init, subdiv_min, subdiv_max, subdiv_limit, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
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
    text = dot_path.read_text()
    n_nodes = sum(1 for line in text.splitlines() if "[label=" in line and "shape=" in line)
    n_edges = sum(1 for line in text.splitlines() if "->" in line)
    n_boxes = 0
    with csv_path.open() as f:
        for line in f:
            if line.strip() and line.strip()[-1].isdigit():
                n_boxes += 1
    return {
        "duration_s": duration_s,
        "morse_nodes": n_nodes,
        "morse_edges": n_edges,
        "morse_total_boxes": n_boxes,
    }


def _load_latent_bounds(cfg, train_file: str, autoencoder, device: torch.device) -> LatentBounds:
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
    return infer_latent_bounds(
        autoencoder.encoder,
        np.vstack(pieces),
        epsilon_frac=cfg.cmgdb.bounds_epsilon_frac,
        device=device,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--train-file", default="train")
    p.add_argument("--subdiv-max", type=int, default=14)
    p.add_argument("--subdiv-gap", type=int, default=1)
    p.add_argument("--subdiv-init", type=int, default=None)
    p.add_argument("--subdiv-limit", type=int, default=10000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--skip-cmgdb", action="store_true", help="microbench only; skip full CMGDB runs")
    args = p.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)

    out_dir = args.out_dir or (CODE_ROOT / "output" / f"{args.config.stem}_optimize")
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_dir = cfg.paths.output_dir / "seed_0"
    autoencoder, _arch = load_checkpoint(seed_dir / "models")
    autoencoder.to(device)
    latent_bounds = _load_latent_bounds(cfg, args.train_file, autoencoder, device)
    system = build_system(cfg.system.name, cfg.system.params)
    ambient_bounds = LatentBounds(
        lower=np.asarray(system.lower_bounds, dtype=np.float64),
        upper=np.asarray(system.upper_bounds, dtype=np.float64),
    )

    # Pick a representative rect for microbench.
    rect_a = (
        ambient_bounds.lower.tolist()
        + ((ambient_bounds.lower + ambient_bounds.upper) / 2).tolist()
    )
    rect_l = (
        latent_bounds.lower.tolist()
        + ((latent_bounds.lower + latent_bounds.upper) / 2).tolist()
    )

    # Build all variants.
    box_maps_analytic = {
        "analytic_default": make_analytic_default(system),
        "analytic_batched": make_analytic_batched(system),
    }
    box_maps_latent: dict[str, Any] = {
        "latent_default": make_latent_default(autoencoder, device),
        "latent_batched": make_latent_batched(autoencoder, device),
    }
    latent_numpy_box_map, _ = make_latent_numpy(autoencoder)
    box_maps_latent["latent_numpy"] = latent_numpy_box_map
    L = lipschitz_upper_bound(autoencoder.latent_map)
    box_maps_latent["latent_centered"] = make_latent_centered(autoencoder, device, L)

    print(f"Lipschitz upper bound for latent map: L = {L:.4f}")
    print(f"ambient rect: dim={len(rect_a)//2} latent rect: dim={len(rect_l)//2}")

    # Microbench all variants.
    print("\n=== microbench per-box wall-clock (median of 200) ===")
    micro: dict[str, float] = {}
    for name, bm in box_maps_analytic.items():
        t = bench_box_map(bm, rect_a)
        micro[name] = t
        print(f"  {name:24s}  {t*1e6:8.2f} us")
    for name, bm in box_maps_latent.items():
        t = bench_box_map(bm, rect_l)
        micro[name] = t
        print(f"  {name:24s}  {t*1e6:8.2f} us")

    cmgdb_results: dict[str, Any] = {}
    if not args.skip_cmgdb:
        smax = args.subdiv_max
        smin = max(1, smax - args.subdiv_gap)
        sinit = args.subdiv_init if args.subdiv_init is not None else max(1, smin - 7)
        print(f"\n=== CMGDB runs at subdiv_init={sinit} min={smin} max={smax} ===")

        def _do(name, bm, bounds):
            o = out_dir / "morse" / name
            r = run_cmgdb(
                bm,
                bounds,
                subdiv_init=sinit,
                subdiv_min=smin,
                subdiv_max=smax,
                subdiv_limit=args.subdiv_limit,
                out_dir=o,
            )
            print(
                f"  {name:24s}  {r['duration_s']:7.2f}s  "
                f"nodes={r['morse_nodes']}  edges={r['morse_edges']}  "
                f"boxes={r['morse_total_boxes']}"
            )
            cmgdb_results[name] = r

        for name, bm in box_maps_analytic.items():
            _do(name, bm, ambient_bounds)
        for name, bm in box_maps_latent.items():
            _do(name, bm, latent_bounds)

    summary = {
        "config": str(args.config),
        "device": str(device),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "torch_version": torch.__version__,
        "latent_lipschitz_upper_bound": L,
        "ambient_bounds": {
            "lower": ambient_bounds.lower.tolist(),
            "upper": ambient_bounds.upper.tolist(),
        },
        "latent_bounds": {
            "lower": latent_bounds.lower.tolist(),
            "upper": latent_bounds.upper.tolist(),
        },
        "microbench_us": {k: v * 1e6 for k, v in micro.items()},
        "cmgdb": cmgdb_results,
        "subdiv": {
            "init": args.subdiv_init if args.subdiv_init is not None else max(1, args.subdiv_max - args.subdiv_gap - 7),
            "min": max(1, args.subdiv_max - args.subdiv_gap),
            "max": args.subdiv_max,
            "limit": args.subdiv_limit,
        },
    }
    (out_dir / "optimize_results.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_dir / 'optimize_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
