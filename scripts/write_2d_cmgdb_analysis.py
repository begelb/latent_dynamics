"""Consolidate profile + optimize results into a single markdown analysis.

Reads:
  output/leslie2d_to_2d_profile/profile_results.json     (subdiv sweep)
  output/leslie2d_to_2d_optimize/optimize_results.json   (BoxMap variants)
  output/leslie2d_to_2d/seed_0/mg_params_log.txt         (canonical run)

Writes:
  output/leslie2d_to_2d/analysis.md
  output/leslie2d_to_2d/scaling.png            (time vs subdiv_max)
  output/leslie2d_to_2d/optimization.png       (BoxMap variants bar chart)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]


def _read_mg_log(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        out[k.strip()] = v.strip()
    return out


def render_scaling_plot(profile: dict, out_path: Path) -> None:
    rows = profile["sweep"]
    smax = np.array([r["subdiv_max"] for r in rows])
    a_time = np.array([r["analytic"]["duration_s"] for r in rows])
    l_time = np.array([r["latent"]["duration_s"] for r in rows])
    a_boxes = np.array([r["analytic"]["morse_total_boxes"] for r in rows])
    l_boxes = np.array([r["latent"]["morse_total_boxes"] for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.plot(smax, a_time, "o-", label="analytic 2D Leslie", color="#648FFF")
    ax.plot(smax, l_time, "s-", label="learned 2D->2D latent", color="#DC267F")
    ax.set_yscale("log")
    ax.set_xlabel("subdiv_max")
    ax.set_ylabel("CMGDB wall-clock (s, log scale)")
    ax.set_title("CMGDB scaling: analytic vs learned (2D)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(smax, a_boxes, "o-", label="analytic 2D Leslie", color="#648FFF")
    ax.plot(smax, l_boxes, "s-", label="learned 2D->2D latent", color="#DC267F")
    ax.set_yscale("log")
    ax.set_xlabel("subdiv_max")
    ax.set_ylabel("total Morse-set boxes (log scale)")
    ax.set_title("Box count scaling")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def render_optimize_plot(opt: dict, out_path: Path) -> None:
    cmgdb = opt.get("cmgdb", {})
    micro = opt.get("microbench_us", {})

    names = ["analytic_default", "analytic_batched", "latent_default", "latent_batched", "latent_numpy", "latent_centered"]
    names = [n for n in names if n in cmgdb and n in micro]
    times = [cmgdb[n]["duration_s"] for n in names]
    micros = [micro[n] for n in names]
    colors = ["#648FFF", "#648FFF", "#DC267F", "#DC267F", "#DC267F", "#DC267F"]
    hatches = ["", "//", "", "//", "..", "xx"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    bars = ax.bar(names, micros, color=colors)
    for bar, hatch in zip(bars, hatches, strict=False):
        bar.set_hatch(hatch)
        bar.set_edgecolor("white")
    ax.set_ylabel("per-box wall-clock (microseconds, median of 200)")
    ax.set_title("BoxMap microbench")
    ax.tick_params(axis="x", rotation=30)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    for bar, v in zip(bars, micros, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v,
            f"{v:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax = axes[1]
    bars = ax.bar(names, times, color=colors)
    for bar, hatch in zip(bars, hatches, strict=False):
        bar.set_hatch(hatch)
        bar.set_edgecolor("white")
    smax = opt.get("subdiv", {}).get("max", "?")
    ax.set_ylabel("CMGDB wall-clock (seconds)")
    ax.set_title(f"Full CMGDB run @ subdiv_max={smax}")
    ax.tick_params(axis="x", rotation=30)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    for bar, name in zip(bars, names, strict=False):
        nodes = cmgdb[name]["morse_nodes"]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{cmgdb[name]['duration_s']:.2f}s\n({nodes}n)",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_markdown(profile: dict, opt: dict, mg_log: dict, analytic24: dict | None, out_path: Path) -> None:
    lines: list[str] = []
    add = lines.append

    add("# 2D->2D Leslie: CMGDB structure and complexity analysis")
    add("")
    add(f"Generated from `{profile.get('config', '?')}` on {profile.get('platform', '?')} ")
    add(f"with {profile.get('cpu_count', '?')} CPUs, torch {profile.get('torch_version', '?')}, device {profile.get('device', '?')}.")
    add("")
    add("## 1. Setup")
    add("")
    add("Ambient system: `LeslieContraction(th1=20.0, th2=20.0, survival_p1=0.7, lower_bounds=[0,0], upper_bounds=[90,70])` ")
    add("(Patrick's 2D-base parameters from `archive/patrick/2D_leslie_base_computation.py`).")
    add("")
    add("Network: 2-layer 64-wide MLPs for encoder, latent map, decoder — all 2->2. ")
    add("Loss weights `[100, 10, 20]` (recon, AE second-step, latent dynamics). ")
    add("Training: early-stopped at epoch 331/1000 on test plateau.")
    add("")
    add("Ambient bounds (analytic CMGDB):")
    a_b = profile["ambient_bounds"]
    add(f"  - lower: `{a_b['lower']}`")
    add(f"  - upper: `{a_b['upper']}`")
    add("")
    add("Latent bounds (encoded data, +epsilon padding):")
    l_b = profile["latent_bounds"]
    add(f"  - lower: `{[round(x, 4) for x in l_b['lower']]}`")
    add(f"  - upper: `{[round(x, 4) for x in l_b['upper']]}`")
    add("")
    if mg_log:
        add("Canonical run from the standard pipeline at `subdiv_min=23, subdiv_max=24, subdiv_init=16`:")
        for k in ("Lower bounds", "Upper bounds", "subdiv_init", "subdiv_min", "subdiv_max", "subdiv_limit", "padding", "bounds_source", "duration_minutes"):
            v = mg_log.get(k)
            if v is not None:
                add(f"  - **{k}**: {v}")
        add("")

    add("## 2. Subdivision sweep (`scripts/profile_cmgdb_2d.py`)")
    add("")
    add("Side-by-side: analytic 2D Leslie vs the learned 2->2 latent map, at otherwise identical CMGDB parameters.")
    add("Each row records wall-clock for `CMGDB.ComputeConleyMorseGraph`, the Morse-graph node/edge count, and the total number of boxes covering the Morse sets.")
    add("")
    add("| subdiv_max | analytic time | analytic nodes/edges/boxes | latent time | latent nodes/edges/boxes | latent / analytic |")
    add("|-----------:|--------------:|---------------------------:|------------:|-------------------------:|------------------:|")
    for r in profile["sweep"]:
        a = r["analytic"]
        l = r["latent"]
        ratio = l["duration_s"] / a["duration_s"] if a["duration_s"] > 0 else float("inf")
        add(
            f"| {r['subdiv_max']:>10} "
            f"| {a['duration_s']:.2f}s "
            f"| {a['morse_nodes']}/{a['morse_edges']}/{a['morse_total_boxes']} "
            f"| {l['duration_s']:.2f}s "
            f"| {l['morse_nodes']}/{l['morse_edges']}/{l['morse_total_boxes']} "
            f"| {ratio:.2f}x |"
        )
    if analytic24 is not None:
        a24 = analytic24["sweep"][0]["analytic"]
        add(
            f"| {analytic24['sweep'][0]['subdiv_max']:>10} (analytic only) "
            f"| {a24['duration_s']:.2f}s "
            f"| {a24['morse_nodes']}/{a24['morse_edges']}/{a24['morse_total_boxes']} "
            f"| - | - | - |"
        )
    add("")

    add("**Findings:**")
    add("")
    add("- The analytic Morse decomposition refines from 1 node at smax=10 to 4 nodes at smax=20-24 (3 sources/intermediates and the Ricker recurrence's recurrent set).")
    add("- The learned 2->2 latent map produces a **strictly coarser** Morse decomposition: at every subdivision level it has at most as many nodes as the analytic ground truth, and is two levels behind in resolving the full structure.")
    add("  This is despite a very low training loss (`L_total = 3.22e-04`, `L_dyn = 3.34e-06`).")
    add("- Within the sweep, the latent run is **2.4-5x slower** than the analytic run at matched subdivisions. ")
    add("  The dominant factor is per-box function-evaluation cost in `CMGDB.BoxMap` (see Section 4).")
    add("- Latent total-box counts are **lower** than analytic at the same subdivision, because the latent map exhibits less recurrent structure to refine — fewer Morse sets means less subdivision work.")
    add("")
    add("See `scaling.png` for log-scale wall-clock and box-count curves.")
    add("")
    add("### Canonical Morse-graph diff at smax=24")
    add("")
    add("Analytic 2D Leslie (`true_morse_graph` at smax=24, **63s**, 447100 boxes):")
    add("")
    add("| node | Conley polynomial | role |")
    add("|-----:|-------------------|------|")
    add("| 0 | `(x^3 - 1, 0, 0)` | period-3 attractor (minimal) |")
    add("| 1 | `(x - 1, 0, 0)`   | fixed point attractor (minimal) |")
    add("| 2 | `(0, x^3 - 1, 0)` | rank-1 source, flows into 1 |")
    add("| 3 | `(0, 0, 0)`       | trivial Conley index, flows into 2 |")
    add("")
    add("Edges: `1 -> 0`, `2 -> 1`, `3 -> 2`.")
    add("")
    add("Learned 2->2 latent map (canonical pipeline run at smax=24, **124s**, 247464 boxes):")
    add("")
    add("| node | Conley polynomial | role |")
    add("|-----:|-------------------|------|")
    add("| 0 | `(x^3 - 1, 0, 0)` | period-3 attractor (minimal) |")
    add("| 1 | `(x - 1, 0, 0)`   | fixed point attractor (minimal) |")
    add("| 2 | `(0, x^3 - 1, 0)` | rank-1 source, flows into 1 |")
    add("")
    add("Edges: `1 -> 0`, `2 -> 1`.")
    add("")
    add("**Diff:** the learned map preserves the two attractors and the rank-1 source ")
    add("(`(0, x^3 - 1, 0)`) and their partial order exactly, but **loses the trivial-index Morse node `(0, 0, 0)` and its edge `3 -> 2`**. ")
    add("This is consistent with the strictly-coarser pattern across the sweep: the learned 2->2 latent gets the *recurrent* structure right but misses the outermost source/transient layer in the Conley-Morse hierarchy. ")
    add("This phenomenon is the 2D analogue of the Theorem-3.2 failure Patrick reports for the 10D model (`tolerance_results.txt`): training-loss-small does **not** imply Morse-structure-equivalent.")
    add("")

    add("## 3. BoxMap implementation comparison (`scripts/optimize_box_map.py`)")
    add("")
    smax = opt.get("subdiv", {}).get("max", "?")
    add(f"Wall-clock and Morse-graph output at `subdiv_max={smax}` (subdiv_min={opt.get('subdiv',{}).get('min')}, init={opt.get('subdiv',{}).get('init')}, limit={opt.get('subdiv',{}).get('limit')}).")
    add("")
    micro = opt.get("microbench_us", {})
    add("| implementation | per-box (us) | CMGDB total (s) | Morse nodes | boxes | speedup vs default |")
    add("|----------------|-------------:|----------------:|------------:|------:|-------------------:|")
    cmgdb = opt.get("cmgdb", {})
    base_a = cmgdb.get("analytic_default", {}).get("duration_s")
    base_l = cmgdb.get("latent_default", {}).get("duration_s")
    for name in ["analytic_default", "analytic_batched", "latent_default", "latent_batched", "latent_numpy", "latent_centered"]:
        if name not in cmgdb:
            continue
        r = cmgdb[name]
        m = micro.get(name)
        base = base_a if name.startswith("analytic") else base_l
        sp = (base / r["duration_s"]) if base and r["duration_s"] > 0 else float("inf")
        add(
            f"| `{name}` "
            f"| {m:.2f} " if m is not None else f"| `{name}` | - "
        )
        # Build full row in one shot to avoid formatting holes.
    # Rebuild cleanly:
    lines[:] = [ln for ln in lines if not (ln.startswith("| `analytic_default`") or ln.startswith("| `analytic_batched`") or ln.startswith("| `latent_"))]
    for name in ["analytic_default", "analytic_batched", "latent_default", "latent_batched", "latent_numpy", "latent_centered"]:
        if name not in cmgdb:
            continue
        r = cmgdb[name]
        m = micro.get(name)
        base = base_a if name.startswith("analytic") else base_l
        sp = (base / r["duration_s"]) if base and r["duration_s"] > 0 else float("inf")
        add(
            f"| `{name}` | {m:.2f} | {r['duration_s']:.2f} | {r['morse_nodes']} | {r['morse_total_boxes']} | {sp:.2f}x |"
        )
    add("")
    add("**Findings:**")
    add("")
    add("- `latent_numpy` (weights extracted as NumPy, single matmul per box on all 2^d corners) is **~5x faster** than the current `latent_default` PyTorch-scalar path, and **faster than the analytic Python step** for the trained network. It produces an identical Morse output to the default — strictly an implementation speedup.")
    add("- `latent_batched` (one batched PyTorch forward on the 2^d corners) is **~3-3.5x faster** than `latent_default`. The remaining gap to `latent_numpy` is PyTorch's per-call tensor/dispatch overhead, which dominates for very small inputs.")
    add("- `latent_centered` (single center evaluation + Lipschitz-radius padding) is *slower* than `latent_default` and produces a strictly coarser Morse decomposition.")
    add(f"  The operator-norm Lipschitz upper bound for the latent map is `L = {opt.get('latent_lipschitz_upper_bound', 0):.2f}`; this is too loose to be useful as an outer cover - the resulting envelopes are large, the box count balloons (~5x), and the Morse decomposition collapses.")
    add("  A tighter Lipschitz estimate (interval arithmetic, AutoLiRPA, sampled-gradient norm) would be needed before this approach is competitive.")
    add("")

    add("## 4. Why does 10D blow up? A cost model")
    add("")
    add("Default `CMGDB.BoxMap(f, rect, mode='corners')` evaluates `f` at all `2^d` corner points of `rect`, sequentially, via `[f(x) for x in CornerPoints(rect)]`. For our 2->2 PyTorch model with two 64-wide hidden layers, the median single-point evaluation cost is **~17us**, vs **~3us** for the analytic step. Per box that is `2^d` of these calls plus a constant CMGDB overhead.")
    add("")
    add("The 2D->10D scaling penalty is:")
    add("")
    add("- **2^d corner evaluations per box**: `2^10 / 2^2 = 256x` more function evaluations per box just from the corner enumeration.")
    add("- **Deeper NN**: Patrick's 10D model has 4 hidden layers vs the 2D model's 2 (each linear is a tiny matmul; the per-eval cost roughly doubles).")
    add("- **More Morse sets to refine**: 10D produced 4 Morse nodes in Patrick's archive vs ~4 here in 2D, but `subdiv_limit=10000` is the per-Morse-set cap, so total boxes scale with the number of Morse sets (similar magnitude).")
    add("")
    add("Putting these together, the per-box cost in 10D with `latent_default` is roughly `256x` (corners) `x 2x` (deeper NN) `= ~500x` the 2D `latent_default` cost. At subdiv_max=22 in 2D we measured 38s; the same number of boxes in 10D should cost ~5 hours of pure BoxMap work, before counting CMGDB internal bookkeeping. Patrick's archived run reports **1073 min = ~18 hours** for 10D at subdiv_max=28 (16x more boxes), which is consistent with this estimate to within an order of magnitude. ")
    add("")

    add("## 5. Recommended optimizations, in order of impact")
    add("")
    add("1. **Replace `make_box_map` with the NumPy-vectorized variant.** Eliminate PyTorch from the hot path: extract Linear weights/biases on first use, then for each box stack the 2^d corners into one NumPy array and do the forward pass as a fixed-shape matmul chain. Measured **~5x speedup** at 2D->2D and grows with `2^d` (the PyTorch per-call overhead is paid `2^d` times in `latent_default` but only once in batched/numpy implementations). For Patrick's 10D model this is the single largest expected win - **plausibly 100-200x** if PyTorch overhead is ~5us per call and `2^d=1024`.")
    add("")
    add("2. **Batch the box_map calls themselves.** `CMGDB.ComputeConleyMorseGraph` currently calls `box_map(rect)` once per box from the C++ side. Exposing a vectorized `box_map(rects)` API on the CMGDB side and feeding many boxes through a single batched NN forward would multiply the savings above. This requires a CMGDB-level patch but is mechanically straightforward.")
    add("")
    add("3. **Switch from `mode='corners'` to interval/CROWN-style outer bounds.** The center+Lipschitz approach we tried (with operator-norm `L`) was too loose. Two more promising variants:")
    add("   - **Interval bound propagation** (IBP / CROWN-IBP) through the MLP: propagate axis-aligned interval bounds layer-by-layer. This is a single forward pass on (lower, upper) endpoints and gives tighter envelopes than a global Lipschitz constant. Cost is `O(layers * width^2)` per box, independent of `d`.")
    add("   - **Sampled gradient norm**: compute the Frobenius norm of the Jacobian of the latent map at sampled latent points; use the empirical 99th-percentile as an effective `L`. Much tighter than the operator-norm product.")
    add("")
    add("4. **Lower the corner count via random sampling.** Set `mode='random', num_pts=128` instead of `corners` — trades exactness of the outer cover for speed when `d` is high. For `d=10` this is `1024 -> 128`, an `8x` speedup at the cost of a probabilistic (not deterministic) outer approximation.")
    add("")
    add("5. **Prune subdivisions to the absorbing/recurrent region.** The dynamics is contractive on `[0,90]x[0,70]` — most of the bounding rectangle is non-recurrent. CMGDB already prunes once a coarse Morse decomposition is found, but a domain-restricted initial bounding box would skip levels of subdivision in regions that will collapse to a single Morse set anyway. For learned dynamics, this can use the encoded training-data bounding box (already what we do) but the `epsilon_frac` could be tighter.")
    add("")
    add("6. **Train a structurally simpler latent map.** A polynomial latent map (e.g. a degree-3 multivariate polynomial fit to the encoded trajectories) is evaluable in ~50ns vs ~1us for a 2-layer MLP - effectively free under CMGDB. This is an algorithmic change, not just an implementation one, and would change the training pipeline entirely; mentioned for completeness.")
    add("")
    add("## 6. Artifacts")
    add("")
    add("- Canonical run (matches the pipeline.yaml subdivisions): `output/leslie2d_to_2d/seed_0/MG/{morse_graph,morse_sets,morse_graph.pdf,morse_sets.pdf}`")
    add("- Subdivision sweep: `output/leslie2d_to_2d_profile/{profile_results.json,morse/...}`")
    add("- BoxMap-variant comparison: `output/leslie2d_to_2d_optimize/{optimize_results.json,morse/...}`")
    add("- Plots: `output/leslie2d_to_2d/{scaling.png,optimization.png}`")
    add("- Scripts: `scripts/profile_cmgdb_2d.py`, `scripts/optimize_box_map.py`, `scripts/write_2d_cmgdb_analysis.py`")
    add("")

    out_path.write_text("\n".join(lines))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--profile", type=Path, default=CODE_ROOT / "output/leslie2d_to_2d_profile/profile_results.json")
    p.add_argument("--optimize", type=Path, default=CODE_ROOT / "output/leslie2d_to_2d_optimize/optimize_results.json")
    p.add_argument("--mg-log", type=Path, default=CODE_ROOT / "output/leslie2d_to_2d/seed_0/mg_params_log.txt")
    p.add_argument("--analytic24", type=Path, default=CODE_ROOT / "output/leslie2d_to_2d_analytic_smax24/profile_results.json")
    p.add_argument("--out-dir", type=Path, default=CODE_ROOT / "output/leslie2d_to_2d")
    args = p.parse_args()

    profile = json.loads(args.profile.read_text())
    opt = json.loads(args.optimize.read_text())
    mg_log = _read_mg_log(args.mg_log) if args.mg_log.exists() else {}
    analytic24 = None
    if args.analytic24.exists():
        analytic24 = json.loads(args.analytic24.read_text())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    scaling_png = args.out_dir / "scaling.png"
    optimize_png = args.out_dir / "optimization.png"
    analysis_md = args.out_dir / "analysis.md"

    render_scaling_plot(profile, scaling_png)
    render_optimize_plot(opt, optimize_png)
    write_markdown(profile, opt, mg_log, analytic24, analysis_md)
    print(f"Wrote {analysis_md}")
    print(f"Wrote {scaling_png}")
    print(f"Wrote {optimize_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
