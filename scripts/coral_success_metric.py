"""Coral success-rate scaling figure with standard-deviation error bars.

Reproduces the §5.4 data-scaling / adaptive-sampling success-rate figure
from saved per-seed Morse artifacts, adding the standard-deviation error
bars noted as a TODO in the paper.

Two modes
---------
adaptive (default)
    x-axis = M, number of adaptive samples (linear scale).
    Datasets: train_500_{M}_adaptive for M in {100,200,300,400,500}.
size
    x-axis = N, number of initial conditions (log scale).
    Datasets: train_{N} for N in {100,200,500,1000,2000,5000}.

Per (dataset, seed) the three binary indicators (a0, a1, r) are computed
by reusing ``latentdynamics.analysis.morse_metrics.check_unique_membership``.
Seeds whose Morse artifacts are absent or 0-byte are silently skipped; the
surviving count n is annotated on each tick with n < 30.

The three series (a0, a1, r) are dodged apart in x and drawn as markers with
error whiskers (no connecting lines). The whisker type is selectable via
``--error``: ``wilson`` (95% binomial confidence interval; the default and the
appropriate uncertainty for a success *probability*), ``se`` (standard error of
the mean), or ``std`` (sample standard deviation of the 0/1 indicators). All
three (mean, std, se) are saved to the summary CSV regardless.

Usage
-----
python code/scripts/coral_success_metric.py --mode adaptive
python code/scripts/coral_success_metric.py --mode size --error se
# Fast plot iteration from already-computed per-seed indicators:
python code/scripts/coral_success_metric.py --mode adaptive \
    --replot-from scratch/coral_success_stdev/coral_success_adaptive_perseed.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Iterator

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Package imports
# ---------------------------------------------------------------------------
from latentdynamics.analysis.morse_metrics import check_unique_membership
from latentdynamics.replay import load_experiment
from latentdynamics.viz import save_figure

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
POINTS = ("a0", "a1", "r")
COLORS = {"a0": "#648FFF", "a1": "#DC267F", "r": "#FFB000"}
MARKERS = {"a0": "o", "a1": "s", "r": "^"}
LABELS = {"a0": r"$a_0$", "a1": r"$a_1$", "r": r"$r$"}
# Alternative legend matching the paper body notation: each series is the
# empirical success probability of that fixed point's indicator.
LABELS_PHAT = {
    "a0": r"$\hat{p}(1_{a_0})$",
    "a1": r"$\hat{p}(1_{a_1})$",
    "r": r"$\hat{p}(1_{r})$",
}

# All 30 seeds declared in both coral configs.
ALL_SEEDS: list[int] = list(range(30))

SIZE_DATASETS: list[tuple[int, str]] = [
    (100, "train_100"),
    (200, "train_200"),
    (500, "train_500"),
    (1000, "train_1000"),
    (2000, "train_2000"),
    (5000, "train_5000"),
]

ADAPTIVE_DATASETS: list[tuple[int, str]] = [
    (100, "train_500_100_adaptive"),
    (200, "train_500_200_adaptive"),
    (300, "train_500_300_adaptive"),
    (400, "train_500_400_adaptive"),
    (500, "train_500_500_adaptive"),
]


# ---------------------------------------------------------------------------
# Per-seed indicator collection
# ---------------------------------------------------------------------------

def _morse_artifacts_present(morse_dir: Path) -> bool:
    """Return True iff both morse_graph and morse_sets exist and are non-empty."""
    for name in ("morse_graph", "morse_sets"):
        p = morse_dir / name
        if not p.is_file() or p.stat().st_size == 0:
            return False
    return True


def collect_seed_indicators(
    config_name: str,
    dataset_name: str,
    seeds: list[int],
) -> Iterator[dict]:
    """Yield one dict per usable seed with keys dataset, x_val, seed, a0, a1, r.

    Seeds are skipped (with a printed warning) when:
    - load_experiment raises,
    - the Morse artifacts are absent or 0-byte,
    - exp.scaler is None (metric requires a scaler; no fallback).

    Parameters
    ----------
    config_name:
        Packaged config name, e.g. ``"coral_adaptive"`` or ``"coral_data_scaling"``.
    dataset_name:
        train_file argument for load_experiment, e.g. ``"train_500_100_adaptive"``.
    seeds:
        Seed integers to try.
    """
    for seed in seeds:
        try:
            exp = load_experiment(config_name, train_file=dataset_name, seed=seed)
        except Exception as exc:
            print(
                f"  SKIP {dataset_name}/seed_{seed}: load_experiment raised: {exc}",
                file=sys.stderr,
            )
            continue

        if not _morse_artifacts_present(exp.morse_dir):
            # 0-byte or absent — expected for cluster stubs
            continue

        if exp.scaler is None:
            print(
                f"  SKIP {dataset_name}/seed_{seed}: scaler is None (metric unavailable)",
                file=sys.stderr,
            )
            continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, metrics = check_unique_membership(
                    exp.model.encoder,
                    exp.scaler,
                    exp.morse_dir / "morse_sets",
                    exp.morse_dir / "morse_graph",
                )
        except Exception as exc:
            print(
                f"  SKIP {dataset_name}/seed_{seed}: check_unique_membership raised: {exc}",
                file=sys.stderr,
            )
            continue

        yield {
            "dataset": dataset_name,
            "seed": seed,
            "a0": int(metrics["a0"]),
            "a1": int(metrics["a1"]),
            "r": int(metrics["r"]),
        }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_perseed(
    perseed_df: pd.DataFrame,
    x_map: dict[str, int],
) -> pd.DataFrame:
    """Compute per-(x_val, point) mean / std / SE / n from the per-seed frame.

    Parameters
    ----------
    perseed_df:
        DataFrame with columns dataset, seed, a0, a1, r.
    x_map:
        Maps dataset name to integer x-axis value.

    Returns
    -------
    DataFrame with columns: x, point, n, mean, std, se.
    """
    rows = []
    for dataset, grp in perseed_df.groupby("dataset", sort=False):
        x = x_map[dataset]
        for pt in POINTS:
            arr = grp[pt].to_numpy(dtype=float)
            n = len(arr)
            mean = float(np.mean(arr))
            std = float(np.std(arr, ddof=1)) if n >= 2 else float("nan")
            se = std / np.sqrt(n) if n >= 2 else float("nan")
            rows.append({"x": x, "point": pt, "n": n, "mean": mean, "std": std, "se": se})
    df = pd.DataFrame(rows)
    df.sort_values(["point", "x"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_Z95 = 1.959963984540054  # standard normal 0.975 quantile


def _wilson_interval(mean: float, n: int) -> tuple[float, float]:
    """95% Wilson score interval (lower, upper) for a binomial proportion.

    Appropriate for a success *probability* estimated from ``n`` Bernoulli
    trials: it is asymmetric and always lies within [0, 1], so it never
    pokes past 0 or 1 the way a symmetric std/SE bar does.
    """
    if n <= 0:
        return mean, mean
    k = round(mean * n)
    p = k / n
    z2 = _Z95 * _Z95
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (_Z95 / denom) * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return max(0.0, center - half), min(1.0, center + half)


def error_bounds(
    mean: float, n: int, std: float, se: float, error: str
) -> tuple[float, float]:
    """Return (lower_err, upper_err) for matplotlib ``yerr``, clamped to [0, 1].

    ``error`` is one of ``"wilson"`` (asymmetric binomial CI; recommended for
    a success probability), ``"se"`` (standard error of the mean), or ``"std"``
    (sample standard deviation of the 0/1 indicators).
    """
    if error == "wilson":
        lo, hi = _wilson_interval(mean, n)
        # At p_hat = 0 or 1 the point estimate can sit just outside its own
        # Wilson interval; clamp the whisker on that side to length 0.
        return max(0.0, mean - lo), max(0.0, hi - mean)
    spread = se if error == "se" else std
    if np.isnan(spread):
        return 0.0, 0.0
    lower = mean - max(0.0, mean - spread)
    upper = min(1.0, mean + spread) - mean
    return max(0.0, lower), max(0.0, upper)


def _dodged_x(x: int, point: str, mode: str) -> float:
    """Nudge the three series apart in x so their markers/bars don't overlap.

    Additive offset on a linear axis (adaptive), multiplicative on a log axis
    (size). ``a0`` shifts left, ``a1`` stays, ``r`` shifts right.
    """
    step = {"a0": -1, "a1": 0, "r": 1}[point]
    if mode == "size":
        return x * (1.045 ** step)  # log axis -> geometric dodge
    return x + step * 9.0  # linear axis -> ~9-unit dodge (gaps are 100)


def build_figure(
    summary_df: pd.DataFrame,
    *,
    mode: str,
    x_values_all: list[int],
    error: str = "wilson",
    legend: str = "points",
) -> plt.Figure:
    """Construct the success-rate figure: x-dodged markers with error whiskers.

    Parameters
    ----------
    summary_df:
        Output of :func:`aggregate_perseed`, columns: x, point, n, mean, std, se.
    mode:
        ``"adaptive"`` or ``"size"``.
    x_values_all:
        All candidate x values in the mode (for setting ticks even if some
        are missing from summary_df).
    error:
        Whisker type: ``"wilson"`` (95% binomial CI; default), ``"se"``, or ``"std"``.
    """
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral"],
            "mathtext.fontset": "stix",
            "axes.labelsize": 28,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 26,
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    labels = LABELS_PHAT if legend == "phat" else LABELS

    # n annotation: how many seeds each x has (same for all points at that x)
    n_at_x: dict[int, int] = {}
    for _, row in summary_df.iterrows():
        n_at_x[int(row["x"])] = int(row["n"])

    for pt in POINTS:
        pt_df = summary_df[summary_df["point"] == pt].sort_values("x")
        if pt_df.empty:
            continue

        xs = pt_df["x"].to_numpy(dtype=int)
        means = pt_df["mean"].to_numpy(dtype=float)
        stds = pt_df["std"].to_numpy(dtype=float)
        ses = pt_df["se"].to_numpy(dtype=float)
        ns = pt_df["n"].to_numpy(dtype=int)

        # Asymmetric yerr (2 x N) per the chosen error model, clamped to [0, 1].
        lower_errs = np.zeros(len(means))
        upper_errs = np.zeros(len(means))
        for i in range(len(means)):
            lower_errs[i], upper_errs[i] = error_bounds(
                float(means[i]), int(ns[i]), float(stds[i]), float(ses[i]), error
            )

        if error == "none":
            # No uncertainty whiskers: markers joined by lines, no x-dodge.
            ax.plot(
                xs,
                means,
                label=labels[pt],
                color=COLORS[pt],
                marker=MARKERS[pt],
                linewidth=2,
                markersize=8,
            )
            continue

        xs_dodged = np.array([_dodged_x(int(x), pt, mode) for x in xs])

        # Markers + whiskers only -- no connecting lines (3 noisy points
        # should not be joined into a trend).
        ax.errorbar(
            xs_dodged,
            means,
            yerr=np.array([lower_errs, upper_errs]),
            label=labels[pt],
            color=COLORS[pt],
            marker=MARKERS[pt],
            linestyle="none",
            markersize=9,
            markeredgecolor="white",
            markeredgewidth=0.8,
            capsize=5,
            capthick=1.6,
            elinewidth=1.8,
        )

    # Axis labels and scale.
    if mode == "adaptive":
        ax.set_xlabel("Number of adaptive samples")
    else:
        ax.set_xlabel("Number of initial conditions for training")
        ax.set_xscale("log")
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    ax.set_ylabel("Success rate")
    ax.set_ylim(-0.05, 1.05)

    # Ticks: only x values present in summary_df (avoids gaps for 0-seed points).
    present_xs = sorted(summary_df["x"].unique().tolist())
    ax.set_xticks(present_xs)
    ax.get_xaxis().set_tick_params(which="minor", size=0)

    if legend != "none":
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
            handletextpad=0.4,
            borderaxespad=0.0,
        )
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Annotate n < 30 below each tick.
    y_annot = -0.08  # in axes fraction, below x-axis
    ax_trans = ax.get_xaxis_transform()
    for x_val in present_xs:
        n = n_at_x.get(x_val, 0)
        if n < 30:
            ax.annotate(
                f"n={n}",
                xy=(x_val, 0),
                xycoords=ax_trans,
                xytext=(0, -28),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=13,
                color="dimgray",
            )

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Cross-check against saved metrics.json
# ---------------------------------------------------------------------------

def cross_check_old_output(old_output_root: Path) -> list[dict]:
    """Re-run check_unique_membership on old_output artifacts and compare to saved metrics.json.

    The old_output tree uses the legacy checkpoint format (encoder.pt / decoder.pt /
    dynamics.pt) with its own scaler (``old_output_root/scalers/<dataset>/scaler.gz``).
    These are DIFFERENT model weights from those in replay_sources, so the cross-check
    must use the old_output's own checkpoints — not the perseed_df from replay_sources.

    Parameters
    ----------
    old_output_root:
        Parent of ``train_<N>/seed_<s>/`` directories, e.g.
        ``scratch/old_output/coral_data_scaling``.
    """
    try:
        import joblib  # noqa: PLC0415

        from latentdynamics._paths import get_repo_root  # noqa: PLC0415
        from latentdynamics.training import load_any_checkpoint  # noqa: PLC0415
    except ImportError as exc:
        print(f"  cross-check imports failed: {exc}", file=sys.stderr)
        return []

    repo_root = get_repo_root()
    scaler_dir = old_output_root / "scalers"
    results = []

    if not old_output_root.is_dir():
        return results

    for dataset_dir in sorted(old_output_root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset_name = dataset_dir.name
        scaler_path = scaler_dir / dataset_name / "scaler.gz"
        if not scaler_path.is_file():
            continue

        for seed_dir in sorted(dataset_dir.iterdir()):
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                continue
            try:
                seed = int(seed_dir.name.split("_")[1])
            except (IndexError, ValueError):
                continue

            mg = seed_dir / "MG" / "morse_graph"
            if not mg.is_file() or mg.stat().st_size == 0:
                continue

            saved_json = seed_dir / "metrics.json"
            if not saved_json.is_file():
                continue
            try:
                saved = json.loads(saved_json.read_text())
            except Exception:
                continue
            saved_metrics = saved.get("metrics", {})

            # Load this run's own checkpoint and scaler.
            try:
                model, _ = load_any_checkpoint(
                    seed_dir / "models",
                    legacy_root=repo_root / "legacy",
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    scaler = joblib.load(scaler_path)
                _, computed_metrics = check_unique_membership(
                    model.encoder,
                    scaler,
                    seed_dir / "MG" / "morse_sets",
                    seed_dir / "MG" / "morse_graph",
                )
            except Exception as exc:
                print(
                    f"  cross-check SKIP {dataset_name}/seed_{seed}: {exc}",
                    file=sys.stderr,
                )
                continue

            computed = {pt: bool(computed_metrics.get(pt)) for pt in POINTS}
            saved_bools = {pt: bool(saved_metrics.get(pt)) for pt in POINTS}
            match = computed == saved_bools
            results.append(
                {
                    "dataset": dataset_name,
                    "seed": seed,
                    "match": match,
                    "computed": computed,
                    "saved": saved_bools,
                }
            )

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Coral success-rate scaling figure with std error bars.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["size", "adaptive"],
        default="adaptive",
        help="Experiment axis: 'size' (N, log x) or 'adaptive' (M, linear x).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "/Users/bdoprad/Work/Projects/latent-dynamics/scratch/coral_success_stdev"
        ),
        help="Output directory for CSVs and figures.",
    )
    parser.add_argument(
        "--error",
        choices=["wilson", "se", "std", "none"],
        default="wilson",
        help="Whisker type: 'wilson' (95%% binomial CI; recommended for a "
        "success probability), 'se' (standard error), 'std' (sample stdev), or "
        "'none' (no whiskers; markers joined by lines, Brittany's original style).",
    )
    parser.add_argument(
        "--legend",
        choices=["points", "phat", "none"],
        default="points",
        help="Legend labels: 'points' ($a_0$, $a_1$, $r$) or 'phat' "
        r"($\hat{p}(1_{a_0})$, ...), matching the paper body notation.",
    )
    parser.add_argument(
        "--replot-from",
        type=Path,
        default=None,
        help="Skip metric recomputation and re-plot from an existing per-seed "
        "CSV (columns: dataset, x, seed, a0, a1, r). Fast for plot iteration.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Override the experiment config used to locate artifacts (a "
        "packaged name or a path to a YAML). Default: the packaged coral config "
        "for the chosen --mode.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point."""
    args = parse_args(argv)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "adaptive":
        config_name = "coral_adaptive"
        datasets = ADAPTIVE_DATASETS
        xlabel_mode = "adaptive"
    else:
        config_name = "coral_data_scaling"
        datasets = SIZE_DATASETS
        xlabel_mode = "size"

    if args.config is not None:
        config_name = args.config

    x_map = {ds: x for x, ds in datasets}
    x_values_all = [x for x, _ in datasets]

    # ------------------------------------------------------------------
    # Collect per-seed indicators (or load them from a saved CSV).
    # ------------------------------------------------------------------
    if args.replot_from is not None:
        perseed_df = pd.read_csv(args.replot_from)
        print(f"Re-plotting from saved per-seed CSV: {args.replot_from} "
              f"({len(perseed_df)} rows)")
        # Reconstruct the dataset->x map from the saved data.
        x_map = (
            perseed_df[["dataset", "x"]].drop_duplicates().set_index("dataset")["x"].to_dict()
        )
    else:
        all_rows: list[dict] = []
        for x_val, dataset_name in datasets:
            print(f"\n--- {dataset_name} (x={x_val}) ---")
            count_before = len(all_rows)
            for row in collect_seed_indicators(config_name, dataset_name, ALL_SEEDS):
                row["x"] = x_val
                all_rows.append(row)
            n_real = len(all_rows) - count_before
            print(f"  {n_real} real seeds collected.")
            if n_real == 0:
                print(f"  WARNING: no real seeds for {dataset_name} — skipping in figure.")

        if not all_rows:
            print("ERROR: no data collected. Exiting.", file=sys.stderr)
            sys.exit(1)

        perseed_df = pd.DataFrame(
            all_rows, columns=["dataset", "x", "seed", "a0", "a1", "r"]
        )
        perseed_path = out_dir / f"coral_success_{args.mode}_perseed.csv"
        perseed_df.to_csv(perseed_path, index=False)
        print(f"\nPer-seed CSV written: {perseed_path}")

    # ------------------------------------------------------------------
    # Aggregate.
    # ------------------------------------------------------------------
    summary_df = aggregate_perseed(perseed_df, x_map)
    summary_path = out_dir / f"coral_success_{args.mode}_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary CSV written: {summary_path}")

    # Print summary table.
    print(f"\nSummary (mode={args.mode}):")
    print(f"{'x':>6}  {'point':>5}  {'n':>3}  {'mean':>6}  {'std':>6}  {'se':>6}")
    print("-" * 42)
    for _, row in summary_df.iterrows():
        std_str = f"{row['std']:.4f}" if not np.isnan(row["std"]) else "  nan "
        se_str = f"{row['se']:.4f}" if not np.isnan(row["se"]) else "  nan "
        print(
            f"{int(row['x']):>6}  {row['point']:>5}  {int(row['n']):>3}  "
            f"{row['mean']:.4f}  {std_str}  {se_str}"
        )

    # ------------------------------------------------------------------
    # Plot.
    # ------------------------------------------------------------------
    present_x_vals = sorted(summary_df["x"].unique().tolist())
    if len(present_x_vals) < 1:
        print("WARNING: no points to plot — figure skipped.", file=sys.stderr)
    elif len(present_x_vals) < 2:
        print(
            f"WARNING: only {len(present_x_vals)} distinct x value(s) — "
            "figure is sparse; emitting anyway.",
            file=sys.stderr,
        )

    fig = build_figure(
        summary_df,
        mode=args.mode,
        x_values_all=x_values_all,
        error=args.error,
        legend=args.legend,
    )
    fig_stem = out_dir / f"morse_metric_plot_{args.mode}_{args.error}"
    written = save_figure(fig, fig_stem, formats=("pdf", "png"), close=True)
    for p in written:
        print(f"Figure written: {p}")

    if args.replot_from is not None:
        return  # re-plot only; skip the (expensive) cross-check

    # ------------------------------------------------------------------
    # Cross-check against saved metrics.json (size mode has old output).
    # ------------------------------------------------------------------
    old_root = (
        Path("/Users/bdoprad/Work/Projects/latent-dynamics/scratch/old_output")
        / "coral_data_scaling"
    )
    xcheck = cross_check_old_output(old_root)
    if xcheck:
        n_match = sum(r["match"] for r in xcheck)
        n_total = len(xcheck)
        print(f"\nCross-check vs old_output metrics.json: {n_match}/{n_total} match.")
        for r in xcheck:
            status = "OK" if r["match"] else "MISMATCH"
            print(
                f"  [{status}] {r['dataset']}/seed_{r['seed']}: "
                f"computed={r['computed']} saved={r['saved']}"
            )
    else:
        print(
            "\nCross-check: no old_output metrics.json with real MG artifacts found "
            f"(expected for mode={args.mode!r} which has no old_output tree)."
        )


if __name__ == "__main__":
    main()
