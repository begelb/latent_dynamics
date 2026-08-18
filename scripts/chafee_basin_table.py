"""Derive the Chafee--Infante basin-classification table from per-IC data.

Reproduces the paper's basins-of-attraction summary table from the
per-initial-condition classification file for the 45 latent-model
computations (3 latent dimensions x 5 datasets x 3 trainings, 10,000 initial
conditions each).  For every latent dimension the script computes, as mean
percentages of all initial conditions: the undetermined share split by the
steady state the initial condition actually converges to, the true-positive
(TP) and false-positive (FP) shares per steady state, and precision and
recall.  Following the published caption, the false negatives of one steady
state are counted as the false positives of the other steady state plus the
undetermined initial conditions converging to the first; equivalently, the
recall denominator for a steady state is the number of initial conditions
that truly converge to it.

Sign convention: the published table's M(0^+) rows correspond to the
classification file's ``negative`` class label and its M(0^-) rows to
``positive``.

Outputs: ``basin_table.csv`` (full precision), ``basin_table.md`` (printed
precision, paper row order), and ``per_run_outcomes.csv`` (per-run counts and
percentages).  By default every derived value is checked against the
published table at its printed precision and the script fails on any
disagreement; pass ``--no-verify`` when running on a freshly recomputed
classification file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    CODE_ROOT
    / "replay_sources"
    / "chafee_infante"
    / "statistics"
    / "ci_completed_10k_raw_classifications_45_runs.csv"
)
DEFAULT_OUTPUT = CODE_ROOT / "output" / "chafee_basin_table"

CLASSES = ("negative", "positive")
PAPER_LABEL = {"negative": "M(0^+)", "positive": "M(0^-)"}

DIMENSIONS = (1, 2, 3)
DATASETS = (1, 2, 3, 4, 5)
RUNS = (1, 2, 3)
ICS_PER_RUN = 10_000
TRUE_COUNTS_PER_RUN = {"negative": 5_030, "positive": 4_970}

# Published table values by (statistic, class) -> {dimension: percent}.
# Outcome shares are printed with two decimals except the FP rows (three);
# precision and recall are printed with two decimals.
PUBLISHED = {
    ("undetermined_to", "negative"): {1: 29.35, 2: 23.16, 3: 19.57},
    ("undetermined_to", "positive"): {1: 27.25, 2: 19.59, 3: 20.25},
    ("tp", "negative"): {1: 20.65, 2: 26.92, 3: 30.68},
    ("tp", "positive"): {1: 22.17, 2: 30.00, 3: 29.43},
    ("fp", "negative"): {1: 0.282, 2: 0.104, 3: 0.023},
    ("fp", "positive"): {1: 0.303, 2: 0.220, 3: 0.048},
    ("precision", "negative"): {1: 98.65, 2: 99.62, 3: 99.93},
    ("precision", "positive"): {1: 98.65, 2: 99.27, 3: 99.84},
    ("recall", "negative"): {1: 41.05, 2: 53.53, 3: 61.00},
    ("recall", "positive"): {1: 44.61, 2: 60.37, 3: 59.21},
}
DECIMALS = {
    "undetermined_to": 2,
    "tp": 2,
    "fp": 3,
    "precision": 2,
    "recall": 2,
}
ROW_ORDER = (
    ("undetermined_to", "negative"),
    ("undetermined_to", "positive"),
    ("tp", "negative"),
    ("tp", "positive"),
    ("fp", "negative"),
    ("fp", "positive"),
    ("precision", "negative"),
    ("precision", "positive"),
    ("recall", "negative"),
    ("recall", "positive"),
)
ROW_TITLE = {
    "undetermined_to": "Undetermined, converges to {label}",
    "tp": "TP, {label}",
    "fp": "FP, {label}",
    "precision": "Precision, {label}",
    "recall": "Recall, {label}",
}


def load_classifications(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        usecols=["dimension", "dataset", "run", "true_class", "predicted_class"],
    )
    unexpected_true = set(frame["true_class"].unique()) - set(CLASSES)
    if unexpected_true:
        raise ValueError(f"unexpected true_class values: {sorted(unexpected_true)}")
    unexpected_pred = set(frame["predicted_class"].unique()) - (
        set(CLASSES) | {"undetermined"}
    )
    if unexpected_pred:
        raise ValueError(
            f"unexpected predicted_class values: {sorted(unexpected_pred)}"
        )
    return frame


def per_run_outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    """Count TP, FP, and split undetermined outcomes for every run."""
    groups = frame.groupby(["dimension", "dataset", "run"], sort=True)
    records = []
    for (dimension, dataset, run), sub in groups:
        record: dict[str, object] = {
            "dimension": int(dimension),
            "dataset": int(dataset),
            "run": int(run),
            "n_initial_conditions": int(len(sub)),
        }
        for cls in CLASSES:
            other = CLASSES[1 - CLASSES.index(cls)]
            true_cls = sub["true_class"] == cls
            record[f"true_{cls}_count"] = int(true_cls.sum())
            record[f"tp_{cls}_count"] = int(
                (true_cls & (sub["predicted_class"] == cls)).sum()
            )
            record[f"fp_{cls}_count"] = int(
                ((sub["true_class"] == other) & (sub["predicted_class"] == cls)).sum()
            )
            record[f"undetermined_to_{cls}_count"] = int(
                (true_cls & (sub["predicted_class"] == "undetermined")).sum()
            )
        records.append(record)
    outcomes = pd.DataFrame.from_records(records)

    expected_runs = len(DIMENSIONS) * len(DATASETS) * len(RUNS)
    if len(outcomes) != expected_runs:
        raise ValueError(f"expected {expected_runs} runs; got {len(outcomes)}")
    for cls in CLASSES:
        # Every initial condition truly converging to a steady state is either
        # its TP, the other steady state's FP, or undetermined-converging-to-it.
        accounted = (
            outcomes[f"tp_{cls}_count"]
            + outcomes[f"fp_{CLASSES[1 - CLASSES.index(cls)]}_count"]
            + outcomes[f"undetermined_to_{cls}_count"]
        )
        if not (accounted == outcomes[f"true_{cls}_count"]).all():
            raise ValueError(f"outcome counts do not partition the true {cls} class")

    for cls in CLASSES:
        for stat in ("tp", "fp", "undetermined_to"):
            outcomes[f"{stat}_{cls}_percent"] = (
                100.0 * outcomes[f"{stat}_{cls}_count"] / outcomes["n_initial_conditions"]
            )
    return outcomes


def dimension_table(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-run outcomes into the per-dimension published table.

    All runs classify the same number of initial conditions, so the mean of
    the per-run percentages equals the aggregate-count percentage.
    """
    rows = []
    for dimension in DIMENSIONS:
        sub = outcomes[outcomes["dimension"] == dimension]
        total = int(sub["n_initial_conditions"].sum())
        values: dict[tuple[str, str], float] = {}
        for cls in CLASSES:
            other = CLASSES[1 - CLASSES.index(cls)]
            tp = int(sub[f"tp_{cls}_count"].sum())
            fp = int(sub[f"fp_{cls}_count"].sum())
            fp_other = int(sub[f"fp_{other}_count"].sum())
            undetermined = int(sub[f"undetermined_to_{cls}_count"].sum())
            values[("undetermined_to", cls)] = 100.0 * undetermined / total
            values[("tp", cls)] = 100.0 * tp / total
            values[("fp", cls)] = 100.0 * fp / total
            values[("precision", cls)] = 100.0 * tp / (tp + fp)
            values[("recall", cls)] = 100.0 * tp / (tp + fp_other + undetermined)
        for (statistic, cls), value in values.items():
            rows.append(
                {
                    "statistic": statistic,
                    "data_class": cls,
                    "paper_label": PAPER_LABEL[cls],
                    "dimension": dimension,
                    "percent": value,
                }
            )
    table = pd.DataFrame(rows).pivot_table(
        index=["statistic", "data_class", "paper_label"],
        columns="dimension",
        values="percent",
        sort=False,
    )
    table.columns = [f"d{dimension}" for dimension in table.columns]
    return table.reset_index()


def verify_against_published(
    table: pd.DataFrame, outcomes: pd.DataFrame
) -> list[str]:
    """Return one message per value that disagrees with the published table."""
    mismatches = []
    truth = outcomes.groupby("dimension")[
        [f"true_{cls}_count" for cls in CLASSES]
    ].sum()
    for cls in CLASSES:
        expected_truth = TRUE_COUNTS_PER_RUN[cls] * len(DATASETS) * len(RUNS)
        for dimension in DIMENSIONS:
            observed_truth = int(truth.loc[dimension, f"true_{cls}_count"])
            if observed_truth != expected_truth:
                mismatches.append(
                    f"d={dimension}: true {cls} count {observed_truth} != "
                    f"{expected_truth} (completed 10k ground truth)"
                )
    indexed = table.set_index(["statistic", "data_class"])
    for (statistic, cls), published_by_dim in PUBLISHED.items():
        tolerance = 0.5 * 10.0 ** (-DECIMALS[statistic])
        for dimension, published in published_by_dim.items():
            computed = float(indexed.loc[(statistic, cls), f"d{dimension}"])
            if abs(computed - published) > tolerance:
                mismatches.append(
                    f"{statistic} {PAPER_LABEL[cls]} d={dimension}: "
                    f"computed {computed:.4f} != published {published}"
                )
    return mismatches


def markdown_table(table: pd.DataFrame) -> str:
    indexed = table.set_index(["statistic", "data_class"])
    lines = [
        "| Initial Conditions | d=1 (%) | d=2 (%) | d=3 (%) |",
        "|---|---|---|---|",
    ]
    for statistic, cls in ROW_ORDER:
        title = ROW_TITLE[statistic].format(label=PAPER_LABEL[cls])
        decimals = DECIMALS[statistic]
        cells = " | ".join(
            f"{float(indexed.loc[(statistic, cls), f'd{dimension}']):.{decimals}f}"
            for dimension in DIMENSIONS
        )
        lines.append(f"| {title} | {cells} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="skip the check against the published table values",
    )
    args = parser.parse_args()

    frame = load_classifications(args.input.resolve())
    outcomes = per_run_outcomes(frame)
    table = dimension_table(outcomes)

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outcomes.to_csv(output_dir / "per_run_outcomes.csv", index=False)
    table.to_csv(output_dir / "basin_table.csv", index=False, float_format="%.6f")
    markdown = markdown_table(table)
    (output_dir / "basin_table.md").write_text(markdown, encoding="utf-8")
    print(markdown)

    if not args.no_verify:
        mismatches = verify_against_published(table, outcomes)
        if mismatches:
            for message in mismatches:
                print(f"MISMATCH {message}", file=sys.stderr)
            return 1
        print("all values match the published table at printed precision")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
