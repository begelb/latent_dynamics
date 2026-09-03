#!/usr/bin/env python3
"""Regenerate every manuscript figure that is a result of a computation.

One driver for what was previously ten render scripts plus manual collection.
It runs only the figure-producing steps -- the sampled residual/tolerance
computations feed a table, not a figure, and add roughly an hour -- then copies
each PDF into one directory under its manuscript figure name.

Which figures are obtainable depends on the installed CMGDB. The coarsened
3-D Leslie panels need the Conley index of a cell subset whose index pair spans
two subdivision depths; only ``ComputeConleyIndexForCells`` can compute that, so
on a build without it the coarsening step stops and the packaged bundle it
feeds cannot be built. Rather than emit a figure carrying a placeholder index,
this driver reports such panels as unavailable and names the reason.

Figures with no in-repo generator, or whose inputs were not preserved, are
listed by ``--list`` alongside what each needs. They are declared rather than
discovered halfway through a run.

Usage:
    python scripts/render_paper_figures.py --list
    python scripts/render_paper_figures.py
    python scripts/render_paper_figures.py --only coral leslie3d_example1
    python scripts/render_paper_figures.py --skip-compute   # re-collect only
    python scripts/render_paper_figures.py --only leslie3d_baseline_morse

Scope: CMGDB computations and the figures that come out of them, nothing
else. Sampled residual/tolerance metrics and the forward-closure check are
analysis, not figures -- they cost minutes each and no panel reads their
output -- so they live in reproduce_paper.py and are run on their own.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from latentdynamics.analysis.cmgdb_features import (  # noqa: E402
    cmgdb_capabilities,
    cmgdb_provenance,
)

DEFAULT_OUTPUT = REPO_ROOT / "output" / "paper_figures_pdf"

# Rendering the orbit overlay is opt-in: it is a diagnostic variant rather than
# a manuscript panel, and it doubles the Morse-set renders.
DEFAULT_FIGURE_GROUPS = "morse,roa,extras"

# The Chafee d=2 replay config records padding=false, matching the published
# run. This driver pads by default at the caller's request; the shipped config
# is left alone so it keeps documenting what produced the paper's figure.
DEFAULT_CHAFEE_PADDING = True



@dataclass(frozen=True)
class Step:
    """One computation that has to run before some figure can be copied."""

    label: str
    argv: list[str]
    #: CMGDB capability the step needs; ``None`` when it runs on any build.
    needs: str | None = None
    #: Argument appended when the build cannot compute a Conley index for an
    #: adaptive cell subset, letting the step proceed and mark what is missing.
    placeholder_flag: str | None = None


@dataclass(frozen=True)
class Figure:
    """One manuscript panel and where its generator leaves the PDF."""

    manuscript: str
    filename: str
    source: str
    #: Capability required to produce it, if any.
    needs: str | None = None


@dataclass(frozen=True)
class Family:
    description: str
    steps: list[Step] = field(default_factory=list)
    figures: list[Figure] = field(default_factory=list)
    #: Excluded from a default run and taken only when asked for by name or
    #: with --with-baselines. The direct references recompute CMGDB on the true
    #: maps at published subdivisions -- roughly 54 and 97 minutes -- which is
    #: far more than the latent families and is rarely what a figure refresh
    #: needs, since nothing about them changes when a latent model does.
    optional: bool = False
    #: Taken only when named with --only: never by a default run, and not by
    #: --with-baselines either. For a family that writes a panel another family
    #: also claims, so which computation supplies that panel stays an explicit
    #: choice rather than a side effect of ordering.
    by_name_only: bool = False


FAMILIES: dict[str, Family] = {
    "coral": Family(
        description="13-D red coral model, 1-D latent",
        steps=[
            Step("CMGDB + render", [
                "pipeline.py", "--config", "coral_basic",
                "--stages", "morse,render,metrics", "--cell-index", "16",
                "--device", "cpu", "--figures", DEFAULT_FIGURE_GROUPS]),
            Step("1-D Morse-set bands", ["scripts/render_coral_morse_sets_1d.py"]),
        ],
        figures=[
            Figure("coral_latent_dynamics(a) Morse graph",
                   "fig_coral_a_morse_graph.pdf",
                   "replay/coral_basic/train_500/seed_16/MG/morse_graph.pdf"),
            Figure("coral_latent_dynamics(b) 1-D bands + fixed points",
                   "fig_coral_b_morse_sets_1d.pdf",
                   "output/coral_figures/morse_sets_1D.pdf"),
        ],
    ),
    "chafee_infante": Family(
        description="Chafee-Infante, 64-mode spectral discretisation, 2-D latent",
        steps=[
            Step("CMGDB + render", [
                "pipeline.py", "--config", "chafee_infante_replay",
                "--stages", "morse,render,metrics", "--device", "cpu",
                "--figures", DEFAULT_FIGURE_GROUPS]),
            Step("theoretical Hasse diagrams",
                 ["scripts/render_chafee_theoretical_morse.py"]),
            # The paper's own d=2 computation: subdivisions (14, 16, 22) on
            # data-derived bounds with 10% per-axis padding, coarsened to the
            # three-node graph. Distinct from the replay block above, which
            # runs the production d=2 subdivisions (14, 16, 22).
            # Fine sets from the adaptive run; the merged fiber completed on the
            # uniform grid at subdiv_min, which is the grid the basins use.
            Step("coarsen (adaptive fine sets, uniform completion)",
                 ["scripts/coarsen_chafee_infante.py",
                  "--computation", "archived"]),
            # Basins on the archived uniform 256x256 grid, drawn as one RGBA
            # image, alone and under the coarsened Morse sets.
            Step("attractor basins + overlay",
                 ["scripts/plot_chafee_coarse_morse_roa_overlay.py"]),
            # d=1 and d=2 standardized panels, and the d=3 level-palette
            # graph + 3-D Morse sets, all drawn by CMGDB's own plotting from
            # the persisted latent-dimension-study artifacts.
            Step("standardized d=1/d=2 panels",
                 ["scripts/render_chafee_infante_standardized.py"]),
            Step("d=3 level-palette graph + 3-D sets",
                 ["scripts/render_chafee_infante_3d_graph_palette.py"]),
        ],
        figures=[
            Figure("ci_MRfull(a) full Morse representation",
                   "fig_ci_MRfull_a_hasse_full.pdf",
                   "output/chafee_infante_theoretical_morse/ci_morse_representation_full.pdf"),
            Figure("ci_MRfull(b) coarse Morse representation",
                   "fig_ci_MRfull_b_hasse_coarse.pdf",
                   "output/chafee_infante_theoretical_morse/ci_morse_representation_coarse.pdf"),
            Figure("ci_morse_graph_dynamics(a) Morse graph before coarsening",
                   "fig_ci_latent_d2_fine_morse_graph.pdf",
                   "output/chafee_coarsened/fine_morse_graph.pdf"),
            Figure("ci_morse_graph_dynamics(b) Morse sets before coarsening",
                   "fig_ci_latent_d2_fine_morse_sets.pdf",
                   "output/chafee_coarsened/fine_morse_sets.pdf"),
            Figure("ci_morse_graph_dynamics(c) coarsened Morse graph, nodes 0/1/2",
                   "fig_ci_latent_d2_coarse_morse_graph.pdf",
                   "output/chafee_coarsened/morse_graph.pdf"),
            Figure("ci_morse_graph_dynamics(d) coarsened Morse sets",
                   "fig_ci_latent_d2_coarse_morse_sets.pdf",
                   "output/chafee_coarsened/morse_sets.pdf"),
            Figure("ci_attractor_basins(a) attraction basins, uniform 256x256",
                   "fig_ci_latent_d2_attractor_basins.pdf",
                   "output/chafee_coarsened/attractor_basins.pdf"),
            Figure("ci_attractor_basins(b) basins + coarsened Morse sets",
                   "fig_ci_latent_d2_basins_with_coarse_sets.pdf",
                   "output/chafee_coarsened/morse_roa_overlay.pdf"),
            Figure("ci_latent_1d Morse graph",
                   "fig_ci_latent_1d_morse_graph.pdf",
                   "output/chafee_standardized/ci_latent_1d_morse_graph.pdf"),
            Figure("ci_latent_1d Morse sets (CMGDB PlotMorseSets1D)",
                   "fig_ci_latent_1d_morse_sets.pdf",
                   "output/chafee_standardized/ci_latent_1d_morse_sets.pdf"),
            Figure("ci_latent_3d Morse graph, level palette",
                   "fig_ci_latent_3d_morse_graph.pdf",
                   "output/chafee_standardized/latent_3d_level_palette/"
                   "ci_latent_3d_morse_graph_level_palette.pdf"),
            Figure("ci_latent_3d Morse sets, 3-D cubical (CMGDB PlotMorseSets3D)",
                   "fig_ci_latent_3d_morse_sets_3d.pdf",
                   "output/chafee_standardized/latent_3d_level_palette/"
                   "ci_latent_3d_morse_sets_3d.pdf"),
        ],
    ),
    "leslie3d_example1": Family(
        description="3-D Leslie model, 2-D latent (spurious attractor)",
        steps=[
            Step("CMGDB + render", [
                "pipeline.py", "--config", "leslie3d_example1_replay",
                "--stages", "morse,render,metrics", "--device", "cpu",
                "--figures", DEFAULT_FIGURE_GROUPS]),
            # Runs on any build: the coarsened sets come from graph
            # reachability. Where the Conley index is unavailable the step is
            # told to continue and mark it, rather than stopping.
            Step("coarsen nodes 4,5",
                 ["scripts/leslie3d_example1_coarsen_morse_graph.py",
                  # coarsen this run's Morse graph, not the shipped one
                  "--morse-dir", "replay/leslie3d_example1_replay/MG"],
                 placeholder_flag="--allow-placeholder-index"),
            # Coarse reference at (22, 22, 24): base grid 22 with refinement
            # allowed to 24. Pinning max to 22 instead reports 24 Morse sets,
            # 20 of them one-cell components with trivial index that refinement
            # resolves away; the four real ones are identical either way.
            # CMGDB's own ComputeConleyMorseGraph, whose indices are correct on
            # the adaptive grid. The fixed-depth study script instead relabels
            # every node with a local per-cell index pair, which upstream CMGDB
            # cannot express across depths 22-24 and silently reports trivial --
            # collapsing nodes 2 and 3 out of the nontrivial skeleton.
            Step("coarse (22,22,24) CMGDB run",
                 ["scripts/leslie3d_example1_coarse_subdiv.py",
                  "--subdiv", "22", "22", "24",
                  "--output", "output/leslie3d_example1_study/coarse"]),
            Step("coarsened sets + graph (CMGDB)",
                 ["scripts/plot_coarsened_morse.py",
                  "--box-scale", "1", "1", "1", "1", "20"]),
            # Fine Morse sets as vector rectangles. Sets 4 and 5 are a few
            # boxes across, so they get a magnified inset rather than an
            # inflation factor: every box stays at its true size in both views.
            Step("fine Morse sets (CMGDB)",
                 ["scripts/plot_coarsened_morse.py",
                  "--sets-csv", "replay/leslie3d_example1_replay/MG/morse_sets",
                  "--graph-dot", "replay/leslie3d_example1_replay/MG/morse_graph",
                  "--out-sets", "output/leslie3d_example1_figures/morse_sets_fine.pdf",
                  "--out-graph", "output/leslie3d_example1_figures/morse_graph_fine.pdf",
                  "--zoom-nodes", "4", "5"]),

        ],
        figures=[
            Figure("3D_Leslie_latent_coarse(a) (22,22,24) Morse graph",
                   "fig_leslie3d_coarse_a_morse_graph.pdf",
                   "output/leslie3d_example1_study/coarse/22_22_24/morse_graph.pdf"),
            Figure("3D_Leslie_latent_coarse(b) (22,22,24) Morse sets",
                   "fig_leslie3d_coarse_b_morse_sets.pdf",
                   "output/leslie3d_example1_study/coarse/22_22_24/morse_sets.pdf"),
            Figure("3D_Leslie_latent(b) coarsened graph (from CMGDB DOT)",
                   "fig_leslie3d_latent_b_coarsened_morse_graph_cmgdb.pdf",
                   "output/leslie3d_example1_study/coarsened_45/morse_graph_coarse.pdf"),
            Figure("3D_Leslie_latent(d) coarsened Morse sets (CMGDB PlotMorseSets)",
                   "fig_leslie3d_latent_d_coarsened_morse_sets_cmgdb.pdf",
                   "output/leslie3d_example1_study/coarsened_45/morse_sets_coarse.pdf"),
            # Available on any build: the pipeline render, not the bundle.
            Figure("3D_Leslie_latent fine Morse graph (unpackaged)",
                   "fig_leslie3d_latent_a_fine_morse_graph_plain.pdf",
                   "replay/leslie3d_example1_replay/MG/morse_graph.pdf"),
            Figure("3D_Leslie_latent fine Morse sets (unpackaged)",
                   "fig_leslie3d_latent_c_fine_morse_sets_plain.pdf",
                   "output/leslie3d_example1_figures/morse_sets_fine.pdf"),
        ],
    ),
    "leslie2d_baseline": Family(
        optional=True,
        description="2-D Leslie baseline: CMGDB on the true map, no autoencoder "
                    "(subdivisions 24/27/28)",
        steps=[Step("CMGDB + plots", ["scripts/plot_leslie2d_baseline.py"])],
        figures=[
            Figure("lesliecontraction_dynamics(a) direct 2-D reference Morse graph",
                   "fig_leslie2gen_a_direct_morse_graph.pdf",
                   "output/leslie2d_baseline/morse_graph.pdf"),
            Figure("lesliecontraction_dynamics(b) direct 2-D reference Morse sets",
                   "fig_leslie2gen_b_direct_morse_sets.pdf",
                   "output/leslie2d_baseline/morse_sets.pdf"),
        ],
    ),
    "leslie3d_baseline": Family(
        optional=True,
        description="3-D Leslie baseline: CMGDB on the true map, 3-D Morse sets "
                    "(published subdivisions 29/33/36; needs a ~100 GB machine)",
        steps=[Step("CMGDB + plots", ["scripts/plot_leslie3d_baseline.py"])],
        figures=[
            Figure("3D_Leslie_direct(a) reference Morse graph",
                   "fig_leslie3d_direct_a_morse_graph.pdf",
                   "output/leslie3d_baseline/morse_graph.pdf"),
            Figure("3D_Leslie_direct(b) reference Morse sets, 3-D cubical",
                   "fig_leslie3d_direct_b_morse_sets_3d.pdf",
                   "output/leslie3d_baseline/morse_sets_3d.pdf"),
        ],
    ),
    "leslie3d_baseline_morse": Family(
        by_name_only=True,
        description="3-D Leslie baseline at the machine-sized screen 24/33/36, "
                    "in the published node colors: supplies panel (b) alone "
                    "(same Morse sets as 29/33/36, on a machine that has 16 GB "
                    "rather than 100)",
        steps=[Step("CMGDB + plots", ["scripts/plot_leslie3d_baseline_morse.py"])],
        # Panel (b) only. A coarse initial grid leaves the transient region
        # under-resolved, so this run's Morse-graph edges collapse the two
        # attractors into a chain; its sets are the published ones, its graph is
        # not, and the graph panel has to come from the published screen above.
        figures=[
            Figure("3D_Leslie_direct(b) reference Morse sets, 3-D cubical "
                   "(24/33/36 screen, published node colors)",
                   "fig_leslie3d_direct_b_morse_sets_3d.pdf",
                   "output/leslie3d_baseline_morse/morse_sets_3d.pdf"),
        ],
    ),
    "leslie_2gen_contraction": Family(
        description="2-D overcompensatory Leslie model embedded in 10-D",
        steps=[
            Step("CMGDB + render", [
                "pipeline.py", "--config", "leslie_2gen_contraction_replay",
                "--stages", "morse,render,metrics", "--device", "cpu",
                "--figures", DEFAULT_FIGURE_GROUPS]),
            Step("latent Morse sets", [
                "scripts/render_leslie_2gen_contraction_morse_sets.py",
                # plot what this run just computed, not the shipped reference CSV
                "--csv", "replay/leslie_2gen_contraction_replay/MG/morse_sets"]),
            Step("aligned Morse graph (DOT)", ["@dot"]),
        ],
        figures=[
            Figure("lesliecontraction_dynamics(c) latent Morse graph, paper alignment",
                   "fig_leslie2gen_c_latent_morse_graph.pdf",
                   "output/leslie_2gen_contraction_figures/morse_graph_aligned.pdf"),
            Figure("lesliecontraction_dynamics(d) latent Morse sets",
                   "fig_leslie2gen_d_latent_morse_sets.pdf",
                   "output/leslie_2gen_contraction_figures/morse_sets.pdf"),
        ],
    ),
}

#: Manuscript figures with no path from what this repository ships. Listed so
#: the gap is visible; each entry says what it would take.
UNREPRODUCIBLE: dict[str, str] = {
    "lesliecontraction_dynamics(a,b) direct 2-D reference":
        "needs the 154 MB morse_sets CSV under output/original_leslie/...; "
        "output/**/*.csv was gitignored. Recompute: scripts/compute_original_leslie.py "
        "--system 2d --subdiv 24 27 28",
    "3D_Leslie_direct(a,b) direct 3-D reference":
        "needs the 158 MB level-33 screen output and the level-24 display cover; "
        "same ignore rule. Recompute: scripts/screen_original_leslie3d_initial.py 29 (~97 min)",
    "ci_morse_graph_dynamics(a,b) fine d=2, (c,d) coarsened, ci_attractor_basins(a,b)":
        "need replay_sources/chafee_infante/reference_inputs/{ci_model_weights.pth,"
        "train_data.csv} from the coauthor archive, never committed",
    "ci_bif_diagram":
        "static manuscript asset; no generating script or source data was preserved",
}


#: Horizontal order the manuscript uses inside each same-rank row of the
#: leslie_2gen latent Morse graph. CMGDB emits the rank groups but not an
#: ordering, so graphviz is free to flip a row; these invisible edges pin it.
PAPER_RANK_ORDER: list[list[str]] = [["0", "1"], ["4", "3"]]

LIVE_2GEN_DOT = "replay/leslie_2gen_contraction_replay/MG/morse_graph"
REFERENCE_2GEN_DOT = "artifacts/reference_results/leslie_2gen_contraction/aligned_morse_graph.dot"


def _graph_content(dot: str) -> tuple[set[str], set[str]]:
    """Node declarations and edges, ignoring layout-only directives.

    Used to confirm the recomputed graph carries the same Conley indices and
    connections as the published one before the paper's layout is applied to it.
    """
    nodes, edges = set(), set()
    for raw in dot.splitlines():
        line = raw.strip()
        if line.startswith("{rank") or not line or line in {"digraph {", "}"}:
            continue
        if "->" in line and "[" not in line:
            edges.add(line.rstrip(";").replace(" ", ""))
        elif "[label=" in line:
            nodes.add(line.rstrip(";"))
    return nodes, edges


def _apply_paper_alignment(dot: str) -> str:
    """Rewrite CMGDB's rank groups into the manuscript's ordered rows."""
    wanted = {frozenset(group): group for group in PAPER_RANK_ORDER}
    out = []
    for raw in dot.splitlines():
        line = raw.strip()
        if line.startswith("{rank=same;"):
            members = line[len("{rank=same;"):].split("}")[0].split()
            order = wanted.get(frozenset(members))
            if order is not None:
                invis = "; ".join(f"{a} -> {b} [style=invis]"
                                  for a, b in zip(order, order[1:]))
                out.append(f"{{rank=same; {' '.join(order)}; {invis}}};")
                continue
        out.append(raw)
    return "\n".join(out)


def render_aligned_morse_graph() -> tuple[bool, str]:
    """Draw the latent Morse graph from THIS run's CMGDB output.

    The manuscript panel differs from raw CMGDB output only in row ordering, so
    the recomputed graph is checked against the published one for content and
    then re-laid-out, rather than the published .dot being drawn directly.
    """
    live = REPO_ROOT / LIVE_2GEN_DOT
    if not live.is_file():
        return False, f"missing recomputed DOT {LIVE_2GEN_DOT}; run the morse stage first"
    dot = live.read_text(encoding="utf-8")

    reference = REPO_ROOT / REFERENCE_2GEN_DOT
    if reference.is_file():
        live_content = _graph_content(dot)
        ref_content = _graph_content(reference.read_text(encoding="utf-8"))
        if live_content != ref_content:
            added = live_content[0] - ref_content[0], live_content[1] - ref_content[1]
            gone = ref_content[0] - live_content[0], ref_content[1] - live_content[1]
            return False, ("recomputed Morse graph differs from the published one "
                           f"(new {added}, missing {gone}); not a layout change")

    destination = REPO_ROOT / "output/leslie_2gen_contraction_figures/morse_graph_aligned.pdf"
    destination.parent.mkdir(parents=True, exist_ok=True)
    aligned = REPO_ROOT / "output/leslie_2gen_contraction_figures/morse_graph_aligned.dot"
    aligned.write_text(_apply_paper_alignment(dot).rstrip("\n") + "\n", encoding="utf-8")
    completed = subprocess.run(
        ["dot", "-Tpdf", str(aligned), "-o", str(destination)], cwd=REPO_ROOT
    )
    return completed.returncode == 0, f"dot exited {completed.returncode} (from recomputed DOT)"


def run_step(step: Step, capabilities: dict[str, bool], verbose: bool) -> tuple[bool, str]:
    """Run one step. Returns (ran, note)."""
    if step.needs is not None and not capabilities.get(step.needs, False):
        return False, f"skipped: CMGDB lacks {step.needs}"
    if step.placeholder_flag and not capabilities.get("conley_index_for_cells", False):
        step = Step(step.label, [*step.argv, step.placeholder_flag], step.needs)
    if step.argv == ["@dot"]:
        return render_aligned_morse_graph()
    argv = [sys.executable, str(REPO_ROOT / step.argv[0]), *step.argv[1:]]
    completed = subprocess.run(argv, cwd=REPO_ROOT,
                               stdout=None if verbose else subprocess.DEVNULL,
                               stderr=None if verbose else subprocess.DEVNULL)
    return completed.returncode == 0, f"exited {completed.returncode}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--only", nargs="+", choices=list(FAMILIES), metavar="FAMILY")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-compute", action="store_true",
                        help="collect from existing outputs without recomputing")
    parser.add_argument("--overlay", action="store_true",
                        help="also render the orbit-overlay variants (off by default)")
    parser.add_argument("--chafee-padding", choices=("true", "false"),
                        default=str(DEFAULT_CHAFEE_PADDING).lower(),
                        help="CMGDB padding for the Chafee d=2 run "
                             "(default true, matching the production "
                             "computation)")
    parser.add_argument("--with-baselines", action="store_true",
                        help="also run the direct 2-D and 3-D Leslie reference "
                             "computations, which are excluded by default "
                             "because they take hours; naming one with --only "
                             "runs it regardless")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    # Captured before any step runs: figures older than this were not made
    # by this run, whatever is sitting on disk.
    run_started = time.time()

    capabilities = cmgdb_capabilities()
    provenance = cmgdb_provenance()

    if args.list:
        print(f"CMGDB {provenance['version']}  ({provenance['module_path']})")
        for feature, present in capabilities.items():
            print(f"   {feature:32} {present}")
        for name, family in FAMILIES.items():
            if family.by_name_only:
                suffix = f"   [on request only: --only {name}]"
            elif family.optional:
                suffix = "   [optional: --with-baselines or --only]"
            else:
                suffix = ""
            print(f"\n{name}: {family.description}{suffix}")
            for figure in family.figures:
                blocked = figure.needs and not capabilities.get(figure.needs, False)
                mark = f"UNAVAILABLE (needs {figure.needs})" if blocked else "ok"
                print(f"   [{mark}] {figure.filename}")
                print(f"        {figure.manuscript}")
        print("\nNot reproducible from what this repository ships:")
        for name, reason in UNREPRODUCIBLE.items():
            print(f"   - {name}\n       {reason}")
        return 0

    groups = "morse,roa,overlay,extras" if args.overlay else DEFAULT_FIGURE_GROUPS
    # --only names families explicitly, so it overrides the default exclusion.
    targets = args.only or [
        name for name, family in FAMILIES.items()
        if not family.by_name_only and (args.with_baselines or not family.optional)
    ]
    print(f"CMGDB {provenance['version']} at {provenance['module_path']}")
    print(f"figure groups: {groups}")
    print(f"chafee padding override: {args.chafee_padding}\n")

    for name in targets:
        family = FAMILIES[name]
        print(f"===== {name} =====")
        if args.skip_compute:
            print("   (compute skipped)")
        else:
            for step in family.steps:
                argv_ = list(step.argv)
                if "--figures" in argv_:
                    argv_[argv_.index("--figures") + 1] = groups
                if name == "chafee_infante" and "--config" in argv_:
                    argv_ += ["--set", f"cmgdb.padding={args.chafee_padding}"]
                # dataclasses.replace, not a positional rebuild: constructing a
                # fresh Step here silently dropped placeholder_flag, so the
                # coarsen step never received --allow-placeholder-index and
                # failed on every run against a CMGDB without the capability.
                ok, note = run_step(replace(step, argv=argv_),
                                    capabilities, not args.quiet)
                print(f"   [{'ok  ' if ok else 'FAIL'}] {step.label}  ({note})")

    args.output.mkdir(parents=True, exist_ok=True)
    collected, unavailable = [], []
    # A figure is only trustworthy if the step that makes it ran and rewrote it
    # in THIS run. Collecting whatever happens to sit on disk silently ships the
    # output of an earlier run when a step fails -- which is exactly how a
    # failing coarsen step went unnoticed while its panels kept being collected.
    stale_cutoff = None if args.skip_compute else run_started
    for name in targets:
        for figure in FAMILIES[name].figures:
            source = REPO_ROOT / figure.source
            if figure.needs and not capabilities.get(figure.needs, False):
                unavailable.append((figure, f"CMGDB lacks {figure.needs}"))
                continue
            if not source.is_file():
                unavailable.append((figure, f"not produced: {figure.source}"))
                continue
            if stale_cutoff is not None and source.stat().st_mtime < stale_cutoff:
                age = (stale_cutoff - source.stat().st_mtime) / 60.0
                unavailable.append((
                    figure,
                    f"STALE: {figure.source} was not rewritten by this run "
                    f"(last written {age:.0f} min before it started); "
                    f"the step that produces it failed or was skipped"
                ))
                continue
            shutil.copy(source, args.output / figure.filename)
            collected.append(figure)

    # Accumulate: a --only run must not erase what a previous one recorded, or
    # the manifest ends up describing a subset of the directory beside it.
    manifest_path = args.output / "manifest.json"
    previous: dict = {}
    if manifest_path.is_file():
        try:
            previous = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            previous = {}
    kept = [
        entry for entry in previous.get("collected", [])
        if entry["file"] not in {f.filename for f in collected}
        and (args.output / entry["file"]).is_file()
    ]
    manifest = {
        "cmgdb": provenance,
        "figure_groups": groups,
        "chafee_padding": args.chafee_padding,
        "collected": kept + [{"file": f.filename, "manuscript": f.manuscript,
                              "source": f.source} for f in collected],
        "unavailable": [{"file": f.filename, "manuscript": f.manuscript,
                         "reason": why} for f, why in unavailable],
        "not_reproducible": UNREPRODUCIBLE,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\n===== {len(collected)} figures -> {args.output} =====")
    for figure in collected:
        print(f"   {figure.filename}")
    if unavailable:
        print(f"\n----- {len(unavailable)} unavailable -----")
        for figure, why in unavailable:
            print(f"   {figure.filename}\n       {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
