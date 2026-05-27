"""Render figures from on-disk artifacts; never invokes CMGDB.

The new pipeline writes ``MG/morse_graph`` (graphviz DOT) and ``MG/morse_sets``
(box CSV) during the ``morse`` stage. The render stage reads those files and
re-emits PDF/PNG plots via the unified palette in :mod:`latentdynamics.viz`.

Experiment-specific extras (e.g. the latent-trajectory plot for the Leslie 3D
spurious case) are dispatched on ``cfg.system.name`` via :func:`render_extras`.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch

from ..config import ExperimentConfig
from ..training import has_legacy_checkpoint, has_new_checkpoint, load_any_checkpoint
from ..viz import render_morse_from_files, render_morse_sets_with_overlay
from ..analysis.cmgdb_roa import EXACT_ROA_FILENAME
from ..viz.regions_of_attraction import render_cell_graph_roa, render_exact_roa_artifact


def _morse_dir_for(cfg: ExperimentConfig) -> Path:
    return cfg.paths.morse_dir


def _bounds_from_log(log_path: Path) -> tuple[list[float] | None, list[float] | None]:
    if not log_path.exists():
        return None, None
    lower: list[float] | None = None
    upper: list[float] | None = None
    for line in log_path.read_text().splitlines():
        if line.lower().startswith("lower bounds:"):
            lower = _parse_bounds(line)
        elif line.lower().startswith("upper bounds:"):
            upper = _parse_bounds(line)
    return lower, upper


def _parse_bounds(line: str) -> list[float]:
    payload = line.split(":", 1)[1].strip()
    payload = payload.strip("[]() ")
    return [float(x.strip()) for x in payload.split(",") if x.strip()]


def render_stage(
    cfg: ExperimentConfig,
    *,
    train_file: str = "train",
    device: torch.device | str | None = None,
    verbose: bool = True,
    out_dir: Path | None = None,
    figures: set[str] | None = None,
) -> dict[str, list[str]]:
    """Re-render Morse plots from saved DOT/CSV plus any system-specific figures.

    ``figures`` selects which groups to render -- a subset of
    ``{"morse", "roa", "overlay", "extras"}``; ``None`` renders all. Only ``roa``
    is expensive (it walks the latent map); the rest just read saved artifacts,
    so an overlay tweak can pass ``figures={"overlay"}`` to regenerate in ~a second.

    Reads source artifacts from ``cfg.paths.morse_dir`` and
    ``cfg.paths.output_dir``. When ``out_dir`` is provided, all rendered
    figures (Morse PDFs/PNGs and system-specific extras) are written under
    ``out_dir`` (``out_dir/MG/...`` and ``out_dir/figures/...``); when it is
    ``None``, output defaults to ``cfg.paths.output_dir``, matching the
    pre-replay-routing behavior.

    Returns ``{"skipped": <reason>}`` when the saved Morse artifacts are missing
    or empty (e.g. partial uploads), so a multi-seed sweep can keep going.
    """
    morse_dir = _morse_dir_for(cfg)
    dot_path = morse_dir / "morse_graph"
    csv_path = morse_dir / "morse_sets"
    if not dot_path.exists() or not csv_path.exists():
        if verbose:
            print(f"render: skipped {morse_dir} (missing morse_graph or morse_sets)")
        return {"skipped": f"missing morse_graph or morse_sets in {morse_dir}"}
    if dot_path.stat().st_size == 0 or csv_path.stat().st_size == 0:
        if verbose:
            print(f"render: skipped {morse_dir} (empty morse_graph or morse_sets)")
        return {"skipped": f"empty morse_graph or morse_sets in {morse_dir}"}

    bounds_lower, bounds_upper = _bounds_from_log(cfg.paths.output_dir / "mg_params_log.txt")

    write_root = Path(out_dir) if out_dir is not None else cfg.paths.output_dir
    want = set(figures) if figures is not None else {"morse", "roa", "overlay", "extras"}
    render_device = _resolve_render_device(device)
    rendered: list[str] = []

    if "morse" in want:
        morse_figs = render_morse_from_files(
            morse_dir,
            bounds_lower=bounds_lower,
            bounds_upper=bounds_upper,
            out_dir=write_root / "MG",
            box_scale="auto",
        )
        rendered += [
            str(morse_figs.morse_graph_pdf),
            str(morse_figs.morse_graph_png),
            *map(str, morse_figs.morse_sets_paths),
        ]

    if "roa" in want:
        roa = _render_roa_overlay(
            cfg,
            dot_path,
            csv_path,
            out_dir=write_root,
            device=render_device,
            verbose=verbose,
        )
        if roa is not None:
            rendered.append(str(roa))

    if "overlay" in want:
        overlay = _render_morse_overlay(
            cfg,
            csv_path,
            dot_path,
            out_dir=write_root,
            device=render_device,
            bounds_lower=bounds_lower,
            bounds_upper=bounds_upper,
            verbose=verbose,
        )
        if overlay is not None:
            rendered.append(str(overlay))

    if "extras" in want:
        extras = render_extras(
            cfg,
            train_file=train_file,
            device=render_device,
            verbose=verbose,
            out_dir=write_root,
        )
        rendered.extend(extras)

    if verbose:
        print(f"render: {len(rendered)} file(s) under {write_root}")
    return {"figures": rendered}


def _resolve_render_device(device: torch.device | str | None) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device is not None:
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _render_roa_overlay(
    cfg: ExperimentConfig,
    dot_path: Path,
    csv_path: Path,
    *,
    out_dir: Path,
    device: torch.device,
    verbose: bool,
) -> Path | None:
    """Cell-graph regions-of-attraction overlay, written to ``out_dir/MG/``.

    Currently 2D-only (the cell-graph backend raises for d!=2); silently skips
    if the latent map is higher-dimensional or the checkpoint is missing.
    """
    if cfg.arch.low_dims != 2:
        if verbose:
            print(f"render: skipping RoA overlay (latent dim {cfg.arch.low_dims} != 2)")
        return None
    exact_artifact = dot_path.with_name(EXACT_ROA_FILENAME)
    if exact_artifact.exists():
        out_path = Path(out_dir) / "MG" / "regions_of_attraction_exact.png"
        # Drop any stale cell-graph fallback render so the two don't coexist
        # (the exact RoA, on CMGDB's grid, supersedes it).
        for stale in ("regions_of_attraction.png", "regions_of_attraction.pdf"):
            stale_path = Path(out_dir) / "MG" / stale
            if stale_path.exists():
                stale_path.unlink()
        return render_exact_roa_artifact(exact_artifact, dot_path, out_path)

    model_dir = cfg.paths.output_dir / "models"
    if not (has_legacy_checkpoint(model_dir) or has_new_checkpoint(model_dir)):
        if verbose:
            print(f"render: skipping RoA overlay (no checkpoint at {model_dir})")
        return None
    model, _arch = load_any_checkpoint(model_dir, arch=cfg.arch)
    model.to(device).eval()

    if verbose:
        print(
            "render: using the approximate uniform-grid RoA -- no exact RoA artifact found; "
            "re-run the morse stage to compute the RoA on CMGDB's own grid"
        )
    out_path = Path(out_dir) / "MG" / "regions_of_attraction.png"
    return render_cell_graph_roa(
        dot_path,
        csv_path,
        model.latent_map,
        out_path,
        resolution=128,
        device=str(device),
    )


def _node_periods_from_dot(dot_path: Path) -> dict[int, int]:
    """Map Morse-graph node id -> orbit period parsed from its Conley index.

    Reads labels like ``0 : (x^6-1, 0, 0)``. A component equal to ``x^n-1``
    gives period ``n``; ``x^n+1`` gives ``2n``; ``x-1`` is period 1 (a fixed
    point). Components that are not a bare cyclotomic-style factor (e.g.
    ``x^2+x+1``) are ignored. Period 1 means no nontrivial orbit to trace.
    """
    import re

    text = dot_path.read_text()
    periods: dict[int, int] = {}
    for m in re.finditer(r'(\d+) \[label="(\d+) : \(([^)]*)\)"', text):
        best = 1
        for comp in m.group(3).split(","):
            mm = re.fullmatch(r"x\^?(\d*)([+-])1", comp.strip().replace(" ", ""))
            if mm:
                n = int(mm.group(1)) if mm.group(1) else 1
                best = max(best, n if mm.group(2) == "-" else 2 * n)
        periods[int(m.group(2))] = best
    return periods


def _box_components(centers: np.ndarray, box_w: float) -> tuple[np.ndarray, int]:
    """Split a Morse set's boxes into spatially connected components.

    Two boxes are linked when their centers lie within ``1.5 * box_w``; the
    components are the connected pieces -- e.g. the n points of a period-n orbit.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    n = len(centers)
    if n <= 1:
        return np.zeros(n, dtype=int), n
    pairs = cKDTree(centers).query_pairs(r=1.5 * box_w, output_type="ndarray")
    if len(pairs) == 0:
        return np.arange(n), n
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    n_comp, comp = connected_components(graph, directed=False)
    return comp, n_comp


def _closest_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The closest box-center pair between two components, returned as ``(in a, in b)``."""
    from scipy.spatial import cKDTree

    dist, idx = cKDTree(b).query(a)
    i = int(np.argmin(dist))
    return a[i], b[int(idx[i])]


def _render_morse_overlay(
    cfg: ExperimentConfig,
    csv_path: Path,
    dot_path: Path,
    *,
    out_dir: Path,
    device: torch.device,
    bounds_lower: list[float] | None,
    bounds_upper: list[float] | None,
    verbose: bool,
) -> Path | None:
    """Morse sets + grey orbit arrows (``morse_sets_with_overlay``).

    2D-only. For each periodic attractor (a Morse-graph sink with a cyclotomic
    ``x^n-1``/``x^n+1`` index), split its boxes into spatial components (the n
    points of the period-n orbit), order the components by the latent dynamics,
    and draw a grey arrow between the closest boxes of each consecutive pair --
    so every arrowhead stops at the next component's edge rather than over its
    boxes. Period-1 sinks (fixed points, invariant annuli) get no arrows. Skips
    silently when the latent map is higher-dimensional or no checkpoint exists.
    """
    if cfg.arch.low_dims != 2:
        return None
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return None
    model_dir = cfg.paths.output_dir / "models"
    if not (has_legacy_checkpoint(model_dir) or has_new_checkpoint(model_dir)):
        if verbose:
            print(f"render: skipping Morse overlay (no checkpoint at {model_dir})")
        return None
    model, _arch = load_any_checkpoint(model_dir, arch=cfg.arch)
    model.to(device).eval()
    advance = _make_advance_callable(model.latent_map, device)

    data = np.loadtxt(csv_path, delimiter=",", ndmin=2)
    if data.shape[1] != 5:
        return None
    labels = data[:, 4].astype(int)
    cx = 0.5 * (data[:, 0] + data[:, 2])
    cy = 0.5 * (data[:, 1] + data[:, 3])

    from .metrics import _minimal_morse_labels

    sinks = _minimal_morse_labels(dot_path) if dot_path.exists() else []
    periods = _node_periods_from_dot(dot_path) if dot_path.exists() else {}
    widths = data[:, 2] - data[:, 0]

    arrows: list[tuple[np.ndarray, np.ndarray]] = []
    for lbl in sinks:
        if periods.get(lbl, 1) < 2:
            continue  # fixed points / annuli: an orbit arrow would only point inward
        mask = labels == lbl
        centers = np.column_stack((cx[mask], cy[mask]))
        if len(centers) < 2:
            continue
        box_w = float(np.median(widths[mask]))
        if box_w <= 0:
            box_w = 0.02 * float(np.ptp(centers, axis=0).max()) + 1e-9
        comp, n_comp = _box_components(centers, box_w)
        if n_comp < 2:
            continue  # one connected blob -- no distinct components to link
        comp_pts = [centers[comp == c] for c in range(n_comp)]
        comp_cent = np.array([p.mean(axis=0) for p in comp_pts])

        # Order the components by following the latent dynamics for one period.
        rep = centers[np.argmin(np.linalg.norm(centers - centers.mean(axis=0), axis=1))]
        order: list[int] = []
        z = rep[None, :]
        for _ in range(int(periods[lbl])):
            c = int(np.argmin(np.linalg.norm(comp_cent - z[0], axis=1)))
            if c not in order:
                order.append(c)
            z = np.asarray(advance(z), dtype=np.float64)
        # Only the components the period-length orbit actually visits (the orbit
        # points) -- not every box-cluster, which for a large fragmented Morse
        # set would be hundreds of spurious arrows.
        if len(order) < 2:
            continue

        # One arrow per consecutive pair, anchored at the closest boxes (cyclic).
        for k in range(len(order)):
            arrows.append(_closest_pair(comp_pts[order[k]], comp_pts[order[(k + 1) % len(order)]]))

    written = render_morse_sets_with_overlay(
        csv_path,
        Path(out_dir) / "MG",
        arrows=arrows,
        bounds_lower=bounds_lower,
        bounds_upper=bounds_upper,
        box_scale="auto",
    )
    return written[0] if written else None


def render_extras(
    cfg: ExperimentConfig,
    *,
    train_file: str = "train",
    device: torch.device | str | None = None,
    verbose: bool = True,
    out_dir: Path | None = None,
) -> list[str]:
    """System-specific render hooks; safe no-op for systems without extras."""
    render_device = _resolve_render_device(device)
    name = cfg.system.name
    if name == "leslie3d":
        return _render_leslie3d_extras(
            cfg,
            train_file=train_file,
            device=render_device,
            verbose=verbose,
            out_dir=out_dir,
        )
    return []


LESLIE3D_PERIODIC_PTS: dict[int, list[list[float]]] = {
    0: [
        [102.59382834, 4.62509476, 0.59276684],
        [6.47696572e-02, 7.18156798e01, 3.23756633e00],
        [1.20972812e00, 4.53387600e-02, 5.02709759e01],
        [6.60727793, 0.84680968, 0.03173713],
    ],
    1: [
        [20.09019989, 2.26201326, 21.10982997],
        [14.41254064, 14.06313992, 1.58340928],
        [43.08128567, 10.08877845, 9.84419795],
        [3.23144751, 30.15689997, 7.06214491],
    ],
}


def _make_encode_callable(encoder: torch.nn.Module, scaler, device: torch.device):
    @torch.no_grad()
    def _encode(points: np.ndarray) -> np.ndarray:
        scaled = scaler.transform(points)
        x = torch.as_tensor(scaled, dtype=torch.float32, device=device)
        return encoder(x).cpu().numpy()

    return _encode


def _make_advance_callable(latent_map: torch.nn.Module, device: torch.device):
    @torch.no_grad()
    def _advance(z: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(z, dtype=torch.float32, device=device)
        return latent_map(x).cpu().numpy()

    return _advance


def _render_leslie3d_extras(
    cfg: ExperimentConfig,
    *,
    train_file: str,
    device: torch.device,
    verbose: bool,
    out_dir: Path | None = None,
) -> list[str]:
    """Render the latent-trajectory overlay (paper Fig. 1.214) from saved artifacts."""
    from ..viz import plot_latent_trajectory

    morse_dir = _morse_dir_for(cfg)
    morse_sets_path = morse_dir / "morse_sets"
    if not morse_sets_path.exists():
        return []

    model_dir = cfg.paths.output_dir / "models"
    if not (has_legacy_checkpoint(model_dir) or has_new_checkpoint(model_dir)):
        if verbose:
            print(f"  no checkpoint at {model_dir}; skipping latent trajectory")
        return []

    model, _arch = load_any_checkpoint(model_dir, arch=cfg.arch)
    model.to(device)

    scaler_path = cfg.paths.scaler_path(train_file)
    if not scaler_path.exists():
        if verbose:
            print(f"  no scaler at {scaler_path}; skipping latent trajectory")
        return []
    scaler = joblib.load(scaler_path)

    morse_set_data = np.loadtxt(morse_sets_path, delimiter=",", ndmin=2)
    encode = _make_encode_callable(model.encoder, scaler, device)
    advance = _make_advance_callable(model.latent_map, device)

    figures_root = Path(out_dir) / "figures" if out_dir is not None else cfg.paths.figures_dir
    save_path = figures_root / "latent_trajectory.png"
    plot_latent_trajectory(
        morse_set_data=morse_set_data,
        periodic_pts=LESLIE3D_PERIODIC_PTS,
        encode=encode,
        advance_latent=advance,
        save_path=save_path,
        trajectory_steps=4,
    )
    return [str(save_path)]
