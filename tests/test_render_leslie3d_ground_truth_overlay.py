"""Focused tests for the parameterized Leslie3D ground-truth overlay."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "render_leslie3d_ground_truth_overlay.py"
SPEC = importlib.util.spec_from_file_location("render_leslie3d_ground_truth_overlay", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RENDERER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RENDERER)


def _summary(nodes: dict[str, int]) -> dict:
    return {
        "morse_membership": {
            name: {
                "assigned_morse_node": node,
                "phase_containing_nodes": [[node]],
            }
            for name, node in nodes.items()
        }
    }


def test_role_assignments_are_read_from_analyzer_not_fixed_ids() -> None:
    nodes = {
        "P0": 19,
        "P1": 3,
        "S2": 41,
        "S4": 7,
        "p_star": 23,
        "origin": 101,
    }
    assert RENDERER._role_assignments(_summary(nodes)) == nodes


def test_membership_recheck_rejects_stale_analyzer_nodes() -> None:
    nodes = {name: index + 10 for index, name in enumerate(RENDERER.OBJECT_ORDER)}
    data = np.asarray(
        [[index, 0.0, index + 0.9, 1.0, nodes[name]] for index, name in enumerate(RENDERER.OBJECT_ORDER)],
        dtype=np.float64,
    )
    encoded = {
        name: np.asarray([[index + 0.5, 0.5]], dtype=np.float64)
        for index, name in enumerate(RENDERER.OBJECT_ORDER)
    }
    summary = _summary(nodes)
    RENDERER._verify_analyzer_membership(data, encoded, summary)
    summary["morse_membership"]["P0"]["phase_containing_nodes"] = [[999]]
    try:
        RENDERER._verify_analyzer_membership(data, encoded, summary)
    except RuntimeError as exc:
        assert "stale" in str(exc)
    else:  # pragma: no cover - protects the provenance gate itself.
        raise AssertionError("stale analyzer membership was accepted")


def test_rasterizer_visits_every_raw_box_without_sampling() -> None:
    data = np.asarray(
        [
            [0.0, 0.0, 0.25, 0.5, 0],
            [0.25, 0.0, 0.5, 0.5, 1],
            [0.5, 0.5, 1.0, 1.0, 2],
        ],
        dtype=np.float64,
    )
    image, extent, metadata = RENDERER._rasterize_morse_boxes(
        data,
        palette=("#ff0000", "#00ff00", "#0000ff"),
        max_pixels=512,
        alpha=255,
    )
    assert extent == (0.0, 1.0, 0.0, 1.0)
    assert metadata["input_rows"] == 3
    assert metadata["rows_visited"] == 3
    assert metadata["row_sampling"] == "none"
    pixels = np.asarray(image)
    observed_rgb = {
        tuple(value)
        for value in np.unique(pixels.reshape(-1, 4), axis=0)
        if value[3] != 0
    }
    assert observed_rgb == {
        (255, 0, 0, 255),
        (0, 255, 0, 255),
        (0, 0, 255, 255),
    }
