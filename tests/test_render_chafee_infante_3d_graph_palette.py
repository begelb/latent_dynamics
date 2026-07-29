from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "render_chafee_infante_3d_graph_palette.py"
)
SPEC = importlib.util.spec_from_file_location(
    "render_chafee_infante_3d_graph_palette",
    SCRIPT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_graph_palette_is_read_by_exact_node_label(tmp_path: Path) -> None:
    dot = tmp_path / "morse_graph"
    dot.write_text(
        "\n".join(
            (
                "digraph {",
                '1 [fillcolor="#DC267FFF"];',
                '0 [fillcolor="#FFB000FF"];',
                "1 -> 0;",
                "}",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    palette = MODULE.graph_palette_from_dot(
        dot,
        expected_labels=frozenset({0, 1}),
    )

    assert palette == ("#FFB000", "#DC267F")


def test_graph_palette_rejects_graph_set_label_mismatch(tmp_path: Path) -> None:
    dot = tmp_path / "morse_graph"
    dot.write_text(
        'digraph { 0 [fillcolor="#FFB000FF"]; }\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="do not match"):
        MODULE.graph_palette_from_dot(
            dot,
            expected_labels=frozenset({0, 1}),
        )


def test_validate_morse_sets_requires_three_dimensions(tmp_path: Path) -> None:
    path = tmp_path / "morse_sets"
    np.savetxt(path, np.zeros((1, 5)), delimiter=",")

    with pytest.raises(ValueError, match="seven columns"):
        MODULE._validate_morse_sets(path)


def test_load_bounds_requires_three_dimensions(tmp_path: Path) -> None:
    path = tmp_path / "bounds.json"
    path.write_text(
        '{"lower": [-1, -1], "upper": [1, 1]}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="three-dimensional"):
        MODULE._load_bounds(path)


def test_persisted_fine_three_dimensional_palette_matches_saved_graph() -> None:
    source = MODULE.DEFAULT_SOURCE
    labels, rows = MODULE._validate_morse_sets(source / "morse_sets")
    palette = MODULE.graph_palette_from_dot(
        source / "morse_graph",
        expected_labels=labels,
    )

    assert labels == frozenset(range(11))
    assert rows == 30_672
    assert palette == (
        "#FFB000",
        "#DC267F",
        "#648FFF",
        "#FE6100",
        "#785EF0",
        "#008080",
        "#FCC2E8",
        "#FFB000",
        "#DC267F",
        "#648FFF",
        "#FE6100",
    )


def test_persisted_fine_three_dimensional_heights_match_reference() -> None:
    heights = MODULE.morse_heights_from_dot(MODULE.DEFAULT_SOURCE / "morse_graph")

    assert {
        height: frozenset(
            label
            for label, actual_height in heights.items()
            if actual_height == height
        )
        for height in sorted(set(heights.values()))
    } == MODULE.EXPECTED_HEIGHT_GROUPS


def test_persisted_fine_three_dimensional_level_palette_matches_reference() -> None:
    source = MODULE.DEFAULT_SOURCE
    labels, _ = MODULE._validate_morse_sets(source / "morse_sets")

    palette = MODULE.chafee_level_palette_from_dot(
        source / "morse_graph",
        expected_labels=labels,
    )

    assert palette == (
        "#FFB000",
        "#DC267F",
        "#648FFF",
        "#648FFF",
        "#FE6100",
        "#785EF0",
        "#008080",
        "#FE6100",
        "#785EF0",
        "#008080",
        "#FCC2E8",
    )
