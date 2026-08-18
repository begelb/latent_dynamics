"""Structural checks for the small, reader-facing Colab notebook set."""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"
EXPECTED = {
    "00_cmgdb_intro.ipynb": None,
    "01_leslie_2d_contraction.ipynb": "leslie_2gen_contraction_replay",
    "02_leslie3d_example1.ipynb": "leslie3d_example1_replay",
    "03_coral.ipynb": "coral_basic",
    "04_chafee_infante.ipynb": "chafee_infante_replay",
}


def _source(notebook: dict) -> str:
    return "".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_exactly_one_notebook_per_paper_example_plus_primer():
    assert {path.name for path in NOTEBOOKS.glob("*.ipynb")} == set(EXPECTED)


def test_notebooks_are_small_clean_replays_with_fresh_colab_setup():
    for name, config in EXPECTED.items():
        notebook = json.loads((NOTEBOOKS / name).read_text())
        cells = notebook["cells"]
        source = _source(notebook)

        assert len(cells) <= 12, name
        assert sum(len(cell.get("source", [])) for cell in cells if cell["cell_type"] == "code") <= 55
        assert all(not cell.get("outputs") for cell in cells if cell["cell_type"] == "code")
        assert all(cell.get("execution_count") is None for cell in cells if cell["cell_type"] == "code")

        assert "git clone" not in source  # subprocess arguments avoid shell/magic state
        assert '"git", "clone"' in source
        assert '"-e", str(root)' in source
        assert "cmgdb==1.3.3+fork.3" in source
        assert "MODE =" not in source
        assert "SUBDIV =" not in source

        if config is not None:
            assert "load_experiment" in source
            assert config in source
            assert "retrain(" not in source


def test_chafee_notebook_covers_all_dimensions_and_static_bifurcation_decision():
    notebook = json.loads((NOTEBOOKS / "04_chafee_infante.ipynb").read_text())
    source = _source(notebook)
    for expected in ("latent_1d", "latent_3d", "coarsened", "updated_paper_statistics"):
        assert expected in source
    assert "bifurcation diagram is intentionally treated as a" in source
    assert "does not claim to reproduce it" in source
