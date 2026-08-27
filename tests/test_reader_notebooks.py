"""Structural checks for the Colab notebook set (primer, baselines, drivers)."""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"

#: Driver notebooks and the replay config each one loads in quick/morse mode.
DRIVERS = {
    "02_leslie_2d_contraction.ipynb": "leslie_2gen_contraction_replay",
    "03_leslie3d_example1.ipynb": "leslie3d_example1_replay",
    "04_chafee_infante.ipynb": "chafee_infante_replay",
    "05_coral.ipynb": "coral_basic",
}
ALL = {"00_cmgdb_intro.ipynb", "01_leslie_baselines.ipynb", *DRIVERS}


def _load(name: str) -> tuple[dict, str]:
    notebook = json.loads((NOTEBOOKS / name).read_text())
    source = "".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    return notebook, source


def test_expected_notebook_set():
    assert {path.name for path in NOTEBOOKS.glob("*.ipynb")} == ALL


def test_every_notebook_installs_cmgdb_and_the_repo_for_colab():
    for name in sorted(ALL):
        _, source = _load(name)
        assert "git clone" in source, name  # the repo tree ships the weights
        assert "pip install -q CMGDB" in source, name
        assert "pip install -q -e latent_dynamics" in source, name
        assert "%cd latent_dynamics" in source, name


def test_notebooks_are_clean_of_outputs():
    for name in sorted(ALL):
        notebook, _ = _load(name)
        code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
        assert all(not cell.get("outputs") for cell in code_cells), name
        assert all(cell.get("execution_count") is None for cell in code_cells), name


def test_driver_notebooks_offer_quick_morse_retrain_and_not_replay():
    for name, config in DRIVERS.items():
        _, source = _load(name)
        assert 'MODE = "quick"' in source, name
        assert "QUICK_SUBDIV" in source, name
        assert "SUBDIV = (" in source, name
        assert "load_experiment" in source, name
        assert config in source, name
        assert "retrain(" in source, name
        assert "recompute_morse" in source, name
        assert '"replay"' not in source, name
        assert "make_precomputed_box_map" not in source, name  # fork-era API


def test_baselines_notebook_computes_both_direct_references():
    _, source = _load("01_leslie_baselines.ipynb")
    assert "SUBDIV_2D" in source
    assert "SUBDIV_3D" in source
    assert "(24, 27, 28)" in source  # the 2-D baseline subdivisions, noted beside the preview
    assert "(29, 33, 36)" in source  # published 3-D reference
    assert "ComputeConleyMorseGraph" in source
    assert "PlotMorseSets3D" in source
    assert "MODE" not in source  # the baselines have no saved model to reuse


def test_chafee_driver_keeps_the_exact_roa_option():
    _, source = _load("04_chafee_infante.ipynb")
    assert "COMPUTE_ROA" in source
    assert "compute_roa" in source
