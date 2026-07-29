"""Source-integrity checks for the standardized Chafee--Infante renderer."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_renderer_module():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "render_chafee_infante_standardized.py"
    )
    spec = importlib.util.spec_from_file_location(
        "render_chafee_infante_standardized",
        script,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RENDERER = _load_renderer_module()


def test_default_specs_use_semantic_colors_and_omit_unverified_d2_fine() -> None:
    specs = RENDERER._build_specs(
        d1_dir=RENDERER.DEFAULT_D1_DIR,
        d1_bounds_path=RENDERER.DEFAULT_D1_BOUNDS,
        d2_coarse_dir=RENDERER.DEFAULT_D2_COARSE_DIR,
        d2_manifest_path=RENDERER.DEFAULT_D2_MANIFEST,
        d2_fine_dir=None,
    )

    assert [spec.key for spec in specs] == ["latent_1d", "latent_2d_coarse"]
    one, coarse = specs
    assert (one.negative_label, one.positive_label, one.connecting_labels) == (
        0,
        1,
        (),
    )
    assert (
        coarse.negative_label,
        coarse.positive_label,
        coarse.connecting_labels,
    ) == (1, 0, (2,))
    RENDERER._validate_spec(one)
    RENDERER._validate_spec(coarse)


def test_saved_replay_is_rejected_as_marcio_adaptive_fine_source() -> None:
    replay = (
        RENDERER.CODE_ROOT
        / "replay_sources"
        / "chafee_infante"
        / "replay"
        / "MG"
    )
    specs = RENDERER._build_specs(
        d1_dir=RENDERER.DEFAULT_D1_DIR,
        d1_bounds_path=RENDERER.DEFAULT_D1_BOUNDS,
        d2_coarse_dir=RENDERER.DEFAULT_D2_COARSE_DIR,
        d2_manifest_path=RENDERER.DEFAULT_D2_MANIFEST,
        d2_fine_dir=replay,
    )
    fine = next(spec for spec in specs if spec.key == "latent_2d_fine")

    with pytest.raises(ValueError, match=r"edges do not match.*missing=.*extra="):
        RENDERER._validate_spec(fine)


def test_default_axis_cleanup_preserves_morse_labels() -> None:
    spec = RENDERER._build_specs(
        d1_dir=RENDERER.DEFAULT_D1_DIR,
        d1_bounds_path=RENDERER.DEFAULT_D1_BOUNDS,
        d2_coarse_dir=RENDERER.DEFAULT_D2_COARSE_DIR,
        d2_manifest_path=RENDERER.DEFAULT_D2_MANIFEST,
        d2_fine_dir=None,
    )[0]
    palette = RENDERER.chafee_semantic_palette(
        3,
        negative_label=0,
        positive_label=1,
    )
    plot = RENDERER.plot_morse_sets_from_csv(
        spec.source_dir / "morse_sets",
        bounds_lower=spec.bounds_lower,
        bounds_upper=spec.bounds_upper,
        palette=palette,
    )
    before = [text.get_text() for text in plot.ax.texts]
    RENDERER._hide_axis_annotations(plot)

    assert plot.ax.get_xlabel() == ""
    assert plot.ax.get_ylabel() == ""
    assert plot.ax.get_xticks().size == 0
    assert plot.ax.get_yticks().size == 0
    assert [text.get_text() for text in plot.ax.texts] == before
