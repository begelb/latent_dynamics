"""Tests for the dense Ives sampled residual/tolerance audit."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# The audit script is run on demand with ``uv run --with shapely``; Shapely is
# deliberately not a project dependency, so skip rather than fail without it.
pytest.importorskip("shapely")

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "compute_ives_myvatn_sampled_residual_tolerance.py"
)
SPEC = importlib.util.spec_from_file_location("compute_ives_sampled_metrics", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
METRICS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = METRICS
SPEC.loader.exec_module(METRICS)


class _AffineInward(torch.nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return 0.5 * values + 0.25


class _Outside(torch.nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + 2.0


class _Identity(torch.nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values


class _Zero(torch.nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(values)


class _Model:
    def __init__(self, *, encoder: torch.nn.Module, latent_map: torch.nn.Module) -> None:
        self.encoder = encoder
        self.latent_map = latent_map


class _IdentityScaler:
    @staticmethod
    def transform(values: np.ndarray) -> np.ndarray:
        return np.asarray(values)


def test_minimal_labels_are_nodes_without_outgoing_edges(tmp_path: Path) -> None:
    graph = tmp_path / "morse_graph"
    graph.write_text(
        "digraph {\n"
        '0 [label="0"];\n'
        '1 [label="1"];\n'
        '2 [label="2"];\n'
        "1 -> 0;\n"
        "1 -> 2;\n"
        "}\n",
        encoding="utf-8",
    )

    assert METRICS.minimal_labels(graph) == [0, 2]


def test_block_geometry_uses_union_boundary_and_closed_membership() -> None:
    block = METRICS.BlockGeometry(
        np.asarray(
            [
                [0.0, 0.0, 1.0, 1.0],
                [1.0, 0.0, 2.0, 1.0],
            ]
        )
    )
    points = np.asarray(
        [
            [1.0, 0.5],  # shared cell face is interior to the union
            [0.0, 0.5],  # outer boundary
            [2.5, 0.5],  # outside
        ]
    )

    assert block.membership(points).tolist() == [True, True, False]
    assert block.membership(points, interior=True).tolist() == [True, False, False]
    np.testing.assert_allclose(block.clearance(points), [0.5, 0.0, 0.0])


def test_dense_tolerance_sampler_covers_target_and_returns_unsquared_distance() -> None:
    block = METRICS.BlockGeometry(np.asarray([[0.0, 0.0, 1.0, 1.0]]))
    model = _Model(encoder=_Identity(), latent_map=_AffineInward())

    result = METRICS.sample_tolerance(
        model=model,
        block=block,
        target_points=100,
        sobol_scrambles=2,
        local_boxes=0,
        sobol_seed=7,
    )

    assert result["explicit_latent_samples"] >= 100
    assert result["sampling"]["deterministic_corners_and_centers"] == 5
    assert result["all_explicit_sample_images_in_interior"] is True
    assert np.isclose(result["sampled_minimum"], 0.25)


def test_tolerance_clearance_is_zero_when_an_image_leaves_block() -> None:
    block = METRICS.BlockGeometry(np.asarray([[0.0, 0.0, 1.0, 1.0]]))
    model = _Model(encoder=_Identity(), latent_map=_Outside())

    result = METRICS.sample_tolerance(
        model=model,
        block=block,
        target_points=5,
        sobol_scrambles=1,
        local_boxes=0,
        sobol_seed=7,
    )

    assert result["sampled_minimum"] == 0.0
    assert result["all_explicit_sample_images_in_interior"] is False


def test_residual_update_uses_closed_block_membership_and_l2_norm() -> None:
    block = METRICS.BlockGeometry(np.asarray([[0.0, 0.0, 1.0, 1.0]]))
    model = _Model(encoder=_Identity(), latent_map=_Zero())
    stats = {0: METRICS.empty_residual_stats()}

    METRICS.update_residual_stats(
        raw_x=np.asarray([[0.0, 0.5]]),  # on block boundary: accepted
        raw_y=np.asarray([[3.0, 4.0]]),
        source="synthetic",
        source_offset=0,
        model=model,
        scaler=_IdentityScaler(),
        blocks={0: block},
        stats=stats,
    )

    assert stats[0]["accepted_samples"] == 1
    assert stats[0]["sampled_maximum"] == 5.0
    assert stats[0]["squared_value_diagnostic"] == 25.0


def test_paper_crosscheck_preserves_all_table_rows_and_direction() -> None:
    rows = METRICS.paper_crosscheck()

    assert len(rows) == 15
    chafee_d2 = [row for row in rows if row["example"] == "Chafee--Infante d=2"]
    assert len(chafee_d2) == 2
    assert all(row["sampled_inequality_holds_as_displayed"] for row in chafee_d2)
