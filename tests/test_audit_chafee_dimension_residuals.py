"""Focused tests for the persisted Chafee--Infante residual audit."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


def _load_audit_module():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "audit_chafee_dimension_residuals.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_chafee_dimension_residuals",
        script,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


AUDIT = _load_audit_module()


def test_dimension_specs_fix_physical_node_mapping() -> None:
    assert AUDIT.DIMENSION_SPECS[1].physical_nodes == {
        "negative": 0,
        "positive": 1,
    }
    assert AUDIT.DIMENSION_SPECS[2].physical_nodes == {
        "negative": 1,
        "positive": 0,
    }
    assert AUDIT.DIMENSION_SPECS[3].physical_nodes == {
        "negative": 0,
        "positive": 1,
    }


def test_membership_mask_uses_closed_union_of_boxes() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
        ]
    )
    boxes = np.asarray(
        [
            [0.0, 0.0, 1.0, 1.0],
            [2.0, 2.0, 2.5, 2.5],
        ]
    )

    mask = AUDIT._membership_mask(points, boxes)

    np.testing.assert_array_equal(mask.numpy(), [True, True, True, False])


def test_physical_mapping_validation_rejects_swapped_nodes() -> None:
    class IdentityEncoder(torch.nn.Module):
        def forward(self, x):
            return x

    model = torch.nn.Module()
    model.encoder = IdentityEncoder()
    stable_roots = np.asarray([[-1.0], [1.0]])
    blocks = {
        0: np.asarray([[-1.1, -0.9]]),
        1: np.asarray([[0.9, 1.1]]),
    }

    passed = AUDIT._validate_physical_node_mapping(
        model,
        stable_roots,
        blocks,
        {"negative": 0, "positive": 1},
    )
    assert passed["passed"] is True

    with pytest.raises(ValueError, match="negative stable root"):
        AUDIT._validate_physical_node_mapping(
            model,
            stable_roots,
            blocks,
            {"negative": 1, "positive": 0},
        )


def test_stored_pair_summary_reports_block_and_global_witnesses(
    monkeypatch,
) -> None:
    monkeypatch.setattr(AUDIT, "HIGH_DIMENSION", 2)
    monkeypatch.setattr(AUDIT, "STEPS_PER_TRAJECTORY", 2)
    raw = np.asarray(
        [
            [-1.0, 0.0, -0.5, 0.0],
            [-0.5, 0.0, 0.0, 0.0],
            [0.5, 0.0, 1.0, 0.0],
            [1.0, 0.0, 1.5, 0.0],
        ]
    )
    evaluation = AUDIT.ModelEvaluation(
        reconstruction_error=torch.zeros((4, 2)),
        prediction_error=torch.zeros((4, 2)),
        encoded_current=torch.tensor([[-1.0], [-0.5], [0.5], [1.0]]),
        encoded_next=torch.zeros((4, 1)),
        predicted_latent=torch.tensor([[0.1], [0.2], [0.3], [0.4]]),
    )
    blocks = {
        0: np.asarray([[-1.1, -0.4]]),
        1: np.asarray([[0.4, 1.1]]),
    }

    physical, global_result = AUDIT._summarize_stored_pair_residuals(
        raw,
        evaluation,
        blocks,
        {"negative": 0, "positive": 1},
    )

    assert physical["negative"]["accepted_pairs"] == 2
    assert physical["negative"]["witness"]["row_index_zero_based"] == 1
    assert physical["positive"]["accepted_pairs"] == 2
    assert physical["positive"]["witness"]["row_index_zero_based"] == 3
    assert global_result["witness"]["row_index_zero_based"] == 3
    assert global_result["witness"]["trajectory_index_zero_based"] == 1
    assert global_result["witness"]["step_index_zero_based"] == 1


def test_dense_reference_marks_one_and_three_dimensions_not_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    payload = {
        "example": "chafee_infante_current",
        "metric": "Euclidean",
        "nodes": {
            "0": {
                "n_boxes": 2,
                "residual": {
                    "evaluated_samples": 10,
                    "accepted_samples": 4,
                    "sampled_maximum": 0.2,
                    "witness": {"source": "positive"},
                },
            },
            "1": {
                "n_boxes": 3,
                "residual": {
                    "evaluated_samples": 10,
                    "accepted_samples": 5,
                    "sampled_maximum": 0.1,
                    "witness": {"source": "negative"},
                },
            },
        },
    }
    path = tmp_path / "dense.json"
    path.write_text(json.dumps(payload))
    monkeypatch.setitem(
        AUDIT.EXPECTED_SHA256,
        "dense_d2_result",
        AUDIT._sha256(path),
    )

    result = AUDIT._dense_d2_reference(path)

    assert result["status_by_dimension"] == {
        "1": "not_run",
        "2": "existing_result_referenced_not_recomputed",
        "3": "not_run",
    }
    assert (
        result["dimension_2_by_physical_attractor"]["negative"][
            "sampled_max_euclidean_residual"
        ]
        == 0.1
    )
