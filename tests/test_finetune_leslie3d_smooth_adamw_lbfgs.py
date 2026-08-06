"""Focused safety tests for the Leslie3D AdamW/L-BFGS fine-tuner."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from scripts import finetune_leslie3d_smooth_adamw_lbfgs as finetune
from torch import nn


def test_cosine_learning_rate_has_expected_endpoints_and_midpoint() -> None:
    start = 1.0e-7
    end = 2.0e-8

    assert finetune._cosine_learning_rate(start, end, 1, 5) == pytest.approx(start)
    assert finetune._cosine_learning_rate(start, end, 3, 5) == pytest.approx(
        0.5 * (start + end)
    )
    assert finetune._cosine_learning_rate(start, end, 5, 5) == pytest.approx(end)
    assert finetune._cosine_learning_rate(start, end, 1, 1) == pytest.approx(end)


def test_paths_overlap_detects_equal_and_nested_paths(tmp_path: Path) -> None:
    source = tmp_path / "accepted_source"
    nested = source / "candidate"
    sibling = tmp_path / "accepted_source_copy"

    assert finetune._paths_overlap(source, source)
    assert finetune._paths_overlap(source, nested)
    assert finetune._paths_overlap(nested, source)
    assert not finetune._paths_overlap(source, sibling)


def test_gradient_guard_accepts_finite_and_rejects_invalid_gradients() -> None:
    model = nn.Module()
    model.latent_map = nn.Linear(2, 2)
    parameters = list(model.latent_map.parameters())
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)

    finetune._assert_finite_gradients(model, stage="test")

    parameters[0].grad = None
    with pytest.raises(FloatingPointError, match=r"missing=\['weight'\]"):
        finetune._assert_finite_gradients(model, stage="test")

    parameters[0].grad = torch.ones_like(parameters[0])
    parameters[0].grad[0, 0] = torch.nan
    with pytest.raises(FloatingPointError, match=r"nonfinite=\['weight'\]"):
        finetune._assert_finite_gradients(model, stage="test")


def test_parser_rejects_removed_force_overwrite_option() -> None:
    parser = finetune._parser()

    assert "--force-overwrite" not in parser.format_help()
    with pytest.raises(SystemExit) as error:
        parser.parse_args(["--force-overwrite"])
    assert error.value.code == 2
