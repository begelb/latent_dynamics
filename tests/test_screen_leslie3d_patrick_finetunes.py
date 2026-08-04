"""Tests for the non-rigorous Patrick fine-tune checkpoint screen."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def _load_module():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "screen_leslie3d_patrick_finetunes.py"
    )
    spec = importlib.util.spec_from_file_location(
        "screen_leslie3d_patrick_finetunes", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SCREEN = _load_module()


def test_canonical_cycle_is_phase_invariant() -> None:
    cycle = np.asarray([[0.8, -0.1], [0.2, 0.7], [-0.4, 0.3], [0.1, -0.8]])
    signature = SCREEN._canonical_cycle(cycle)

    for shift in range(cycle.shape[0]):
        np.testing.assert_allclose(SCREEN._canonical_cycle(np.roll(cycle, shift, axis=0)), signature)


def test_cycle_signature_clustering_separates_near_and_far_cycles() -> None:
    signatures = [
        np.asarray([-0.5, 0.2, 0.5, -0.2]),
        np.asarray([-0.501, 0.201, 0.499, -0.199]),
        np.asarray([-0.1, 0.8, 0.1, -0.8]),
    ]

    clusters = SCREEN._cluster_signatures(signatures, rms_tolerance=0.01)

    assert [cluster["support"] for cluster in clusters] == [2, 1]
    assert clusters[0]["member_indices"] == [0, 1]


class AffineContraction(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("offset", torch.tensor([0.2, -0.1], dtype=torch.float32))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return 0.5 * value + self.offset


class FixedErrorAutoencoder(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> SimpleNamespace:
        z_tau = y[:, :2]
        z_tau_pred = z_tau + 1.0
        return SimpleNamespace(
            x_t=x,
            x_tau=y,
            x_t_hat=x + 1.0,
            x_tau_hat=y + 2.0,
            z_tau=z_tau,
            z_tau_pred=z_tau_pred,
            z_tau_pred_cycle=z_tau_pred + 3.0,
        )


def test_recomputed_loss_ignores_explicit_zero_cycle_weight(tmp_path: Path) -> None:
    x = np.arange(12, dtype=np.float64).reshape(4, 3)
    y = x + 0.5
    data = SCREEN.FixedValidationData(
        x_scaled=x,
        y_scaled=y,
        initial_scaled=x,
        metadata={},
        csv_path=tmp_path / "val.csv",
        metadata_path=tmp_path / "val_metadata.json",
        scaler_path=tmp_path / "scaler",
    )

    losses = SCREEN.recompute_validation_losses(
        FixedErrorAutoencoder(),
        data,
        loss_weights=[2.0, 3.0, 5.0, 0.0],
        device=torch.device("cpu"),
        batch_size=3,
    )

    assert losses["loss_reconstruction"] == pytest.approx(1.0)
    assert losses["loss_prediction"] == pytest.approx(4.0)
    assert losses["loss_semiconjugacy"] == pytest.approx(1.0)
    assert "loss_cycle" not in losses
    assert losses["loss_total"] == pytest.approx(19.0)


def test_root_probe_recovers_and_classifies_stable_fixed_point() -> None:
    result = SCREEN.find_periodic_roots(
        AffineContraction(),
        (np.asarray([-1.0, -1.0]), np.asarray([1.0, 1.0])),
        periods=[1],
        n_grid_starts=9,
        extra_starts=None,
        device=torch.device("cpu"),
        residual_tolerance_relative=1e-5,
        dedupe_tolerance_relative=1e-4,
        hyperbolicity_margin=1e-3,
        max_nfev=50,
    )

    assert result["attempted_solver_calls"] == 9
    assert len(result["cycles"]) == 1
    cycle = result["cycles"][0]
    assert cycle["primitive_period"] == 1
    assert cycle["stability"] == "attractor"
    assert cycle["stable_dimension"] == 2
    assert cycle["unstable_dimension"] == 0
    np.testing.assert_allclose(cycle["representative"], [0.4, -0.2], atol=2e-5)
    np.testing.assert_allclose(
        [entry["modulus"] for entry in cycle["eigenvalues"]],
        [0.5, 0.5],
        atol=1e-7,
    )


def _candidate_record(
    *,
    p4_attractors: int,
    other_attractors: int,
    p2_saddle: bool,
    p1_saddle: bool,
    unclassified: float,
    heldout: float,
) -> dict:
    clusters = [
        {"period": 4, "supported": True} for _ in range(p4_attractors)
    ] + [{"period": 3, "supported": True} for _ in range(other_attractors)]
    roots = [
        {"primitive_period": 4, "stability": "attractor"}
        for _ in range(p4_attractors)
    ]
    if p2_saddle:
        roots.append({"primitive_period": 2, "stability": "saddle"})
    if p1_saddle:
        roots.append({"primitive_period": 1, "stability": "saddle"})
    return {
        "recurrent_orbit_probe": {
            "cycle_clusters": clusters,
            "unclassified_fraction": unclassified,
        },
        "periodic_root_probe": {"cycles": roots},
        "validation_loss_recomputed": {"loss_total": heldout},
    }


def test_ranking_prioritizes_recurrent_pattern_before_loss() -> None:
    topology_match = SCREEN._screening_criteria(
        _candidate_record(
            p4_attractors=2,
            other_attractors=0,
            p2_saddle=True,
            p1_saddle=True,
            unclassified=0.01,
            heldout=0.08,
        )
    )
    lower_loss_wrong_pattern = SCREEN._screening_criteria(
        _candidate_record(
            p4_attractors=1,
            other_attractors=0,
            p2_saddle=True,
            p1_saddle=True,
            unclassified=0.0,
            heldout=0.01,
        )
    )

    assert tuple(topology_match["lexicographic_sort_key"]) < tuple(
        lower_loss_wrong_pattern["lexicographic_sort_key"]
    )


def test_training_summary_extracts_saved_checkpoint_loss(tmp_path: Path) -> None:
    path = tmp_path / "training_summary.json"
    path.write_text(
        json.dumps(
            {
                "best_epoch": 12,
                "initial_val": {"loss_total": 0.08},
                "val": {
                    "loss_total": {"best_epoch_value": 0.04, "final": 0.05},
                    "loss_semiconjugacy": {"best_epoch_value": 0.001},
                },
            }
        ),
        encoding="utf-8",
    )

    report = SCREEN.reported_validation_losses(path)

    assert report is not None
    assert report["best_epoch"] == 12
    assert report["initial_validation"] == {"loss_total": 0.08}
    assert report["saved_checkpoint_validation"] == {
        "loss_total": pytest.approx(0.04),
        "loss_semiconjugacy": pytest.approx(0.001),
    }
    assert report["saved_checkpoint_validation_source"] == "val.*.best_epoch_value"
    assert len(report["sha256"]) == 64


def test_training_summary_prefers_selected_warm_start_baseline(tmp_path: Path) -> None:
    path = tmp_path / "training_summary.json"
    path.write_text(
        json.dumps(
            {
                "best_epoch": -1,
                "best_source": "warm_start_initial",
                "selected_val": {
                    "loss_total": 0.066,
                    "loss_semiconjugacy": 0.00027,
                },
                "initial_val": {
                    "loss_total": 0.066,
                    "loss_semiconjugacy": 0.00027,
                },
                "val": {
                    "loss_total": {"best_epoch_value": float("nan")},
                    "loss_semiconjugacy": {"best_epoch_value": float("nan")},
                },
            }
        ),
        encoding="utf-8",
    )

    report = SCREEN.reported_validation_losses(path)

    assert report is not None
    assert report["best_epoch"] == -1
    assert report["best_source"] == "warm_start_initial"
    assert report["saved_checkpoint_validation_source"] == "selected_val"
    assert report["saved_checkpoint_validation"] == {
        "loss_total": pytest.approx(0.066),
        "loss_semiconjugacy": pytest.approx(0.00027),
    }


def test_parser_defaults_make_non_cmgdb_scope_explicit() -> None:
    args = SCREEN.build_parser().parse_args([])

    assert args.config == "leslie3d_example2_patrick_finetune_4x"
    assert args.root_periods == [1, 2, 4]
    assert args.output is None
    assert "CMGDB" in SCREEN.__doc__
    assert "not" in SCREEN.__doc__.lower()
