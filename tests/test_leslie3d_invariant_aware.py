from __future__ import annotations

import json
import runpy
from pathlib import Path

import numpy as np
import pytest
import torch

from latentdynamics.config import load_config
from latentdynamics.models import build_autoencoder

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILDER = runpy.run_path(str(REPO_ROOT / "scripts/build_leslie3d_invariant_aware_dataset.py"))
ANALYZER = runpy.run_path(str(REPO_ROOT / "scripts/analyze_leslie3d_invariant_aware.py"))
SMOOTH_TRAINER = runpy.run_path(str(REPO_ROOT / "scripts/train_leslie3d_smooth_topology.py"))


def test_inventory_has_every_known_recurrent_phase_and_exact_successor() -> None:
    objects = BUILDER["_object_arrays"]()
    assert {name: len(points) for name, points in objects.items()} == {
        "P0": 4,
        "P1": 4,
        "S2": 2,
        "S4": 4,
        "p_star": 1,
        "origin": 1,
    }
    errors = BUILDER["_phase_step_error"](objects)
    assert max(errors.values()) < 1e-10


def test_catalogued_saddles_have_a_resolved_unstable_direction() -> None:
    objects = BUILDER["_object_arrays"]()
    for name in ("S2", "S4", "p_star"):
        direction, multiplier = BUILDER["_unstable_direction"](objects[name])
        assert np.isfinite(direction).all()
        assert np.isclose(np.linalg.norm(direction / BUILDER["UPPER"]), 1.0)
        assert abs(multiplier) > 1.0


def test_invariant_aware_config_preserves_warm_start_coordinates() -> None:
    cfg = load_config("leslie3d_invariant_aware")
    assert cfg.arch.high_dims == 3
    assert cfg.arch.low_dims == 2
    assert cfg.training.loss_weights == [100.0, 20.0, 100.0, 20.0]
    assert str(cfg.training.warm_start_checkpoint_dir).endswith(
        "replay_sources/leslie3d_example2/models"
    )
    assert cfg.paths.flat_scaler is True
    assert cfg.paths.scaler_read_only is True
    assert cfg.seeds == [20260803]

    refined = load_config("leslie3d_invariant_aware_refined")
    assert str(refined.training.warm_start_checkpoint_dir).endswith(
        "output/leslie3d_invariant_aware/seed_20260803/models"
    )
    assert refined.paths.output_dir != cfg.paths.output_dir
    assert refined.seeds == [20260804]


def test_smooth_config_changes_only_the_latent_hidden_activation() -> None:
    base = load_config("leslie3d_invariant_aware")
    smooth = load_config("leslie3d_invariant_aware_smooth")

    assert smooth.arch.component("encoder") == base.arch.component("encoder")
    assert smooth.arch.component("decoder") == base.arch.component("decoder")
    assert smooth.arch.component("latent_map").hidden_shapes == (64, 64)
    assert smooth.arch.component("latent_map").activation == "gelu"
    assert base.arch.component("latent_map").activation == "relu"
    assert smooth.arch.component("latent_map").out_activation == "tanh"
    assert str(smooth.training.warm_start_checkpoint_dir).endswith(
        "output/leslie3d_invariant_aware/seed_20260803/models"
    )
    assert smooth.paths.output_dir not in {
        base.paths.output_dir,
        load_config("leslie3d_invariant_aware_refined").paths.output_dir,
    }
    assert smooth.seeds == [20260805]


def test_sharp_gelu_transfer_preserves_the_relu_map_to_small_error() -> None:
    torch.manual_seed(12)
    source = build_autoencoder(load_config("leslie3d_invariant_aware").arch)
    target = build_autoencoder(load_config("leslie3d_invariant_aware_smooth").arch)
    transferred = SMOOTH_TRAINER["_transfer_components"](
        source,
        target,
        gelu_sharpness=100.0,
    )
    points = torch.tensor([[-1.0, -1.0], [-0.5, 0.25], [0.0, 0.0], [0.3, -0.7], [1.0, 1.0]])

    with torch.no_grad():
        error = torch.max(torch.abs(source.latent_map(points) - target.latent_map(points)))

    assert transferred == sorted(target.latent_map.state_dict())
    assert float(error) < 5e-4
    for name, value in source.encoder.state_dict().items():
        assert torch.equal(value, target.encoder.state_dict()[name])
    for name, value in source.decoder.state_dict().items():
        assert torch.equal(value, target.decoder.state_dict()[name])


def test_analyzer_refuses_a_stale_promotion_after_rejected_smooth_run(
    tmp_path: Path,
) -> None:
    (tmp_path / "smooth_topology_summary.json").write_text(
        json.dumps(
            {
                "status": "rejected_by_strict_numerical_gates_candidate_only",
                "promoted_checkpoint": None,
            }
        )
    )

    with pytest.raises(RuntimeError, match="possibly stale promoted checkpoint"):
        ANALYZER["_validate_smooth_promotion"](tmp_path)


def test_analyzer_verifies_promoted_smooth_checkpoint_hashes(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    checkpoint = models / "autoencoder.pt"
    sidecar = models / "autoencoder.json"
    checkpoint.write_bytes(b"checkpoint")
    sidecar.write_text('{"version": 1}')
    hashes = {path.name: ANALYZER["_sha256"](path) for path in (checkpoint, sidecar)}
    (tmp_path / "smooth_topology_summary.json").write_text(
        json.dumps(
            {
                "status": "accepted_numerical_candidate_not_a_conley_certificate",
                "promoted_checkpoint": [str(checkpoint), str(sidecar)],
                "promoted_checkpoint_sha256": hashes,
            }
        )
    )

    provenance = ANALYZER["_validate_smooth_promotion"](tmp_path)
    assert provenance is not None
    assert provenance["promoted_checkpoint_sha256"] == hashes


def test_analyzer_can_verify_rejected_smooth_candidate_without_promoting_it(
    tmp_path: Path,
) -> None:
    models = tmp_path / "models"
    models.mkdir()
    checkpoint = models / "smooth_candidate.pt"
    sidecar = models / "smooth_candidate.json"
    checkpoint.write_bytes(b"candidate checkpoint")
    sidecar.write_text('{"version": 1}')
    hashes = {path.name: ANALYZER["_sha256"](path) for path in (checkpoint, sidecar)}
    (tmp_path / "smooth_topology_summary.json").write_text(
        json.dumps(
            {
                "status": "rejected_by_strict_numerical_gates_candidate_only",
                "candidate_checkpoint_sha256": hashes,
                "promoted_checkpoint": None,
            }
        )
    )

    provenance = ANALYZER["_validate_smooth_candidate"](tmp_path)

    assert provenance is not None
    assert provenance["status"] == "rejected_by_strict_numerical_gates_candidate_only"
    assert provenance["checkpoint_role"] == "candidate_not_assumed_promoted"
    assert provenance["candidate_checkpoint_sha256"] == hashes


def test_analyzer_checkpoint_artifacts_keep_candidate_separate() -> None:
    suffix = ANALYZER["_checkpoint_artifact_suffix"]

    assert suffix("autoencoder") == ""
    assert suffix("smooth_candidate") == "_smooth_candidate"
    with pytest.raises(ValueError, match="plain non-empty"):
        suffix("../smooth_candidate")


def test_recurrent_exclusion_defaults_disabled_and_uses_dividing_period_roles() -> None:
    assert SMOOTH_TRAINER["DEFAULT_RECURRENT_EXCLUSION_WEIGHT"] == 0.0
    targets = {
        name: torch.full((period, 2), float(index))
        for index, (name, period) in enumerate(SMOOTH_TRAINER["PERIODS"].items(), start=1)
    }

    allowed, labels = SMOOTH_TRAINER["_allowed_recurrent_anchor_sets"](targets)

    assert labels == {
        "1": ["p_star", "origin"],
        "2": ["S2", "p_star", "origin"],
        "4": ["P0", "P1", "S2", "S4", "p_star", "origin"],
    }
    assert {period: len(points) for period, points in allowed.items()} == {
        1: 2,
        2: 4,
        4: 16,
    }


def test_trajectory_shadowing_defaults_disabled_with_odd_even_long_horizons() -> None:
    assert SMOOTH_TRAINER["DEFAULT_TRAJECTORY_SHADOWING_WEIGHT"] == 0.0
    groups = SMOOTH_TRAINER["DEFAULT_TRAJECTORY_SHADOWING_HORIZON_GROUPS"]

    assert groups["short"] == (1, 2, 3, 4)
    assert groups["medium"] == (7, 8, 15, 16, 31, 32)
    assert groups["long"] == (63, 64, 127, 128, 255, 256, 319, 320)
    assert any(horizon % 2 for horizons in groups.values() for horizon in horizons)
    assert any(horizon % 2 == 0 for horizons in groups.values() for horizon in horizons)


def test_trajectory_shadowing_reconstructs_time_major_blocks_and_filters_horizons() -> None:
    short_states = torch.arange(65 * 2 * 3, dtype=torch.float32).reshape(65, 2, 3)
    long_states = 1000.0 + torch.arange(321 * 3, dtype=torch.float32).reshape(321, 1, 3)
    x = torch.cat((short_states[:-1].reshape(-1, 3), long_states[:-1].reshape(-1, 3)))
    y = torch.cat((short_states[1:].reshape(-1, 3), long_states[1:].reshape(-1, 3)))
    encoded_x = (0.01 * x[:, :2]).requires_grad_()
    encoded_y = (0.01 * y[:, :2]).requires_grad_()
    metadata = {
        "components": [
            {
                "name": "saddle_tangent_transition_tubes",
                "row_start_inclusive": 0,
                "row_stop_exclusive": 128,
                "rows": 128,
                "trajectories": 2,
                "steps": 64,
            },
            {
                "name": "audited_origin_p_star_s2_transition_tubes",
                "row_start_inclusive": 128,
                "row_stop_exclusive": 448,
                "rows": 320,
                "trajectories": 1,
                "steps": 320,
            },
        ]
    }

    blocks, report = SMOOTH_TRAINER["_prepare_trajectory_shadowing_blocks"](
        x,
        y,
        encoded_x,
        encoded_y,
        metadata,
        SMOOTH_TRAINER["DEFAULT_TRAJECTORY_SHADOWING_HORIZON_GROUPS"],
        split_name="test",
    )

    assert [block["name"] for block in blocks] == [
        "saddle_tangent_transition_tubes",
        "audited_origin_p_star_s2_transition_tubes",
    ]
    assert blocks[0]["latent_states"].shape == (65, 2, 2)
    assert blocks[1]["latent_states"].shape == (321, 1, 2)
    assert torch.equal(blocks[0]["latent_states"], 0.01 * short_states[:, :, :2])
    assert torch.equal(blocks[1]["latent_states"], 0.01 * long_states[:, :, :2])
    assert blocks[0]["latent_states"].requires_grad is False
    assert blocks[0]["horizon_groups"]["long"] == (63, 64)
    assert blocks[1]["horizon_groups"]["long"][-2:] == (319, 320)
    assert report["row_order"] == "time_major_step_then_trajectory"
    assert report["components"]["saddle_tangent_transition_tubes"][
        "scaled_state_continuity_max_abs"
    ] == 0.0

    broken_y = y.clone()
    broken_y[0, 0] += 1.0
    with pytest.raises(ValueError, match="continuous time-major trajectory block"):
        SMOOTH_TRAINER["_prepare_trajectory_shadowing_blocks"](
            x,
            broken_y,
            encoded_x,
            encoded_y,
            metadata,
            {"short": (1, 2)},
            split_name="broken",
        )


def test_trajectory_shadowing_balances_components_and_trains_only_the_map() -> None:
    latent_map = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        latent_map.weight.zero_()
    blocks = [
        {
            "name": "saddle_tangent_transition_tubes",
            "steps": 4,
            "trajectories": 1,
            "latent_states": torch.ones((5, 1, 2)),
            "horizon_groups": {"short": (1, 2)},
        },
        {
            "name": "audited_origin_p_star_s2_transition_tubes",
            "steps": 320,
            "trajectories": 27,
            "latent_states": 3.0 * torch.ones((321, 27, 2)),
            "horizon_groups": {"short": (1, 2)},
        },
    ]

    loss, report = SMOOTH_TRAINER["_trajectory_shadowing_loss"](
        latent_map,
        blocks,
        diagnostics=True,
    )
    loss.backward()

    assert float(loss.detach()) == pytest.approx(5.0)
    assert report["components"]["saddle_tangent_transition_tubes"][
        "balanced_mse"
    ] == pytest.approx(1.0)
    assert report["components"]["audited_origin_p_star_s2_transition_tubes"][
        "balanced_mse"
    ] == pytest.approx(9.0)
    assert report["components"]["audited_origin_p_star_s2_transition_tubes"][
        "horizons"
    ]["1"]["starting_windows"] == 320 * 27
    assert latent_map.weight.grad is not None
    assert torch.isfinite(latent_map.weight.grad).all()
    assert float(torch.linalg.vector_norm(latent_map.weight.grad)) > 0.0
    assert all(block["latent_states"].grad is None for block in blocks)


def test_recurrent_exclusion_probes_are_deterministic_global_and_local() -> None:
    lower = torch.tensor([-0.4, -0.3])
    upper = torch.tensor([0.3, 0.4])
    anchors = torch.tensor([[-0.1, -0.1], [0.1, 0.1]])
    kwargs = {
        "global_count": 32,
        "local_radius_count": 4,
        "local_direction_count": 8,
        "local_min_radius": 2e-5,
        "local_max_radius": 2e-3,
        "seed": 101,
    }

    first, first_meta = SMOOTH_TRAINER["_sample_recurrent_exclusion_probes"](
        lower, upper, anchors, **kwargs
    )
    second, second_meta = SMOOTH_TRAINER["_sample_recurrent_exclusion_probes"](
        lower, upper, anchors, **kwargs
    )

    assert torch.equal(first, second)
    assert first_meta == second_meta
    assert first_meta["global_count"] == 32
    assert first_meta["local_retained_count"] == 2 * 4 * 8
    assert len(first) == 32 + 2 * 4 * 8
    assert torch.all(first >= lower) and torch.all(first <= upper)
    local = first[32:].reshape(2, 4, 8, 2)
    radii = torch.linalg.vector_norm(local - anchors[:, None, None, :], dim=3)
    assert torch.allclose(radii[:, 0], torch.full((2, 8), 2e-5), atol=2e-7)
    assert torch.allclose(radii[:, -1], torch.full((2, 8), 2e-3), atol=2e-7)


def test_recurrent_exclusion_score_is_differentiable_and_strictly_gated() -> None:
    latent_map = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        latent_map.weight.copy_(0.95 * torch.eye(2))
    probes = torch.tensor([[0.10, 0.10], [0.20, -0.10], [-0.15, 0.12]])
    allowed = {
        period: torch.zeros((1, 2)) for period in SMOOTH_TRAINER["RECURRENT_EXCLUSION_PERIODS"]
    }

    loss, report = SMOOTH_TRAINER["_recurrent_exclusion_term"](
        latent_map,
        probes,
        allowed,
        core_radius=1e-5,
        distance_epsilon=1e-8,
        score_margin=0.10,
        temperature=0.01,
        optimization_probe_count=None,
        diagnostics=True,
    )
    loss.backward()

    assert float(loss.detach()) > 0.0
    assert latent_map.weight.grad is not None
    assert torch.isfinite(latent_map.weight.grad).all()
    assert float(torch.linalg.vector_norm(latent_map.weight.grad)) > 0.0
    assert set(report["by_period"]) == {"1", "2", "4"}
    for diagnostic in report["by_period"].values():
        assert diagnostic["minimum_return_distance_score"] > 0.0
        assert (
            diagnostic["q01_return_distance_score"] >= diagnostic["minimum_return_distance_score"]
        )

    passed, no_violation = SMOOTH_TRAINER["_recurrent_exclusion_gate"](
        report, enabled=True, acceptance_score=0.01
    )
    failed, violation = SMOOTH_TRAINER["_recurrent_exclusion_gate"](
        report, enabled=True, acceptance_score=0.10
    )
    assert passed is True and no_violation == 0.0
    assert failed is False and violation > 0.0

    with torch.no_grad():
        latent_map.weight.copy_(torch.eye(2))
    _identity_loss, identity_report = SMOOTH_TRAINER["_recurrent_exclusion_term"](
        latent_map,
        probes,
        allowed,
        core_radius=1e-5,
        distance_epsilon=1e-8,
        score_margin=0.10,
        temperature=0.01,
        optimization_probe_count=None,
        diagnostics=True,
    )
    identity_passed, _identity_violation = SMOOTH_TRAINER["_recurrent_exclusion_gate"](
        identity_report, enabled=True, acceptance_score=0.005
    )
    assert identity_report["minimum_return_distance_score_over_all_periods"] == 0.0
    assert identity_passed is False


def test_manifest_component_weights_are_explicit_and_cover_replay_rows() -> None:
    overrides = SMOOTH_TRAINER["_parse_replay_component_weights"](
        [
            "saddle_tangent_transition_tubes=4",
            "origin_positive_cone_transition_fan=3",
        ]
    )
    metadata = {
        "components": [
            {
                "name": "background",
                "row_start_inclusive": 0,
                "row_stop_exclusive": 2,
            },
            {
                "name": "saddle_tangent_transition_tubes",
                "row_start_inclusive": 2,
                "row_stop_exclusive": 5,
            },
            {
                "name": "origin_positive_cone_transition_fan",
                "row_start_inclusive": 5,
                "row_stop_exclusive": 7,
            },
        ]
    }

    weights, report = SMOOTH_TRAINER["_manifest_component_sample_weights"](
        metadata,
        row_count=7,
        overrides=overrides,
        device=torch.device("cpu"),
    )

    assert torch.equal(weights, torch.tensor([1.0, 1.0, 4.0, 4.0, 4.0, 3.0, 3.0]))
    assert report["background"]["explicit_override"] is False
    assert report["saddle_tangent_transition_tubes"]["weight"] == 4.0
    assert report["origin_positive_cone_transition_fan"]["weight"] == 3.0
    with pytest.raises(ValueError, match="unknown replay component"):
        SMOOTH_TRAINER["_manifest_component_sample_weights"](
            metadata,
            row_count=7,
            overrides={"misspelled": 2.0},
            device=torch.device("cpu"),
        )


def test_fixed_recurrent_exclusion_negatives_load_with_hash(tmp_path: Path) -> None:
    path = tmp_path / "census_negatives.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cycles": [
                    {
                        "cycle_id": "extra-p2",
                        "least_period": 2,
                        "inside_cmgdb_bounds": True,
                        "phase_points": [[0.01, -0.02], [0.03, 0.04]],
                        "association": {"is_intended_catalogue_cycle": False},
                    },
                    {
                        "cycle_id": "known",
                        "least_period": 1,
                        "inside_cmgdb_bounds": True,
                        "phase_points": [[0.10, 0.10]],
                        "association": {"is_intended_catalogue_cycle": True},
                    },
                    {
                        "cycle_id": "outside",
                        "least_period": 1,
                        "inside_cmgdb_bounds": False,
                        "phase_points": [[9.0, 9.0]],
                        "association": {"is_intended_catalogue_cycle": False},
                    },
                ],
            }
        )
    )

    probes, provenance = SMOOTH_TRAINER["_load_extra_recurrent_probes"](
        path,
        device=torch.device("cpu"),
    )

    assert torch.allclose(probes, torch.tensor([[0.01, -0.02], [0.03, 0.04]]))
    assert provenance["count"] == 2
    assert provenance["sha256"] == SMOOTH_TRAINER["_sha256"](path)
    assert provenance["census_selection"]["selected_cycle_ids"] == ["extra-p2"]
    assert "inside_cmgdb_bounds" in provenance["census_selection"]["selection_rule"]


def test_model_selection_validation_rows_are_disjoint_from_training() -> None:
    data_dir = REPO_ROOT / "data" / "leslie3d_invariant_aware"
    train = np.loadtxt(data_dir / "train.csv", delimiter=",", skiprows=1)
    validation = np.loadtxt(data_dir / "val.csv", delimiter=",", skiprows=1)
    train_keys = {tuple(f"{value:.15e}" for value in row) for row in train}
    validation_keys = {tuple(f"{value:.15e}" for value in row) for row in validation}

    assert len(validation_keys) == len(validation)
    assert train_keys.isdisjoint(validation_keys)


def test_collapsed_morse_roles_do_not_count_as_a_reachability_match(tmp_path: Path) -> None:
    graph_path = tmp_path / "morse_graph"
    graph_path.write_text(
        """digraph {
0 [label="0 : (x^4-1, 0, 0)"];
1 [label="1 : (x^4-1, 0, 0)"];
2 [label="2 : (0, x^4-1, 0)"];
3 [label="3 : (0, x+1, 0)"];
4 [label="4 : (0, 0, 0)"];
2 -> 0;
2 -> 1;
3 -> 1;
4 -> 2;
4 -> 3;
}
"""
    )
    assignments = {
        "P0": {"assigned_morse_node": 0},
        "P1": {"assigned_morse_node": 1},
        "S2": {"assigned_morse_node": 1},  # deliberate role collapse
        "S4": {"assigned_morse_node": 2},
        "p_star": {"assigned_morse_node": 3},
        "origin": {"assigned_morse_node": 4},
    }
    manifest = {
        "expected_direct_indices": BUILDER["EXPECTED_DIRECT_INDICES"],
        "orbit_manifold_informed_reduced_edges": BUILDER["EXPECTED_REDUCED_EDGES"],
    }

    comparison = ANALYZER["_graph_comparison"](graph_path, assignments, manifest)
    s2_to_p1 = next(
        check
        for check in comparison["orbit_manifold_reachability_checks"]
        if check["source_object"] == "S2" and check["target_object"] == "P1"
    )
    assert s2_to_p1["reachable"] is False
    assert comparison["all_objects_in_distinct_nodes"] is False
    assert comparison["exact_role_aligned_morse_graph_match"] is False


def test_exact_role_aligned_graph_comparison_checks_positive_and_negative_relations(
    tmp_path: Path,
) -> None:
    graph_path = tmp_path / "morse_graph"
    graph_path.write_text(
        """digraph {
0 [label="0 : (x^4-1, 0, 0)"];
1 [label="1 : (x^4-1, 0, 0)"];
2 [label="2 : (0, x^2+1, 0)"];
3 [label="3 : (0, x^4-1, 0)"];
4 [label="4 : (0, x+1, 0)"];
5 [label="5 : (0, 0, 0)"];
2 -> 1;
3 -> 0;
3 -> 1;
4 -> 2;
5 -> 3;
5 -> 4;
}
"""
    )
    assignments = {
        name: {"assigned_morse_node": node}
        for node, name in enumerate(("P0", "P1", "S2", "S4", "p_star", "origin"))
    }
    manifest = {
        "expected_direct_indices": BUILDER["EXPECTED_DIRECT_INDICES"],
        "orbit_manifold_informed_reduced_edges": BUILDER["EXPECTED_REDUCED_EDGES"],
    }

    comparison = ANALYZER["_graph_comparison"](graph_path, assignments, manifest)
    assert comparison["all_role_reachability_and_nonreachability_match"] is True
    assert comparison["all_object_minimality_matches"] is True
    assert comparison["exact_role_aligned_morse_graph_match"] is True
