from __future__ import annotations

import copy
import hashlib
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scripts import train_leslie3d_alternating_chart as alternating
from torch import nn

from latentdynamics.config import load_config
from latentdynamics.models import build_autoencoder


class _IdentityScaler:
    def transform(self, values):
        return values


def _toy_targets() -> dict[str, torch.Tensor]:
    """Return six distinct objects with the production catalogue periods."""

    targets: dict[str, torch.Tensor] = {}
    cursor = 0
    generator = torch.Generator().manual_seed(314159)
    base = 1.5 * torch.rand((16, 2), generator=generator) - 0.75
    for name in alternating.topology.OBJECT_ORDER:
        period = alternating.topology.PERIODS[name]
        targets[name] = base[cursor : cursor + period].clone()
        cursor += period
    assert cursor == 16
    return targets


def test_alternating_config_isolated_and_uses_primary_v2_checkpoint() -> None:
    config = load_config("leslie3d_invariant_aware_v2_alternating")

    assert str(config.training.warm_start_checkpoint_dir).endswith(
        "output/leslie3d_invariant_aware_v2_smooth/seed_20260809/models"
    )
    assert config.paths.output_dir.name == "leslie3d_invariant_aware_v2_alternating"
    assert config.cmgdb.lower_bounds is None
    assert config.cmgdb.upper_bounds is None
    assert (config.cmgdb.subdiv_min, config.cmgdb.subdiv_max) == (25, 30)


def test_edge_chart_scope_exposes_only_encoder_output_and_decoder_input() -> None:
    model = build_autoencoder(load_config("leslie3d_invariant_aware_v2_alternating").arch)

    selected = alternating._configure_chart_parameters(model, "edge")

    assert selected == [
        "encoder.net.4.weight",
        "encoder.net.4.bias",
        "decoder.net.0.weight",
        "decoder.net.0.bias",
    ]
    assert all(not parameter.requires_grad for parameter in model.latent_map.parameters())


def test_cross_role_margin_and_antifold_terms_penalize_collapse() -> None:
    reference = torch.tensor([[0.0, 0.0], [0.2, 0.0], [0.0, 0.3]])
    collapsed = torch.zeros_like(reference)
    labels = ("A", "B", "C")
    margin = alternating._cross_role_margin_loss(
        collapsed,
        reference,
        labels,
        expansion=1.0,
        minimum_margin=0.01,
    )
    bank = alternating.SecantBank(
        left=torch.tensor([0, 1]),
        right=torch.tensor([1, 2]),
        reference_distance=torch.tensor([0.2, 0.36]),
    )
    anti_fold = alternating._anti_fold_loss(collapsed, bank, retained_fraction=0.8)

    assert float(margin) > 0.0
    assert torch.isclose(anti_fold, torch.tensor(0.64))


def test_local_secant_bank_is_deterministic_and_detached() -> None:
    physical = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.4, 0.0, 0.0], [1.0, 0.0, 0.0]])
    latent = torch.tensor([[0.0, 0.0], [0.1, 0.0], [0.4, 0.0], [1.0, 0.0]])

    first = alternating._make_local_secant_bank(physical, latent, sample_count=4, neighbors=1)
    second = alternating._make_local_secant_bank(physical, latent, sample_count=4, neighbors=1)

    assert torch.equal(first.left, second.left)
    assert torch.equal(first.right, second.right)
    assert torch.equal(first.reference_distance, second.reference_distance)
    assert not first.reference_distance.requires_grad


def test_chart_gates_reject_a_folded_validation_chart() -> None:
    baseline = {
        "terms": {"reconstruction": 1.0, "prediction": 1.0, "semiconjugacy": 1.0},
    }
    folded = {
        "score": 1.0,
        "terms": {"reconstruction": 1.0, "prediction": 1.0, "semiconjugacy": 1.0},
        "encoder_drift_rmse": 0.01,
        "cross_role": {"minimum_ratio_to_reference": 0.2},
        "local_secant_ratio": {"p01": 0.1},
    }

    report = alternating._chart_gate_report(
        folded,
        baseline,
        reconstruction_ratio_limit=1.02,
        prediction_ratio_limit=1.05,
        semiconjugacy_ratio_limit=1.05,
        drift_limit=0.03,
        cross_role_ratio_floor=0.95,
        secant_p01_ratio_floor=0.8,
    )

    assert report["accepted"] is False
    assert report["gates"]["cross_role_separation"] is False
    assert report["gates"]["local_secant_anti_fold"] is False


def test_chart_objective_sends_gradients_only_to_selected_chart_layers() -> None:
    model = build_autoencoder(load_config("leslie3d_invariant_aware_v2_alternating").arch)
    reference = copy.deepcopy(model).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    alternating._configure_chart_parameters(model, "edge")
    x = torch.rand(8, 3)
    y = torch.rand(8, 3)
    with torch.no_grad():
        reference_x = reference.encoder(x)
        reference_y = reference.encoder(y)
    known = x[:3]
    known_reference = reference.encoder(known).detach()
    bank = alternating.SecantBank(
        left=torch.tensor([0, 1, 2]),
        right=torch.tensor([1, 2, 3]),
        reference_distance=torch.linalg.vector_norm(
            reference_x[torch.tensor([0, 1, 2])] - reference_x[torch.tensor([1, 2, 3])],
            dim=1,
        ),
    )

    terms = alternating._chart_terms(
        model,
        reference,
        x,
        y,
        reference_x,
        reference_y,
        known,
        known_reference,
        ("A", "B", "C"),
        bank,
        None,
        margin_expansion=1.0,
        minimum_margin=0.01,
        anti_fold_fraction=0.8,
    )
    sum(terms.values()).backward()

    selected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert selected == {
        "encoder.net.4.weight",
        "encoder.net.4.bias",
        "decoder.net.0.weight",
        "decoder.net.0.bias",
    }
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    assert all(parameter.grad is None for parameter in model.latent_map.parameters())


def test_map_gate_rejects_rollout_regression_and_nonfinite_diagnostics() -> None:
    candidate = {
        "replay": {"total": 1.0},
        "max_anchor_normalized_l2": 1e-5,
        "max_characteristic_relative_error": 0.01,
        "max_role_margin_violation": 0.0,
        "trust_rmse": 0.01,
        "rollout_loss": 2.0,
    }
    report = alternating._map_gate_report(
        candidate,
        1.0,
        1.0,
        rollout_enabled=True,
        validation_ratio_limit=1.05,
        rollout_ratio_limit=1.05,
        rollout_absolute_limit=1.5,
        anchor_acceptance=0.001,
        characteristic_acceptance=0.05,
    )

    assert report["accepted"] is False
    assert report["gates"]["validation_rollout"] is False
    assert report["rollout_ratio"] == 2.0

    candidate["rollout_loss"] = float("nan")
    nonfinite = alternating._map_gate_report(
        candidate,
        1.0,
        1.0,
        rollout_enabled=True,
        validation_ratio_limit=1.05,
        rollout_ratio_limit=1.05,
        rollout_absolute_limit=1.5,
        anchor_acceptance=0.001,
        characteristic_acceptance=0.05,
    )
    assert nonfinite["gates"]["finite_diagnostics"] is False


def test_map_gate_cannot_game_a_degraded_post_chart_rollout_baseline() -> None:
    candidate = {
        "replay": {
            "total": 1.0,
            "reconstruction": 0.02,
            "prediction": 0.03,
        },
        "max_anchor_normalized_l2": 1.0e-5,
        "max_characteristic_relative_error": 0.01,
        "max_role_margin_violation": 0.0,
        "trust_rmse": 0.01,
        "rollout_loss": 0.02,
    }

    report = alternating._map_gate_report(
        candidate,
        1.0,
        0.05,
        rollout_enabled=True,
        validation_ratio_limit=1.05,
        rollout_ratio_limit=1.5,
        rollout_absolute_limit=0.00697,
        anchor_acceptance=0.001,
        characteristic_acceptance=0.05,
        source_physical_baseline={"reconstruction": 0.01, "prediction": 0.03},
        source_reconstruction_ratio_limit=1.02,
        source_prediction_ratio_limit=1.05,
    )

    assert report["gates"]["validation_rollout"] is True
    assert report["gates"]["validation_rollout_absolute"] is False
    assert report["gates"]["source_chart_reconstruction"] is False
    assert report["accepted"] is False


def test_latent_cache_is_rebuilt_from_current_encoder_and_detached() -> None:
    model = build_autoencoder(load_config("leslie3d_invariant_aware_v2_alternating").arch)
    states = torch.tensor(
        [
            [0.10, 0.20, 0.30],
            [0.20, 0.25, 0.35],
            [0.30, 0.30, 0.40],
            [0.40, 0.35, 0.45],
            [0.50, 0.40, 0.50],
        ]
    )
    x = states[:-1]
    y = states[1:]
    manifest = {
        "known_objects": {
            name: {"points": [[0.05 * (index + 1)] * 3]}
            for index, name in enumerate(alternating.topology.OBJECT_ORDER)
        }
    }
    metadata = {
        "components": [
            {
                "name": "saddle_tangent_transition_tubes",
                "row_start_inclusive": 0,
                "row_stop_exclusive": 4,
                "rows": 4,
                "trajectories": 1,
                "steps": 4,
            }
        ]
    }
    before = alternating._rebuild_latent_cache(
        model, _IdentityScaler(), manifest, metadata, metadata, x, y, x, y
    )
    with torch.no_grad():
        alternating._linear_layers(model.encoder)[-1].bias.add_(0.2)
    after = alternating._rebuild_latent_cache(
        model, _IdentityScaler(), manifest, metadata, metadata, x, y, x, y
    )

    assert not torch.equal(before.z_train, after.z_train)
    assert all(not tensor.requires_grad for tensor in after.targets.values())
    assert all(not block["latent_states"].requires_grad for block in after.train_shadowing_blocks)


def test_map_optimizer_excludes_projection_owned_final_affine_layer() -> None:
    model = build_autoencoder(load_config("leslie3d_invariant_aware_v2_alternating").arch)

    selected = alternating._configure_map_parameters(model)
    final_affine = alternating._linear_layers(model.latent_map)[-1]

    assert selected
    assert all(not parameter.requires_grad for parameter in final_affine.parameters())
    assert all(parameter.requires_grad for parameter in alternating._linear_layers(model.latent_map)[0].parameters())


def test_constrained_map_scope_exposes_final_affine_but_keeps_it_out_of_adam() -> None:
    model = build_autoencoder(load_config("leslie3d_invariant_aware_v2_alternating").arch)

    selected = alternating._configure_map_parameters(
        model,
        constrained_output_update=True,
    )
    final_affine = alternating._linear_layers(model.latent_map)[-1]
    adam_parameters = [
        parameter
        for parameter in model.latent_map.parameters()
        if parameter.requires_grad
        and parameter is not final_affine.weight
        and parameter is not final_affine.bias
    ]

    assert selected
    assert all(parameter.requires_grad for parameter in final_affine.parameters())
    assert all(parameter is not final_affine.weight for parameter in adam_parameters)
    assert all(parameter is not final_affine.bias for parameter in adam_parameters)


def test_nullspace_output_step_preserves_anchor_preactivation_constraints() -> None:
    torch.manual_seed(42)
    model = build_autoencoder(load_config("leslie3d_invariant_aware_v2_alternating").arch)
    alternating._configure_map_parameters(model, constrained_output_update=True)
    targets = _toy_targets()
    alternating.topology._project_anchor_equalities(model.latent_map, targets)
    geometry = alternating._output_constraint_geometry(model.latent_map, targets)
    layer = geometry.output_layer
    before = torch.cat([layer.weight.T, layer.bias.unsqueeze(0)], dim=0).double().clone()
    tangent_direction = geometry.nullspace_basis[:, 0].to(layer.weight.dtype)
    parameters = torch.cat([layer.weight.T, layer.bias.unsqueeze(0)], dim=0)
    loss = torch.sum(parameters * tangent_direction.unsqueeze(1))
    named_parameters = [
        (name, parameter)
        for name, parameter in model.latent_map.named_parameters()
        if parameter.requires_grad
    ]

    term_report = alternating._backward_terms_with_caps(
        {"probe": loss},
        named_parameters,
        per_term_max_norm=10.0,
        output_constraint_geometry=geometry,
    )
    step_report = alternating._apply_constrained_output_sgd(
        geometry,
        learning_rate=1.0e-3,
    )
    after = torch.cat([layer.weight.T, layer.bias.unsqueeze(0)], dim=0).double()
    installed_constraint_delta = geometry.design @ (after - before)

    assert geometry.rank <= geometry.design.shape[0]
    assert geometry.nullspace_basis.shape[1] == geometry.design.shape[1] - geometry.rank
    assert step_report["ideal_F_delta_max_abs"] < 1.0e-15
    # The ideal tangent residual is machine-zero. Installing it into a random,
    # poorly conditioned float32 toy layer can amplify the final cast, but the
    # leakage remains small and is removed by the subsequent exact projection.
    assert float(torch.max(torch.abs(installed_constraint_delta.detach()))) < 1.0e-3
    assert step_report["F_delta_max_abs"] == pytest.approx(
        float(torch.max(torch.abs(installed_constraint_delta.detach())))
    )
    projection = term_report["probe"]["output_tangent_projection"]
    assert projection["F_projected_gradient_max_abs"] < 1.0e-12
    assert projection["projected_gradient_l2"] > 0.0


def test_float64_gradient_norm_and_safe_clip_handle_huge_finite_elements() -> None:
    parameter = nn.Parameter(torch.tensor([0.0]))
    parameter.grad = torch.tensor([1.0e30])

    statistics = alternating._gradient_tensor_statistics([("p", parameter.grad)])
    clipped = alternating._safe_clip_grad_norm_([("p", parameter)], 1.0)

    assert statistics["finite"] is True
    assert statistics["norm_accumulator_dtype"] == "float64"
    assert 0.99e30 <= statistics["total_l2"] <= 1.01e30
    assert clipped["clip_coefficient"] < 1.0e-20
    assert float(torch.abs(parameter.grad)) <= 1.00001


def test_rng_checkpoint_round_trip_restores_python_numpy_and_torch() -> None:
    original = alternating._capture_rng_state()
    random.seed(101)
    np.random.seed(202)
    torch.manual_seed(303)
    state = alternating._capture_rng_state()
    expected = (random.random(), float(np.random.rand()), float(torch.rand(())))
    random.random()
    np.random.rand()
    torch.rand(())

    alternating._restore_rng_state(state)
    actual = (random.random(), float(np.random.rand()), float(torch.rand(())))

    assert actual == expected
    alternating._restore_rng_state(original)


def test_format_v2_continuation_inherits_fixed_output_learning_rate() -> None:
    state = {
        "format_version": 2,
        "continuation": {
            "output_learning_rate_policy": "fixed_explicit",
            "fixed_output_learning_rate": 3.0e-7,
        },
    }

    inherited, source = alternating._resolve_continuation_output_learning_rate(
        state,
        None,
    )
    overridden, override_source = (
        alternating._resolve_continuation_output_learning_rate(state, 9.0e-7)
    )

    assert inherited == 3.0e-7
    assert source == "inherited_fixed_from_format_v2_state"
    assert overridden == 9.0e-7
    assert override_source == "explicit_cli_override"


def test_input_provenance_change_is_rejected() -> None:
    expected = {
        "schema_version": 1,
        "data_provenance": {"train_csv_sha256": "abc"},
        "training_configuration": {"loss_weights": [1.0, 2.0]},
    }
    observed = copy.deepcopy(expected)
    observed["data_provenance"]["train_csv_sha256"] = "changed"

    with pytest.raises(ValueError, match="input provenance mismatch"):
        alternating._validate_input_provenance(expected, observed)


def test_infeasible_continuation_boundary_is_rejected() -> None:
    torch.manual_seed(71)
    model = build_autoencoder(load_config("leslie3d_invariant_aware_v2_alternating").arch)
    targets = _toy_targets()
    projection = alternating.topology._project_anchor_equalities(
        model.latent_map,
        targets,
    )
    layer = alternating._linear_layers(model.latent_map)[-1]
    with torch.no_grad():
        layer.bias.add_(0.1)
    geometry = alternating._output_constraint_geometry(model.latent_map, targets)
    report = alternating._anchor_preactivation_report(geometry)

    with pytest.raises(ValueError, match="violates anchor preactivation"):
        alternating._validate_continuation_boundary(
            geometry,
            report,
            {"last_anchor_projection": projection},
        )


def test_safe_gradient_clip_rejects_nonfinite_elements_without_nan_diagnostics() -> None:
    parameter = nn.Parameter(torch.tensor([0.0, 0.0]))
    parameter.grad = torch.tensor([float("nan"), 2.0])

    statistics = alternating._gradient_tensor_statistics([("p", parameter.grad)])

    assert statistics["finite"] is False
    assert statistics["maximum_absolute_element"] == 2.0
    json.dumps(statistics, allow_nan=False)
    with pytest.raises(FloatingPointError, match="non-finite gradient"):
        alternating._safe_clip_grad_norm_([("p", parameter)], 1.0)


def test_per_term_gradient_caps_prevent_one_term_from_erasing_others() -> None:
    parameter = nn.Parameter(torch.tensor(1.0))
    terms = {
        "large": 1.0e10 * parameter,
        "small": parameter,
    }

    report = alternating._backward_terms_with_caps(
        terms,
        [("p", parameter)],
        per_term_max_norm=2.0,
    )

    assert report["large"]["total_l2"] == 1.0e10
    assert report["large"]["clip_coefficient"] < 1.0e-9
    assert report["small"]["clip_coefficient"] == 1.0
    assert torch.isclose(parameter.grad, torch.tensor(3.0))


def test_tbptt_preserves_forward_rollout_value_and_reduces_unstable_gradient() -> None:
    latent_map = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        latent_map.weight.fill_(1.5)
    blocks = [
        {
            "name": "unstable",
            "latent_states": torch.ones(17, 1),
            "horizon_groups": {"medium": (16,)},
        }
    ]

    full, _ = alternating.topology._trajectory_shadowing_loss(
        latent_map,
        blocks,
        diagnostics=False,
    )
    full.backward()
    full_gradient = float(torch.abs(latent_map.weight.grad))
    latent_map.weight.grad = None
    truncated, report = alternating._trajectory_shadowing_loss_truncated(
        latent_map,
        blocks,
        backprop_steps=8,
        diagnostics=True,
    )
    truncated.backward()
    truncated_gradient = float(torch.abs(latent_map.weight.grad))

    assert torch.equal(full.detach(), truncated.detach())
    assert report["forward_values_are_unmodified"] is True
    assert truncated_gradient < full_gradient


def test_rollout_horizon_limiter_keeps_long_targets_evaluation_only() -> None:
    blocks = [
        {
            "name": "tube",
            "latent_states": torch.zeros(65, 2),
            "horizon_groups": {
                "short": (1, 2, 3, 4),
                "medium": (7, 8, 15, 16, 31, 32),
                "long": (63, 64),
            },
        }
    ]

    limited = alternating._limit_shadowing_horizons(blocks, 16)

    assert limited[0]["horizon_groups"] == {
        "short": (1, 2, 3, 4),
        "medium": (7, 8, 15, 16),
    }
    assert blocks[0]["horizon_groups"]["long"] == (63, 64)


def test_recoverability_screen_is_distinct_from_strict_promotion_gate() -> None:
    candidate = {
        "replay": {"total": 1.1},
        "rollout_loss": 5.0,
        "max_anchor_normalized_l2": 1.0e-4,
        "max_characteristic_relative_error": 0.7,
        "max_role_margin_violation": 0.046,
    }

    report = alternating._map_recoverability_report(
        candidate,
        baseline_replay_total=1.0,
        baseline_rollout_loss=1.0,
    )

    assert report["recoverable"] is True
    assert report["purpose"] == "pretraining_screen_not_promotion_gate"
    candidate["max_characteristic_relative_error"] = 1.1
    rejected = alternating._map_recoverability_report(
        candidate,
        baseline_replay_total=1.0,
        baseline_rollout_loss=1.0,
    )
    assert rejected["recoverable"] is False


def test_map_phase_non_eval_branch_and_optimizer_checkpoint(
    monkeypatch,
    tmp_path,
) -> None:
    config = load_config("leslie3d_invariant_aware_v2_alternating")
    model = build_autoencoder(config.arch)
    z_train = torch.rand(4, 2)
    z_next = torch.rand(4, 2)
    cache = alternating.LatentCache(
        z_train=z_train,
        z_train_next=z_next,
        z_val=z_train,
        z_val_next=z_next,
        targets={},
        scales={},
        train_shadowing_blocks=[],
        val_shadowing_blocks=[],
        shadowing_report={},
    )

    def fake_replay(latent_model, z, z_target, *_args, **_kwargs):
        loss = nn.functional.mse_loss(latent_model.latent_map(z), z_target)
        return {"total": loss}

    def fake_spectral(latent_map, *_args, **_kwargs):
        parameter = next(latent_map.parameters())
        characteristic = 1.0e-4 * torch.mean(parameter**2)
        return characteristic, characteristic, {}

    def fake_evaluation(*_args, **_kwargs):
        return {
            "replay": {"total": 1.0},
            "max_anchor_normalized_l2": 0.0,
            "anchor_quadratic": 0.0,
            "characteristic_loss": 0.0,
            "topology_loss": 0.0,
            "max_characteristic_relative_error": 0.0,
            "max_role_margin_violation": 0.0,
            "trust_rmse": 0.0,
            "rollout_loss": 0.0,
        }

    monkeypatch.setattr(alternating.topology, "_replay_losses", fake_replay)
    monkeypatch.setattr(
        alternating.topology,
        "_anchor_residuals",
        lambda *_args, **_kwargs: (torch.zeros(1, 2), {}),
    )
    monkeypatch.setattr(alternating.topology, "_spectral_terms", fake_spectral)
    monkeypatch.setattr(
        alternating.topology,
        "_project_anchor_equalities",
        lambda *_args, **_kwargs: {"maximum_correction": 0.0},
    )
    monkeypatch.setattr(alternating, "_map_evaluation", fake_evaluation)
    models_dir = tmp_path / "models"
    logs_dir = tmp_path / "logs"
    models_dir.mkdir()
    logs_dir.mkdir()
    settings = alternating.MapTrainingSettings(
        epochs=2,
        learning_rate=1.0e-6,
        rollout_learning_rate=1.0e-7,
        rollout_min_topology_epochs=1,
        anchor_weight=1.0,
        characteristic_weight=1.0,
        topology_weight=1.0,
        trust_weight=1.0,
        rollout_weight=0.0,
        rollout_ratio_limit=1.5,
        rollout_absolute_limit=0.00697,
        rollout_backprop_steps=8,
        rollout_short_epochs=1,
        rollout_medium_max_horizon=16,
        spectral_start_epoch=0,
        spectral_ramp_epochs=2,
        per_term_gradient_clip_norm=1.0,
        gradient_diagnostics_every=1,
        gradient_diagnostics_threshold=1.0e6,
        eval_every=2,
        stable_ceiling=0.98,
        unstable_floor=1.05,
        jury_buffer=0.005,
        validation_ratio_limit=1.05,
        source_reconstruction_ratio_limit=1.02,
        source_prediction_ratio_limit=1.05,
        anchor_acceptance=0.001,
        characteristic_acceptance=0.05,
    )
    cfg = SimpleNamespace(training=config.training, arch=config.arch)

    result = alternating._run_map_phase(
        model,
        cfg,
        cache,
        torch.rand(4, 3),
        torch.rand(4, 3),
        torch.rand(4, 3),
        torch.rand(4, 3),
        torch.ones(4),
        models_dir,
        logs_dir,
        settings,
    )

    history = json.loads((logs_dir / "map_history.json").read_text())
    checkpoint = torch.load(
        models_dir / "map_training_state.pt",
        weights_only=False,
    )
    assert len(history) == 1
    assert history[0]["epoch"] == 1
    assert result["optimizer_checkpoint"].endswith("map_training_state.pt")
    assert checkpoint["next_epoch"] == 2
    assert checkpoint["last_anchor_projection"] == {"maximum_correction": 0.0}
    assert checkpoint["settings"]["rollout_medium_max_horizon"] == 16
    assert "optimizer_state_dict" in checkpoint
    assert "scheduler_state_dict" in checkpoint


def test_map_continuation_restores_state_and_isolates_nullspace_updates(
    monkeypatch,
    tmp_path,
) -> None:
    config = load_config("leslie3d_invariant_aware_v2_alternating")
    cfg = SimpleNamespace(training=config.training, arch=config.arch)
    torch.manual_seed(7)
    model = build_autoencoder(config.arch)
    reference_model = copy.deepcopy(model)
    targets = _toy_targets()
    scales = alternating.topology._phase_scales(targets)
    z_train = torch.rand(8, 2) * 1.5 - 0.75
    z_next = torch.rand(8, 2) * 1.5 - 0.75
    cache = alternating.LatentCache(
        z_train=z_train,
        z_train_next=z_next,
        z_val=z_train,
        z_val_next=z_next,
        targets=targets,
        scales=scales,
        train_shadowing_blocks=[],
        val_shadowing_blocks=[],
        shadowing_report={},
    )

    def fake_replay(latent_model, z, z_target, *_args, **_kwargs):
        loss = nn.functional.mse_loss(latent_model.latent_map(z), z_target)
        return {"total": loss}

    def fake_spectral(latent_map, *_args, **_kwargs):
        parameter = next(latent_map.parameters())
        characteristic = 1.0e-4 * torch.mean(parameter**2)
        return characteristic, characteristic, {}

    def fake_evaluation(*_args, **_kwargs):
        return {
            "replay": {"total": 1.0},
            "max_anchor_normalized_l2": 0.0,
            "anchor_quadratic": 0.0,
            "characteristic_loss": 0.0,
            "topology_loss": 0.0,
            "max_characteristic_relative_error": 0.0,
            "max_role_margin_violation": 0.0,
            "trust_rmse": 0.0,
            "rollout_loss": 0.0,
        }

    monkeypatch.setattr(alternating.topology, "_replay_losses", fake_replay)
    monkeypatch.setattr(alternating.topology, "_spectral_terms", fake_spectral)
    monkeypatch.setattr(alternating, "_map_evaluation", fake_evaluation)
    settings = alternating.MapTrainingSettings(
        epochs=1,
        learning_rate=1.0e-4,
        rollout_learning_rate=1.0e-5,
        rollout_min_topology_epochs=1,
        anchor_weight=1.0,
        characteristic_weight=1.0,
        topology_weight=1.0,
        trust_weight=1.0,
        rollout_weight=0.0,
        rollout_ratio_limit=1.5,
        rollout_absolute_limit=0.00697,
        rollout_backprop_steps=8,
        rollout_short_epochs=1,
        rollout_medium_max_horizon=16,
        spectral_start_epoch=0,
        spectral_ramp_epochs=1,
        per_term_gradient_clip_norm=1.0,
        gradient_diagnostics_every=1,
        gradient_diagnostics_threshold=1.0e6,
        eval_every=1,
        stable_ceiling=0.98,
        unstable_floor=1.05,
        jury_buffer=0.005,
        validation_ratio_limit=1.05,
        source_reconstruction_ratio_limit=1.02,
        source_prediction_ratio_limit=1.05,
        anchor_acceptance=0.001,
        characteristic_acceptance=0.05,
    )
    source_models = tmp_path / "source" / "models"
    source_logs = tmp_path / "source" / "logs"
    source_models.mkdir(parents=True)
    source_logs.mkdir()
    tensors3 = torch.rand(8, 3)
    alternating._run_map_phase(
        model,
        cfg,
        cache,
        tensors3,
        tensors3,
        tensors3,
        tensors3,
        torch.ones(8),
        source_models,
        source_logs,
        settings,
    )
    source_path = source_models / "map_training_state.pt"
    source_bytes = source_path.read_bytes()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    source_state = torch.load(source_path, weights_only=False)
    source_steps = [
        int(value["step"].item())
        for value in source_state["optimizer_state_dict"]["state"].values()
    ]

    continued_model = build_autoencoder(config.arch)
    continued_model.load_state_dict(source_state["model_state_dict"])
    continuation_models = tmp_path / "continuation" / "models"
    continuation_logs = tmp_path / "continuation" / "logs"
    continuation_models.mkdir(parents=True)
    continuation_logs.mkdir()
    random_state_before = alternating._capture_rng_state()
    torch.manual_seed(9999)
    result = alternating._run_map_phase(
        continued_model,
        cfg,
        cache,
        tensors3,
        tensors3,
        tensors3,
        tensors3,
        torch.ones(8),
        continuation_models,
        continuation_logs,
        settings,
        resume_state=source_state,
        reference_model=reference_model,
        continuation_epochs=1,
        constrained_output_learning_rate=None,
        source_training_state={"path": str(source_path), "sha256": source_sha},
        input_provenance={"schema_version": 1, "test": True},
    )
    continued_state = torch.load(
        continuation_models / "map_training_state.pt",
        weights_only=False,
    )
    continued_steps = [
        int(value["step"].item())
        for value in continued_state["optimizer_state_dict"]["state"].values()
    ]
    constraint_history = json.loads(
        (continuation_logs / "output_constraint_history.json").read_text()
    )

    assert source_path.read_bytes() == source_bytes
    assert source_sha == hashlib.sha256(source_path.read_bytes()).hexdigest()
    assert continued_state["format_version"] == 2
    assert continued_state["next_epoch"] == 2
    assert continued_state["continuation"]["source_history_length"] == 1
    assert continued_state["map_history"][0] == source_state["map_history"][0]
    assert min(continued_steps) > max(source_steps)
    assert (
        continued_state["scheduler_state_dict"]["last_epoch"]
        == source_state["scheduler_state_dict"]["last_epoch"] + 1
    )
    assert result["start_epoch"] == 1
    assert result["epochs_executed"] == 1
    assert len(constraint_history) == 1
    assert constraint_history[0]["ideal_F_delta_max_abs"] < 1.0e-15
    assert constraint_history[0]["F_delta_max_abs"] < 1.0e-3
    assert constraint_history[0]["preactivation_residual_after_exact_projection_max_abs"] < 1.0e-3
    # Avoid leaking this test's restored checkpoint RNG into later tests.
    alternating._restore_rng_state(random_state_before)
