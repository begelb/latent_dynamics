"""Focused contracts for fixed-epoch full-batch curriculum training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from latentdynamics.cli import make_data, pipeline, scale_data, train
from latentdynamics.config import (
    ArchConfig,
    CMGDBConfig,
    CurriculumLBFGSPolishConfig,
    CurriculumOptimizerConfig,
    CurriculumStageConfig,
    DataConfig,
    ExperimentConfig,
    PathsConfig,
    SystemConfig,
    TrainingConfig,
)
from latentdynamics.models import build_autoencoder
from latentdynamics.training import load_checkpoint, train_curriculum_full_batch


def _tiny_arch() -> ArchConfig:
    return ArchConfig(
        high_dims=3,
        low_dims=2,
        encoder={"hidden_shapes": [5], "activation": "tanh", "out_activation": "none"},
        latent_map={
            "hidden_shapes": [4],
            "activation": "tanh",
            "out_activation": "none",
        },
        decoder={"hidden_shapes": [5], "activation": "tanh", "out_activation": "none"},
    )


def _tiny_pairs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2026)
    x_train = rng.normal(size=(8, 3)).astype(np.float64)
    y_train = (0.6 * x_train + rng.normal(scale=0.1, size=x_train.shape)).astype(np.float64)
    x_holdout = rng.normal(size=(4, 3)).astype(np.float64)
    y_holdout = (0.6 * x_holdout + rng.normal(scale=0.1, size=x_holdout.shape)).astype(np.float64)
    return x_train, y_train, x_holdout, y_holdout


def _three_stage_curriculum() -> list[CurriculumStageConfig]:
    return [
        CurriculumStageConfig(
            name="chart",
            epochs=1,
            learning_rate=1e-2,
            loss_weights=[1.0, 0.0, 0.0],
            trainable_components=["encoder", "decoder"],
        ),
        CurriculumStageConfig(
            name="decoded_prediction",
            epochs=1,
            learning_rate=5e-3,
            loss_weights=[1.0, 1.0, 0.0],
            trainable_components=["encoder", "latent_map", "decoder"],
        ),
        CurriculumStageConfig(
            name="semiconjugacy",
            epochs=1,
            learning_rate=1e-3,
            loss_weights=[1.0, 1.0, 2.0],
            trainable_components=["encoder", "latent_map", "decoder"],
        ),
    ]


def _state_for_component(state: dict[str, torch.Tensor], component: str) -> dict[str, torch.Tensor]:
    prefix = f"{component}."
    return {name: value for name, value in state.items() if name.startswith(prefix)}


def _tiny_adamw() -> CurriculumOptimizerConfig:
    return CurriculumOptimizerConfig(
        name="adamw",
        betas=[0.9, 0.999],
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=False,
        foreach=False,
        fused=False,
    )


def _tiny_polish(*, outer_steps: int = 2) -> CurriculumLBFGSPolishConfig:
    return CurriculumLBFGSPolishConfig(
        outer_steps=outer_steps,
        learning_rate=0.2,
        max_iter=2,
        max_eval=5,
        history_size=5,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        loss_weights=[1.0, 1.0, 2.0],
    )


@torch.no_grad()
def _losses(
    model: torch.nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    weights: list[float],
) -> dict[str, float]:
    """Independently reproduce the checkpoint's three-term float32 loss."""

    x_tensor = torch.as_tensor(x, dtype=torch.float32)
    y_tensor = torch.as_tensor(y, dtype=torch.float32)
    fp = model(x_tensor, y_tensor)
    reconstruction = torch.nn.functional.mse_loss(fp.x_t_hat, fp.x_t)
    prediction = torch.nn.functional.mse_loss(fp.x_tau_hat, fp.x_tau)
    semiconjugacy = torch.nn.functional.mse_loss(fp.z_tau_pred, fp.z_tau)
    total = weights[0] * reconstruction + weights[1] * prediction + weights[2] * semiconjugacy
    return {
        "loss_reconstruction": float(reconstruction),
        "loss_prediction": float(prediction),
        "loss_semiconjugacy": float(semiconjugacy),
        "loss_total": float(total),
    }


def test_no_polish_adam_compatibility_uses_one_optimizer_across_stages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One tiny run exercises phase boundaries without doing material training."""

    import latentdynamics.training.curriculum as curriculum_module

    arch = _tiny_arch()
    x_train, y_train, x_holdout, y_holdout = _tiny_pairs()
    torch.manual_seed(41)
    model = build_autoencoder(arch)
    initial_state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }

    real_adam = curriculum_module.Adam
    optimizer_instances: list[torch.optim.Optimizer] = []

    def recording_adam(*args, **kwargs):
        optimizer = real_adam(*args, **kwargs)
        optimizer_instances.append(optimizer)
        return optimizer

    monkeypatch.setattr(curriculum_module, "Adam", recording_adam)
    result = train_curriculum_full_batch(
        arch=arch,
        stages=_three_stage_curriculum(),
        x=x_train,
        y=y_train,
        x_validation=x_holdout,
        y_validation=y_holdout,
        seed=17,
        device="cpu",
        output_dir=tmp_path,
        model=model,
    )

    # One Adam instance is retained across all three stages. There is no
    # scheduler object, stopping rule, or optimizer reset at a boundary.
    assert len(optimizer_instances) == 1
    optimizer_summary = result.summary["optimizer"]
    assert optimizer_summary["sequence"] == ["Adam"]
    assert optimizer_summary["polish"] is None
    assert optimizer_summary["first_order"] == {
        "name": "Adam",
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.0,
        "amsgrad": False,
        "foreach": False,
        "fused": False,
        "state_continues_across_stages": True,
        "stage_learning_rates": [1e-2, 5e-3, 1e-3],
        "updates_completed": 3,
        "device": "cpu",
        "dtype": "float32",
    }
    assert result.summary["scheduler"] is None
    assert result.summary["scheduler_used"] is False
    assert result.summary["patience_used"] is False
    assert result.summary["early_stopping_used"] is False

    stage_one, loaded_arch = load_checkpoint(tmp_path / "stage_checkpoints" / "01_chart" / "models")
    assert loaded_arch == arch
    stage_one_state = stage_one.state_dict()
    for name, initial_value in _state_for_component(initial_state, "latent_map").items():
        torch.testing.assert_close(stage_one_state[name], initial_value, rtol=0, atol=0)
    for component in ("encoder", "decoder"):
        assert any(
            not torch.equal(stage_one_state[name], initial_value)
            for name, initial_value in _state_for_component(initial_state, component).items()
        )

    records = result.summary["curriculum"]
    assert [record["name"] for record in records] == [
        "chart",
        "decoded_prediction",
        "semiconjugacy",
    ]
    assert [record["start_epoch_one_based"] for record in records] == [1, 2, 3]
    assert [record["end_epoch_one_based"] for record in records] == [1, 2, 3]
    assert [record["optimizer_state_continued_from_previous_stage"] for record in records] == [
        False,
        True,
        True,
    ]
    assert records[0]["trainable_components"] == ["encoder", "decoder"]
    assert all(parameter.requires_grad for parameter in result.model.parameters())


def test_adamw_then_cpu_float64_lbfgs_saves_exact_float32_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the production optimizer sequence with deliberately tiny budgets."""

    import latentdynamics.training.curriculum as curriculum_module

    arch = _tiny_arch()
    stages = _three_stage_curriculum()
    x_train, y_train, x_holdout, y_holdout = _tiny_pairs()
    torch.manual_seed(41)
    model = build_autoencoder(arch)
    initial_state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }

    real_adamw = curriculum_module.AdamW
    real_lbfgs = curriculum_module.LBFGS
    adamw_instances: list[torch.optim.Optimizer] = []
    lbfgs_initializations: list[dict[str, object]] = []

    def recording_adamw(*args, **kwargs):
        optimizer = real_adamw(*args, **kwargs)
        adamw_instances.append(optimizer)
        return optimizer

    def recording_lbfgs(parameters, *args, **kwargs):
        parameters = list(parameters)
        lbfgs_initializations.append(
            {
                "devices": {parameter.device.type for parameter in parameters},
                "dtypes": {parameter.dtype for parameter in parameters},
                "kwargs": dict(kwargs),
            }
        )
        return real_lbfgs(parameters, *args, **kwargs)

    monkeypatch.setattr(curriculum_module, "AdamW", recording_adamw)
    monkeypatch.setattr(curriculum_module, "LBFGS", recording_lbfgs)
    result = train_curriculum_full_batch(
        arch=arch,
        stages=stages,
        x=x_train,
        y=y_train,
        x_validation=x_holdout,
        y_validation=y_holdout,
        seed=17,
        device="cpu",
        output_dir=tmp_path,
        first_order_optimizer=_tiny_adamw(),
        polish=_tiny_polish(),
        model=model,
    )

    assert len(adamw_instances) == 1
    assert len(lbfgs_initializations) == 1
    assert lbfgs_initializations[0]["devices"] == {"cpu"}
    assert lbfgs_initializations[0]["dtypes"] == {torch.float64}
    assert lbfgs_initializations[0]["kwargs"] == {
        "lr": 0.2,
        "max_iter": 2,
        "max_eval": 5,
        "tolerance_grad": 1e-9,
        "tolerance_change": 1e-12,
        "history_size": 5,
        "line_search_fn": "strong_wolfe",
    }

    # The first saved stage is still the chart-only phase: AdamW has not
    # touched G, while both chart modules have received an update.
    stage_one, loaded_arch = load_checkpoint(tmp_path / "stage_checkpoints" / "01_chart" / "models")
    assert loaded_arch == arch
    stage_one_state = stage_one.state_dict()
    for name, initial_value in _state_for_component(initial_state, "latent_map").items():
        torch.testing.assert_close(stage_one_state[name], initial_value, rtol=0, atol=0)
    for component in ("encoder", "decoder"):
        assert any(
            not torch.equal(stage_one_state[name], initial_value)
            for name, initial_value in _state_for_component(initial_state, component).items()
        )

    optimizer_summary = result.summary["optimizer"]
    assert optimizer_summary["sequence"] == ["AdamW", "LBFGS"]
    assert optimizer_summary["first_order"]["updates_completed"] == 3
    assert optimizer_summary["first_order"]["state_continues_across_stages"] is True
    polish_summary = optimizer_summary["polish"]
    assert polish_summary["device"] == "cpu"
    assert polish_summary["dtype"] == "float64"
    assert polish_summary["outer_steps_requested"] == 2
    assert polish_summary["outer_steps_completed"] == 2
    assert polish_summary["closure_evaluations"] >= 2
    assert polish_summary["internal_iterations"] >= 0

    polish_history = result.history["polish"]
    assert len(polish_history["records"]) == 2
    assert (
        sum(record["closure_evaluations"] for record in polish_history["records"])
        == (polish_summary["closure_evaluations"])
    )
    # Quasi-Newton outer steps, internal iterations, and closure calls are
    # separately accounted for; none are mislabeled as first-order epochs.
    assert result.summary["n_epochs_run"] == 3
    assert result.summary["epochs_completed"] == 3
    assert result.summary["first_order_epochs_completed"] == 3
    assert result.summary["checkpoint_epoch"] is None
    assert result.summary["checkpoint_selection"] == "final_lbfgs_float32_endpoint"

    assert result.summary["validation_evaluated"] is True
    assert result.summary["validation_used_for_optimization"] is False
    assert result.summary["validation_used_for_checkpoint_selection"] is False
    assert result.summary["best_weight_restoration_used"] is False

    loaded, loaded_arch = load_checkpoint(tmp_path / "models")
    assert loaded_arch == arch
    assert {parameter.dtype for parameter in loaded.parameters()} == {torch.float32}
    assert {parameter.dtype for parameter in result.model.parameters()} == {torch.float32}
    for name, live_value in result.model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], live_value.cpu(), rtol=0, atol=0)

    for observed, expected in (
        (
            _losses(loaded, x_train, y_train, [1.0, 1.0, 2.0]),
            result.summary["final_checkpoint_train"],
        ),
        (_losses(loaded, x_holdout, y_holdout, [1.0, 1.0, 2.0]), result.summary["final_holdout"]),
    ):
        assert observed == pytest.approx(expected, rel=1e-7, abs=1e-9)


def test_curriculum_artifacts_history_and_final_checkpoint_semantics(
    tmp_path: Path,
) -> None:
    arch = _tiny_arch()
    x_train, y_train, x_holdout, y_holdout = _tiny_pairs()
    result = train_curriculum_full_batch(
        arch=arch,
        stages=_three_stage_curriculum(),
        x=x_train,
        y=y_train,
        x_validation=x_holdout,
        y_validation=y_holdout,
        seed=23,
        device="cpu",
        output_dir=tmp_path,
    )

    expected_files = (
        tmp_path / "models" / "autoencoder.pt",
        tmp_path / "models" / "autoencoder.json",
        tmp_path / "logs" / "history.json",
        tmp_path / "training_summary.json",
        tmp_path / "final_losses.txt",
        tmp_path / "adamw_endpoint" / "models" / "autoencoder.pt",
        tmp_path / "adamw_endpoint" / "models" / "autoencoder.json",
        tmp_path / "stage_checkpoints" / "01_chart" / "models" / "autoencoder.pt",
        tmp_path / "stage_checkpoints" / "02_decoded_prediction" / "models" / "autoencoder.pt",
        tmp_path / "stage_checkpoints" / "03_semiconjugacy" / "models" / "autoencoder.pt",
    )
    assert all(path.is_file() and path.stat().st_size > 0 for path in expected_files)

    history = json.loads(result.history_path.read_text())
    summary = json.loads(result.summary_path.read_text())
    assert history == result.history
    assert summary == result.summary
    assert history["schema_version"] == 2
    assert summary["schema_version"] == 2
    assert history["training_method"] == "curriculum_full_batch"
    assert history["stage_index"] == [1, 2, 3]
    assert history["stage_name"] == [
        "chart",
        "decoded_prediction",
        "semiconjugacy",
    ]
    assert history["learning_rate"] == pytest.approx([1e-2, 5e-3, 1e-3])
    assert history["loss_weights"] == [
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 2.0],
    ]
    for split in ("train", "val"):
        for loss_name in (
            "loss_reconstruction",
            "loss_prediction",
            "loss_semiconjugacy",
            "loss_total",
        ):
            assert len(history[split][loss_name]) == 3

    assert summary["n_epochs_run"] == 3
    assert summary["epochs_requested"] == 3
    assert summary["epochs_completed"] == 3
    assert summary["checkpoint_epoch"] == 3
    assert summary["checkpoint_selection"] == "final_epoch"
    assert summary["best_epoch"] is None
    assert summary["best_source"] == "not_applicable_final_epoch_selected"
    assert summary["validation_evaluated"] is True
    assert summary["validation_used_for_optimization"] is False
    assert summary["validation_used_for_checkpoint_selection"] is False
    assert summary["best_weight_restoration_used"] is False
    assert summary["gradient_clipping_used"] is False
    assert summary["data"] == {
        "n_training_pairs": 8,
        "n_validation_pairs": 4,
        "high_dims": 3,
        "dtype": "float32",
        "full_batch": True,
    }
    assert summary["artifacts"] == {
        "checkpoint": "models/autoencoder.pt",
        "checkpoint_metadata": "models/autoencoder.json",
        "adamw_checkpoint": "adamw_endpoint/models/autoencoder.pt",
        "adamw_checkpoint_metadata": "adamw_endpoint/models/autoencoder.json",
        "history": "logs/history.json",
    }

    final_train = summary["final_epoch_train"]
    assert final_train["loss_total"] == pytest.approx(
        final_train["loss_reconstruction"]
        + final_train["loss_prediction"]
        + 2.0 * final_train["loss_semiconjugacy"]
    )
    final_holdout = summary["final_holdout"]
    assert final_holdout["loss_total"] == pytest.approx(
        final_holdout["loss_reconstruction"]
        + final_holdout["loss_prediction"]
        + 2.0 * final_holdout["loss_semiconjugacy"]
    )

    loaded, loaded_arch = load_checkpoint(tmp_path / "models")
    assert loaded_arch == arch
    for name, live_value in result.model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], live_value.cpu())

    final_text = result.final_losses_path.read_text()
    assert "checkpoint_selection: final_epoch" in final_text
    assert "lbfgs_closure_evaluations: 0" in final_text
    assert "scheduler_used: false" in final_text
    assert "early_stopping_used: false" in final_text


def test_holdout_is_reporting_only_for_adamw_and_lbfgs(tmp_path: Path) -> None:
    arch = _tiny_arch()
    x_train, y_train, x_holdout, y_holdout = _tiny_pairs()
    common = {
        "arch": arch,
        "stages": _three_stage_curriculum(),
        "x": x_train,
        "y": y_train,
        "seed": 71,
        "device": "cpu",
        "first_order_optimizer": _tiny_adamw(),
        "polish": _tiny_polish(outer_steps=1),
    }
    reference = train_curriculum_full_batch(
        **common,
        x_validation=x_holdout,
        y_validation=y_holdout,
        output_dir=tmp_path / "reference",
    )
    changed_holdout = train_curriculum_full_batch(
        **common,
        x_validation=x_holdout + 7.0,
        y_validation=-3.0 * y_holdout,
        output_dir=tmp_path / "changed_holdout",
    )

    for name, reference_value in reference.model.state_dict().items():
        torch.testing.assert_close(
            changed_holdout.model.state_dict()[name],
            reference_value,
            rtol=0,
            atol=0,
        )
    assert reference.summary["final_checkpoint_train"] == pytest.approx(
        changed_holdout.summary["final_checkpoint_train"], rel=0, abs=0
    )
    assert reference.summary["final_holdout"]["loss_total"] != pytest.approx(
        changed_holdout.summary["final_holdout"]["loss_total"]
    )
    for summary in (reference.summary, changed_holdout.summary):
        assert summary["validation_used_for_optimization"] is False
        assert summary["validation_used_for_checkpoint_selection"] is False


def test_curriculum_optimizer_schema_rejects_ambiguous_or_invalid_protocols() -> None:
    with pytest.raises(ValueError, match="betas"):
        CurriculumOptimizerConfig(betas=[1.0, 0.999])
    with pytest.raises(ValueError, match="max_eval"):
        CurriculumLBFGSPolishConfig(max_iter=3, max_eval=2)
    with pytest.raises(ValueError, match="duplicates"):
        CurriculumLBFGSPolishConfig(trainable_components=["encoder", "encoder"])
    with pytest.raises(ValueError, match=r"require training\.curriculum"):
        TrainingConfig(
            learning_rate=1e-3,
            batch_size=8,
            epochs=3,
            patience=4,
            lr_patience=1,
            curriculum_optimizer=_tiny_adamw(),
        )
    with pytest.raises(ValueError, match=r"curriculum_polish\.loss_weights"):
        TrainingConfig(
            learning_rate=1e-3,
            batch_size=8,
            epochs=3,
            patience=4,
            lr_patience=1,
            loss_weights=[1.0, 1.0, 2.0],
            curriculum=_three_stage_curriculum(),
            curriculum_optimizer=_tiny_adamw(),
            curriculum_polish=CurriculumLBFGSPolishConfig(loss_weights=[1.0, 1.0, 1.0]),
        )


def test_config_driven_train_cli_dispatches_to_curriculum(tmp_path: Path) -> None:
    """The pipeline-facing CLI reads/scales CSVs and selects the new trainer."""

    arch = _tiny_arch()
    stages = _three_stage_curriculum()
    cfg = ExperimentConfig(
        system=SystemConfig(
            name="leslie3d",
            params={"lower_bounds": [0, 0, 0], "upper_bounds": [2, 2, 2]},
        ),
        arch=arch,
        training=TrainingConfig(
            learning_rate=1e-2,
            batch_size=2,
            epochs=3,
            patience=4,
            lr_patience=1,
            loss_weights=[1, 1, 2],
            gradient_clip_norm=None,
            curriculum=stages,
            curriculum_optimizer=_tiny_adamw(),
            curriculum_polish=_tiny_polish(outer_steps=1),
        ),
        data=DataConfig(
            sampling_method="uniform",
            scaling="minmax",
            n_samples_train=2,
            n_samples_val=2,
            n_iterations=1,
            skip=0,
            train_seed=11,
            val_seed=12,
        ),
        cmgdb=CMGDBConfig(),
        paths=PathsConfig(data_dir=tmp_path / "data", output_dir=tmp_path / "output"),
        seeds=[5],
        experiment_name="tiny_curriculum_dispatch",
    )

    make_data.run(cfg, verbose=False)
    scale_data.run(cfg, "train", verbose=False)
    train.run(cfg, seed=5, device=torch.device("cpu"), verbose=False)

    summary = json.loads((cfg.paths.output_dir / "training_summary.json").read_text())
    assert summary["training_method"] == "curriculum_full_batch"
    assert summary["n_epochs_run"] == 3
    assert summary["checkpoint_selection"] == "final_lbfgs_float32_endpoint"
    assert summary["optimizer"]["sequence"] == ["AdamW", "LBFGS"]
    assert summary["data"]["n_training_pairs"] == 2
    assert (cfg.paths.model_dir / "autoencoder.pt").is_file()
    assert pipeline._curriculum_training_complete(cfg, cfg.paths.output_dir)
    (cfg.paths.output_dir / "final_losses.txt").unlink()
    assert not pipeline._curriculum_training_complete(cfg, cfg.paths.output_dir)
