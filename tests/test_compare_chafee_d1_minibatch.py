from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.optim import Adam

from latentdynamics.config import ArchConfig
from latentdynamics.models import build_autoencoder
from latentdynamics.training import load_checkpoint


def _load_module():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "compare_chafee_d1_minibatch.py"
    )
    spec = importlib.util.spec_from_file_location(
        "compare_chafee_d1_minibatch",
        script,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MINIBATCH = _load_module()


def _tiny_arch() -> ArchConfig:
    return ArchConfig(
        high_dims=3,
        low_dims=1,
        encoder={"hidden_shapes": [4], "activation": "tanh", "out_activation": "none"},
        latent_map={"hidden_shapes": [4], "activation": "tanh", "out_activation": "none"},
        decoder={"hidden_shapes": [4], "activation": "tanh", "out_activation": "none"},
    )


def test_seeded_batches_cover_every_row_once_without_dropping() -> None:
    generators = [torch.Generator().manual_seed(17) for _ in range(2)]
    runs = [
        MINIBATCH._shuffled_batch_indices(7, 3, generator=generator)
        for generator in generators
    ]

    assert [batch.tolist() for batch in runs[0]] == [
        batch.tolist() for batch in runs[1]
    ]
    assert [len(batch) for batch in runs[0]] == [3, 3, 1]
    assert sorted(torch.cat(runs[0]).tolist()) == list(range(7))


def test_one_full_sized_batch_matches_manual_two_term_adam_update() -> None:
    arch = _tiny_arch()
    rng = np.random.default_rng(2026)
    x = torch.tensor(rng.normal(size=(7, 3)), dtype=torch.float32)
    y = torch.tensor(rng.normal(size=(7, 3)), dtype=torch.float32)
    torch.manual_seed(11)
    actual = build_autoencoder(arch)
    expected = copy.deepcopy(actual)

    expected_optimizer = Adam(expected.parameters(), lr=1e-3)
    expected_optimizer.zero_grad(set_to_none=True)
    reconstruction, prediction, total = MINIBATCH._two_term_losses(expected, x, y)
    total.backward()
    expected_optimizer.step()

    actual_optimizer = Adam(actual.parameters(), lr=1e-3)
    metrics, updates = MINIBATCH._train_epoch(
        actual,
        x,
        y,
        optimizer=actual_optimizer,
        batch_size=7,
        shuffle_generator=torch.Generator().manual_seed(99),
        device=torch.device("cpu"),
    )

    assert updates == 1
    assert metrics == pytest.approx(
        {
            "loss_reconstruction": float(reconstruction.detach()),
            "loss_prediction": float(prediction.detach()),
            "loss_total": float(total.detach()),
        }
    )
    for name, expected_value in expected.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[name], expected_value)


class _IdentityPart(nn.Module):
    def forward(self, values):
        return values


class _IdentityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _IdentityPart()
        self.latent_map = _IdentityPart()
        self.decoder = _IdentityPart()


def test_evaluation_reports_zero_residual_for_identity_dynamics() -> None:
    values = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    metrics = MINIBATCH._evaluate_model(
        _IdentityModel(),
        values,
        values,
        batch_size=3,
        device=torch.device("cpu"),
    )

    assert metrics["loss_reconstruction"] == 0.0
    assert metrics["loss_prediction"] == 0.0
    assert metrics["loss_total"] == 0.0
    assert metrics["latent_semiconjugacy_mse"] == 0.0
    assert metrics["normalized_latent_residual_rmse"] == 0.0
    assert metrics["max_euclidean_latent_residual"] == 0.0
    assert metrics["latent_span_q99_q01"] > 0.0


def test_checkpoint_selection_uses_latent_metric_only_within_validation_band() -> None:
    selection = MINIBATCH._select_checkpoint_epochs(
        [2.0, 1.0, 1.04, 1.2],
        [0.1, 0.5, 0.2, 0.01],
        tolerance_fraction=0.05,
    )

    assert selection == {
        "best_validation_epoch": 2,
        "selected_epoch": 3,
        "validation_cutoff": pytest.approx(1.05),
        "eligible_epoch_count": 2,
    }


def test_tiny_run_persists_selected_equal_update_and_final_checkpoints(
    tmp_path: Path,
) -> None:
    arch = _tiny_arch()
    rng = np.random.default_rng(55)
    x_train = rng.normal(size=(9, 3))
    y_train = rng.normal(size=(9, 3))
    x_val = rng.normal(size=(6, 3))
    y_val = rng.normal(size=(6, 3))
    output = tmp_path / "run"
    plan = MINIBATCH._write_json(output / "run_plan.json", {"frozen": True})
    plan_hash = MINIBATCH._sha256(plan)
    settings = MINIBATCH.MiniBatchSettings(
        batch_size=4,
        max_epochs=3,
        minimum_epochs=1,
        equal_update_epoch=1,
        scheduler_patience=1,
        early_stopping_patience=2,
    )

    summary = MINIBATCH.train_minibatch(
        arch=arch,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        settings=settings,
        device=torch.device("cpu"),
        output_dir=output,
        run_plan_sha256=plan_hash,
        verbose=False,
    )

    assert summary["run_plan_sha256"] == plan_hash
    assert summary["optimizer_updates"] == summary["epochs_completed"] * 3
    assert (output / "candidates" / "equal_update" / "models" / "autoencoder.pt").is_file()
    assert (output / "candidates" / "final" / "models" / "autoencoder.pt").is_file()
    selected, selected_arch = load_checkpoint(
        output / "models",
        basename="selected",
    )
    assert selected_arch == arch
    assert all(torch.isfinite(value).all() for value in selected.state_dict().values())
    assert (output / "selection_record.json").is_file()


@pytest.mark.parametrize(
    ("x_shape", "y_shape", "message"),
    [
        ((3,), (3,), "rank-2"),
        ((2, 3), (3, 3), "shapes differ"),
        ((2, 2), (2, 2), r"shape \(n, 3\)"),
    ],
)
def test_pair_validation_rejects_invalid_shapes(x_shape, y_shape, message) -> None:
    with pytest.raises(ValueError, match=message):
        MINIBATCH._validate_pair_arrays(
            np.zeros(x_shape),
            np.zeros(y_shape),
            high_dimension=3,
            name="test",
        )
