from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.optim import Adam

from latentdynamics.config import ArchConfig
from latentdynamics.models import build_autoencoder


def _load_module():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    script = scripts / "compare_chafee_d1_grad_accum.py"
    spec = importlib.util.spec_from_file_location(
        "compare_chafee_d1_grad_accum",
        script,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ACCUM = _load_module()


def _tiny_arch() -> ArchConfig:
    return ArchConfig(
        high_dims=3,
        low_dims=1,
        encoder={"hidden_shapes": [4], "activation": "tanh", "out_activation": "none"},
        latent_map={"hidden_shapes": [4], "activation": "tanh", "out_activation": "none"},
        decoder={"hidden_shapes": [4], "activation": "tanh", "out_activation": "none"},
    )


@pytest.mark.parametrize("microbatch_size", [1, 2, 3, 6, 20])
def test_one_accumulated_update_matches_direct_full_batch(
    microbatch_size: int,
) -> None:
    arch = _tiny_arch()
    rng = np.random.default_rng(123)
    x = torch.tensor(rng.normal(size=(7, 3)), dtype=torch.float32)
    y = torch.tensor(rng.normal(size=(7, 3)), dtype=torch.float32)
    torch.manual_seed(44)
    accumulated = build_autoencoder(arch)
    direct = copy.deepcopy(accumulated)

    direct_optimizer = Adam(direct.parameters(), lr=3e-3)
    direct_optimizer.zero_grad(set_to_none=True)
    _, _, direct_total = ACCUM.common._two_term_losses(direct, x, y)
    direct_total.backward()
    direct_optimizer.step()

    accumulated_optimizer = Adam(accumulated.parameters(), lr=3e-3)
    metrics, batches = ACCUM._accumulated_epoch(
        accumulated,
        x,
        y,
        optimizer=accumulated_optimizer,
        microbatch_size=microbatch_size,
        device=torch.device("cpu"),
    )

    assert batches == (7 + microbatch_size - 1) // microbatch_size
    assert metrics["loss_total"] == pytest.approx(float(direct_total.detach()))
    assert all(parameter.grad is None for parameter in accumulated.parameters())
    for name, expected in direct.state_dict().items():
        torch.testing.assert_close(
            accumulated.state_dict()[name],
            expected,
            rtol=2e-6,
            atol=2e-7,
        )


def test_settings_require_fixed_checkpoint_divisibility() -> None:
    with pytest.raises(ValueError, match="resume_interval"):
        ACCUM.GradientAccumulationSettings(
            epochs=10,
            resume_interval=3,
            milestone_interval=5,
        )
    with pytest.raises(ValueError, match="milestone_interval"):
        ACCUM.GradientAccumulationSettings(
            epochs=10,
            resume_interval=2,
            milestone_interval=6,
        )
