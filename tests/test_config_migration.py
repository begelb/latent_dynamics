"""
Regression test: pin resolved-arch snapshot for every checked-in config.

Captures the current per-component (encoder / latent_map / decoder) resolution
of every YAML under configs/ to ensure subsequent refactor tasks preserve the
structure without silent changes to architecture specifications.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from latentdynamics.config import load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

EXPECTED: dict[str, dict[str, dict[str, object]]] = {
    "chafee_infante.yaml": {
        "encoder": {
            "hidden_shapes": (64, 32),
            "activation": "tanh",
            "out_activation": "none",
        },
        "latent_map": {
            "hidden_shapes": (32, 32),
            "activation": "tanh",
            "out_activation": "none",
        },
        "decoder": {
            "hidden_shapes": (32, 64),
            "activation": "tanh",
            "out_activation": "none",
        },
    },
    "coral_adaptive.yaml": {
        "encoder": {
            "hidden_shapes": (64, 64, 64),
            "activation": "relu",
            "out_activation": "tanh",
        },
        "latent_map": {
            "hidden_shapes": (64, 64, 64),
            "activation": "relu",
            "out_activation": "tanh",
        },
        "decoder": {
            "hidden_shapes": (64, 64, 64),
            "activation": "relu",
            "out_activation": "sigmoid",
        },
    },
    "coral_basic.yaml": {
        "encoder": {
            "hidden_shapes": (64, 64, 64),
            "activation": "relu",
            "out_activation": "tanh",
        },
        "latent_map": {
            "hidden_shapes": (64, 64, 64),
            "activation": "relu",
            "out_activation": "tanh",
        },
        "decoder": {
            "hidden_shapes": (64, 64, 64),
            "activation": "relu",
            "out_activation": "sigmoid",
        },
    },
    "coral_data_scaling.yaml": {
        "encoder": {
            "hidden_shapes": (64, 64, 64),
            "activation": "relu",
            "out_activation": "tanh",
        },
        "latent_map": {
            "hidden_shapes": (64, 64, 64),
            "activation": "relu",
            "out_activation": "tanh",
        },
        "decoder": {
            "hidden_shapes": (64, 64, 64),
            "activation": "relu",
            "out_activation": "sigmoid",
        },
    },
    "leslie3d.yaml": {
        "encoder": {
            "hidden_shapes": (32, 32, 32),
            "activation": "relu",
            "out_activation": "tanh",
        },
        "latent_map": {
            "hidden_shapes": (32, 32, 32),
            "activation": "relu",
            "out_activation": "tanh",
        },
        "decoder": {
            "hidden_shapes": (32, 32, 32),
            "activation": "relu",
            "out_activation": "sigmoid",
        },
    },
    "leslie3d_spurious.yaml": {
        "encoder": {
            "hidden_shapes": (32, 32, 32),
            "activation": "relu",
            "out_activation": "tanh",
        },
        "latent_map": {
            "hidden_shapes": (32, 32, 32),
            "activation": "relu",
            "out_activation": "tanh",
        },
        "decoder": {
            "hidden_shapes": (32, 32, 32),
            "activation": "relu",
            "out_activation": "sigmoid",
        },
    },
    "leslie3d_success.yaml": {
        "encoder": {
            "hidden_shapes": (64, 64),
            "activation": "relu",
            "out_activation": "tanh",
        },
        "latent_map": {
            "hidden_shapes": (64, 64),
            "activation": "relu",
            "out_activation": "tanh",
        },
        "decoder": {
            "hidden_shapes": (64, 64),
            "activation": "relu",
            "out_activation": "sigmoid",
        },
    },
    "leslie_contraction.yaml": {
        "encoder": {
            "hidden_shapes": (64, 64, 64, 64),
            "activation": "relu",
            "out_activation": "tanh",
        },
        "latent_map": {
            "hidden_shapes": (64, 64, 64, 64),
            "activation": "relu",
            "out_activation": "tanh",
        },
        "decoder": {
            "hidden_shapes": (64, 64, 64, 64),
            "activation": "relu",
            "out_activation": "sigmoid",
        },
    },
    "leslie2d_to_2d_test_011.yaml": {
        "encoder": {
            "hidden_shapes": (64, 32),
            "activation": "tanh",
            "out_activation": "none",
        },
        "latent_map": {
            "hidden_shapes": (32, 32),
            "activation": "tanh",
            "out_activation": "none",
        },
        "decoder": {
            "hidden_shapes": (32, 64),
            "activation": "tanh",
            "out_activation": "none",
        },
    },
    "leslie2d_to_2d_test_101.yaml": {
        "encoder": {
            "hidden_shapes": (64, 32),
            "activation": "tanh",
            "out_activation": "none",
        },
        "latent_map": {
            "hidden_shapes": (32, 32),
            "activation": "tanh",
            "out_activation": "none",
        },
        "decoder": {
            "hidden_shapes": (32, 64),
            "activation": "tanh",
            "out_activation": "none",
        },
    },
    "leslie2d_to_2d_test_110.yaml": {
        "encoder": {
            "hidden_shapes": (64, 32),
            "activation": "tanh",
            "out_activation": "none",
        },
        "latent_map": {
            "hidden_shapes": (32, 32),
            "activation": "tanh",
            "out_activation": "none",
        },
        "decoder": {
            "hidden_shapes": (32, 64),
            "activation": "tanh",
            "out_activation": "none",
        },
    },
    "scratch/coral_adaptive.yaml": {
        "encoder": {
            "hidden_shapes": (64, 64, 64),
            "activation": "relu",
            "out_activation": "tanh",
        },
        "latent_map": {
            "hidden_shapes": (64, 64, 64),
            "activation": "relu",
            "out_activation": "tanh",
        },
        "decoder": {
            "hidden_shapes": (64, 64, 64),
            "activation": "relu",
            "out_activation": "sigmoid",
        },
    },
    "scratch/coral_basic.yaml": {
        "encoder": {
            "hidden_shapes": (64, 64, 64),
            "activation": "relu",
            "out_activation": "tanh",
        },
        "latent_map": {
            "hidden_shapes": (64, 64, 64),
            "activation": "relu",
            "out_activation": "tanh",
        },
        "decoder": {
            "hidden_shapes": (64, 64, 64),
            "activation": "relu",
            "out_activation": "sigmoid",
        },
    },
    "scratch/coral_data_scaling.yaml": {
        "encoder": {
            "hidden_shapes": (64, 64, 64),
            "activation": "relu",
            "out_activation": "tanh",
        },
        "latent_map": {
            "hidden_shapes": (64, 64, 64),
            "activation": "relu",
            "out_activation": "tanh",
        },
        "decoder": {
            "hidden_shapes": (64, 64, 64),
            "activation": "relu",
            "out_activation": "sigmoid",
        },
    },
}


@pytest.mark.parametrize("rel_path", sorted(EXPECTED.keys()))
def test_resolved_arch_matches_snapshot(rel_path: str) -> None:
    """Verify resolved architecture matches snapshot for the given config."""
    config_path = CONFIG_DIR / rel_path
    assert config_path.exists(), f"Config not found: {config_path}"

    cfg = load_config(config_path)
    expected = EXPECTED[rel_path]

    for component_name in ("encoder", "latent_map", "decoder"):
        component = cfg.arch.component(component_name)
        component_expected = expected[component_name]

        assert component.hidden_shapes == component_expected["hidden_shapes"], (
            f"{rel_path} {component_name}: hidden_shapes mismatch: got {component.hidden_shapes}, expected {component_expected['hidden_shapes']}"
        )

        assert component.activation == component_expected["activation"], (
            f"{rel_path} {component_name}: activation mismatch: got {component.activation!r}, expected {component_expected['activation']!r}"
        )

        assert component.out_activation == component_expected["out_activation"], (
            f"{rel_path} {component_name}: out_activation mismatch: got {component.out_activation!r}, expected {component_expected['out_activation']!r}"
        )
