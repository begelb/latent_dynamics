"""Convert Marcio's Chafee-Infante weights into the package checkpoint format.

His ``DynamicsAutoencoder`` (archive/marcio/scripts/autoencoder_model.py) stores
each component as a bare ``nn.Sequential`` -> state-dict keys ``encoder.0.weight``,
``encoder.2.weight``, ``encoder.4.weight`` (and likewise ``latent_map.*`` /
``decoder.*``). The package's ``LatentDynamicsAutoencoder`` wraps each component
in a ``.net`` Sequential, so its keys are ``encoder.net.0.weight`` etc. The only
transform is inserting ``.net`` after the component name; the architectures are
otherwise identical (encoder [64,32], latent_map [32,32], decoder [32,64], tanh
hidden, linear output).

Run once from code/:
    ../.venv/bin/python scripts/convert_marcio_chafee.py
"""

from __future__ import annotations

from pathlib import Path

import torch

from latentdynamics.config import load_config
from latentdynamics.models import build_autoencoder
from latentdynamics.training.checkpoints import save_checkpoint

CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_WEIGHTS = CODE_ROOT.parent / "archive" / "marcio" / "scripts" / "ci_model_weights.pth"
CONFIG = CODE_ROOT / "configs" / "chafee_infante_marcio.yaml"
OUT_DIR = CODE_ROOT / "replay_sources" / "chafee_infante" / "marcio" / "models"


def convert() -> None:
    raw = torch.load(SRC_WEIGHTS, map_location="cpu", weights_only=True)

    remapped: dict[str, torch.Tensor] = {}
    for key, tensor in raw.items():
        component, rest = key.split(".", 1)  # e.g. "encoder", "0.weight"
        remapped[f"{component}.net.{rest}"] = tensor

    cfg = load_config(CONFIG)
    model = build_autoencoder(cfg.arch)
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    if missing or unexpected:
        raise SystemExit(
            f"state_dict mismatch -- the conversion assumption is wrong.\n"
            f"  missing (in model, absent from weights): {sorted(missing)}\n"
            f"  unexpected (in weights, not in model):  {sorted(unexpected)}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pt_path, json_path = save_checkpoint(model, cfg.arch, OUT_DIR)
    print(f"converted {SRC_WEIGHTS.name} -> {pt_path}")
    print(f"arch sidecar -> {json_path}")


if __name__ == "__main__":
    convert()
