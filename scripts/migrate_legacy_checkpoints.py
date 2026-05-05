"""One-shot migration from legacy three-file checkpoints to the new format.

Walks ``--output-root`` (default: ``code/output``) and for every directory
containing ``models/encoder.pt``, ``models/dynamics.pt``, ``models/decoder.pt``
emitted by the legacy ``Training.save_models``, produces a new
``<dest_dir>/models/autoencoder.pt`` (state_dict only) plus
``<dest_dir>/models/autoencoder.json`` (architecture sidecar).

The legacy files are *pickled nn.Module objects* and require the original
class definitions to be importable, so we add ``code/src`` to ``sys.path``
before loading. Once migrated, the legacy ``.pt`` files can be deleted.

Usage:
    python scripts/migrate_legacy_checkpoints.py \
        --output-root code/output \
        --dest-root output/legacy_migrated \
        --high-dims 13 --low-dims 1 --num-layers 3 --hidden-shape 64
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_SRC = REPO_ROOT / "legacy" / "src"


def _enable_legacy_imports() -> None:
    if str(LEGACY_SRC) not in sys.path:
        sys.path.insert(0, str(LEGACY_SRC))


def _translate_keys(
    legacy_state: dict[str, torch.Tensor], legacy_root: str, new_root: str
) -> dict[str, torch.Tensor]:
    """Convert ``encoder.linear_i.{weight,bias}`` to ``encoder.net.{2*i}.{weight,bias}``."""
    out: dict[str, torch.Tensor] = {}
    prefix = f"{legacy_root}."
    for key, tensor in legacy_state.items():
        if not key.startswith(prefix):
            continue
        suffix = key[len(prefix) :]
        if not suffix.startswith("linear_"):
            continue
        idx_str, _, param_name = suffix[len("linear_") :].partition(".")
        idx = int(idx_str)
        out[f"{new_root}.net.{2 * idx}.{param_name}"] = tensor
    return out


def _migrate_one(seed_dir: Path, dest_dir: Path, arch_payload: dict, *, dry_run: bool) -> None:
    legacy = seed_dir / "models"
    enc_path = legacy / "encoder.pt"
    dyn_path = legacy / "dynamics.pt"
    dec_path = legacy / "decoder.pt"

    encoder = torch.load(enc_path, weights_only=False, map_location="cpu")
    dynamics = torch.load(dyn_path, weights_only=False, map_location="cpu")
    decoder = torch.load(dec_path, weights_only=False, map_location="cpu")

    state: dict[str, torch.Tensor] = {}
    state.update(_translate_keys(encoder.state_dict(), "encoder", "encoder"))
    state.update(_translate_keys(dynamics.state_dict(), "dynamics", "latent_map"))
    state.update(_translate_keys(decoder.state_dict(), "decoder", "decoder"))

    out_models = dest_dir / "models"
    if dry_run:
        print(f"[dry-run] would write {out_models / 'autoencoder.pt'} ({len(state)} tensors)")
        return

    out_models.mkdir(parents=True, exist_ok=True)
    torch.save(state, out_models / "autoencoder.pt")
    (out_models / "autoencoder.json").write_text(
        f'{{"version": 1, "arch": {arch_payload}}}'.replace("'", '"')
    )
    print(f"wrote {out_models / 'autoencoder.pt'} ({len(state)} tensors)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--dest-root", required=True, type=Path)
    parser.add_argument("--num-layers", required=True, type=int)
    parser.add_argument("--hidden-shape", required=True, type=int)
    parser.add_argument("--high-dims", required=True, type=int)
    parser.add_argument("--low-dims", required=True, type=int)
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--encoder-out-activation", default="tanh")
    parser.add_argument("--decoder-out-activation", default="sigmoid")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    _enable_legacy_imports()
    arch_payload = {
        "num_layers": args.num_layers,
        "hidden_shape": args.hidden_shape,
        "high_dims": args.high_dims,
        "low_dims": args.low_dims,
        "activation": args.activation,
        "encoder_out_activation": args.encoder_out_activation,
        "decoder_out_activation": args.decoder_out_activation,
    }

    n_migrated = 0
    for enc_path in args.output_root.rglob("models/encoder.pt"):
        seed_dir = enc_path.parent.parent
        rel = seed_dir.relative_to(args.output_root)
        dest_dir = args.dest_root / rel
        try:
            _migrate_one(seed_dir, dest_dir, arch_payload, dry_run=args.dry_run)
            n_migrated += 1
        except Exception as exc:
            print(f"[skip] {seed_dir}: {exc}")
    print(f"done: {n_migrated} legacy checkpoint(s) processed")


if __name__ == "__main__":
    main()
