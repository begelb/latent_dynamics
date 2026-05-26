"""Safe state-dict checkpointing with an architecture sidecar.

Each checkpoint is a pair of files:

- ``<name>.pt``    : state_dict (loadable with ``weights_only=True``)
- ``<name>.json``  : architecture spec (the :class:`ArchConfig` model dump)

The :func:`load_legacy_checkpoint` helper additionally reads the legacy
three-file format produced by ``code/legacy/main_scripts/train.py``
(``encoder.pt`` + ``dynamics.pt`` + ``decoder.pt``) and assembles a unified
:class:`LatentDynamicsAutoencoder` without rewriting the files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from ..config.schema import ArchConfig
from ..models.autoencoder import LatentDynamicsAutoencoder, build_autoencoder

CHECKPOINT_VERSION = 1
DEFAULT_BASENAME = "autoencoder"

LEGACY_FILES = ("encoder.pt", "dynamics.pt", "decoder.pt")


def _nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def save_checkpoint(
    model: LatentDynamicsAutoencoder,
    arch: ArchConfig,
    out_dir: str | Path,
    *,
    basename: str = DEFAULT_BASENAME,
) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pt_path = out / f"{basename}.pt"
    json_path = out / f"{basename}.json"
    torch.save(model.state_dict(), pt_path)
    json_path.write_text(
        json.dumps(
            {"version": CHECKPOINT_VERSION, "arch": arch.model_dump()},
            indent=2,
        )
    )
    return pt_path, json_path


def load_checkpoint(
    in_dir: str | Path,
    *,
    basename: str = DEFAULT_BASENAME,
    map_location: str | torch.device = "cpu",
) -> tuple[LatentDynamicsAutoencoder, ArchConfig]:
    in_path = Path(in_dir)
    pt_path = in_path / f"{basename}.pt"
    json_path = in_path / f"{basename}.json"
    sidecar = json.loads(json_path.read_text())
    if sidecar.get("version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"unsupported checkpoint version {sidecar.get('version')!r} "
            f"in {json_path} (expected {CHECKPOINT_VERSION})"
        )
    arch = ArchConfig.model_validate(sidecar["arch"])
    model = build_autoencoder(arch)
    state = torch.load(pt_path, map_location=map_location, weights_only=True)
    model.load_state_dict(state)
    return model, arch


def has_legacy_checkpoint(in_dir: str | Path) -> bool:
    in_path = Path(in_dir)
    return all(_nonempty_file(in_path / name) for name in LEGACY_FILES)


def has_new_checkpoint(in_dir: str | Path, *, basename: str = DEFAULT_BASENAME) -> bool:
    in_path = Path(in_dir)
    return _nonempty_file(in_path / f"{basename}.pt") and _nonempty_file(
        in_path / f"{basename}.json"
    )


def _legacy_module_to_state_dict(
    legacy_module: torch.nn.Module, legacy_root: str, new_root: str
) -> dict[str, torch.Tensor]:
    """Translate ``encoder.linear_i.{w,b}`` keys to ``encoder.net.{2*i}.{w,b}``."""
    src = legacy_module.state_dict()
    out: dict[str, torch.Tensor] = {}
    inner_key = (
        legacy_root  # inside the legacy nn.Module, the Sequential lives at attr ``legacy_root``
    )
    prefix = f"{inner_key}."
    for key, tensor in src.items():
        if not key.startswith(prefix):
            continue
        suffix = key[len(prefix) :]
        if not suffix.startswith("linear_"):
            continue
        idx_str, _, param_name = suffix[len("linear_") :].partition(".")
        out[f"{new_root}.net.{2 * int(idx_str)}.{param_name}"] = tensor
    return out


def load_legacy_checkpoint(
    in_dir: str | Path,
    arch: ArchConfig,
    *,
    map_location: str | torch.device = "cpu",
    legacy_root: str | Path | None = None,
) -> LatentDynamicsAutoencoder:
    """Load a three-file pickled-nn.Module checkpoint into a LatentDynamicsAutoencoder.

    The legacy files were saved with ``torch.save(obj)`` of full ``nn.Module``
    instances whose qualified class name is ``src.models.Encoder`` (etc.).
    Unpickling therefore needs ``src.models`` importable. We add the
    *parent* of the legacy ``src/`` directory (default: ``code/legacy``) to
    ``sys.path`` so ``import src.models`` resolves there.
    """
    in_path = Path(in_dir)
    if not has_legacy_checkpoint(in_path):
        raise FileNotFoundError(f"missing one or more legacy files {LEGACY_FILES} in {in_path}")

    default_root = Path(__file__).resolve().parents[3] / "legacy"
    candidate = Path(legacy_root) if legacy_root is not None else default_root
    if candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

    encoder = torch.load(in_path / "encoder.pt", weights_only=False, map_location=map_location)
    dynamics = torch.load(in_path / "dynamics.pt", weights_only=False, map_location=map_location)
    decoder = torch.load(in_path / "decoder.pt", weights_only=False, map_location=map_location)

    state: dict[str, torch.Tensor] = {}
    state.update(_legacy_module_to_state_dict(encoder, "encoder", "encoder"))
    state.update(_legacy_module_to_state_dict(dynamics, "dynamics", "latent_map"))
    state.update(_legacy_module_to_state_dict(decoder, "decoder", "decoder"))

    model = build_autoencoder(arch)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"legacy/new state_dict mismatch: missing={list(missing)} unexpected={list(unexpected)}"
        )
    return model


def load_any_checkpoint(
    in_dir: str | Path,
    *,
    arch: ArchConfig | None = None,
    basename: str = DEFAULT_BASENAME,
    map_location: str | torch.device = "cpu",
    legacy_root: str | Path | None = None,
) -> tuple[LatentDynamicsAutoencoder, ArchConfig]:
    """Try the new format first; fall back to the legacy three-file format.

    For the legacy format an explicit ``arch`` must be supplied (since the
    sidecar JSON does not exist in old runs); typically pass ``cfg.arch``.
    """
    in_path = Path(in_dir)
    if has_new_checkpoint(in_path, basename=basename):
        return load_checkpoint(in_path, basename=basename, map_location=map_location)
    if has_legacy_checkpoint(in_path):
        if arch is None:
            raise ValueError("legacy checkpoint requires an explicit ArchConfig")
        model = load_legacy_checkpoint(
            in_path, arch, map_location=map_location, legacy_root=legacy_root
        )
        return model, arch
    raise FileNotFoundError(f"no checkpoint (new or legacy) found in {in_path}")
