"""The in-repo minimal checkpoints that make quick/morse work from a bare clone.

``artifacts/reference_models/<key>/`` mirrors the ``replay_sources/`` layout
with just the network weights, arch sidecar, scaler, and recorded latent
bounds for each paper experiment -- a few hundred kilobytes committed to git
so ``load_experiment`` can recompute Morse graphs (quick/morse modes, Colab
included) without the released artifact bundles.
"""

from __future__ import annotations

from pathlib import Path

from latentdynamics.replay import _seed_reference_models
from latentdynamics.replay.fetch import _normalize_experiment_name

STAGE = Path(__file__).resolve().parents[1] / "artifacts" / "reference_models"

#: Experiment key -> files every bare clone needs for quick/morse recomputes.
EXPECTED = {
    "leslie_2gen_contraction": [
        "leslie_2gen_contraction/models/autoencoder.pt",
        "leslie_2gen_contraction/models/autoencoder.json",
        "leslie_2gen_contraction/scalers/train/scaler.gz",
        "leslie_2gen_contraction/mg_params_log.txt",
    ],
    "leslie3d_example1": [
        "leslie3d_example1/spurious_attractor_ex/models/autoencoder.pt",
        "leslie3d_example1/spurious_attractor_ex/models/autoencoder.json",
        "leslie3d_example1/28.9_29.8_22.0/scalers/scaler.gz",
        "leslie3d_example1/spurious_attractor_ex/mg_params_log.txt",
    ],
    "chafee_infante": [
        "chafee_infante/replay/models/autoencoder.pt",
        "chafee_infante/replay/models/autoencoder.json",
        "chafee_infante/replay/scalers/train/scaler.gz",
        "chafee_infante/replay/mg_params_log.txt",
    ],
    "coral": [
        "coral/train_500/seed_16/models/autoencoder.pt",
        "coral/train_500/seed_16/models/autoencoder.json",
        "coral/data/scalers/train_500/scaler.gz",
        "coral/train_500/seed_16/mg_params_log.txt",
    ],
}


def test_every_experiment_ships_its_minimal_checkpoint_tree():
    for key, rels in EXPECTED.items():
        for rel in rels:
            path = STAGE / key / rel
            assert path.is_file() and path.stat().st_size > 0, path


def test_staged_trees_stay_small_enough_for_git():
    total = sum(p.stat().st_size for key in EXPECTED for p in (STAGE / key).rglob("*") if p.is_file())
    assert total < 2 * 1024 * 1024  # weights, not artifacts: keep it far below GitHub's limits


def test_replay_names_resolve_to_staged_trees():
    for name in ("leslie_2gen_contraction_replay", "leslie3d_example1_replay",
                 "chafee_infante_replay", "coral_basic", "coral"):
        assert _normalize_experiment_name(name) in EXPECTED
        # Seeding is a no-op copy when replay_sources is fully populated, but
        # it must report that a staged tree exists for every paper experiment.
        assert _seed_reference_models(name) is True


def test_unknown_experiment_has_no_staged_tree():
    assert _seed_reference_models("not_a_real_experiment") is False
