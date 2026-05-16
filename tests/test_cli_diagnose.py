"""Unit tests for the diagnose CLI stage helpers.

The helpers are tested with synthetic torch modules and numpy arrays so we
do not need a config or trained checkpoint on disk.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from latentdynamics.cli import diagnose


def test_module_imports():
    # Sanity check that the helpers we plan to add are exported.
    # This will fail until Task 2 lands the first helper.
    assert hasattr(diagnose, "_encoder_extent_report")


def test_encoder_extent_report_tanh_healthy():
    # Encoded data spans most of [-1, 1]^2: max_extent ~ 1.8, reference 2.0.
    encoded = np.array([[-0.9, -0.9], [0.9, 0.9], [-0.9, 0.9], [0.9, -0.9]])
    block, collapsed = diagnose._encoder_extent_report(
        encoded, out_activation="tanh", collapse_thresh=0.02
    )
    assert collapsed is False
    assert block["out_activation"] == "tanh"
    assert block["reference_span"] == 2.0
    assert block["max_extent"] == pytest.approx(1.8)
    assert block["max_extent_relative"] == pytest.approx(0.9)
    assert block["extent_per_axis"] == pytest.approx([1.8, 1.8])


def test_encoder_extent_report_tanh_collapsed():
    # Encoded data clustered in a 0.01-wide region around 0: max_extent 0.01,
    # ratio 0.005, below the 0.02 threshold.
    encoded = np.array([[0.0, 0.0], [0.01, 0.01], [0.005, -0.005]])
    block, collapsed = diagnose._encoder_extent_report(
        encoded, out_activation="tanh", collapse_thresh=0.02
    )
    assert collapsed is True
    assert block["max_extent_relative"] < 0.02


def test_encoder_extent_report_sigmoid_collapsed():
    # Reference span 1.0; same 0.005 max_extent now reads relative=0.005.
    encoded = np.array([[0.5, 0.5], [0.505, 0.5], [0.5, 0.505]])
    block, collapsed = diagnose._encoder_extent_report(
        encoded, out_activation="sigmoid", collapse_thresh=0.02
    )
    assert collapsed is True
    assert block["reference_span"] == 1.0


def test_encoder_extent_report_linear_healthy():
    # Linear out: reference_span is null, flag uses absolute max_extent.
    # max_extent = 0.5 here, well above 0.02.
    encoded = np.array([[-0.25, -0.25], [0.25, 0.25]])
    block, collapsed = diagnose._encoder_extent_report(
        encoded, out_activation="none", collapse_thresh=0.02
    )
    assert collapsed is False
    assert block["reference_span"] is None
    assert block["max_extent_relative"] is None
    assert block["max_extent"] == pytest.approx(0.5)


def test_encoder_extent_report_linear_collapsed():
    # max_extent = 0.01 absolute, below the 0.02 threshold.
    encoded = np.array([[0.0, 0.0], [0.01, 0.005]])
    block, collapsed = diagnose._encoder_extent_report(
        encoded, out_activation="none", collapse_thresh=0.02
    )
    assert collapsed is True
    assert block["reference_span"] is None


def _make_grid_2d(low=-1.0, high=1.0, n=10) -> np.ndarray:
    axis = np.linspace(low, high, n)
    g1, g2 = np.meshgrid(axis, axis, indexing="ij")
    return np.stack([g1.ravel(), g2.ravel()], axis=-1)


def test_latent_map_one_step_identity():
    # G = identity: contraction_ratio ~ 1.0, mean_step ~ 0.
    grid = _make_grid_2d()
    G = nn.Identity()
    bounds = diagnose.LatentBounds(
        lower=np.array([-1.0, -1.0]), upper=np.array([1.0, 1.0])
    )
    block, image, overcontracted = diagnose._latent_map_one_step_report(
        G, grid, device=torch.device("cpu"), bounds=bounds,
        contraction_thresh=0.05, near_identity_thresh=0.01,
    )
    assert overcontracted is False
    assert block["contraction_ratio"] == pytest.approx(1.0, abs=1e-6)
    assert block["mean_step_relative"] == pytest.approx(0.0, abs=1e-6)
    assert block["near_identity"] is True
    assert image.shape == grid.shape


def test_latent_map_one_step_overcontracted():
    # G maps everything to 0: image diameter is 0, ratio 0, far below 0.05.
    grid = _make_grid_2d()
    G = nn.Linear(2, 2, bias=True)
    with torch.no_grad():
        G.weight.zero_()
        G.bias.zero_()
    bounds = diagnose.LatentBounds(
        lower=np.array([-1.0, -1.0]), upper=np.array([1.0, 1.0])
    )
    block, _, overcontracted = diagnose._latent_map_one_step_report(
        G, grid, device=torch.device("cpu"), bounds=bounds,
        contraction_thresh=0.05, near_identity_thresh=0.01,
    )
    assert overcontracted is True
    assert block["contraction_ratio"] == pytest.approx(0.0, abs=1e-6)
    assert block["near_identity"] is False


def test_latent_map_one_step_non_identity_non_collapsed():
    # G = 0.5 * id: contraction_ratio ~ 0.5, near_identity False, not flagged.
    grid = _make_grid_2d()
    G = nn.Linear(2, 2, bias=True)
    with torch.no_grad():
        G.weight.copy_(torch.eye(2) * 0.5)
        G.bias.zero_()
    bounds = diagnose.LatentBounds(
        lower=np.array([-1.0, -1.0]), upper=np.array([1.0, 1.0])
    )
    block, _, overcontracted = diagnose._latent_map_one_step_report(
        G, grid, device=torch.device("cpu"), bounds=bounds,
        contraction_thresh=0.05, near_identity_thresh=0.01,
    )
    assert overcontracted is False
    assert block["contraction_ratio"] == pytest.approx(0.5, abs=1e-6)
    assert block["near_identity"] is False


class _MockModel:
    """Minimal stand-in with .encoder and .decoder attributes."""

    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        self.encoder = encoder
        self.decoder = decoder


class _ArchStub:
    def __init__(self, high_dims: int, low_dims: int,
                 enc_out: str, dec_out: str) -> None:
        self.high_dims = high_dims
        self.low_dims = low_dims
        self.encoder_out_activation = enc_out
        self.decoder_out_activation = dec_out


def test_matched_dim_identity_report_not_applicable_unmatched_dims():
    model = _MockModel(nn.Identity(), nn.Identity())
    arch = _ArchStub(high_dims=3, low_dims=2, enc_out="none", dec_out="none")
    bounds = diagnose.LatentBounds(
        lower=np.array([-1.0, -1.0]), upper=np.array([1.0, 1.0])
    )
    block = diagnose._matched_dim_identity_report(
        model, arch=arch, data_sample_scaled=np.zeros((4, 3)),
        grid=_make_grid_2d(), bounds=bounds, device=torch.device("cpu"),
        near_identity_thresh=0.01,
    )
    assert block == {
        "matched_dims": False,
        "encoder_near_identity": None,
        "decoder_near_identity": None,
        "mean_step_E_relative": None,
        "mean_step_D_relative": None,
    }


def test_matched_dim_identity_report_not_applicable_tanh_out():
    model = _MockModel(nn.Identity(), nn.Identity())
    arch = _ArchStub(high_dims=2, low_dims=2, enc_out="tanh", dec_out="none")
    bounds = diagnose.LatentBounds(
        lower=np.array([-1.0, -1.0]), upper=np.array([1.0, 1.0])
    )
    block = diagnose._matched_dim_identity_report(
        model, arch=arch, data_sample_scaled=np.zeros((4, 2)),
        grid=_make_grid_2d(), bounds=bounds, device=torch.device("cpu"),
        near_identity_thresh=0.01,
    )
    assert block["matched_dims"] is True  # dims match...
    # ... but bounded encoder activation disqualifies that side only.
    assert block["encoder_near_identity"] is None
    assert block["mean_step_E_relative"] is None
    # Decoder side still has linear out, so it IS computed.
    assert block["decoder_near_identity"] is not None
    assert block["mean_step_D_relative"] is not None


def test_matched_dim_identity_report_identity_modules():
    # E=id, D=id, matched dims, linear out: both soft notes fire True.
    model = _MockModel(nn.Identity(), nn.Identity())
    arch = _ArchStub(high_dims=2, low_dims=2, enc_out="none", dec_out="none")
    bounds = diagnose.LatentBounds(
        lower=np.array([-1.0, -1.0]), upper=np.array([1.0, 1.0])
    )
    data = np.array([[0.1, 0.2], [0.3, 0.4]])
    block = diagnose._matched_dim_identity_report(
        model, arch=arch, data_sample_scaled=data,
        grid=_make_grid_2d(), bounds=bounds, device=torch.device("cpu"),
        near_identity_thresh=0.01,
    )
    assert block["matched_dims"] is True
    assert block["encoder_near_identity"] is True
    assert block["decoder_near_identity"] is True
    assert block["mean_step_E_relative"] == pytest.approx(0.0, abs=1e-6)
    assert block["mean_step_D_relative"] == pytest.approx(0.0, abs=1e-6)


def test_save_one_step_plot_2d(tmp_path):
    grid = _make_grid_2d()
    image = grid * 0.5
    bounds = diagnose.LatentBounds(
        lower=np.array([-1.0, -1.0]), upper=np.array([1.0, 1.0])
    )
    out_path = tmp_path / "figures" / "latent_map_one_step.png"
    diagnose._save_one_step_plot(grid, image, bounds, out_path)
    assert out_path.is_file()
    assert out_path.stat().st_size > 0


def test_save_one_step_plot_1d(tmp_path):
    grid = np.linspace(-1.0, 1.0, 50).reshape(-1, 1)
    image = grid * 0.7
    bounds = diagnose.LatentBounds(lower=np.array([-1.0]), upper=np.array([1.0]))
    out_path = tmp_path / "figures" / "latent_map_one_step.png"
    diagnose._save_one_step_plot(grid, image, bounds, out_path)
    assert out_path.is_file()


import json
from pathlib import Path


def test_run_produces_new_schema(tmp_path):
    """Integration test against the existing leslie2d_to_2d_test_110 checkpoint.

    Skips if the checkpoint isn't on disk (CI environments without it).
    """
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "configs" / "leslie2d_to_2d_test_110.yaml"
    checkpoint_dir = repo_root / "output" / "leslie2d_to_2d_test_110" / "seed_0"
    if not (checkpoint_dir / "models").is_dir():
        pytest.skip("test_110 checkpoint not present")

    from latentdynamics.config import load_config
    import copy
    cfg = load_config(config_path)
    # Make paths absolute so they work regardless of pytest's cwd
    cfg.paths.data_dir = (repo_root / cfg.paths.data_dir).resolve()
    cfg.paths.output_dir = (repo_root / cfg.paths.output_dir).resolve()
    # The models live in seed_0, scalers live in parent. Deep copy and
    # patch output_dir to seed_0 for the diagnose run (models are there),
    # but scaler_path will still use the parent dir via the original cfg.
    cfg_seed0 = copy.deepcopy(cfg)
    cfg_seed0.paths.output_dir = cfg.paths.output_dir / "seed_0"
    write_root = tmp_path / "diag_out"
    write_root.mkdir()
    # Use the original cfg for scaler paths but seed0 cfg for models
    # Actually, we need to call the function with the seed0 config and fix scaler loading
    # Monkey-patch _load_train_data_scaled to use the correct scaler path
    from latentdynamics.cli import diagnose as diag_module
    original_load = diag_module._load_train_data_scaled

    def patched_load(cfg_arg, train_file):
        # Use the original cfg's paths for scaler, but the data from cfg_seed0
        high = cfg_arg.arch.high_dims
        import numpy as np
        import joblib
        train = np.loadtxt(cfg_arg.paths.data_dir / f"{train_file}.csv", delimiter=",", skiprows=1)
        val = np.loadtxt(cfg_arg.paths.val_csv(), delimiter=",", skiprows=1)
        # Use parent output dir's scaler
        scaler_path = cfg.paths.output_dir / "scalers" / train_file / "scaler.gz"
        scaler = joblib.load(scaler_path)
        pieces = [
            scaler.transform(train[:, :high]),
            scaler.transform(val[:, :high]),
            scaler.transform(train[:, high:]),
            scaler.transform(val[:, high:]),
        ]
        return np.vstack(pieces).astype(np.float64)

    diag_module._load_train_data_scaled = patched_load
    try:
        payload = diagnose.run(cfg_seed0, train_file="train", out_dir=write_root, verbose=False)
    finally:
        diag_module._load_train_data_scaled = original_load

    diag_path = write_root / "diagnose.json"
    assert diag_path.is_file()
    saved = json.loads(diag_path.read_text())
    # New fields present
    assert "diagnostic" in saved
    assert saved["diagnostic"] in {
        "ok", "encoder_collapsed", "latent_map_overcontracted",
        "encoder_collapsed_and_latent_overcontracted",
    }
    assert "hard_flags" in saved
    assert set(saved["hard_flags"].keys()) == {
        "encoder_collapsed", "latent_map_overcontracted"
    }
    assert "encoder" in saved
    assert "latent_map" in saved
    assert {"max_extent_relative", "max_extent", "extent_per_axis",
            "out_activation", "reference_span"} <= saved["encoder"].keys()
    assert {"contraction_ratio", "mean_step_relative", "near_identity",
            "n_grid_points", "grid_diameter", "image_diameter"} <= saved["latent_map"].keys()
    # Old fields gone
    for legacy in ("n_distinct_limit_points", "n_terminal_clusters_all",
                   "n_terminal_clusters_converged", "frac_unconverged",
                   "mean_iter_to_convergence", "max_iter_to_convergence",
                   "convergence_eps", "cluster_eps", "n_iter"):
        assert legacy not in saved, f"legacy field {legacy} still present"
    # Figures
    assert (write_root / "figures" / "latent_pointcloud.png").is_file()
    assert (write_root / "figures" / "latent_map_one_step.png").is_file()
    # Old figure gone
    assert not (write_root / "figures" / "latent_orbits.png").is_file()
