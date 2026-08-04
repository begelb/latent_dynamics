from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from scripts.census_leslie3d_latent_cycles import (
    DecoderClosedLeslie3DMap,
    NumpySequentialMap,
    _checkpoint_paths,
    _decoder_closed_torch_map,
    _default_output_name,
    canonical_cycle,
    cycles_equivalent,
)
from torch import nn


def test_checkpoint_basename_controls_paths_and_default_artifact_name(tmp_path) -> None:
    checkpoint, sidecar = _checkpoint_paths(tmp_path, "smooth_candidate")

    assert checkpoint == tmp_path / "smooth_candidate.pt"
    assert sidecar == tmp_path / "smooth_candidate.json"
    assert (
        _default_output_name(map_mode="latent", checkpoint_basename="smooth_candidate")
        == "dense_periodic_root_census_smooth_candidate.json"
    )
    assert (
        _default_output_name(map_mode="decoder_closed", checkpoint_basename="smooth_candidate")
        == "dense_periodic_root_census_smooth_candidate_decoder_closed.json"
    )
    assert (
        _default_output_name(map_mode="latent", checkpoint_basename="autoencoder")
        == "dense_periodic_root_census.json"
    )


def test_numpy_smooth_mlp_matches_torch_values_and_jacobians() -> None:
    torch.manual_seed(41)
    model = nn.Sequential(
        nn.Linear(2, 5),
        nn.GELU(),
        nn.Linear(5, 4),
        nn.Tanh(),
        nn.Linear(4, 2),
        nn.Tanh(),
    ).double()
    points = torch.tensor([[-0.3, 0.2], [0.1, -0.4], [0.7, 0.5]], dtype=torch.float64)
    evaluator = NumpySequentialMap(model)

    actual_values, actual_jacobians = evaluator.value_and_jacobian(points.numpy())
    expected_values = model(points).detach().numpy()
    expected_jacobians = np.stack(
        [
            torch.autograd.functional.jacobian(model, point, vectorize=True).detach().numpy()
            for point in points
        ]
    )

    np.testing.assert_allclose(actual_values, expected_values, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual_jacobians, expected_jacobians, rtol=1e-11, atol=1e-12)


def test_canonical_cycle_is_invariant_under_cyclic_rotation() -> None:
    cycle = np.asarray([[0.2, -0.1], [1.0, 0.3], [-0.4, 0.8]], dtype=np.float64)
    canonical = canonical_cycle(cycle)

    for shift in range(len(cycle)):
        rotated = canonical_cycle(np.roll(cycle, shift, axis=0))
        np.testing.assert_array_equal(rotated, canonical)
        assert cycles_equivalent(rotated, canonical, tolerance=1e-12)


def test_decoder_closed_map_matches_torch_values_and_jacobians() -> None:
    torch.manual_seed(73)
    decoder = nn.Sequential(
        nn.Linear(2, 5),
        nn.ReLU(),
        nn.Linear(5, 3),
        nn.Sigmoid(),
    ).double()
    encoder = nn.Sequential(
        nn.Linear(3, 6),
        nn.ReLU(),
        nn.Linear(6, 2),
        nn.Tanh(),
    ).double()
    model = nn.Module()
    model.decoder = decoder
    model.encoder = encoder
    scaler = SimpleNamespace(
        scale_=np.asarray([0.013, 0.021, 0.034]),
        min_=np.asarray([-0.17, 0.08, -0.31]),
    )
    params = {
        "th1": 28.9,
        "th2": 29.8,
        "th3": 22.0,
        "survival_p1": 0.7,
        "survival_p2": 0.7,
    }
    points = torch.tensor([[-0.31, 0.22], [0.14, -0.37], [0.48, 0.19]], dtype=torch.float64)
    evaluator = DecoderClosedLeslie3DMap(
        decoder=decoder,
        encoder=encoder,
        scaler=scaler,
        params=params,
    )
    torch_map = _decoder_closed_torch_map(model, scaler, params)

    actual_values, actual_jacobians = evaluator.value_and_jacobian(points.numpy())
    expected_values = torch_map(points).detach().numpy()
    expected_jacobians = np.stack(
        [
            torch.autograd.functional.jacobian(torch_map, point, vectorize=True).detach().numpy()
            for point in points
        ]
    )

    np.testing.assert_allclose(actual_values, expected_values, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual_jacobians, expected_jacobians, rtol=1e-11, atol=1e-12)
