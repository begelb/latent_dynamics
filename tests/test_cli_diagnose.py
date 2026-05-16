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
