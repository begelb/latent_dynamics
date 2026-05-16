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
