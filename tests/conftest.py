"""Shared pytest fixtures for the latentdynamics test suite."""

import pytest


@pytest.fixture(scope="session")
def fixed_seed() -> int:
    return 42
