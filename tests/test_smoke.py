"""Milestone-1 smoke check: package is importable."""

import latentdynamics


def test_package_imports():
    assert latentdynamics.__version__ == "0.1.0"


def test_subpackages_importable():
    from latentdynamics import (
        analysis,
        cli,
        config,
        models,
        sampling,
        systems,
        training,
        viz,
    )

    for pkg in (analysis, cli, config, models, sampling, systems, training, viz):
        assert pkg is not None
