"""The fork check must accept a wheel install, not only the local checkout."""

from __future__ import annotations

from pathlib import Path

import pytest

from latentdynamics.analysis import cmgdb_fork


def test_installed_cmgdb_is_the_fork():
    module_path = cmgdb_fork.require_fork_cmgdb()
    assert module_path.name == "__init__.py"


def test_missing_fork_api_is_rejected(monkeypatch):
    monkeypatch.delattr(cmgdb_fork.CMGDB, "PrecomputedBoxMap", raising=False)
    with pytest.raises(RuntimeError, match="PrecomputedBoxMap"):
        cmgdb_fork.require_fork_cmgdb()


def test_provenance_records_git_state_for_a_checkout():
    checkout = Path(cmgdb_fork.CMGDB.__file__).resolve().parents[2]
    state = cmgdb_fork.cmgdb_provenance(checkout)
    assert state["version"]
    if (checkout / ".git").is_dir():
        assert state["repository"] == str(checkout)
        assert "revision" in state and "dirty" in state
    else:  # an installed wheel has no git state to record
        assert state["source"] == "installed distribution"


def test_provenance_of_a_wheel_install_reports_no_repository(tmp_path):
    # A checkout that the module does not live under stands in for the wheel
    # case, whatever the developer's environment happens to be.
    state = cmgdb_fork.cmgdb_provenance(tmp_path)
    assert state["source"] == "installed distribution"
    assert "repository" not in state
    assert "revision" not in state
