"""The fork check must accept a wheel install, not only the local checkout."""

from __future__ import annotations

import subprocess
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


def test_provenance_records_git_state_for_a_checkout(tmp_path):
    # A real git checkout records its revision and dirtiness.
    repo = tmp_path / "checkout"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--allow-empty",
            "-m",
            "init",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    state = cmgdb_fork._git_state(repo)
    assert len(state["revision"]) == 40
    assert state["dirty"] is False

    # A directory that is not a git checkout (a tarball install) yields a
    # marker instead of an exception.
    plain = tmp_path / "plain"
    plain.mkdir()
    assert cmgdb_fork._git_state(plain) == {"git": "not a git checkout"}

    # End to end: whichever tree the installed module lives under.
    checkout = Path(cmgdb_fork.CMGDB.__file__).resolve().parents[2]
    state = cmgdb_fork.cmgdb_provenance(checkout)
    assert state["version"]
    assert state["repository"] == str(checkout)
    if (checkout / ".git").exists():
        assert "revision" in state and "dirty" in state
    else:
        assert state["git"] == "not a git checkout"


def test_provenance_of_a_wheel_install_reports_no_repository(tmp_path):
    # A checkout that the module does not live under stands in for the wheel
    # case, whatever the developer's environment happens to be.
    state = cmgdb_fork.cmgdb_provenance(tmp_path)
    assert state["source"] == "installed distribution"
    assert "repository" not in state
    assert "revision" not in state
