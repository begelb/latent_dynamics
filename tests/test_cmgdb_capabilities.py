"""The CMGDB check reports capabilities, and works on any build.

These must pass against any installed CMGDB: the point of the capability probe
is that "which build is this?" stops being the question a caller has to
answer.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import latentdynamics.analysis.cmgdb_features as caps


def test_capabilities_cover_every_declared_feature():
    capabilities = caps.cmgdb_capabilities()
    assert set(capabilities) == set(caps.CAPABILITIES)
    assert all(isinstance(value, bool) for value in capabilities.values())


def test_precomputed_box_map_class_is_distinguished_from_the_module():
    # The attribute may resolve to the submodule of the same name rather than
    # the class. The capability must report what the object actually is.
    import inspect

    candidate = getattr(caps.CMGDB, "PrecomputedBoxMap", None)
    reported = caps.cmgdb_capabilities()["precomputed_box_map_class"]
    assert reported == (
        inspect.isclass(candidate)
        and "subdiv_max" in inspect.signature(candidate.__init__).parameters
    )


def test_a_universally_present_routine_is_requirable():
    # ComputeMorseGraph is in every CMGDB release this project supports, so
    # nothing that only needs it should ever be gated. Requiring an empty
    # feature set must also pass.
    assert hasattr(caps.CMGDB, "ComputeMorseGraph")
    assert caps.require_cmgdb_features().name == "__init__.py"


def test_missing_feature_is_rejected_by_name(monkeypatch):
    monkeypatch.delattr(caps.CMGDB, "ComputeConleyIndexForCells", raising=False)
    assert caps.missing_cmgdb_features("conley_index_for_cells") == [
        "conley_index_for_cells"
    ]
    with pytest.raises(RuntimeError, match="conley_index_for_cells"):
        caps.require_cmgdb_features("conley_index_for_cells")


def test_unknown_feature_name_is_an_error():
    with pytest.raises(KeyError, match="not_a_feature"):
        caps.missing_cmgdb_features("not_a_feature")


def test_provenance_records_capabilities_without_gating(monkeypatch):
    # A manifest must be writable for a build missing newer routines:
    # provenance records, it does not refuse.
    monkeypatch.delattr(caps.CMGDB, "MorseDirectedPathCells", raising=False)
    state = caps.cmgdb_provenance()
    assert state["capabilities"]["morse_directed_path_cells"] is False
    assert state["version"]


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
    state = caps._git_state(repo)
    assert len(state["revision"]) == 40
    assert state["dirty"] is False

    # A directory that is not a git checkout (a tarball install) yields a
    # marker instead of an exception.
    plain = tmp_path / "plain"
    plain.mkdir()
    assert caps._git_state(plain) == {"git": "not a git checkout"}

    # End to end: whichever tree the installed module lives under.
    checkout = Path(caps.CMGDB.__file__).resolve().parents[2]
    state = caps.cmgdb_provenance(checkout)
    assert state["version"]
    assert state["repository"] == str(checkout)
    if (checkout / ".git").exists():
        assert "revision" in state and "dirty" in state
    else:
        assert state["git"] == "not a git checkout"


def test_provenance_of_a_wheel_install_reports_no_repository(tmp_path):
    # A checkout that the module does not live under stands in for the wheel
    # case, whatever the developer's environment happens to be.
    state = caps.cmgdb_provenance(tmp_path)
    assert state["source"] == "installed distribution"
    assert "repository" not in state
    assert "revision" not in state
