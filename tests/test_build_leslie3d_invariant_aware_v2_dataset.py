from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scripts import build_leslie3d_invariant_aware_v2_dataset as builder

from latentdynamics.config import load_config


def _synthetic_transition_neighborhoods() -> dict[str, np.ndarray]:
    s2 = np.asarray(builder.base.KNOWN_OBJECTS["S2"]["points"], dtype=np.float64)
    return {
        "origin": np.asarray([[0.0, 0.0, 0.0, 0.0537109375, 0.03759765625, 0.0263671875]]),
        "p_star": np.asarray(
            [
                [
                    17.5634765625,
                    12.2568359375,
                    8.5693359375,
                    19.9267578125,
                    13.986328125,
                    9.80859375,
                ]
            ]
        ),
        "S2": np.hstack((s2 - 1.0, s2 + 1.0)),
    }


def _write_synthetic_morse_sets(path: Path) -> None:
    boxes = _synthetic_transition_neighborhoods()
    rows = [
        [*boxes["S2"][0], 2],
        [*boxes["S2"][1], 2],
        [*boxes["p_star"][0], 4],
        [*boxes["origin"][0], 5],
    ]
    np.savetxt(path, np.asarray(rows), delimiter=",")


def test_v2_config_uses_isolated_paths_and_expected_counts() -> None:
    config = load_config("leslie3d_invariant_aware_v2")

    assert config.experiment_name == "leslie3d_invariant_aware_v2"
    assert config.paths.data_dir == Path("data/leslie3d_invariant_aware_v2")
    assert config.paths.output_dir == Path("output/leslie3d_invariant_aware_v2")
    assert config.paths.data_dir != load_config("leslie3d_invariant_aware").paths.data_dir
    assert (config.data.n_samples_train, config.data.n_samples_val) == (45_512, 20_661)

    smooth = load_config("leslie3d_invariant_aware_v2_smooth")
    assert smooth.experiment_name == "leslie3d_invariant_aware_v2_smooth"
    assert smooth.paths.data_dir == config.paths.data_dir
    assert smooth.paths.output_dir == Path("output/leslie3d_invariant_aware_v2_smooth")
    assert smooth.paths.output_dir != config.paths.output_dir
    assert smooth.arch.component("latent_map").hidden_shapes == (64, 64)
    assert smooth.arch.component("latent_map").activation == "gelu"
    assert config.arch.component("latent_map").activation == "relu"
    assert (
        smooth.arch.component("latent_map").out_activation
        == config.arch.component("latent_map").out_activation
    )
    assert smooth.arch.component("encoder") == config.arch.component("encoder")
    assert smooth.arch.component("decoder") == config.arch.component("decoder")
    assert smooth.training.warm_start_checkpoint_dir == Path(
        "output/leslie3d_invariant_aware/seed_20260803/models"
    )
    assert smooth.cmgdb.lower_bounds == [-0.34096482396125793, -0.2952786684036255]
    assert smooth.cmgdb.upper_bounds == [0.25832921266555786, 0.41028958559036255]
    assert smooth.seeds == [20260809]


def test_v2_builder_refuses_the_original_dataset_path() -> None:
    with pytest.raises(ValueError, match="refuses to overwrite"):
        builder.build(builder.base.DEFAULT_OUTPUT, builder.DEFAULT_MORSE_SETS)


def test_witness_banks_are_positive_deterministic_and_split_disjoint() -> None:
    train = builder.audited_witness_starts("train")
    validation = builder.audited_witness_starts("validation")

    assert train.shape == validation.shape == (27, 3)
    assert np.all(train > 0.0) and np.all(validation > 0.0)
    assert len(np.unique(train, axis=0)) == len(train)
    assert len(np.unique(validation, axis=0)) == len(validation)
    np.testing.assert_array_equal(train, builder.audited_witness_starts("train"))
    np.testing.assert_allclose(
        builder.reproduce_discovery_bases("train"),
        builder.TRAIN_WITNESS_BASES,
        rtol=0.0,
        atol=1e-18,
    )
    np.testing.assert_allclose(
        builder.reproduce_discovery_bases("validation"),
        builder.VALIDATION_WITNESS_BASES,
        rtol=0.0,
        atol=1e-18,
    )

    train_states = builder.trajectory_states(train)
    validation_states = builder.trajectory_states(validation)
    assert train_states.shape == validation_states.shape == (321, 27, 3)
    np.testing.assert_allclose(
        train_states[1:],
        builder.base.leslie(train_states[:-1]),
        rtol=1e-14,
        atol=1e-14,
    )
    disjointness = builder._row_disjointness(
        builder._trajectory_pair_rows(train_states),
        builder._trajectory_pair_rows(validation_states),
    )
    assert disjointness == {
        "training_rows": 8_640,
        "training_unique_rows": 8_640,
        "validation_rows": 8_640,
        "validation_unique_rows": 8_640,
        "exact_overlap_rows_at_15_digit_csv_precision": 0,
    }


def test_all_witnesses_follow_origin_p_star_s2_and_continue() -> None:
    neighborhoods = _synthetic_transition_neighborhoods()

    for role in ("train", "validation"):
        starts = builder.audited_witness_starts(role)
        audit = builder.audit_witnesses(
            role=role,
            starts=starts,
            states=builder.trajectory_states(starts),
            neighborhoods=neighborhoods,
        )

        assert audit["trajectory_count"] == 27
        assert audit["pair_rows"] == 8_640
        assert audit["all_start_in_origin_cell"] is True
        assert audit["all_enter_and_leave_p_star_before_S2"] is True
        assert audit["all_enter_and_leave_S2_within_horizon"] is True
        assert audit["worst_minimum_p_star_scaled_l2"] < 0.0027
        assert audit["worst_minimum_S2_scaled_l2"] < 5.1e-5
        for record in audit["records"]:
            assert record["p_star"]["first_entry_time"] < record["p_star"]["last_member_time"]
            assert record["p_star"]["last_member_time"] < record["S2"]["first_entry_time"]
            assert record["S2"]["last_member_time"] < builder.WITNESS_STEPS


def test_build_writes_v2_provenance_without_using_original_path(
    tmp_path: Path, monkeypatch
) -> None:
    morse_sets = tmp_path / "morse_sets"
    _write_synthetic_morse_sets(morse_sets)

    def tiny_base_split(role: str, _morse_sets_path: Path):
        accumulator = builder.base.PairAccumulator.empty()
        point = (
            np.asarray([[70.0, 1.0, 1.0]]) if role == "train" else np.asarray([[60.0, 2.0, 2.0]])
        )
        accumulator.add("v1_fixture", point)
        return accumulator, {"role": role, "fixture": True}

    monkeypatch.setattr(builder, "_base_split", tiny_base_split)
    output = tmp_path / "leslie3d_invariant_aware_v2"
    manifest = builder.build(output, morse_sets)

    assert manifest["schema_version"] == 2
    assert manifest["dataset_version"] == 2
    assert manifest["splits"]["train"]["rows"] == 8_641
    assert manifest["splits"]["validation"]["rows"] == 8_641
    assert manifest["audited_origin_p_star_s2_witnesses"]["steps"] == 320
    assert manifest["audited_origin_p_star_s2_witnesses"]["train"]["trajectory_count"] == 27
    assert manifest["audited_origin_p_star_s2_witnesses"]["validation"]["trajectory_count"] == 27
    assert manifest["final_split_disjointness"]["exact_overlap_rows_at_15_digit_csv_precision"] == 0
    assert manifest["direct_morse_sets_source"]["sha256"] == builder.base.sha256(morse_sets)
    assert Path(manifest["builder"]).name == Path(builder.__file__).name
    assert Path(manifest["base_builder"]["path"]).name == Path(builder.base.__file__).name

    on_disk = json.loads((output / "dataset_manifest.json").read_text())
    assert on_disk["builder_sha256"] == builder.base.sha256(Path(builder.__file__))
    for split, filename in (("train", "train.csv"), ("validation", "val.csv")):
        rows = np.loadtxt(output / filename, delimiter=",", skiprows=1)
        np.testing.assert_allclose(
            rows[:, 3:], builder.base.leslie(rows[:, :3]), rtol=1e-13, atol=1e-13
        )
        component = next(
            item
            for item in manifest["splits"][split]["components"]
            if item["name"] == builder.WITNESS_COMPONENT
        )
        assert component["rows"] == 8_640
        assert component["route"] == ["origin", "p_star", "S2", "onward"]
