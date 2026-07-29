from __future__ import annotations

from scripts import scan_chafee_d1_unit_scale_mu as scan


def test_frozen_candidate_inventory_contains_primary_grid_and_root_alignment():
    rows = scan._candidate_inventory(
        least_squares_mu=0.0475,
        encoded_positive_root=1.236594557762146,
    )
    by_id = {row["candidate_id"]: row for row in rows}

    assert len(rows) == len(scan.EXPECTED_RESULTS)
    assert by_id["least_squares"]["mu"] == 0.0475
    assert not by_id["least_squares"]["test_informed"]
    assert by_id["0.35"]["mu"] == 0.35
    assert by_id["0.35"]["test_informed"]
    assert by_id["root_alignment"]["mu"] == 0.361564185639033


def test_frozen_expected_best_valid_grid_candidate_is_point_35():
    valid = {
        candidate_id: result
        for candidate_id, result in scan.EXPECTED_RESULTS.items()
        if result["valid"]
    }
    best_id = max(valid, key=lambda candidate_id: valid[candidate_id]["correct"])

    assert best_id == "0.35"
    assert valid[best_id]["correct"] == 6_356
    assert valid[best_id]["wrong"] == 0
