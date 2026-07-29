from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_module():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    script = scripts / "analyze_chafee_d1_checkpoint.py"
    spec = importlib.util.spec_from_file_location(
        "analyze_chafee_d1_checkpoint",
        script,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ANALYZE = _load_module()


def test_safe_target_rejects_source_and_descendants(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()

    with pytest.raises(ValueError, match="overlaps protected"):
        ANALYZE._assert_safe_target(source, source)
    with pytest.raises(ValueError, match="overlaps protected"):
        ANALYZE._assert_safe_target(source, source / "analysis")


def test_statistics_validation_checks_count_conservation(tmp_path: Path) -> None:
    payload = {
        "statistics": {
            "total_trajectories": 10_000,
            "excluded_zero_trajectories": 2_138,
            "conditioned_trajectories": 7_862,
            "counts": {"inside": 3_932, "outside": 3_930},
            "percentages": {
                "inside": 100.0 * 3_932 / 7_862,
                "outside": 100.0 * 3_930 / 7_862,
            },
        },
        "uniform_is_bistable": True,
        "roots_define_two_distinct_attractor_basins": True,
        "eligible_for_bistable_dimension_table": True,
    }
    path = tmp_path / "stats.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert ANALYZE._validate_statistics(path) == payload

    payload["statistics"]["counts"]["outside"] -= 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="do not conserve"):
        ANALYZE._validate_statistics(path)
