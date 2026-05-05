"""Tests for the unique-membership Morse metric on synthetic fixtures."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

from latentdynamics.analysis.morse_metrics import (
    check_unique_membership,
    find_morse_label_1d,
    find_seed_subdirs,
    get_minimal_labels,
)


def _write_morse_sets(path, rows):
    text = "a,b,label\n" + "\n".join(",".join(str(c) for c in row) for row in rows) + "\n"
    path.write_text(text)


def _write_morse_graph(path, edges):
    """Edges given as ``[(src, dst), ...]``; nodes inferred."""
    nodes = sorted({n for e in edges for n in e})
    body = "\n".join(f"  {n};" for n in nodes)
    body += "\n" + "\n".join(f"  {s} -> {d};" for s, d in edges)
    path.write_text(f"digraph {{\n{body}\n}}\n")


class TestMorseGraphHelpers:
    def test_minimal_labels_are_sinks(self, tmp_path):
        graph = tmp_path / "morse_graph"
        _write_morse_graph(graph, [(0, 1), (2, 1)])
        # 1 is a sink (no outgoing), 0 and 2 are sources.
        assert get_minimal_labels(graph) == {1}

    def test_find_morse_label_1d(self):
        import pandas as pd

        df = pd.DataFrame({"a": [-1.0, 0.5], "b": [0.0, 1.0], "label": [0, 1]})
        assert find_morse_label_1d(-0.5, df) == 0
        assert find_morse_label_1d(0.7, df) == 1
        assert find_morse_label_1d(0.25, df) is None


class TestCheckUniqueMembership:
    def test_unique_membership_with_identity_encoder(self, tmp_path):
        morse_sets = tmp_path / "morse_sets.csv"
        morse_graph = tmp_path / "morse_graph"
        # Intervals are in scaled (encoder-output) space [0, 1].
        _write_morse_sets(morse_sets, [[0.0, 0.15, 0], [0.85, 1.0, 1], [0.4, 0.6, 2]])
        _write_morse_graph(morse_graph, [(2, 0), (2, 1)])  # 0 and 1 are sinks; 2 is the saddle.

        class FirstColumnEncoder(torch.nn.Module):
            dummy = torch.nn.Parameter(torch.zeros(1))

            def forward(self, x):
                return x[:, :1]

        encoder = FirstColumnEncoder()
        scaler = MinMaxScaler().fit(np.array([[-1.0], [0.0], [1.0]]))

        fixed = {
            "a0": np.array([-1.0]),  # scaled -> 0.0  -> label 0 (sink)
            "a1": np.array([1.0]),  # scaled -> 1.0  -> label 1 (sink)
            "r": np.array([0.0]),  # scaled -> 0.5  -> label 2 (non-sink)
        }
        labels, metrics = check_unique_membership(
            encoder=encoder,
            scaler=scaler,
            morse_sets_path=morse_sets,
            morse_graph_path=morse_graph,
            fixed_points=fixed,
            device=torch.device("cpu"),
        )
        assert labels == {"a0": 0, "a1": 1, "r": 2}
        assert metrics == {"a0": True, "a1": True, "r": True}

    def test_returns_false_when_a0_not_in_any_morse_set(self, tmp_path):
        morse_sets = tmp_path / "morse_sets.csv"
        morse_graph = tmp_path / "morse_graph"
        _write_morse_sets(morse_sets, [[0.85, 1.0, 1]])
        _write_morse_graph(morse_graph, [])  # 1 is trivially a sink

        class FirstColumnEncoder(torch.nn.Module):
            dummy = torch.nn.Parameter(torch.zeros(1))

            def forward(self, x):
                return x[:, :1]

        encoder = FirstColumnEncoder()
        scaler = MinMaxScaler().fit(np.array([[-1.0], [1.0]]))
        fixed = {
            "a0": np.array([-1.0]),  # scaled 0.0 -> not in any interval
            "a1": np.array([1.0]),  # scaled 1.0 -> label 1
            "r": np.array([0.0]),  # scaled 0.5 -> not in any interval
        }
        labels, metrics = check_unique_membership(
            encoder=encoder,
            scaler=scaler,
            morse_sets_path=morse_sets,
            morse_graph_path=morse_graph,
            fixed_points=fixed,
            device=torch.device("cpu"),
        )
        assert labels["a0"] is None
        assert metrics["a0"] is False


class TestFindSeedSubdirs:
    def test_finds_only_complete_runs(self, tmp_path):
        good = tmp_path / "seed_0"
        (good / "models").mkdir(parents=True)
        (good / "models" / "encoder.pt").write_bytes(b"")
        (good / "MG").mkdir(parents=True)
        (good / "MG" / "morse_sets").write_text("")
        (good / "MG" / "morse_graph").write_text("")

        bad = tmp_path / "seed_1"
        (bad / "models").mkdir(parents=True)
        # Missing MG dir entirely.

        ignore = tmp_path / "not_a_seed.txt"
        ignore.write_text("")

        result = find_seed_subdirs(tmp_path)
        assert result == ["seed_0"]
