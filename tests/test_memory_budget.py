"""Tests for memory budget env-var parsing in morse.py."""

from __future__ import annotations

import os
import pytest

from latentdynamics.analysis.morse import _get_memory_budget_bytes


class TestGetMemoryBudgetBytes:
    """Cover unit semantics and edge cases for memory budget env-var resolution."""

    def test_unset_env_var_returns_none(self, monkeypatch):
        """No env var set returns None."""
        monkeypatch.delenv("LATENTDYNAMICS_MEM_BUDGET_BYTES", raising=False)
        assert _get_memory_budget_bytes() is None

    def test_empty_env_var_returns_none(self, monkeypatch):
        """Empty string env var returns None."""
        monkeypatch.setenv("LATENTDYNAMICS_MEM_BUDGET_BYTES", "")
        assert _get_memory_budget_bytes() is None

    def test_whitespace_only_returns_none(self, monkeypatch):
        """Whitespace-only env var returns None."""
        monkeypatch.setenv("LATENTDYNAMICS_MEM_BUDGET_BYTES", "   ")
        assert _get_memory_budget_bytes() is None

    def test_gigabyte_suffix_uppercase(self, monkeypatch):
        """4G (uppercase) -> 4 * 1024^3 bytes."""
        monkeypatch.setenv("LATENTDYNAMICS_MEM_BUDGET_BYTES", "4G")
        assert _get_memory_budget_bytes() == 4 * 1024**3

    def test_gigabyte_suffix_lowercase(self, monkeypatch):
        """4g (lowercase) -> 4 * 1024^3 bytes."""
        monkeypatch.setenv("LATENTDYNAMICS_MEM_BUDGET_BYTES", "4g")
        assert _get_memory_budget_bytes() == 4 * 1024**3

    def test_megabyte_suffix_uppercase(self, monkeypatch):
        """512M (uppercase) -> 512 * 1024^2 bytes."""
        monkeypatch.setenv("LATENTDYNAMICS_MEM_BUDGET_BYTES", "512M")
        assert _get_memory_budget_bytes() == 512 * 1024**2

    def test_megabyte_suffix_lowercase(self, monkeypatch):
        """512m (lowercase) -> 512 * 1024^2 bytes."""
        monkeypatch.setenv("LATENTDYNAMICS_MEM_BUDGET_BYTES", "512m")
        assert _get_memory_budget_bytes() == 512 * 1024**2

    def test_kilobyte_suffix_uppercase(self, monkeypatch):
        """2K (uppercase) -> 2 * 1024 bytes."""
        monkeypatch.setenv("LATENTDYNAMICS_MEM_BUDGET_BYTES", "2K")
        assert _get_memory_budget_bytes() == 2 * 1024

    def test_kilobyte_suffix_lowercase(self, monkeypatch):
        """2k (lowercase) -> 2 * 1024 bytes."""
        monkeypatch.setenv("LATENTDYNAMICS_MEM_BUDGET_BYTES", "2k")
        assert _get_memory_budget_bytes() == 2 * 1024

    def test_terabyte_suffix_uppercase(self, monkeypatch):
        """1T (uppercase) -> 1024^4 bytes."""
        monkeypatch.setenv("LATENTDYNAMICS_MEM_BUDGET_BYTES", "1T")
        assert _get_memory_budget_bytes() == 1024**4

    def test_terabyte_suffix_lowercase(self, monkeypatch):
        """1t (lowercase) -> 1024^4 bytes."""
        monkeypatch.setenv("LATENTDYNAMICS_MEM_BUDGET_BYTES", "1t")
        assert _get_memory_budget_bytes() == 1024**4

    def test_bare_integer_is_bytes(self, monkeypatch):
        """Bare integer defaults to bytes (not MB). 1024 -> 1024 BYTES."""
        monkeypatch.setenv("LATENTDYNAMICS_MEM_BUDGET_BYTES", "1024")
        assert _get_memory_budget_bytes() == 1024

    def test_bare_zero_returns_none(self, monkeypatch):
        """Zero is not a meaningful budget; returns None."""
        monkeypatch.setenv("LATENTDYNAMICS_MEM_BUDGET_BYTES", "0")
        assert _get_memory_budget_bytes() is None

    def test_negative_integer_returns_none(self, monkeypatch):
        """Negative value is malformed; returns None."""
        monkeypatch.setenv("LATENTDYNAMICS_MEM_BUDGET_BYTES", "-1")
        assert _get_memory_budget_bytes() is None

    def test_negative_with_suffix_returns_none(self, monkeypatch):
        """Negative value with suffix is malformed; returns None."""
        monkeypatch.setenv("LATENTDYNAMICS_MEM_BUDGET_BYTES", "-2G")
        assert _get_memory_budget_bytes() is None

    def test_non_numeric_prefix_returns_none(self, monkeypatch):
        """Non-numeric prefix (abc) is malformed; returns None."""
        monkeypatch.setenv("LATENTDYNAMICS_MEM_BUDGET_BYTES", "abc")
        assert _get_memory_budget_bytes() is None

    def test_non_numeric_prefix_with_suffix_returns_none(self, monkeypatch):
        """Non-numeric prefix with a valid suffix is malformed; returns None."""
        monkeypatch.setenv("LATENTDYNAMICS_MEM_BUDGET_BYTES", "xyzG")
        assert _get_memory_budget_bytes() is None

    def test_float_with_suffix_returns_none(self, monkeypatch):
        """Float value (e.g., '1.5G') is malformed; returns None."""
        monkeypatch.setenv("LATENTDYNAMICS_MEM_BUDGET_BYTES", "1.5G")
        assert _get_memory_budget_bytes() is None

    def test_suffix_in_middle_returns_none(self, monkeypatch):
        """Suffix in the middle (e.g., 'G4') is malformed; returns None."""
        monkeypatch.setenv("LATENTDYNAMICS_MEM_BUDGET_BYTES", "G4")
        assert _get_memory_budget_bytes() is None

    def test_leading_trailing_whitespace_stripped(self, monkeypatch):
        """Leading/trailing whitespace is stripped before parsing."""
        monkeypatch.setenv("LATENTDYNAMICS_MEM_BUDGET_BYTES", "  512M  ")
        assert _get_memory_budget_bytes() == 512 * 1024**2

    def test_large_value_preserves_magnitude(self, monkeypatch):
        """Large values (e.g., 100G) compute correctly."""
        monkeypatch.setenv("LATENTDYNAMICS_MEM_BUDGET_BYTES", "100G")
        assert _get_memory_budget_bytes() == 100 * 1024**3

    def test_one_byte(self, monkeypatch):
        """Minimum nonzero value: 1 byte."""
        monkeypatch.setenv("LATENTDYNAMICS_MEM_BUDGET_BYTES", "1")
        assert _get_memory_budget_bytes() == 1
