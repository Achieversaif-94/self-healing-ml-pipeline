"""
Test for PSI drift detection.

Verifies that:
1. Identical distributions produce ~0 PSI
2. Known-shifted distributions produce PSI > 0.2
3. PSI calculation handles moderate shifts
4. PSI handles edge cases (empty bins)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from importlib import import_module
fastapi_module = import_module("05_fastapi_app")
calculate_psi = fastapi_module.calculate_psi


def test_psi_identical_distributions():
    """PSI of identical distributions should be ~0."""
    data = np.random.normal(5, 1, 1000)
    psi = calculate_psi(data, data)
    assert psi < 0.01, f"Expected PSI ~0, got {psi}"


def test_psi_shifted_distribution():
    """PSI of strongly shifted distribution should exceed 0.2."""
    np.random.seed(42)
    expected = np.random.normal(5, 1, 1000)
    actual = np.random.normal(7, 1, 1000)
    psi = calculate_psi(expected, actual)
    assert psi > 0.2, f"Expected PSI > 0.2 for shifted data, got {psi}"


def test_psi_moderate_shift():
    """PSI of slight shift should be in moderate range."""
    np.random.seed(42)
    expected = np.random.normal(5, 1, 1000)
    actual = np.random.normal(5.5, 1, 1000)
    psi = calculate_psi(expected, actual)
    assert 0.01 < psi < 0.6, f"Expected moderate PSI, got {psi}"


def test_psi_handles_empty_bins():
    """PSI should handle bins with zero samples (epsilon guard)."""
    expected = np.array([1, 1, 1, 1, 1])
    actual = np.array([100, 100, 100, 100, 100])
    psi = calculate_psi(expected, actual)
    assert psi >= 0, "PSI should never be negative"
    assert not np.isnan(psi), "PSI should not be NaN"