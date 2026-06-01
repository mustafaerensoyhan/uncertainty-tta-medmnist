"""
Unit tests for the McNemar p-value used in scripts/significance.py.

Run with:  pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from significance import mcnemar  # noqa: E402


def test_no_discordant_pairs_is_one():
    assert mcnemar(0, 0) == 1.0


def test_symmetric_is_not_significant():
    # equal disagreement both ways -> no evidence of difference
    assert mcnemar(40, 40) > 0.5


def test_large_lopsided_is_significant():
    # strategy fixes 60, breaks 10 -> strong evidence it helps
    p = mcnemar(10, 60)
    assert p < 0.001


def test_small_sample_uses_exact_and_is_symmetric_in_args():
    # below n=25 we use the exact binomial; p should not depend on order
    assert abs(mcnemar(2, 9) - mcnemar(9, 2)) < 1e-9


def test_monotonic_in_imbalance():
    # more imbalance (same total) -> smaller p
    p_mild = mcnemar(25, 35)
    p_strong = mcnemar(10, 50)
    assert p_strong < p_mild
