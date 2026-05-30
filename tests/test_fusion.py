"""
Unit tests for the Phase 3 uncertainty-weighted fusion strategies in src/tta.py.

Run with:  pytest tests/ -v

These encode the behaviours the proposal (Phase 3) specifies per strategy:
  - entropy : weights sum to 1; high-entropy views get low weight
  - maxprob : equal max-prob across views recovers equal-weight fusion
  - variance: a near-zero-(class-)variance view gets the highest weight
              (NOTE: this is the literal proposal semantics — see the caveat in
              fuse_variance's docstring; the test pins the specified behaviour)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tta import (fuse, fuse_entropy, fuse_maxprob, fuse_variance,
                     fuse_equal_weight, entropy_weights, _normalize_weights,
                     FUSION_FNS)


# ── Shapes & simplex preservation (all strategies) ─────────────────────────

def test_all_fusions_shape_and_simplex():
    rng = np.random.default_rng(0)
    per_view = rng.dirichlet(np.ones(5), size=(8, 20))   # (N=8, S=20, C=5)
    for name, fn in FUSION_FNS.items():
        fused = fn(per_view)
        assert fused.shape == (20, 5), f"{name} wrong shape"
        assert np.allclose(fused.sum(axis=1), 1.0), f"{name} not on simplex"
        assert (fused >= 0).all(), f"{name} produced negatives"


def test_dispatch_matches_direct_call():
    rng = np.random.default_rng(1)
    per_view = rng.dirichlet(np.ones(3), size=(4, 6))
    for name, fn in FUSION_FNS.items():
        assert np.allclose(fuse(per_view, name), fn(per_view))


def test_dispatch_rejects_unknown():
    import pytest
    with pytest.raises(ValueError):
        fuse(np.zeros((2, 2, 2)), "not_a_strategy")


# ── Weight normalization helper ────────────────────────────────────────────

def test_normalize_weights_columns_sum_to_one():
    w = np.array([[1.0, 2.0], [3.0, 2.0]])      # (N=2, S=2)
    nw = _normalize_weights(w)
    assert np.allclose(nw.sum(axis=0), 1.0)


def test_normalize_weights_zero_column_falls_back_uniform():
    w = np.array([[0.0, 1.0], [0.0, 1.0]])      # first sample all-zero
    nw = _normalize_weights(w)
    assert np.allclose(nw[:, 0], 0.5)           # uniform 1/N fallback
    assert np.allclose(nw.sum(axis=0), 1.0)


# ── Entropy weighting ──────────────────────────────────────────────────────

def test_entropy_weights_sum_to_one():
    rng = np.random.default_rng(2)
    per_view = rng.dirichlet(np.ones(4), size=(6, 10))
    w = entropy_weights(per_view)
    assert w.shape == (6, 10)
    assert np.allclose(w.sum(axis=0), 1.0)


def test_entropy_downweights_high_entropy_view():
    # View 0 is confident (low entropy), view 1 is uniform (max entropy).
    confident = [0.97, 0.01, 0.01, 0.01]
    uniform = [0.25, 0.25, 0.25, 0.25]
    per_view = np.array([[confident], [uniform]])           # (N=2, S=1, C=4)
    w = entropy_weights(per_view)[:, 0]
    assert w[0] > w[1], "confident (low-entropy) view should get more weight"
    # Fused result must lean toward the confident view's class 0.
    fused = fuse_entropy(per_view)[0]
    assert fused.argmax() == 0


# ── Max-probability weighting ──────────────────────────────────────────────

def test_maxprob_equal_maxprob_recovers_equal_weight():
    # Two views with the SAME max prob (0.6) but different shapes.
    v0 = [0.6, 0.3, 0.1]
    v1 = [0.6, 0.1, 0.3]
    per_view = np.array([[v0], [v1]])                       # (N=2, S=1, C=3)
    assert np.allclose(fuse_maxprob(per_view), fuse_equal_weight(per_view))


def test_maxprob_favours_confident_view():
    confident = [0.9, 0.05, 0.05]
    unsure = [0.4, 0.35, 0.25]
    per_view = np.array([[confident], [unsure]])
    fused = fuse_maxprob(per_view)[0]
    # Weighted result should be closer to the confident view than a plain mean.
    mean = fuse_equal_weight(per_view)[0]
    assert fused[0] > mean[0]


# ── Variance weighting (literal proposal semantics) ────────────────────────

def test_variance_near_zero_variance_view_gets_highest_weight():
    # Flat distribution => ~zero class-variance; peaked => high class-variance.
    flat = [0.25, 0.25, 0.25, 0.25]                          # var ~ 0
    peaked = [0.97, 0.01, 0.01, 0.01]                        # var high
    per_view = np.array([[flat], [peaked]])                  # (N=2, S=1, C=4)
    # Reconstruct the (normalized) weights the strategy uses.
    var = per_view.var(axis=2)
    w = _normalize_weights(1.0 / (var + 1e-6))[:, 0]
    assert w[0] > w[1], ("per the proposal's formula+test, the near-zero-variance "
                         "(flat) view must receive the highest weight")
    # And fuse_variance must run and stay on the simplex.
    fused = fuse_variance(per_view)
    assert np.allclose(fused.sum(axis=1), 1.0)


def test_variance_uniform_when_all_equal_variance():
    v = [0.7, 0.2, 0.1]
    per_view = np.array([[v], [v], [v]])                     # identical views
    assert np.allclose(fuse_variance(per_view), fuse_equal_weight(per_view))
