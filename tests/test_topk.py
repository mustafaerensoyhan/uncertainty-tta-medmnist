"""
Unit tests for the Top-K (hard entropy filter) TTA fusion (Phase 5 / VMV plan).

Run with:  pytest tests/test_topk.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tta import (TOP_K_VALUES, FUSION_FNS, top_k_keep_indices, fuse_top_k,
                     top_k_strategy, fuse_equal_weight)


def _peaked(c, n_classes):
    v = np.full(n_classes, 0.01)
    v[c] = 1.0 - 0.01 * (n_classes - 1)
    return v


def _per_view_mixed(n_views=10, n_classes=4, seed=0):
    """N views for 1 sample: first half confident (peaked), second half noisy."""
    rng = np.random.default_rng(seed)
    views = []
    for i in range(n_views):
        if i < n_views // 2:
            v = _peaked(0, n_classes)
        else:
            v = rng.dirichlet(np.ones(n_classes))  # high-entropy-ish
        views.append(v)
    return np.array(views)[:, None, :]  # (N, 1, C)


# ── keep-index selection ───────────────────────────────────────────────────

def test_keep_indices_count_and_uniqueness():
    pv = _per_view_mixed(n_views=10, n_classes=4)
    for k in (1, 3, 5, 7, 10):
        idx = top_k_keep_indices(pv, k)
        assert idx.shape == (k, 1)
        assert len(set(idx[:, 0].tolist())) == k  # no repeats


def test_keep_indices_pick_lowest_entropy_views():
    # confident views (indices 0..4) have lowest entropy -> Top-5 must be those.
    pv = _per_view_mixed(n_views=10, n_classes=4, seed=1)
    keep = set(top_k_keep_indices(pv, 5)[:, 0].tolist())
    assert keep == {0, 1, 2, 3, 4}


def test_k_capped_at_n_views():
    pv = _per_view_mixed(n_views=6, n_classes=3)
    idx = top_k_keep_indices(pv, 99)  # k > N
    assert idx.shape[0] == 6  # capped to all views


# ── fusion output ──────────────────────────────────────────────────────────

def test_fuse_top_k_on_simplex():
    pv = _per_view_mixed(n_views=10, n_classes=4)
    for k in (3, 5, 7):
        fused = fuse_top_k(pv, k)
        assert fused.shape == (1, 4)
        assert np.allclose(fused.sum(axis=1), 1.0, atol=1e-6)
        assert (fused >= 0).all()


def test_top_n_equals_equal_weight():
    # Keeping ALL views must reduce to plain mean fusion.
    pv = _per_view_mixed(n_views=8, n_classes=5, seed=3)
    assert np.allclose(fuse_top_k(pv, 8), fuse_equal_weight(pv), atol=1e-6)


def test_top_k_sharpens_toward_confident_class():
    # With half the views confident in class 0, Top-5 should put more mass on
    # class 0 than averaging all 10 (which is diluted by the noisy half).
    pv = _per_view_mixed(n_views=10, n_classes=4, seed=2)
    p_all = fuse_equal_weight(pv)[0, 0]
    p_top5 = fuse_top_k(pv, 5)[0, 0]
    assert p_top5 > p_all


def test_registry_has_top_k_entries():
    for k in TOP_K_VALUES:
        name = top_k_strategy(k)
        assert name == f"top{k}"
        assert name in FUSION_FNS
        # callable and returns simplex
        out = FUSION_FNS[name](_per_view_mixed())
        assert np.allclose(out.sum(axis=1), 1.0, atol=1e-6)


def test_batch_of_samples():
    # (N=6, S=3, C=4) — independent per sample.
    rng = np.random.default_rng(7)
    pv = rng.dirichlet(np.ones(4), size=(6, 3))
    fused = fuse_top_k(pv, 3)
    assert fused.shape == (3, 4)
    assert np.allclose(fused.sum(axis=1), 1.0, atol=1e-6)
