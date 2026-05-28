"""
Unit tests for src/augmentations.py and src/tta.py (Phase 2).

Run with:  pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.augmentations import (get_augmentation_pipeline, BASE_AUGMENTATIONS,
                               augmentation_names, aug_identity)
from src.tta import fuse_equal_weight


# ── Augmentation pipeline ──────────────────────────────────────────────────

def test_pipeline_length_matches_n_views():
    for n in [1, 5, 10, 20, 50]:
        pipe = get_augmentation_pipeline(n_views=n, seed=0)
        assert len(pipe) == n


def test_pipeline_first_view_is_original():
    pipe = get_augmentation_pipeline(n_views=10, include_original=True)
    assert pipe[0][1] == "original"
    assert pipe[0][0] is aug_identity


def test_pipeline_without_original():
    pipe = get_augmentation_pipeline(n_views=5, include_original=False)
    assert all(name != "original" for _, name in pipe)


def test_pipeline_n_views_one():
    pipe = get_augmentation_pipeline(n_views=1)
    assert len(pipe) == 1
    assert pipe[0][1] == "original"


def test_pipeline_rejects_zero():
    import pytest
    with pytest.raises(ValueError):
        get_augmentation_pipeline(n_views=0)


def test_all_augmentations_preserve_shape_and_range():
    img = torch.rand(3, 64, 64)
    for name, fn in BASE_AUGMENTATIONS:
        out = fn(img)
        assert out.shape == img.shape, f"{name} changed shape"
        # Allow a tiny epsilon for float rounding around the clamp bounds
        assert out.min() >= -0.02 and out.max() <= 1.02, f"{name} out of [0,1]"


def test_identity_is_noop():
    img = torch.rand(3, 32, 32)
    assert torch.equal(aug_identity(img), img)


def test_ten_base_augmentations():
    assert len(BASE_AUGMENTATIONS) == 10
    assert len(augmentation_names()) == 10


# ── Equal-weight fusion ────────────────────────────────────────────────────

def test_fuse_equal_weight_shape():
    # (N=4 views, S=10 samples, C=3 classes)
    per_view = np.random.dirichlet(np.ones(3), size=(4, 10))
    fused = fuse_equal_weight(per_view)
    assert fused.shape == (10, 3)


def test_fuse_equal_weight_is_mean():
    per_view = np.array([
        [[0.8, 0.2]],   # view 0, sample 0
        [[0.4, 0.6]],   # view 1, sample 0
    ])  # shape (2, 1, 2)
    fused = fuse_equal_weight(per_view)
    # mean of [0.8,0.2] and [0.4,0.6] = [0.6,0.4]
    assert np.allclose(fused, [[0.6, 0.4]])


def test_fuse_preserves_probability_simplex():
    # If each view sums to 1, the average also sums to 1
    per_view = np.random.dirichlet(np.ones(5), size=(8, 20))
    fused = fuse_equal_weight(per_view)
    assert np.allclose(fused.sum(axis=1), 1.0)


def test_fuse_single_view_is_identity():
    per_view = np.random.dirichlet(np.ones(4), size=(1, 6))
    fused = fuse_equal_weight(per_view)
    assert np.allclose(fused, per_view[0])
