"""
Unit tests for the backbone-aware naming helpers (src/utils.py, Phase 5).

The whole point of these helpers is that ResNet-18 at the canonical seed keeps
EXACTLY the original Phase 1-4 filenames (so nothing breaks), while EfficientNet
and seed-tagged runs live in a non-colliding parallel namespace.

Run with:  pytest tests/test_naming.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import (DEFAULT_ARCH, checkpoint_filename, result_stem,
                       default_ckpt_tag)


# ── checkpoint_filename ────────────────────────────────────────────────────

def test_default_arch_is_resnet18():
    assert DEFAULT_ARCH == "resnet18"


def test_resnet_canonical_checkpoint_unchanged():
    # This MUST equal the original Phase 1 name or all existing checkpoints break.
    assert checkpoint_filename("pathmnist", "resnet18", "") == "pathmnist_resnet18.pth"


def test_resnet_seed_tagged_checkpoint():
    assert checkpoint_filename("bloodmnist", "resnet18", "_seed0") == "bloodmnist_resnet18_seed0.pth"


def test_effb0_checkpoint_namespace():
    assert checkpoint_filename("pathmnist", "effb0", "") == "pathmnist_effb0.pth"
    assert checkpoint_filename("pathmnist", "effb0", "_seed42") == "pathmnist_effb0_seed42.pth"


# ── result_stem ────────────────────────────────────────────────────────────

def test_resnet_canonical_stem_is_archless():
    # ResNet-18 canonical results stay archless: e.g. results/pathmnist_weighted_tta.csv
    assert result_stem("pathmnist", "resnet18", "") == "pathmnist"


def test_resnet_seed_stem():
    assert result_stem("pathmnist", "resnet18", "_seed0") == "pathmnist_seed0"


def test_effb0_stem_has_arch_infix():
    assert result_stem("pathmnist", "effb0", "") == "pathmnist_effb0"
    assert result_stem("pathmnist", "effb0", "_seed1") == "pathmnist_effb0_seed1"


def test_effb0_glob_does_not_collide_with_resnet():
    # The resnet seed glob 'pathmnist_seed*' must NOT match the effb0 stem.
    effb0_stem = result_stem("pathmnist", "effb0", "_seed0")     # pathmnist_effb0_seed0
    assert not effb0_stem.startswith("pathmnist_seed")


# ── default_ckpt_tag ───────────────────────────────────────────────────────

def test_resnet_canonical_seed_is_untagged():
    assert default_ckpt_tag("resnet18", 42, canonical_seed=42) == ""


def test_resnet_noncanonical_seed_tagged():
    assert default_ckpt_tag("resnet18", 0, canonical_seed=42) == "_seed0"
    assert default_ckpt_tag("resnet18", 123, canonical_seed=42) == "_seed123"


def test_effb0_always_tagged_even_at_canonical_seed():
    # effb0 has no untagged canonical, so even seed 42 is tagged to stay parallel.
    assert default_ckpt_tag("effb0", 42, canonical_seed=42) == "_seed42"
    assert default_ckpt_tag("effb0", 0, canonical_seed=42) == "_seed0"


def test_tag_feeds_back_into_filenames_consistently():
    # tag from default_ckpt_tag must round-trip through checkpoint_filename/result_stem.
    for arch in ("resnet18", "effb0"):
        for seed in (0, 42, 123):
            tag = default_ckpt_tag(arch, seed, canonical_seed=42)
            ckpt = checkpoint_filename("organamnist", arch, tag)
            stem = result_stem("organamnist", arch, tag)
            assert ckpt.endswith(".pth") and ckpt.startswith("organamnist")
            assert stem.startswith("organamnist")
            # resnet canonical stays clean; everything else is namespaced
            if arch == "resnet18" and seed == 42:
                assert ckpt == "organamnist_resnet18.pth" and stem == "organamnist"
            else:
                assert ("effb0" in ckpt) or ("seed" in ckpt)
