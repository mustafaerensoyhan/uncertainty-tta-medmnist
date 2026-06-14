"""
Unit tests for the multi-backbone factory (src/model.py) and MC Dropout on
EfficientNet-B0 (src/mc_dropout.py), added in Phase 5.

Includes a regression guard for a subtle TESTING gotcha discovered during Phase
5: an *untrained* EfficientNet-B0 (pretrained=False) outputs all-zero
penultimate features in eval mode (random init + eval-mode BatchNorm with
default running stats), so MC Dropout on the head shows ZERO stochasticity —
not a bug, just an artifact of having no trained weights. Once BatchNorm running
stats are populated (here via a few train-mode forwards, a proxy for a real
checkpoint), MC Dropout produces real per-pass variation. ResNet-18 features are
non-zero even untrained, so it never hit this.

Run with:  pytest tests/test_model_arch.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model import (build_model, build_resnet18, build_efficientnet_b0,
                       ARCHITECTURES, ARCH_LABELS, count_parameters)
from src.mc_dropout import _find_head, mc_dropout_per_pass_probs


# ── factory ────────────────────────────────────────────────────────────────

def test_architectures_registry():
    assert "resnet18" in ARCHITECTURES and "effb0" in ARCHITECTURES
    assert set(ARCH_LABELS) == set(ARCHITECTURES)


def test_build_model_dispatches():
    r = build_model("resnet18", num_classes=5, pretrained=False)
    e = build_model("effb0", num_classes=5, pretrained=False)
    assert isinstance(r, type(build_resnet18(num_classes=5, pretrained=False)))
    assert isinstance(e, type(build_efficientnet_b0(num_classes=5, pretrained=False)))


def test_build_model_rejects_unknown_arch():
    try:
        build_model("vit_huge", num_classes=2, pretrained=False)
    except (ValueError, KeyError):
        return
    raise AssertionError("build_model should reject an unknown architecture")


def test_head_class_count_matches():
    for arch in ("resnet18", "effb0"):
        for nc in (2, 7, 11):
            m = build_model(arch, num_classes=nc, pretrained=False).eval()
            out = m(torch.randn(2, 3, 64, 64))
            assert out.shape == (2, nc)


def test_param_counts_sane():
    # EfficientNet-B0 (~5.3M) is much larger than ResNet-18's classifier swap is small;
    # just assert both are in a plausible millions range and effb0 < resnet18.
    r = count_parameters(build_resnet18(num_classes=9, pretrained=False))
    e = count_parameters(build_efficientnet_b0(num_classes=9, pretrained=False))
    assert 3e6 < e < 8e6
    assert 10e6 < r < 13e6


# ── MC Dropout head discovery ──────────────────────────────────────────────

def test_find_head_on_both_backbones():
    # Contract: return the head module whose forward-pre-hook receives the pooled
    # feature vector — ResNet's bare Linear `.fc`, EfficientNet's `.classifier`.
    r = build_resnet18(num_classes=3, pretrained=False)
    e = build_efficientnet_b0(num_classes=3, pretrained=False)
    assert _find_head(r) is r.fc
    assert _find_head(e) is e.classifier
    # Both heads' first input is a 2-D (batch, features) tensor we can dropout.
    for model, head in ((r, _find_head(r)), (e, _find_head(e))):
        seen = {}
        h = head.register_forward_pre_hook(lambda mod, inp: seen.__setitem__("x", inp[0]))
        model.eval()
        with torch.no_grad():
            model(torch.randn(2, 3, 64, 64))
        h.remove()
        assert seen["x"].dim() == 2


def _warmup_batchnorm(model, steps=10, bs=16):
    """Populate BatchNorm running stats (proxy for a trained checkpoint)."""
    model.train()
    with torch.no_grad():
        for _ in range(steps):
            model(torch.randn(bs, 3, 64, 64))
    model.eval()


def test_mc_dropout_zero_on_untrained_effb0_is_expected():
    # Documents the gotcha: untrained effb0 -> zero penultimate features -> no MC spread.
    m = build_efficientnet_b0(num_classes=3, pretrained=False).eval()
    feats = {}
    h = _find_head(m).register_forward_pre_hook(lambda mod, inp: feats.__setitem__("x", inp[0].detach()))
    with torch.no_grad():
        m(torch.randn(4, 3, 64, 64))
    h.remove()
    # Effectively zero — float noise ~1e-14, far below any real activation. This
    # is why masking does nothing and MC Dropout shows no spread on an untrained net.
    assert float(feats["x"].abs().mean()) < 1e-9  # the gotcha


def test_mc_dropout_varies_on_effb0_with_features():
    m = build_efficientnet_b0(num_classes=3, pretrained=False)
    _warmup_batchnorm(m)  # now penultimate features are non-zero
    ds = TensorDataset(torch.randn(8, 3, 64, 64), torch.randint(0, 3, (8,)))
    dl = DataLoader(ds, batch_size=4)
    per_pass, labels = mc_dropout_per_pass_probs(m, dl, torch.device("cpu"), T=6, p=0.3)
    assert per_pass.shape[0] == 6
    assert float(per_pass.std(axis=0).mean()) > 0.0  # real per-pass stochasticity


def test_mc_dropout_varies_on_resnet():
    m = build_resnet18(num_classes=3, pretrained=False)  # nonzero even untrained
    ds = TensorDataset(torch.randn(8, 3, 64, 64), torch.randint(0, 3, (8,)))
    dl = DataLoader(ds, batch_size=4)
    per_pass, _ = mc_dropout_per_pass_probs(m, dl, torch.device("cpu"), T=6, p=0.3)
    assert float(per_pass.std(axis=0).mean()) > 0.0
