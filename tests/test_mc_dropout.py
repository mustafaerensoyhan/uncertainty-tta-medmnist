"""
Unit tests for src/mc_dropout.py.

Run with:  pytest tests/ -v

Uses a tiny CPU model with a .fc head (mirroring build_resnet18's structure) so
the tests are fast and need no checkpoint or dataset download.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mc_dropout import (mc_dropout_per_pass_probs, mc_dropout_fuse,
                            mc_dropout_predict)


class TinyNet(nn.Module):
    """Minimal classifier with a .fc head and a BatchNorm, like ResNet's tail."""

    def __init__(self, in_dim=8, n_classes=3):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_dim)
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.fc(self.bn(x))


def _loader(n=24, in_dim=8, n_classes=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, in_dim, generator=g)
    y = torch.randint(0, n_classes, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=8, shuffle=False)


def test_per_pass_shapes():
    model = TinyNet()
    loader = _loader()
    per_pass, labels = mc_dropout_per_pass_probs(model, loader, torch.device("cpu"),
                                                 T=5, p=0.5)
    assert per_pass.shape == (5, 24, 3)
    assert labels.shape == (24,)
    # Each pass is a valid probability distribution.
    assert np.allclose(per_pass.sum(axis=2), 1.0, atol=1e-5)


def test_dropout_actually_introduces_variation():
    # With p>0 the T passes must differ (dropout is active); the fused mean
    # therefore differs from any single pass in general.
    model = TinyNet()
    loader = _loader()
    per_pass, _ = mc_dropout_per_pass_probs(model, loader, torch.device("cpu"),
                                            T=10, p=0.5)
    spread = per_pass.std(axis=0).mean()
    assert spread > 1e-4, "expected stochastic variation across MC passes"


def test_p_zero_is_deterministic():
    # p=0 => no dropout => all passes identical.
    model = TinyNet()
    loader = _loader()
    per_pass, _ = mc_dropout_per_pass_probs(model, loader, torch.device("cpu"),
                                            T=4, p=0.0)
    assert np.allclose(per_pass[0], per_pass[1])
    assert np.allclose(per_pass[0], per_pass[3])


def test_hook_is_removed_after_call():
    # The pre-hook must not linger on the model after the function returns.
    model = TinyNet()
    loader = _loader()
    mc_dropout_per_pass_probs(model, loader, torch.device("cpu"), T=2, p=0.5)
    assert len(model.fc._forward_pre_hooks) == 0


def test_predict_wrapper_shapes():
    model = TinyNet()
    loader = _loader()
    fused, per_pass, labels = mc_dropout_predict(model, loader, torch.device("cpu"),
                                                 T=6, p=0.3)
    assert fused.shape == (24, 3)
    assert per_pass.shape == (6, 24, 3)
    assert np.allclose(fused.sum(axis=1), 1.0, atol=1e-5)
    assert np.allclose(fused, mc_dropout_fuse(per_pass))
