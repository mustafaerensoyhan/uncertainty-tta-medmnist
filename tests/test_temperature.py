"""
Unit tests for temperature scaling (src/temperature.py) and the softmax_np /
temperature helper in src/tta.py.

Run with:  pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tta import softmax_np
from src.temperature import fit_temperature, predict_probs_with_temperature


def _entropy(p, axis=-1):
    return -np.sum(p * np.log(p + 1e-12), axis=axis)


# ── softmax_np / temperature ───────────────────────────────────────────────

def test_softmax_np_is_valid_distribution():
    logits = np.random.randn(4, 6, 5)
    p = softmax_np(logits, axis=2)
    assert np.allclose(p.sum(axis=2), 1.0)
    assert (p >= 0).all()


def test_temperature_above_one_softens():
    logits = np.array([[3.0, 0.0, -1.0]])
    p1 = softmax_np(logits, temperature=1.0)
    p2 = softmax_np(logits, temperature=2.5)
    assert _entropy(p2)[0] > _entropy(p1)[0], "higher T should increase entropy"


def test_temperature_preserves_argmax():
    logits = np.random.randn(20, 7)
    for T in (0.5, 1.0, 2.0, 5.0):
        assert np.array_equal(softmax_np(logits, T).argmax(1), logits.argmax(1))


# ── fit_temperature on a tiny model ────────────────────────────────────────

class TinyNet(nn.Module):
    def __init__(self, in_dim=8, n_classes=4, scale=1.0):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)
        with torch.no_grad():
            self.fc.weight.mul_(scale)  # large scale => overconfident logits
    def forward(self, x):
        return self.fc(x)


def _loader(n=64, in_dim=8, n_classes=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, in_dim, generator=g)
    y = torch.randint(0, n_classes, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=16, shuffle=False)


def test_fit_temperature_returns_positive_finite():
    model = TinyNet(scale=5.0)  # overconfident
    T = fit_temperature(model, _loader(), torch.device("cpu"))
    assert np.isfinite(T) and T > 0


def test_predict_with_temperature_shapes_and_simplex():
    model = TinyNet()
    probs, labels = predict_probs_with_temperature(model, _loader(), torch.device("cpu"), T=1.5)
    assert probs.shape == (64, 4)
    assert labels.shape == (64,)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_ts_only_preserves_accuracy_vs_T1():
    # TS never changes the predicted class — argmax is T-invariant.
    model = TinyNet(scale=3.0)
    loader = _loader()
    p1, y = predict_probs_with_temperature(model, loader, torch.device("cpu"), T=1.0)
    pT, _ = predict_probs_with_temperature(model, loader, torch.device("cpu"), T=2.3)
    assert np.array_equal(p1.argmax(1), pT.argmax(1))
