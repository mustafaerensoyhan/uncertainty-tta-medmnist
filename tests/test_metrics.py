"""
Unit tests for src/metrics.py.

Run with:
    pytest tests/ -v

These guard against regressions when we extend the metrics module in Phase 2
(adding inference time) and Phase 3 (adding per-strategy aggregation).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import (accuracy, auc_roc, expected_calibration_error,
                          negative_log_likelihood, compute_all_metrics)


# ──────────────────────────────────────────────────────────────────────────
#   Accuracy
# ──────────────────────────────────────────────────────────────────────────

def test_accuracy_perfect():
    probs = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2]])
    labels = np.array([0, 1, 0])
    assert accuracy(probs, labels) == 1.0


def test_accuracy_all_wrong():
    probs = np.array([[0.9, 0.1], [0.1, 0.9]])
    labels = np.array([1, 0])  # argmax says 0,1 — but truth is 1,0
    assert accuracy(probs, labels) == 0.0


def test_accuracy_half():
    probs = np.array([[0.9, 0.1], [0.9, 0.1]])
    labels = np.array([0, 1])  # first right, second wrong
    assert accuracy(probs, labels) == 0.5


# ──────────────────────────────────────────────────────────────────────────
#   Expected Calibration Error
# ──────────────────────────────────────────────────────────────────────────

def test_ece_perfect_calibration():
    # Model says 100% with all correct — perfectly calibrated → ECE = 0
    probs = np.array([[0.0, 1.0]] * 10)
    labels = np.ones(10, dtype=int)
    assert expected_calibration_error(probs, labels) == pytest.approx(0.0, abs=1e-9)


def test_ece_perfect_miscalibration():
    # Model says 100% with all wrong — maximal miscalibration → ECE = 1.0
    probs = np.array([[0.0, 1.0]] * 10)
    labels = np.zeros(10, dtype=int)
    assert expected_calibration_error(probs, labels) == pytest.approx(1.0, abs=1e-9)


def test_ece_in_range():
    # ECE is always in [0, 1]
    rng = np.random.default_rng(0)
    n, c = 100, 5
    logits = rng.normal(size=(n, c))
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    labels = rng.integers(0, c, size=n)
    ece = expected_calibration_error(probs, labels)
    assert 0.0 <= ece <= 1.0


# ──────────────────────────────────────────────────────────────────────────
#   NLL
# ──────────────────────────────────────────────────────────────────────────

def test_nll_perfect_predictions():
    probs = np.array([[0.0, 1.0], [1.0, 0.0]])
    labels = np.array([1, 0])
    # -log(1) = 0 for both → mean = 0
    assert negative_log_likelihood(probs, labels) == pytest.approx(0.0, abs=1e-6)


def test_nll_uniform():
    # Uniform over C classes → NLL = log(C)
    C = 4
    probs = np.full((10, C), 1.0 / C)
    labels = np.zeros(10, dtype=int)
    assert negative_log_likelihood(probs, labels) == pytest.approx(np.log(C), abs=1e-6)


def test_nll_punishes_confident_wrongness():
    # Two cases: confident-correct vs confident-wrong. Confident-wrong NLL is larger.
    probs_right = np.array([[0.01, 0.99]])
    probs_wrong = np.array([[0.99, 0.01]])
    labels = np.array([1])
    assert negative_log_likelihood(probs_wrong, labels) > negative_log_likelihood(probs_right, labels)


# ──────────────────────────────────────────────────────────────────────────
#   AUC
# ──────────────────────────────────────────────────────────────────────────

def test_auc_binary_perfect():
    # Model ranks all positives above all negatives → AUC = 1
    probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]])
    labels = np.array([0, 0, 1, 1])
    assert auc_roc(probs, labels, task="binary-class") == pytest.approx(1.0)


def test_auc_multiclass_returns_macro():
    # Smoke test — returns a finite value in [0, 1]
    rng = np.random.default_rng(1)
    n, c = 60, 4
    logits = rng.normal(size=(n, c))
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    labels = rng.integers(0, c, size=n)
    auc = auc_roc(probs, labels, task="multi-class")
    assert auc is not None
    assert 0.0 <= auc <= 1.0


def test_auc_single_class_returns_none():
    # If only one class is present, AUC is undefined → return None
    probs = np.array([[0.9, 0.1], [0.8, 0.2]])
    labels = np.array([0, 0])
    assert auc_roc(probs, labels, task="binary-class") is None


# ──────────────────────────────────────────────────────────────────────────
#   compute_all_metrics  — integration sanity
# ──────────────────────────────────────────────────────────────────────────

def test_compute_all_metrics_keys():
    rng = np.random.default_rng(2)
    probs = rng.dirichlet(np.ones(3), size=20)
    labels = rng.integers(0, 3, size=20)
    metrics = compute_all_metrics(probs, labels, task="multi-class")
    assert set(metrics.keys()) == {"accuracy", "auc_roc", "ece", "nll"}
    for k in ("accuracy", "ece", "nll"):
        assert isinstance(metrics[k], float), f"{k} should be float"
