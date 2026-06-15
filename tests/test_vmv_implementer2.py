"""
Lightweight tests for the VMV Implementer-2 deliverables.

These never touch MedMNIST data or checkpoints — they exercise the pure
numeric/IO helpers on tiny synthetic inputs, and confirm the new scripts import.

Run with:
    pytest tests/test_vmv_implementer2.py -v
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import bootstrap_ece_ci, expected_calibration_error


# ──────────────────────────────────────────────────────────────────────────
#   bootstrap_ece_ci (deliverable 1 core)
# ──────────────────────────────────────────────────────────────────────────
def _synthetic_overconfident(n=400, seed=0):
    """High-confidence probs that are only ~half right -> clearly non-zero ECE."""
    rng = np.random.default_rng(seed)
    p_top = rng.uniform(0.9, 0.99, size=n)
    probs = np.column_stack([p_top, 1.0 - p_top])
    # Only ~50% correct despite ~0.95 confidence => large miscalibration.
    labels = (rng.random(n) > 0.5).astype(int)
    return probs, labels


def test_bootstrap_ci_brackets_point_estimate():
    probs, labels = _synthetic_overconfident()
    ece, lo, hi = bootstrap_ece_ci(probs, labels, n_bins=10, n_boot=500, seed=0)
    point = expected_calibration_error(probs, labels, n_bins=10)
    assert ece == pytest.approx(point, abs=1e-9)   # point matches the plain ECE
    assert 0.0 <= lo <= ece <= hi <= 1.0           # CI brackets the estimate
    assert hi > lo                                 # interval has width


def test_bootstrap_ci_is_reproducible():
    probs, labels = _synthetic_overconfident()
    a = bootstrap_ece_ci(probs, labels, n_boot=300, seed=123)
    b = bootstrap_ece_ci(probs, labels, n_boot=300, seed=123)
    assert a == b                                  # same seed -> identical CI


def test_bootstrap_ci_detects_large_miscalibration():
    probs, labels = _synthetic_overconfident()
    _, lo, _ = bootstrap_ece_ci(probs, labels, n_boot=500, seed=0)
    # ~0.95 confidence at ~50% accuracy => ECE well above 0.2; CI should sit high.
    assert lo > 0.2


# ──────────────────────────────────────────────────────────────────────────
#   import the new scripts (deliverable 7)
# ──────────────────────────────────────────────────────────────────────────
def test_scripts_import():
    boot = importlib.import_module("scripts.bootstrap_ece_ci")
    figs = importlib.import_module("scripts.make_vmv_figures")
    assert hasattr(boot, "main") and callable(boot.main)
    assert hasattr(boot, "ci_disjoint")
    for fn in ("fig2_ece", "fig3_reliability", "fig4_heatmap", "fig5_mechanism"):
        assert callable(getattr(figs, fn))


def test_ci_disjoint_logic():
    boot = importlib.import_module("scripts.bootstrap_ece_ci")
    assert boot.ci_disjoint(0.0, 0.1, 0.2, 0.3) is True     # a fully below b
    assert boot.ci_disjoint(0.2, 0.3, 0.0, 0.1) is True     # a fully above b
    assert boot.ci_disjoint(0.0, 0.25, 0.2, 0.3) is False   # overlapping


# ──────────────────────────────────────────────────────────────────────────
#   Fig 4 augmentation-importance math (deliverable 4)
# ──────────────────────────────────────────────────────────────────────────
def test_aug_importance_full_minus_remove():
    figs = importlib.import_module("scripts.make_vmv_figures")
    df = pd.DataFrame({
        "removed": ["(none/full)", "hflip", "rotate"],
        "accuracy": [0.90, 0.85, 0.92],
    })
    imp = figs._aug_importance(df)
    assert imp["hflip"] == pytest.approx(0.05)    # 0.90 - 0.85 (helped)
    assert imp["rotate"] == pytest.approx(-0.02)  # 0.90 - 0.92 (hurt)
    assert "(none/full)" not in imp


def test_stability_for_arch_filters_combined_schema(tmp_path):
    """New combined seed_stability.csv (with an `arch` column) is filtered by arch."""
    figs = importlib.import_module("scripts.make_vmv_figures")
    rdir = tmp_path / "results"
    rdir.mkdir()
    pd.DataFrame({
        "dataset": ["pathmnist", "pathmnist", "pathmnist", "pathmnist"],
        "arch": ["resnet18", "resnet18", "effb0", "effb0"],
        "strategy": ["baseline", "entropy", "baseline", "entropy"],
        "ece_mean": [0.025, 0.016, 0.034, 0.022],
        "ece_std": [0.003, 0.002, 0.001, 0.001],
    }).to_csv(rdir / "seed_stability.csv", index=False)

    r = figs._stability_for_arch(rdir, "resnet18")
    e = figs._stability_for_arch(rdir, "effb0")
    assert float(r.loc[("pathmnist", "entropy"), "ece_mean"]) == pytest.approx(0.016)
    assert float(e.loc[("pathmnist", "baseline"), "ece_mean"]) == pytest.approx(0.034)
    # index is unique per (dataset, strategy) after arch filtering
    assert r.index.is_unique and e.index.is_unique


def test_stability_for_arch_legacy_archless(tmp_path):
    """Legacy archless seed_stability.csv is treated as resnet18; effb0 -> None."""
    figs = importlib.import_module("scripts.make_vmv_figures")
    rdir = tmp_path / "results"
    rdir.mkdir()
    pd.DataFrame({
        "dataset": ["pathmnist"], "strategy": ["baseline"],
        "ece_mean": [0.025], "ece_std": [0.003],
    }).to_csv(rdir / "seed_stability.csv", index=False)
    assert figs._stability_for_arch(rdir, "resnet18") is not None
    assert figs._stability_for_arch(rdir, "effb0") is None


def test_aug_importance_requires_full_row():
    figs = importlib.import_module("scripts.make_vmv_figures")
    df = pd.DataFrame({"removed": ["hflip"], "accuracy": [0.85]})
    with pytest.raises(ValueError):
        figs._aug_importance(df)


# ──────────────────────────────────────────────────────────────────────────
#   Fig 4 fails clearly (returns None) when data missing; --allow-partial path
# ──────────────────────────────────────────────────────────────────────────
def test_fig4_returns_none_when_incomplete(tmp_path):
    figs = importlib.import_module("scripts.make_vmv_figures")
    rdir = tmp_path / "results"
    rdir.mkdir()
    # No ablation CSVs at all -> incomplete -> must refuse (return None).
    out = figs.fig4_heatmap(results_dir=str(rdir), figures_dir=str(tmp_path / "figs"))
    assert out is None


# ──────────────────────────────────────────────────────────────────────────
#   Fig 3 fails clearly when prediction arrays are absent
# ──────────────────────────────────────────────────────────────────────────
def test_fig3_returns_none_when_arrays_missing(tmp_path):
    figs = importlib.import_module("scripts.make_vmv_figures")
    out = figs.fig3_reliability(predictions_dir=str(tmp_path / "predictions"),
                                figures_dir=str(tmp_path / "figs"))
    assert out is None
