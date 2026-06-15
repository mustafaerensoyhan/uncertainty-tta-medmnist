"""
Evaluation metrics — Accuracy, AUC-ROC, Expected Calibration Error (ECE),
Negative Log-Likelihood (NLL).

The ECE implementation follows Section 3.4 / Metric 3 of the proposal verbatim:
equal-width binning with 10 bins, weighted mean of |acc - conf| per bin.

All functions accept numpy arrays and return Python floats so JSON serialisation
is trivial.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import roc_auc_score


def _to_numpy(x) -> np.ndarray:
    """Best-effort conversion to a numpy array (torch optional)."""
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
    """Top-1 accuracy. probs: (N, C), labels: (N,)."""
    probs, labels = _to_numpy(probs), _to_numpy(labels).ravel()
    preds = probs.argmax(axis=1)
    return float((preds == labels).mean())


def auc_roc(probs: np.ndarray, labels: np.ndarray, task: str) -> float | None:
    """
    AUC-ROC. For binary tasks returns the positive-class AUC; for multi-class
    returns macro-averaged one-vs-rest AUC. Returns None if AUC is undefined
    (e.g. only one class present in `labels`).
    """
    probs, labels = _to_numpy(probs), _to_numpy(labels).ravel()
    # If the labels contain only one class, AUC is undefined.
    # Older sklearn raises ValueError; newer sklearn returns NaN with a warning.
    # We handle both paths and uniformly return None.
    if np.unique(labels).size < 2:
        return None
    try:
        if task == "binary-class":
            # probs[:, 1] is the probability assigned to class "1"
            val = roc_auc_score(labels, probs[:, 1])
        else:
            val = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        return None if not np.isfinite(val) else float(val)
    except ValueError:
        return None


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray,
                                n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE).

    Implementation follows the proposal's Section 3.4 — equal-width binning
    of the max-softmax confidence into n_bins bins, weighted average of
    |accuracy(bin) - confidence(bin)|.

    Note: the proposal uses an exclusive lower bound on bin 0 (i.e.
    p > 0 AND p <= 0.1). We follow that exactly so our numbers match the
    snippet in the proposal text.
    """
    probs, labels = _to_numpy(probs), _to_numpy(labels).ravel()
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(labels)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def _ece_from_conf(confidences: np.ndarray, correct: np.ndarray,
                   n_bins: int = 10) -> float:
    """
    ECE from pre-computed per-image max-confidence and correctness flags.

    Identical binning to expected_calibration_error (equal-width, exclusive
    lower bound). Factored out so the bootstrap can resample indices without
    re-running argmax on every replicate.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(confidences)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        c = int(mask.sum())
        if c == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (c / n) * abs(bin_acc - bin_conf)
    return float(ece)


def bootstrap_ece_ci(probs: np.ndarray, labels: np.ndarray,
                     n_bins: int = 10, n_boot: int = 2000,
                     ci: float = 0.95, seed: int = 0):
    """
    Bootstrap confidence interval for ECE (Implementer-2 deliverable 1).

    Resamples the test set with replacement `n_boot` times and recomputes ECE
    on each replicate, then takes the percentile interval. Returns the point
    estimate plus the CI bounds:

        (ece, ci_low, ci_high)

    The per-image confidences/correctness are computed once and only the
    binning is repeated per replicate, so 2000 resamples run in well under a
    second for a typical MedMNIST test split.

    Args:
        probs: (N, C) softmax probabilities.
        labels: (N,) true labels (or (N, 1); ravelled).
        n_bins: equal-width confidence bins (default 10, matches the proposal).
        n_boot: number of bootstrap resamples (default 2000).
        ci: central interval coverage (default 0.95).
        seed: RNG seed for reproducible resampling.
    """
    probs, labels = _to_numpy(probs), _to_numpy(labels).ravel()
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(np.float64)

    point = _ece_from_conf(confidences, correct, n_bins)

    n = len(labels)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = _ece_from_conf(confidences[idx], correct[idx], n_bins)

    lo_pct = (1.0 - ci) / 2.0 * 100.0
    hi_pct = (1.0 + ci) / 2.0 * 100.0
    ci_low = float(np.percentile(boots, lo_pct))
    ci_high = float(np.percentile(boots, hi_pct))
    return point, ci_low, ci_high


def negative_log_likelihood(probs: np.ndarray, labels: np.ndarray,
                            eps: float = 1e-12) -> float:
    """
    Negative log-likelihood of the correct class. Lower = better.
    eps guards against log(0) for zeroed-out probabilities.
    """
    probs, labels = _to_numpy(probs), _to_numpy(labels).ravel().astype(np.int64)
    n = len(labels)
    p_true = probs[np.arange(n), labels]
    return float(-np.mean(np.log(p_true + eps)))


def compute_all_metrics(probs: np.ndarray, labels: np.ndarray,
                         task: str) -> Dict[str, float | None]:
    """
    Compute all four metrics in one go. Returns a dict ready to dump as JSON
    or paste into the results tracker.
    """
    return {
        "accuracy": accuracy(probs, labels),
        "auc_roc": auc_roc(probs, labels, task),  # may be None
        "ece": expected_calibration_error(probs, labels),
        "nll": negative_log_likelihood(probs, labels),
    }
