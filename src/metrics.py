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
    """Best-effort conversion to a numpy array."""
    import torch
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
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
