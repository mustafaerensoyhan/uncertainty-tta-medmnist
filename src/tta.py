"""
Test-Time Augmentation (TTA) inference.

Phase 2 implements *standard* TTA: generate N augmented views of each test
image, run each through the model, and average the softmax outputs with EQUAL
weight (w_i = 1/N). The goal is to replicate the finding that equal-weight TTA
often hurts medical-image classifiers.

Design note for Phase 3: the expensive part — computing the per-view softmax
probabilities — is separated from the cheap part — fusing them into one
prediction. Phase 3's uncertainty-weighted strategies (entropy, max-prob,
variance) will reuse `tta_per_view_probs()` unchanged and only swap in a
different fusion function. So `fuse_equal_weight()` here is the first of several
fusion strategies that will live in this module.

The TTA loop expects a loader that yields UN-normalized [0, 1] RGB images
(see data.get_tta_test_loader). Augmentations are applied on [0, 1], then each
view is normalized with ImageNet statistics before the forward pass.
"""

from __future__ import annotations

from functools import partial
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .augmentations import AugFn
from .data import normalize_imagenet


@torch.no_grad()
def tta_per_view_logits(model: nn.Module, loader: DataLoader, device: torch.device,
                        augmentations: List[Tuple[AugFn, str]]
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run TTA over a test loader and return per-view raw LOGITS (pre-softmax).

    Returning logits (not probabilities) lets a caller derive both the plain
    softmax and a temperature-scaled softmax (logits / T) from a single set of
    forward passes — needed for the TS + Entropy TTA column (addendum Addition 2)
    without paying for the augmented forward passes twice.

    Returns:
        per_view_logits: (N, num_samples, num_classes)
        labels: (num_samples,)
    """
    model.eval()
    n_views = len(augmentations)
    per_view_batches: List[np.ndarray] = []
    label_batches: List[np.ndarray] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)  # (B, 3, H, W) in [0,1]
        view_logits = []
        for aug_fn, _name in augmentations:
            aug_imgs = torch.stack([aug_fn(img) for img in images], dim=0)
            aug_imgs = normalize_imagenet(aug_imgs)
            logits = model(aug_imgs)
            view_logits.append(logits.cpu().numpy())
        per_view_batches.append(np.stack(view_logits, axis=0))  # (N, B, C)
        label_batches.append(labels.numpy().ravel())

    per_view_logits = np.concatenate(per_view_batches, axis=1)  # (N, num_samples, C)
    labels = np.concatenate(label_batches, axis=0)
    assert per_view_logits.shape[0] == n_views
    return per_view_logits, labels


def softmax_np(logits: np.ndarray, temperature: float = 1.0,
               axis: int = -1) -> np.ndarray:
    """Numerically-stable softmax over `axis`, with optional temperature."""
    z = logits / float(temperature)
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def tta_per_view_probs(model: nn.Module, loader: DataLoader, device: torch.device,
                       augmentations: List[Tuple[AugFn, str]],
                       temperature: float = 1.0
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run TTA over a test loader and return per-view softmax probabilities.

    Thin wrapper over tta_per_view_logits + softmax. Pass temperature > 1 to get
    temperature-scaled probabilities (used by the TS + Entropy column). Default
    temperature=1.0 reproduces the Phase 2 behaviour exactly.

    Returns:
        per_view_probs: (N, num_samples, num_classes)
        labels: (num_samples,)
    """
    per_view_logits, labels = tta_per_view_logits(model, loader, device, augmentations)
    return softmax_np(per_view_logits, temperature=temperature, axis=2), labels


# ── Fusion strategies ──────────────────────────────────────────────────────
# Phase 2 ships only equal-weight fusion. Phase 3 will add entropy / max-prob /
# variance weighting here, each with the same (N, S, C) -> (S, C) signature.

def fuse_equal_weight(per_view_probs: np.ndarray) -> np.ndarray:
    """
    Standard TTA fusion: average the N views with equal weight 1/N.

    Args:
        per_view_probs: (N, num_samples, num_classes)
    Returns:
        fused probabilities: (num_samples, num_classes)
    """
    return per_view_probs.mean(axis=0)


# ── Uncertainty-weighted fusion strategies (Phase 3) ───────────────────────
# All share fuse_equal_weight's (N, S, C) -> (S, C) signature, so they are
# drop-in replacements that reuse the cached per-view probabilities from
# tta_per_view_probs. Each computes a per-(view, sample) weight w[i, s] from that
# view's softmax vector, normalizes the weights across the N views for each
# sample, and returns the weighted average. The expensive forward passes happen
# once; trying another strategy is just a different reduction over (N, S, C).

def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    """
    Normalize per-view weights to sum to 1 across the view axis (axis 0).

    weights: (N, S) -> (N, S), each column (sample) summing to 1. Any sample
    whose weights sum to ~0 falls back to uniform 1/N (numerically safe).
    """
    n_views = weights.shape[0]
    col_sums = weights.sum(axis=0, keepdims=True)            # (1, S)
    safe = col_sums > 0
    return np.where(safe, weights / np.where(safe, col_sums, 1.0), 1.0 / n_views)


def _weighted_average(per_view_probs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted average of views. per_view_probs (N,S,C), weights (N,S) -> (S,C)."""
    w = _normalize_weights(weights)                          # (N, S)
    return np.einsum("ns,nsc->sc", w, per_view_probs)


def fuse_maxprob(per_view_probs: np.ndarray) -> np.ndarray:
    """
    Max-probability weighting: w_i = max_k p_i[k] (proposal §3.2).

    Confident views (high peak probability) dominate. When every view shares the
    same max probability the weights become uniform and this reduces exactly to
    equal-weight fusion.
    """
    weights = per_view_probs.max(axis=2)                     # (N, S)
    return _weighted_average(per_view_probs, weights)


def fuse_entropy(per_view_probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Softmax-entropy weighting: w_i = exp(-H(p_i)), H(p) = -Σ_k p_k log p_k
    (proposal §3.2). Low-entropy (high-certainty) views get exponentially more
    weight. eps guards the log against zero probabilities.
    """
    p = per_view_probs
    entropy = -np.sum(p * np.log(p + eps), axis=2)           # (N, S)
    return _weighted_average(p, np.exp(-entropy))


def entropy_weights(per_view_probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Return the NORMALIZED entropy weights (N, S) rather than the fused result.

    Exposed for the Augmentation Confidence Strip (proposal §3.5), which needs
    the per-view weight values themselves to draw the weight bars.
    """
    p = per_view_probs
    entropy = -np.sum(p * np.log(p + eps), axis=2)           # (N, S)
    return _normalize_weights(np.exp(-entropy))


def fuse_variance(per_view_probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Predictive-variance weighting (confidence-aligned): w_i = var(p_i).

    var(p_i) is the variance of the per-view probability vector across classes.
    A confident, peaked prediction has HIGH class-variance; a flat/uncertain one
    has LOW class-variance. So weighting by var lets the sharp (confident) views
    dominate — matching the proposal's stated intuition ("confident views
    dominate").

    NOTE: this is the OPPOSITE direction from the proposal's literal formula
    w_i = 1/(var+ε). The literal version (which upweights uncertain views and
    empirically hurts) is kept as `fuse_variance_inv` and reported as an
    ablation / negative finding. Supervisor (M. Hafez) approved making the
    confidence-aligned version the headline `variance` strategy.
    """
    weights = per_view_probs.var(axis=2) + eps                # (N, S)
    return _weighted_average(per_view_probs, weights)


def fuse_variance_inv(per_view_probs: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Literal proposal formula (ablation / negative finding): w_i = 1/(var(p_i)+ε).

    Kept exactly as written in proposal §3.2. Because var is taken across the
    class dimension, this upweights FLAT (uncertain) views and empirically
    degrades calibration/accuracy — reported as the "naive variance weighting is
    unstable" result (addendum, Key Message 3). Not the headline `variance`.
    """
    var = per_view_probs.var(axis=2)                          # (N, S)
    return _weighted_average(per_view_probs, 1.0 / (var + eps))


# ── Top-K TTA (Phase 5 / VMV plan): hard entropy filter ────────────────────
# Motivated by the confidence strips (Fig 1): entropy weighting assigns near-zero
# weight to systematically distrusted augmentations (e.g. flips of chest X-rays).
# Top-K makes that filtering EXPLICIT and inspectable — instead of softly
# down-weighting low-confidence views, it discards all but the K lowest-entropy
# (most confident) views and averages them equally. We evaluate K ∈ {3,5,7} at
# N=10. Like the soft fusions it is a pure reduction over the cached (N,S,C)
# per-view probabilities, so it adds no forward passes.

def top_k_keep_indices(per_view_probs: np.ndarray, k: int,
                       eps: float = 1e-12) -> np.ndarray:
    """
    Indices of the K lowest-entropy (most confident) views per sample.

    per_view_probs: (N, S, C). Returns an (k_eff, S) int array, where
    k_eff = clip(k, 1, N). Column s lists the kept view indices for sample s,
    ordered from most to least confident. Exposed so the confidence strip can
    gold-outline the views Top-K would keep (VMV plan, Implementer 1 Task 5).
    """
    n_views = per_view_probs.shape[0]
    k_eff = max(1, min(int(k), n_views))
    entropy = -np.sum(per_view_probs * np.log(per_view_probs + eps), axis=2)  # (N, S)
    # argsort ascending => lowest entropy first; take the first k_eff rows.
    return np.argsort(entropy, axis=0, kind="stable")[:k_eff, :]              # (k_eff, S)


def fuse_top_k(per_view_probs: np.ndarray, k: int, eps: float = 1e-12) -> np.ndarray:
    """
    Top-K fusion: keep the K lowest-entropy views per sample and average them.

    Args:
        per_view_probs: (N, S, C)
        k: number of views to keep (clipped to [1, N]).
    Returns:
        fused probabilities (S, C) — a mean of K simplex vectors, so still on
        the simplex.
    """
    keep = top_k_keep_indices(per_view_probs, k, eps=eps)                     # (k_eff, S)
    s_idx = np.broadcast_to(np.arange(per_view_probs.shape[1]), keep.shape)   # (k_eff, S)
    selected = per_view_probs[keep, s_idx, :]                                 # (k_eff, S, C)
    return selected.mean(axis=0)


# Default K values for the Top-K sweep (VMV plan).
TOP_K_VALUES = (3, 5, 7)


def top_k_strategy(k: int) -> str:
    """Canonical strategy name for a given K, e.g. 5 -> 'top5'."""
    return f"top{k}"


# ── Registry of probability-fusion strategies ──────────────────────────────
# Everything that reduces a cached (N, S, C) array. MC Dropout is intentionally
# NOT here: it needs its own stochastic forward passes (src/mc_dropout.py) and is
# dispatched separately.
FUSION_FNS = {
    "baseline": fuse_equal_weight,
    "maxprob": fuse_maxprob,
    "entropy": fuse_entropy,
    "variance": fuse_variance,
    "variance_inv": fuse_variance_inv,
}
# Register Top-K as bound (N,S,C)->(S,C) fusions so fuse("top5", ...),
# ablate_n, and the reliability/significance tooling can all reuse them.
for _k in TOP_K_VALUES:
    FUSION_FNS[top_k_strategy(_k)] = partial(fuse_top_k, k=_k)

# Human-readable labels aligned with the tracker's Sheet 3 column headers.
STRATEGY_LABELS = {
    "baseline": "Baseline TTA (w=1/N)",
    "maxprob": "Max-Prob Weight (w=max p_i)",
    "entropy": "Entropy Weight (w=exp(-H))",
    "variance": "Variance Weight (w=var, sharp up)",
    "variance_inv": "Variance Inv (w=1/(var+e), ablation)",
    "mc_dropout": "MC Dropout (T stochastic passes)",
    "ts_only": "TS Only (softmax(logits/T))",
    "ts_entropy": "TS + Entropy TTA",
    "top3": "Top-K TTA (K=3, hard filter)",
    "top5": "Top-K TTA (K=5, hard filter)",
    "top7": "Top-K TTA (K=7, hard filter)",
}


def fuse(per_view_probs: np.ndarray, strategy: str) -> np.ndarray:
    """
    Dispatch to a probability-fusion strategy by name.

    strategy ∈ {baseline, maxprob, entropy, variance}. mc_dropout is handled
    separately (src/evaluate.py) because it is not a reduction over per-view
    probabilities.
    """
    if strategy not in FUSION_FNS:
        valid = ", ".join(FUSION_FNS)
        raise ValueError(f"Unknown fusion strategy '{strategy}'. Valid: {valid}")
    return FUSION_FNS[strategy](per_view_probs)


def tta_predict(model: nn.Module, loader: DataLoader, device: torch.device,
                augmentations: List[Tuple[AugFn, str]]
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience wrapper: run TTA and return everything a caller needs.

    Returns:
        fused_probs: (num_samples, num_classes) — equal-weight averaged
        per_view_probs: (N, num_samples, num_classes) — for downstream analysis
        labels: (num_samples,)
    """
    per_view_probs, labels = tta_per_view_probs(model, loader, device, augmentations)
    fused = fuse_equal_weight(per_view_probs)
    return fused, per_view_probs, labels
