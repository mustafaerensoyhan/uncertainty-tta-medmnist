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

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .augmentations import AugFn
from .data import normalize_imagenet


@torch.no_grad()
def tta_per_view_probs(model: nn.Module, loader: DataLoader, device: torch.device,
                       augmentations: List[Tuple[AugFn, str]]
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run TTA over a test loader and return per-view softmax probabilities.

    Args:
        model: trained classifier (eval mode is set internally)
        loader: yields UN-normalized [0, 1] RGB image batches + labels
        device: torch device
        augmentations: list of (fn, name) pairs of length N (from
                       augmentations.get_augmentation_pipeline)

    Returns:
        per_view_probs: float array of shape (N, num_samples, num_classes)
        labels: int array of shape (num_samples,)

    Memory note: this materializes (N × num_samples × C) floats. For the
    MedMNIST test splits (a few thousand images, <=11 classes) that's small
    even at N=50. If a future dataset is much larger, stream the fusion instead.
    """
    model.eval()
    n_views = len(augmentations)

    per_view_batches: List[np.ndarray] = []  # each: (N, B, C)
    label_batches: List[np.ndarray] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)  # (B, 3, H, W) in [0,1]
        B = images.shape[0]

        view_probs = []  # list of (B, C) per view
        for aug_fn, _name in augmentations:
            # Apply augmentation per-image (augs are defined on single images),
            # then stack back into a batch.
            aug_imgs = torch.stack([aug_fn(img) for img in images], dim=0)
            aug_imgs = normalize_imagenet(aug_imgs)
            logits = model(aug_imgs)
            probs = F.softmax(logits, dim=1)  # (B, C)
            view_probs.append(probs.cpu().numpy())

        per_view_batches.append(np.stack(view_probs, axis=0))  # (N, B, C)
        label_batches.append(labels.numpy().ravel())

    # Concatenate along the sample axis
    per_view_probs = np.concatenate(per_view_batches, axis=1)  # (N, num_samples, C)
    labels = np.concatenate(label_batches, axis=0)
    assert per_view_probs.shape[0] == n_views
    return per_view_probs, labels


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
