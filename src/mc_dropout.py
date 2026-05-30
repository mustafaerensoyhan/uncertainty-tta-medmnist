"""
MC Dropout TTA baseline (proposal §3.2, Phase 3).

The other four strategies (baseline / max-prob / entropy / variance) draw their
uncertainty signal from *augmentation* views. MC Dropout draws it from the
*model*: it runs the SAME un-augmented image through the network T times with
dropout left active, then averages the resulting softmax vectors. The spread
across the T passes is an epistemic-uncertainty proxy.

Why a forward hook instead of `model.train()`:
    The Phase 1 baselines were trained with dropout_p=0, so a ResNet-18
    checkpoint has NO dropout layers to enable — calling model.train() would
    only switch BatchNorm into batch-statistics mode (wrong at test time) and
    still produce zero dropout stochasticity. Instead we register a forward
    PRE-hook on the final FC layer that applies functional dropout to its input
    features on every pass, while the model itself stays in eval() so BatchNorm
    keeps its running statistics. This works on the existing checkpoints with no
    retraining and no state_dict changes.

    If a model is built with build_resnet18(dropout_p>0) the head is
    Sequential(Dropout, Linear); the hook still applies cleanly (it adds dropout
    to the head's input) and is consistent across both checkpoint kinds.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def _find_head(model: nn.Module) -> nn.Module:
    """
    Return the classifier head module to inject dropout *before*.

    For build_resnet18 this is `model.fc` (a bare Linear, or Sequential(Dropout,
    Linear)). We hook the head so dropout is applied to the pooled feature vector
    feeding the classifier.
    """
    if hasattr(model, "fc"):
        return model.fc
    raise AttributeError(
        "Model has no .fc head; pass a ResNet-style model or adapt _find_head."
    )


@torch.no_grad()
def mc_dropout_per_pass_probs(model: nn.Module, loader: DataLoader,
                              device: torch.device, T: int = 20, p: float = 0.2
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run T stochastic forward passes with dropout active on the FC input.

    Args:
        model: trained classifier. Kept in eval() so BatchNorm uses running
               stats; dropout is injected functionally by a forward pre-hook.
        loader: the NORMALIZED test loader (standard eval loader — NOT the TTA
                un-normalized loader). MC Dropout does not augment, so it needs
                images normalized exactly as in training. Must be shuffle=False
                so sample order is identical across passes.
        device: torch device
        T: number of stochastic passes (proposal: 20)
        p: dropout probability applied to the head's input features

    Returns:
        per_pass_probs: (T, num_samples, num_classes)
        labels:         (num_samples,)
    """
    model.eval()
    head = _find_head(model)

    def _pre_hook(_module, inputs):
        # inputs is a tuple of positional args to the head; the first is the
        # pooled feature tensor. Replace it with a dropout-masked version.
        x = inputs[0]
        return (F.dropout(x, p=p, training=True),) + inputs[1:]

    handle = head.register_forward_pre_hook(_pre_hook)
    labels_out: np.ndarray | None = None
    try:
        per_pass = []
        for t in range(T):
            probs_batches, label_batches = [], []
            for images, labels in loader:
                images = images.to(device, non_blocking=True)
                logits = model(images)
                probs = F.softmax(logits, dim=1)
                probs_batches.append(probs.cpu().numpy())
                if t == 0:
                    label_batches.append(labels.numpy().ravel())
            per_pass.append(np.concatenate(probs_batches, axis=0))
            if t == 0:
                labels_out = np.concatenate(label_batches, axis=0)
    finally:
        handle.remove()  # always remove the hook, even if a pass raises

    assert labels_out is not None
    return np.stack(per_pass, axis=0), labels_out


def mc_dropout_fuse(per_pass_probs: np.ndarray) -> np.ndarray:
    """Mean over the T stochastic passes. (T, S, C) -> (S, C)."""
    return per_pass_probs.mean(axis=0)


def mc_dropout_predict(model: nn.Module, loader: DataLoader, device: torch.device,
                       T: int = 20, p: float = 0.2
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience wrapper.

    Returns:
        fused_probs:    (S, C) — mean over T passes
        per_pass_probs: (T, S, C) — kept for optional predictive-variance analysis
        labels:         (S,)
    """
    per_pass, labels = mc_dropout_per_pass_probs(model, loader, device, T=T, p=p)
    return mc_dropout_fuse(per_pass), per_pass, labels
