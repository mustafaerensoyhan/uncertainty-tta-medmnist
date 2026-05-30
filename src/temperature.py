"""
Temperature scaling (addendum Addition 2).

A single scalar T, fit once per dataset on the validation split by minimizing
NLL, is divided into the logits before softmax: probs = softmax(logits / T).
T > 1 softens overconfident predictions and improves calibration (ECE/NLL)
WITHOUT changing accuracy — dividing every logit by the same T preserves the
argmax, so the predicted class is unchanged.

This is the standard calibration baseline reviewers expect. We use it two ways
in Phase 3:
  - ts_only    : single forward pass with softmax(logits/T), no augmentation.
  - ts_entropy : the full entropy-weighted TTA pipeline, but every per-view
                 softmax uses logits/T (see src/evaluate.py).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def fit_temperature(model: nn.Module, val_loader: DataLoader, device: torch.device,
                    max_iter: int = 50, lr: float = 0.01) -> float:
    """
    Fit the temperature T on the validation set by minimizing cross-entropy
    (== NLL) of softmax(logits / T) against the true labels, using LBFGS.

    val_loader must yield NORMALIZED images (the standard eval loader), since T
    is a property of the trained model's logits, not of any augmentation.

    Returns the scalar T (typically ~1.2–2.5).
    """
    model.eval()

    # Gather all validation logits + labels once (cheap; val splits are small).
    all_logits, all_labels = [], []
    with torch.no_grad():
        for img, label in val_loader:
            img = img.to(device)
            all_logits.append(model(img).detach())
            all_labels.append(torch.as_tensor(label).reshape(-1).long())
    logits = torch.cat(all_logits).to(device)
    labels = torch.cat(all_labels).to(device)

    log_T = torch.zeros(1, device=device, requires_grad=True)  # optimize log T > 0
    optimizer = torch.optim.LBFGS([log_T], lr=lr, max_iter=max_iter)
    criterion = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = criterion(logits / log_T.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    T = float(log_T.exp().item())
    # Guard against degenerate optima.
    if not np.isfinite(T) or T <= 0:
        T = 1.0
    return T


@torch.no_grad()
def predict_probs_with_temperature(model: nn.Module, loader: DataLoader,
                                   device: torch.device, T: float
                                   ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Single forward pass per image with calibrated softmax (no augmentation).
    Used for the `ts_only` column.

    Returns (probs (S, C), labels (S,)).
    """
    model.eval()
    probs_batches, label_batches = [], []
    for img, label in loader:
        img = img.to(device, non_blocking=True)
        logits = model(img)
        probs = F.softmax(logits / float(T), dim=1)
        probs_batches.append(probs.cpu().numpy())
        label_batches.append(np.asarray(label).reshape(-1))
    return np.concatenate(probs_batches, axis=0), np.concatenate(label_batches, axis=0)
