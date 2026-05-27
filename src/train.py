"""
Training loop and evaluation routine.

The training loop is intentionally minimal — Adam, cross-entropy, fixed LR,
best-checkpoint selection by validation accuracy. We avoid extras (LR schedule,
mixup, label smoothing) for the baseline because those would conflate the
baseline number with downstream TTA gains.

Phase 2 (standard TTA) and Phase 3 (uncertainty-weighted TTA) live in
separate modules and operate on the checkpoints this script produces.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import DatasetConfig
from .metrics import compute_all_metrics


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             task: str) -> Dict[str, float | None]:
    """
    Run model over a loader, collect softmax probabilities, compute all
    four metrics in one pass. Returns a dict suitable for the results tracker.
    """
    model.eval()
    all_probs, all_labels = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = F.softmax(logits, dim=1)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy().ravel())

    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return compute_all_metrics(probs, labels, task=task)


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                    criterion: nn.Module, device: torch.device) -> float:
    """One pass over the training set. Returns mean training loss."""
    model.train()
    running_loss, n_samples = 0.0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long().ravel()

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        n_samples += images.size(0)
    return running_loss / max(n_samples, 1)


def fit(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
        cfg: DatasetConfig, device: torch.device, *,
        epochs: int = 30, lr: float = 1e-4, weight_decay: float = 0.0,
        checkpoint_path: str | Path | None = None) -> Dict[str, float | None]:
    """
    Train ResNet-18 on `cfg`'s dataset for `epochs` epochs.

    Saves the best checkpoint (by val accuracy) to checkpoint_path. Returns
    the validation metrics of the best checkpoint.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_acc, best_metrics = -1.0, {}
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device, task=cfg.task)
        val_acc = val_metrics["accuracy"] or 0.0

        flag = "  ✓ best" if val_acc > best_acc else ""
        print(
            f"  Epoch {epoch:3d}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_acc={val_acc*100:.2f}% | "
            f"val_ece={val_metrics['ece']:.4f}"
            f"{flag}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_metrics = val_metrics
            if checkpoint_path is not None:
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": model.state_dict(),
                            "val_metrics": val_metrics,
                            "epoch": epoch,
                            "dataset": cfg.key}, checkpoint_path)

    return best_metrics
