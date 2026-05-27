"""
Visualization utilities — reliability diagrams and training curves.

Used by scripts/train_baseline.py to produce per-dataset visual artifacts
alongside the JSON metrics. The same `reliability_diagram` function will be
reused in Phase 4 to produce the 30 reliability diagrams required by the
proposal (6 datasets × 5 TTA strategies), so its signature is kept generic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

# Use the non-interactive Agg backend so plots save fine on headless
# environments (Kaggle, Colab, CI) and inside long-running scripts.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def reliability_diagram(probs: np.ndarray, labels: np.ndarray,
                        n_bins: int = 10,
                        save_path: str | Path | None = None,
                        title: str = "Reliability diagram") -> None:
    """
    Plot a calibration reliability diagram (proposal §3.4, Metric 6).

    X-axis: predicted confidence in n_bins equal-width bins.
    Y-axis: actual accuracy of predictions falling in each bin.
    Perfect calibration follows the diagonal y = x. Points below the diagonal
    indicate overconfidence; points above indicate underconfidence.

    Args:
        probs: (N, C) softmax probabilities
        labels: (N,) true class labels (or (N, 1), we ravel)
        n_bins: number of confidence bins (proposal §3.4 uses 10)
        save_path: where to save the PNG; if None, no save (plot is closed)
        title: figure title
    """
    labels = np.asarray(labels).ravel()
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float32)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    mids, accs, confs = [], [], []
    n = len(labels)
    for i in range(n_bins):
        m = (conf > bin_edges[i]) & (conf <= bin_edges[i + 1])
        if m.sum() == 0:
            continue
        mids.append((bin_edges[i] + bin_edges[i + 1]) / 2)
        accs.append(correct[m].mean())
        confs.append(conf[m].mean())

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.bar(mids, accs, width=1 / n_bins, alpha=0.6, edgecolor="black",
           label="Observed accuracy")
    if confs:  # scatter only if any non-empty bin
        ax.scatter(confs, accs, color="red", s=22, zorder=3,
                   label="Bin centroids")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def training_curves(log: list[dict[str, Any]] | pd.DataFrame,
                    save_path: str | Path | None = None,
                    title: str = "Training curves") -> None:
    """
    Plot training loss and validation loss/accuracy over epochs.

    Args:
        log: list of per-epoch dicts (or DataFrame) with at minimum
             'epoch', 'train_loss', 'val_loss', 'val_acc'
        save_path: where to save the PNG
        title: overall figure title
    """
    if not isinstance(log, pd.DataFrame):
        log = pd.DataFrame(log)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    # Loss panel
    axes[0].plot(log["epoch"], log["train_loss"], label="train",
                 marker="o", markersize=3)
    if "val_loss" in log.columns:
        axes[0].plot(log["epoch"], log["val_loss"], label="val",
                     marker="s", markersize=3)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].set_title("Loss")
    axes[0].grid(True, alpha=0.3)

    # Validation accuracy panel
    axes[1].plot(log["epoch"], log["val_acc"] * 100.0, color="C1",
                 marker="s", markersize=3, label="val")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Validation accuracy (%)")
    axes[1].set_title("Accuracy"); axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle(title, fontsize=11, y=1.02)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
