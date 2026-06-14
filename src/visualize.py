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


def augmentation_grid(image: "np.ndarray | Any", augmentations,
                      save_path: str | Path | None = None,
                      title: str = "Augmentation preview") -> None:
    """
    Plot a [0,1] image under each augmentation in the pipeline — the Phase 2
    Day 4-5 "test each augmentation visually" task.

    Args:
        image: a single [0,1] RGB tensor (C, H, W) or numpy array
        augmentations: list of (fn, name) pairs (from get_augmentation_pipeline)
        save_path: where to save the PNG
        title: figure title
    """
    import torch

    if not isinstance(image, torch.Tensor):
        image = torch.as_tensor(image)

    n = len(augmentations)
    ncols = min(n, 5)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.0, nrows * 2.2))
    axes = np.array(axes).reshape(-1)

    for i, (fn, name) in enumerate(augmentations):
        aug = fn(image).clamp(0, 1)
        axes[i].imshow(aug.permute(1, 2, 0).numpy())
        axes[i].set_title(name, fontsize=8)
        axes[i].axis("off")
    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=11, y=1.0)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def accuracy_vs_n(df: pd.DataFrame, save_path: str | Path | None = None,
                  title: str = "Standard TTA: Accuracy vs N views") -> None:
    """
    Plot accuracy (and ECE on a twin axis) against number of TTA views.

    Expects a DataFrame with columns 'n_views', 'accuracy', and optionally 'ece'.
    A horizontal dashed line marks the no-TTA baseline (n_views == 1) if present.
    """
    df = df.sort_values("n_views")
    fig, ax1 = plt.subplots(figsize=(6, 4))

    ax1.plot(df["n_views"], df["accuracy"] * 100, "o-", color="C0", label="Accuracy")
    ax1.set_xlabel("Number of TTA views (N)")
    ax1.set_ylabel("Accuracy (%)", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(df["n_views"])
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    # Mark the no-TTA baseline if it's in the data (n_views == 1)
    if (df["n_views"] == 1).any():
        base_acc = df.loc[df["n_views"] == 1, "accuracy"].iloc[0] * 100
        ax1.axhline(base_acc, ls="--", color="gray", lw=1,
                    label=f"No-TTA baseline ({base_acc:.1f}%)")

    if "ece" in df.columns:
        ax2 = ax1.twinx()
        ax2.plot(df["n_views"], df["ece"], "s--", color="C3", alpha=0.7, label="ECE")
        ax2.set_ylabel("ECE", color="C3")
        ax2.tick_params(axis="y", labelcolor="C3")

    ax1.legend(loc="best", fontsize=8)
    ax1.set_title(title)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def confidence_strip(views: "list", weights: np.ndarray, aug_names: "list[str]",
                     dataset_name: str, save_path: str | Path | None = None,
                     highlight_idx: "list[int] | np.ndarray | None" = None) -> None:
    """
    Augmentation Confidence Strip — the VMV visual contribution (proposal §3.5).

    Two-row figure: the top row shows each augmented view as a thumbnail; the
    bottom row shows a coloured weight bar (height + Blues intensity) encoding
    the entropy weight w_i assigned to that view. High-weight (trusted) views
    get a tall dark-blue bar; low-weight (rejected) views get a short pale bar.

    Args:
        views: list of N augmented [0,1] image tensors (C, H, W) or arrays
        weights: (N,) normalized entropy weights (sum to 1), one per view
        aug_names: list of N augmentation names (column labels)
        dataset_name: title suffix, e.g. "PathMNIST (Histology)"
        save_path: where to save the PDF/PNG (figures/strip/<name>_strip.pdf)
        highlight_idx: optional view indices to gold-outline — the views Top-K
            TTA would KEEP (the K lowest-entropy views). Visually ties Top-K TTA
            to Figure 1 at zero extra cost (VMV plan, Implementer 1 Task 5).
    """
    import torch

    weights = np.asarray(weights, dtype=np.float64)
    highlight = set(int(i) for i in highlight_idx) if highlight_idx is not None else set()
    n = len(views)
    fig, axes = plt.subplots(2, n, figsize=(n * 1.6, 3.5),
                             gridspec_kw={"height_ratios": [3, 1.4]})
    fig.suptitle(f"Augmentation Confidence Strip — {dataset_name}",
                 fontsize=11, fontweight="bold", y=1.02)

    cmap = plt.get_cmap("Blues")
    span = weights.max() - weights.min()
    norm_w = (weights - weights.min()) / (span + 1e-8)  # 0..1 for colour mapping

    for i in range(n):
        view = views[i]
        if isinstance(view, torch.Tensor):
            img_np = view.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        else:
            img_np = np.asarray(view)
            if img_np.ndim == 3 and img_np.shape[0] in (1, 3):
                img_np = np.transpose(img_np, (1, 2, 0))
        # Per-image min-max stretch purely for display contrast.
        rng = img_np.max() - img_np.min()
        img_np = (img_np - img_np.min()) / (rng + 1e-8)

        ax_img = axes[0, i]
        is_gray = img_np.ndim == 2 or img_np.shape[-1] == 1
        ax_img.imshow(img_np.squeeze(), cmap="gray" if is_gray else None)
        ax_img.set_title(aug_names[i], fontsize=7, rotation=30, ha="right")
        ax_img.axis("off")

        ax_bar = axes[1, i]
        kept = i in highlight
        # Kept (Top-K) bars get a thick gold edge; others keep the navy edge.
        ax_bar.bar(0, weights[i], color=cmap(0.3 + 0.7 * norm_w[i]),
                   width=0.6, edgecolor=("gold" if kept else "navy"),
                   linewidth=(2.6 if kept else 0.5), zorder=3)
        ax_bar.set_ylim(0, weights.max() * 1.15 + 1e-6)
        ax_bar.set_xlim(-0.5, 0.5)
        ax_bar.set_xticks([])
        ax_bar.text(0, weights[i] + weights.max() * 0.03, f"{weights[i]:.2f}",
                    ha="center", va="bottom", fontsize=7,
                    color=("darkgoldenrod" if kept else "black"),
                    fontweight=("bold" if kept else "normal"))
        if i == 0:
            ax_bar.set_ylabel("weight $w_i$", fontsize=8)
        else:
            ax_bar.set_yticks([])

    if highlight:
        # Tiny legend cue so the gold outline is self-explanatory in the paper.
        fig.text(0.5, -0.02, f"gold outline = Top-{len(highlight)} kept (lowest-entropy) views",
                 ha="center", va="top", fontsize=7, color="darkgoldenrod")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def training_curves(log: "list[dict[str, Any]] | pd.DataFrame",
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
