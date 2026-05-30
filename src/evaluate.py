"""
Phase 3 — unified evaluation across all five TTA strategies.

Two entry points:

  tta_evaluate(model, dataset_name, strategy, n_views, device)
      Evaluate ONE strategy and return its full metric dict. This is the
      signature the proposal (Phase 3 / S5) specifies.

  run_all_strategies(model, dataset_name, device, n_views=10, ...)
      Efficient batch path used by scripts/run_weighted_tta.py. Computes the
      expensive per-view probabilities ONCE and reuses them for the four
      reduction strategies (baseline / max-prob / entropy / variance), then runs
      MC Dropout on its own stochastic path. Returns {strategy: metric_dict}.

All metrics come from src.metrics.compute_all_metrics, so each dict has keys
{accuracy, auc_roc, ece, nll}. AUC is None for multi-class-with-one-class edge
cases and meaningful for the binary datasets.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch.nn as nn

from .augmentations import get_augmentation_pipeline
from .config import get_config
from .data import get_dataloaders, get_tta_test_loader
from .mc_dropout import mc_dropout_predict
from .metrics import compute_all_metrics
from .tta import FUSION_FNS, fuse, tta_per_view_probs
from .utils import set_seed

# All five strategies, in the tracker's / paper's Sheet-3 column order.
ALL_STRATEGIES: List[str] = ["baseline", "maxprob", "entropy", "variance", "mc_dropout"]


def run_all_strategies(model: nn.Module, dataset_name: str, device, *,
                       n_views: int = 10, seed: int = 42, batch_size: int = 64,
                       img_size: int = 64, num_workers: int = 2,
                       data_root: str = "./data", mc_T: int = 20, mc_p: float = 0.2
                       ) -> Tuple[Dict[str, Dict], int]:
    """
    Evaluate all five strategies on one dataset at a fixed N.

    Returns (results, n_test) where results maps each strategy name to its
    metric dict, and n_test is the number of test samples (handy for the
    tracker / inference-time bookkeeping).
    """
    cfg = get_config(dataset_name)

    # ── Shared expensive part: per-view softmax probabilities (used by the four
    #    reduction strategies). Computed exactly once. ──
    set_seed(seed)  # reproducible augmentation selection
    augs = get_augmentation_pipeline(n_views=n_views, seed=seed, include_original=True)
    tta_loader, _ = get_tta_test_loader(dataset_name, batch_size=batch_size,
                                        img_size=img_size, num_workers=num_workers,
                                        root=data_root)
    per_view_probs, labels = tta_per_view_probs(model, tta_loader, device, augs)
    n_test = int(labels.shape[0])

    results: Dict[str, Dict] = {}
    for strat in ("baseline", "maxprob", "entropy", "variance"):
        fused = fuse(per_view_probs, strat)
        results[strat] = compute_all_metrics(fused, labels, task=cfg.task)

    # ── MC Dropout: separate stochastic path on the NORMALIZED test loader ──
    _, _, test_loader, _ = get_dataloaders(dataset_name, batch_size=batch_size,
                                           img_size=img_size, num_workers=num_workers,
                                           root=data_root)
    set_seed(seed)
    mc_fused, _, mc_labels = mc_dropout_predict(model, test_loader, device,
                                                T=mc_T, p=mc_p)
    results["mc_dropout"] = compute_all_metrics(mc_fused, mc_labels, task=cfg.task)

    return results, n_test


def tta_evaluate(model: nn.Module, dataset_name: str, strategy: str,
                 n_views: int, device, *, seed: int = 42, batch_size: int = 64,
                 img_size: int = 64, num_workers: int = 2, data_root: str = "./data",
                 mc_T: int = 20, mc_p: float = 0.2) -> Dict[str, float | None]:
    """
    Evaluate a SINGLE strategy and return its metric dict
    {accuracy, auc_roc, ece, nll} (proposal Phase 3 / S5 signature).

    strategy ∈ {baseline, maxprob, entropy, variance, mc_dropout}.
    """
    cfg = get_config(dataset_name)

    if strategy == "mc_dropout":
        _, _, test_loader, _ = get_dataloaders(dataset_name, batch_size=batch_size,
                                               img_size=img_size,
                                               num_workers=num_workers, root=data_root)
        set_seed(seed)
        fused, _, labels = mc_dropout_predict(model, test_loader, device,
                                              T=mc_T, p=mc_p)
        return compute_all_metrics(fused, labels, task=cfg.task)

    if strategy not in FUSION_FNS:
        valid = ", ".join(list(FUSION_FNS) + ["mc_dropout"])
        raise ValueError(f"Unknown strategy '{strategy}'. Valid: {valid}")

    set_seed(seed)
    augs = get_augmentation_pipeline(n_views=n_views, seed=seed, include_original=True)
    tta_loader, _ = get_tta_test_loader(dataset_name, batch_size=batch_size,
                                        img_size=img_size, num_workers=num_workers,
                                        root=data_root)
    per_view_probs, labels = tta_per_view_probs(model, tta_loader, device, augs)
    fused = fuse(per_view_probs, strategy)
    return compute_all_metrics(fused, labels, task=cfg.task)
