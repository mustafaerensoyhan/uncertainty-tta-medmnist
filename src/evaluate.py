"""
Phase 3 — unified evaluation across all TTA strategies (+ addendum additions).

Strategies (8):
    baseline, maxprob, entropy, variance (confidence-aligned),
    variance_inv (literal, ablation), mc_dropout,
    ts_only (temperature scaling, no TTA), ts_entropy (TS + entropy TTA).

run_all_strategies() computes the expensive per-view LOGITS once and reuses them
for every augmentation-based strategy (plain and temperature-scaled), fits the
temperature T on the validation split, runs MC Dropout and TS-only on their own
single/stochastic passes, measures inference time per strategy, and returns the
fused per-image probabilities so the caller can save prediction arrays
(addendum Addition 5) for the Phase 4 statistical tests.

Every metric dict has keys {accuracy, auc_roc, ece, nll}; inference time is
returned alongside as inf_ms.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch.nn as nn

from .augmentations import get_augmentation_pipeline
from .config import get_config
from .data import get_dataloaders, get_tta_test_loader
from .mc_dropout import mc_dropout_per_pass_probs, mc_dropout_fuse, mc_dropout_predict
from .metrics import compute_all_metrics
from .perf import measure_ms_per_image
from .temperature import fit_temperature, predict_probs_with_temperature
from .tta import (FUSION_FNS, TOP_K_VALUES, fuse, fuse_top_k, softmax_np,
                  top_k_strategy, tta_per_view_logits, tta_per_view_probs)
from .utils import set_seed

# All strategies in tracker / paper column order.
ALL_STRATEGIES: List[str] = [
    "baseline", "maxprob", "entropy", "variance", "variance_inv",
    "mc_dropout", "ts_only", "ts_entropy",
]
# The augmentation-based fusions that share one set of per-view forward passes.
_AUG_FUSIONS = ["baseline", "maxprob", "entropy", "variance", "variance_inv"]


def strategies_in_order(top_ks: Tuple[int, ...] = ()) -> List[str]:
    """The 8 core strategies, then the requested Top-K columns (top3/top5/top7)."""
    return ALL_STRATEGIES + [top_k_strategy(k) for k in top_ks]


def run_all_strategies(model: nn.Module, dataset_name: str, device, *,
                       n_views: int = 10, seed: int = 42, batch_size: int = 64,
                       img_size: int = 64, num_workers: int = 2,
                       data_root: str = "./data", mc_T: int = 20, mc_p: float = 0.2,
                       measure_time: bool = True,
                       top_ks: Tuple[int, ...] = TOP_K_VALUES
                       ) -> Tuple[Dict[str, Dict], Dict[str, np.ndarray], np.ndarray, float, int]:
    """
    Evaluate all 8 strategies (+ Top-K columns) on one dataset at a fixed N.

    top_ks: which Top-K (hard entropy filter) columns to add, e.g. (3, 5, 7).
            Pass () to skip them. They reuse the same per-view forward passes as
            the soft fusions, so they cost no extra inference.

    Returns (results, fused_probs, labels, T, n_test):
      results     : {strategy: {accuracy, auc_roc, ece, nll, inf_ms}}
      fused_probs : {strategy: (n_test, C)}  — for saving per-image arrays
      labels      : (n_test,)
      T           : fitted temperature for this dataset
      n_test      : number of test images
    """
    cfg = get_config(dataset_name)

    # ── Fit temperature T on the validation split (once per dataset) ──
    train_loader, val_loader, test_loader, _ = get_dataloaders(
        dataset_name, batch_size=batch_size, img_size=img_size,
        num_workers=num_workers, root=data_root)
    T = fit_temperature(model, val_loader, device)

    # ── Shared per-view LOGITS for the augmentation-based strategies ──
    set_seed(seed)
    augs = get_augmentation_pipeline(n_views=n_views, seed=seed, include_original=True)
    tta_loader, _ = get_tta_test_loader(dataset_name, batch_size=batch_size,
                                        img_size=img_size, num_workers=num_workers,
                                        root=data_root)
    per_view_logits, labels = tta_per_view_logits(model, tta_loader, device, augs)
    n_test = int(labels.shape[0])

    probs_T1 = softmax_np(per_view_logits, temperature=1.0, axis=2)   # plain
    probs_Ts = softmax_np(per_view_logits, temperature=T, axis=2)     # calibrated

    results: Dict[str, Dict] = {}
    fused: Dict[str, np.ndarray] = {}

    for strat in _AUG_FUSIONS:
        fp = FUSION_FNS[strat](probs_T1)
        fused[strat] = fp
        results[strat] = compute_all_metrics(fp, labels, task=cfg.task)

    # Top-K (hard entropy filter) — reuses the same per-view probs (no extra fwd).
    for k in top_ks:
        name = top_k_strategy(k)
        fp_k = fuse_top_k(probs_T1, k)
        fused[name] = fp_k
        results[name] = compute_all_metrics(fp_k, labels, task=cfg.task)

    # ts_entropy: entropy fusion on the temperature-scaled per-view probs
    fp_tse = FUSION_FNS["entropy"](probs_Ts)
    fused["ts_entropy"] = fp_tse
    results["ts_entropy"] = compute_all_metrics(fp_tse, labels, task=cfg.task)

    # ── MC Dropout: stochastic passes on the normalized test loader ──
    set_seed(seed)
    mc_pp, mc_labels = mc_dropout_per_pass_probs(model, test_loader, device,
                                                 T=mc_T, p=mc_p)
    fp_mc = mc_dropout_fuse(mc_pp)
    fused["mc_dropout"] = fp_mc
    results["mc_dropout"] = compute_all_metrics(fp_mc, mc_labels, task=cfg.task)

    # ── TS only: single calibrated forward pass, no augmentation ──
    fp_ts, ts_labels = predict_probs_with_temperature(model, test_loader, device, T)
    fused["ts_only"] = fp_ts
    results["ts_only"] = compute_all_metrics(fp_ts, ts_labels, task=cfg.task)

    # ── Inference time per strategy ──
    for strat in results:
        results[strat]["inf_ms"] = None
    if measure_time:
        # N-view forward cost is shared by all augmentation fusions (incl ts_entropy
        # and every Top-K column); fusion itself is microseconds, so they report the
        # same forward-dominated time.
        nview_ms = measure_ms_per_image(
            lambda: tta_per_view_logits(model, tta_loader, device, augs), n_test, device)
        for strat in _AUG_FUSIONS + ["ts_entropy"] + [top_k_strategy(k) for k in top_ks]:
            results[strat]["inf_ms"] = nview_ms
        results["mc_dropout"]["inf_ms"] = measure_ms_per_image(
            lambda: mc_dropout_per_pass_probs(model, test_loader, device, T=mc_T, p=mc_p),
            n_test, device)
        results["ts_only"]["inf_ms"] = measure_ms_per_image(
            lambda: predict_probs_with_temperature(model, test_loader, device, T),
            n_test, device)

    return results, fused, labels, T, n_test


def tta_evaluate(model: nn.Module, dataset_name: str, strategy: str,
                 n_views: int, device, *, seed: int = 42, batch_size: int = 64,
                 img_size: int = 64, num_workers: int = 2, data_root: str = "./data",
                 mc_T: int = 20, mc_p: float = 0.2, temperature: float | None = None
                 ) -> Dict[str, float | None]:
    """
    Evaluate a SINGLE strategy and return its metric dict (proposal S5 signature).

    strategy ∈ ALL_STRATEGIES. For ts_only / ts_entropy, T is fit on val unless
    `temperature` is passed explicitly.
    """
    cfg = get_config(dataset_name)

    if strategy in ("ts_only", "ts_entropy") and temperature is None:
        _, val_loader, _, _ = get_dataloaders(dataset_name, batch_size=batch_size,
                                              img_size=img_size, num_workers=num_workers,
                                              root=data_root)
        temperature = fit_temperature(model, val_loader, device)

    if strategy == "mc_dropout":
        _, _, test_loader, _ = get_dataloaders(dataset_name, batch_size=batch_size,
                                               img_size=img_size,
                                               num_workers=num_workers, root=data_root)
        set_seed(seed)
        fused, _, labels = mc_dropout_predict(model, test_loader, device, T=mc_T, p=mc_p)
        return compute_all_metrics(fused, labels, task=cfg.task)

    if strategy == "ts_only":
        _, _, test_loader, _ = get_dataloaders(dataset_name, batch_size=batch_size,
                                               img_size=img_size,
                                               num_workers=num_workers, root=data_root)
        fused, labels = predict_probs_with_temperature(model, test_loader, device, temperature)
        return compute_all_metrics(fused, labels, task=cfg.task)

    # augmentation-based strategies (plain or temperature-scaled)
    set_seed(seed)
    augs = get_augmentation_pipeline(n_views=n_views, seed=seed, include_original=True)
    tta_loader, _ = get_tta_test_loader(dataset_name, batch_size=batch_size,
                                        img_size=img_size, num_workers=num_workers,
                                        root=data_root)
    per_view_logits, labels = tta_per_view_logits(model, tta_loader, device, augs)
    if strategy == "ts_entropy":
        probs = softmax_np(per_view_logits, temperature=temperature, axis=2)
        fused = fuse(probs, "entropy")
    else:
        if strategy not in FUSION_FNS:
            valid = ", ".join(ALL_STRATEGIES)
            raise ValueError(f"Unknown strategy '{strategy}'. Valid: {valid}")
        probs = softmax_np(per_view_logits, temperature=1.0, axis=2)
        fused = fuse(probs, strategy)
    return compute_all_metrics(fused, labels, task=cfg.task)
