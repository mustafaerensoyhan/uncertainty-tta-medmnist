"""Utility helpers — seeding, device detection, checkpoint I/O."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed every RNG we use. Determinism is best-effort on CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # cuDNN deterministic mode trades a bit of speed for reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return a CUDA device if available, else CPU."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(model: torch.nn.Module, path: str | Path,
                    extra: Dict[str, Any] | None = None) -> None:
    """Save model state_dict plus any metadata we want to keep with the weights."""
    payload: Dict[str, Any] = {"state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(model: torch.nn.Module, path: str | Path,
                    device: torch.device | None = None) -> Dict[str, Any]:
    """Load weights into `model` in place, return the rest of the payload."""
    ckpt = torch.load(path, map_location=device or "cpu")
    model.load_state_dict(ckpt["state_dict"])
    extra = {k: v for k, v in ckpt.items() if k != "state_dict"}
    return extra


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def print_tracker_row(dataset_key: str, student: str, metrics: Dict[str, Any],
                       ckpt_path: str) -> None:
    """
    Print a one-line summary in the order the results tracker spreadsheet expects:
    Dataset | Modality | Student | Accuracy(%) | AUC | ECE | NLL | Checkpoint
    so Mustafa can copy it straight into Sheet 1️⃣  Baselines.
    """
    acc = metrics.get("accuracy")
    auc = metrics.get("auc_roc")
    ece = metrics.get("ece")
    nll = metrics.get("nll")

    def fmt(v, places=4, pct=False):
        if v is None:
            return "N/A"
        return f"{v*100:.2f}" if pct else f"{v:.{places}f}"

    print()
    print("=" * 70)
    print(f"📊  Tracker row for {dataset_key}  (paste into Sheet 1️⃣ Baselines)")
    print("=" * 70)
    print(f"  Student        : {student}")
    print(f"  Accuracy (%)   : {fmt(acc, pct=True)}")
    print(f"  AUC-ROC        : {fmt(auc)}")
    print(f"  ECE ↓          : {fmt(ece)}")
    print(f"  NLL ↓          : {fmt(nll)}")
    print(f"  Checkpoint     : {ckpt_path}")
    print("=" * 70)
