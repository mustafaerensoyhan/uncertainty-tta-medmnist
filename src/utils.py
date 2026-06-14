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


# ── File-naming policy (backbone-aware) ────────────────────────────────────
# Phase 5 adds a second backbone (EfficientNet-B0). To avoid colliding with the
# existing ResNet-18 artifacts AND to keep every Phase 1-4 file/glob working
# unchanged, ResNet-18 keeps its original "archless" result/prediction stem and
# its `{ds}_resnet18{tag}.pth` checkpoint name; any OTHER backbone gets an
# explicit `_{arch}` infix. So:
#   checkpoint_filename("pathmnist", "resnet18")          -> pathmnist_resnet18.pth     (existing)
#   checkpoint_filename("pathmnist", "effb0", "_seed0")   -> pathmnist_effb0_seed0.pth  (new)
#   result_stem("pathmnist", "resnet18", "_seed0")        -> pathmnist_seed0            (existing)
#   result_stem("pathmnist", "effb0", "_seed0")           -> pathmnist_effb0_seed0      (new)
# Everything that consumes ResNet-18 files (build_full_matrix, aggregate_seeds,
# significance, the tracker) therefore needs no change; EfficientNet lives in a
# parallel namespace.

DEFAULT_ARCH = "resnet18"


def checkpoint_filename(dataset_key: str, arch: str = DEFAULT_ARCH,
                        tag: str = "") -> str:
    """Checkpoint filename for a (dataset, backbone, tag). See policy note above."""
    return f"{dataset_key}_{arch}{tag}.pth"


def result_stem(dataset_key: str, arch: str = DEFAULT_ARCH, tag: str = "") -> str:
    """
    Stem for result/prediction files: `<stem>_weighted_tta.csv`, `<stem>_labels.npy`,
    etc. ResNet-18 keeps the original archless stem for backward compatibility;
    other backbones get an `_{arch}` infix.
    """
    if arch == DEFAULT_ARCH:
        return f"{dataset_key}{tag}"
    return f"{dataset_key}_{arch}{tag}"


def default_ckpt_tag(arch: str, seed: int, canonical_seed: int = 42) -> str:
    """
    The tag the multi-backbone runner uses for a (arch, seed). ResNet-18 at the
    canonical seed (42) writes the untagged canonical files (so the headline
    Phase 1-3 numbers stay put); every other (arch, seed) is seed-tagged.
    """
    if arch == DEFAULT_ARCH and seed == canonical_seed:
        return ""
    return f"_seed{seed}"


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
