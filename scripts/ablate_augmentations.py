"""
Ablation: leave-one-out over augmentation types (proposal Phase 4) — Phase 4.
Which individual augmentations drive the gains (or do harm)? For a chosen
strategy, run the full pipeline, then re-run with each base augmentation removed
one at a time, and report the change vs the full pipeline. Runs on the existing
checkpoint — no retraining.

A LARGE accuracy drop when an augmentation is removed => that augmentation was
helping. An accuracy RISE when removed => that augmentation was harmful (the
kind of finding this paper is about).

Usage:
    python -m scripts.ablate_augmentations --dataset pathmnist
    python -m scripts.ablate_augmentations --dataset dermamnist --strategy entropy

Outputs:
    results/{dataset}_ablation_aug.csv
    figures/ablation_aug/{dataset}_{strategy}_loo.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.augmentations import BASE_AUGMENTATIONS, aug_identity
from src.config import all_dataset_keys, get_config
from src.data import get_tta_test_loader
from src.metrics import compute_all_metrics
from src.model import build_resnet18
from src.tta import fuse, softmax_np, tta_per_view_logits
from src.utils import get_device, load_checkpoint, set_seed


def _pipeline(exclude: str | None):
    """Original view + every base augmentation except `exclude`, as (fn, name)."""
    pipe = [(aug_identity, "original")]
    for name, fn in BASE_AUGMENTATIONS:
        if name != exclude:
            pipe.append((fn, name))
    return pipe


def _run(model, loader, device, pipe, strategy, task, seed):
    set_seed(seed)
    logits, labels = tta_per_view_logits(model, loader, device, pipe)
    fused = fuse(softmax_np(logits, 1.0, axis=2), strategy)
    return compute_all_metrics(fused, labels, task=task)


def main() -> int:
    ap = argparse.ArgumentParser(description="Leave-one-out augmentation ablation (Phase 4).")
    ap.add_argument("--dataset", required=True, choices=all_dataset_keys())
    ap.add_argument("--strategy", default="entropy")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--img-size", type=int, default=64, choices=[28, 64])
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--checkpoints-dir", default="./checkpoints")
    ap.add_argument("--results-dir", default="./results")
    ap.add_argument("--figures-dir", default="./figures")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    cfg = get_config(args.dataset)
    device = get_device(prefer_cuda=not args.cpu)
    ckpt = Path(args.checkpoints_dir) / f"{cfg.key}_resnet18.pth"
    if not ckpt.exists():
        print(f"ERROR: checkpoint not found at {ckpt}")
        return 1
    model = build_resnet18(num_classes=cfg.n_classes, pretrained=False)
    load_checkpoint(model, ckpt, device=device)
    model.to(device)

    loader, _ = get_tta_test_loader(cfg.key, batch_size=args.batch_size,
                                    img_size=args.img_size,
                                    num_workers=args.num_workers, root=args.data_root)

    full = _run(model, loader, device, _pipeline(None), args.strategy, cfg.task, args.seed)
    print(f"\nLeave-one-out ablation: {cfg.key} / {args.strategy}")
    print(f"full pipeline: acc={full['accuracy']*100:.2f}%  ece={full['ece']:.4f}\n")
    print(f"{'removed':<16}{'acc%':>8}{'Δacc(pp)':>10}{'ECE':>9}{'ΔECE':>10}")
    print("-" * 53)

    rows = [{"dataset": cfg.key, "strategy": args.strategy, "removed": "(none/full)",
             "accuracy": full["accuracy"], "ece": full["ece"], "nll": full["nll"],
             "d_acc_pp": 0.0, "d_ece": 0.0}]
    for name, _fn in BASE_AUGMENTATIONS:
        m = _run(model, loader, device, _pipeline(name), args.strategy, cfg.task, args.seed)
        d_acc = (m["accuracy"] - full["accuracy"]) * 100
        d_ece = m["ece"] - full["ece"]
        flag = "  <- harmful (removing helps)" if d_acc > 0.05 else ""
        print(f"{name:<16}{m['accuracy']*100:>8.2f}{d_acc:>+10.2f}{m['ece']:>9.4f}{d_ece:>+10.4f}{flag}")
        rows.append({"dataset": cfg.key, "strategy": args.strategy, "removed": name,
                     "accuracy": m["accuracy"], "ece": m["ece"], "nll": m["nll"],
                     "d_acc_pp": round(d_acc, 3), "d_ece": round(d_ece, 4)})

    df = pd.DataFrame(rows)
    out = Path(args.results_dir) / f"{cfg.key}_ablation_aug.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    loo = df[df.removed != "(none/full)"].copy()
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["tab:green" if v > 0 else "tab:red" for v in loo.d_acc_pp]
    ax.barh(loo.removed, loo.d_acc_pp, color=colors)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("Δ accuracy when removed (pp) — positive = augmentation was harmful")
    ax.set_title(f"{cfg.key} — {args.strategy}: leave-one-out", fontweight="bold")
    fig_path = Path(args.figures_dir) / "ablation_aug" / f"{cfg.key}_{args.strategy}_loo.pdf"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(fig_path, dpi=300, bbox_inches="tight"); plt.close(fig)

    print(f"\nSaved -> {out} and {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
