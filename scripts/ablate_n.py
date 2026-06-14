"""
Ablation: effect of N (number of TTA views) on accuracy and ECE for a chosen
strategy (proposal Phase 4) — Phase 4. Runs on the existing checkpoint, no
retraining.

Usage:
    python -m scripts.ablate_n --dataset pathmnist                 # entropy by default
    python -m scripts.ablate_n --dataset pathmnist --strategy variance --n-views 2 5 10 20 50

Outputs:
    results/{dataset}_ablation_N.csv
    figures/ablation_N/{dataset}_{strategy}_N.pdf   (accuracy & ECE vs N)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.augmentations import get_augmentation_pipeline
from src.config import all_dataset_keys, get_config
from src.data import get_tta_test_loader
from src.metrics import compute_all_metrics
from src.model import build_model, ARCHITECTURES
from src.tta import fuse, softmax_np, tta_per_view_logits
from src.utils import (get_device, load_checkpoint, set_seed,
                       checkpoint_filename, result_stem)


def main() -> int:
    ap = argparse.ArgumentParser(description="N-views ablation (Phase 4).")
    ap.add_argument("--dataset", required=True, choices=all_dataset_keys())
    ap.add_argument("--arch", default="resnet18", choices=list(ARCHITECTURES),
                    help="Backbone: resnet18 (default) or effb0.")
    ap.add_argument("--strategy", default="entropy")
    ap.add_argument("--n-views", type=int, nargs="+", default=[2, 5, 10, 20, 50])
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
    ckpt = Path(args.checkpoints_dir) / checkpoint_filename(cfg.key, args.arch)
    if not ckpt.exists():
        print(f"ERROR: checkpoint not found at {ckpt}")
        return 1
    model = build_model(args.arch, num_classes=cfg.n_classes, pretrained=False)
    load_checkpoint(model, ckpt, device=device)
    model.to(device)

    tta_loader, _ = get_tta_test_loader(cfg.key, batch_size=args.batch_size,
                                        img_size=args.img_size,
                                        num_workers=args.num_workers, root=args.data_root)

    print(f"\nN-ablation: {cfg.key} / {args.strategy}\n{'N':>4}{'acc%':>9}{'ECE':>9}{'NLL':>9}")
    print("-" * 31)
    rows = []
    for n in args.n_views:
        set_seed(args.seed)
        augs = get_augmentation_pipeline(n_views=n, seed=args.seed, include_original=True)
        logits, labels = tta_per_view_logits(model, tta_loader, device, augs)
        fused = fuse(softmax_np(logits, 1.0, axis=2), args.strategy)
        m = compute_all_metrics(fused, labels, task=cfg.task)
        print(f"{n:>4}{m['accuracy']*100:>9.2f}{m['ece']:>9.4f}{m['nll']:>9.4f}")
        rows.append({"dataset": cfg.key, "strategy": args.strategy, "n_views": n, **m})

    df = pd.DataFrame(rows)
    stem = result_stem(cfg.key, args.arch)
    out = Path(args.results_dir) / f"{stem}_ablation_N.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(df.n_views, df.accuracy * 100, "o-", color="tab:blue", label="accuracy")
    ax1.set_xlabel("N (views)"); ax1.set_ylabel("accuracy (%)", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(df.n_views, df.ece, "s--", color="tab:red", label="ECE")
    ax2.set_ylabel("ECE", color="tab:red")
    ax1.set_title(f"{cfg.key} — {args.strategy}: accuracy & ECE vs N", fontweight="bold")
    ax1.set_xscale("log"); ax1.set_xticks(args.n_views); ax1.set_xticklabels(args.n_views)
    fig_path = Path(args.figures_dir) / "ablation_N" / f"{stem}_{args.strategy}_N.pdf"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(fig_path, dpi=300, bbox_inches="tight"); plt.close(fig)

    print(f"\nSaved -> {out} and {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
