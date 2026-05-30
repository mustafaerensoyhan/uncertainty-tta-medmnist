"""
Run standard (equal-weight) Test-Time Augmentation on one MedMNIST dataset
across multiple view counts N, and record how accuracy/ECE/NLL change.

This is the Phase 2 deliverable: it confirms whether equal-weight TTA helps or
hurts, replicating the "I Can't Believe TTA Is Not Better" finding.

Usage from the repo root:
    python -m scripts.run_standard_tta --dataset pathmnist
    python -m scripts.run_standard_tta --dataset bloodmnist --n-views 5 10 20 50

Outputs:
    results/{dataset}_standard_tta.csv          — one row per N (plus the N=1 no-TTA baseline)
    figures/tta/{dataset}_accuracy_vs_n.png     — Accuracy (and ECE) vs N curve

Requires the baseline checkpoint from Phase 1 at
    checkpoints/{dataset}_resnet18.pth
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_config, all_dataset_keys
from src.data import get_tta_test_loader
from src.model import build_resnet18
from src.augmentations import get_augmentation_pipeline
from src.tta import tta_per_view_probs, fuse_equal_weight
from src.metrics import compute_all_metrics
from src.perf import measure_ms_per_image
from src.visualize import accuracy_vs_n
from src.utils import set_seed, get_device, load_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run standard equal-weight TTA on one dataset.")
    p.add_argument("--dataset", required=True, choices=all_dataset_keys())
    p.add_argument("--n-views", type=int, nargs="+", default=[5, 10, 20, 50],
                   help="View counts to evaluate (default: 5 10 20 50).")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--img-size", type=int, default=64, choices=[28, 64])
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-root", default="./data")
    p.add_argument("--checkpoints-dir", default="./checkpoints")
    p.add_argument("--results-dir", default="./results")
    p.add_argument("--figures-dir", default="./figures")
    p.add_argument("--no-time", action="store_true",
                   help="Skip inference-time measurement (leaves Inf.ms blank).")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    cfg = get_config(args.dataset)
    device = get_device(prefer_cuda=not args.cpu)

    ckpt_path = Path(args.checkpoints_dir) / f"{cfg.key}_resnet18.pth"
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found at {ckpt_path}")
        print("Run Phase 1 first:  python -m scripts.train_baseline --dataset "
              f"{cfg.key}")
        return 1

    print(f"\n{'='*70}")
    print(f"  Standard TTA: {cfg.modality}  ({cfg.key})")
    print(f"  Task          : {cfg.task} ({cfg.n_classes} classes)")
    print(f"  Owner         : {cfg.student}")
    print(f"  Device        : {device}")
    print(f"  View counts N : {args.n_views}")
    print(f"  Checkpoint    : {ckpt_path}")
    print(f"{'='*70}\n")

    # Load model + checkpoint
    model = build_resnet18(num_classes=cfg.n_classes, pretrained=False)
    load_checkpoint(model, ckpt_path, device=device)
    model.to(device)

    # Un-normalized test loader (TTA normalizes each view itself)
    test_loader, _ = get_tta_test_loader(
        args.dataset, batch_size=args.batch_size, img_size=args.img_size,
        num_workers=args.num_workers, root=args.data_root,
    )

    rows = []

    # N=1 no-TTA baseline (just the original view) for a clean reference point.
    # We compute this once and prepend it to the table.
    print("Computing N=1 no-TTA baseline (original view only)...")
    base_aug = get_augmentation_pipeline(n_views=1, include_original=True)
    base_probs, labels = tta_per_view_probs(model, test_loader, device, base_aug)
    base_fused = fuse_equal_weight(base_probs)
    base_metrics = compute_all_metrics(base_fused, labels, task=cfg.task)
    n_test = int(labels.shape[0])
    if not args.no_time:
        base_metrics["inf_ms"] = measure_ms_per_image(
            lambda: tta_per_view_probs(model, test_loader, device, base_aug),
            n_test, device)
    else:
        base_metrics["inf_ms"] = None
    rows.append({"dataset": cfg.key, "n_views": 1, **base_metrics})
    print(f"  baseline acc={base_metrics['accuracy']*100:.2f}%  "
          f"ece={base_metrics['ece']:.4f}\n")

    # Each requested N
    for n in args.n_views:
        print(f"Running standard TTA with N={n} views...")
        set_seed(args.seed)  # reproducible augmentation selection per N
        augs = get_augmentation_pipeline(n_views=n, seed=args.seed,
                                         include_original=True)
        start = time.perf_counter()
        per_view, labels = tta_per_view_probs(model, test_loader, device, augs)
        fused = fuse_equal_weight(per_view)
        metrics = compute_all_metrics(fused, labels, task=cfg.task)
        elapsed = time.perf_counter() - start

        # Proper per-image latency for Sheet 2 (warmup + cuda.synchronize).
        if not args.no_time:
            metrics["inf_ms"] = measure_ms_per_image(
                lambda: tta_per_view_probs(model, test_loader, device, augs),
                n_test, device)
        else:
            metrics["inf_ms"] = None

        delta_acc = (metrics["accuracy"] - base_metrics["accuracy"]) * 100
        arrow = "↓ HURTS" if delta_acc < 0 else "↑ helps"
        infms = f" | {metrics['inf_ms']:.2f} ms/img" if metrics["inf_ms"] is not None else ""
        print(f"  N={n:2d} | acc={metrics['accuracy']*100:.2f}% "
              f"(Δ {delta_acc:+.2f}pp {arrow}) | "
              f"ece={metrics['ece']:.4f} | nll={metrics['nll']:.4f} | "
              f"{elapsed:.1f}s{infms}\n")
        rows.append({"dataset": cfg.key, "n_views": n, **metrics})

    # Save results CSV
    df = pd.DataFrame(rows)
    results_path = Path(args.results_dir) / f"{cfg.key}_standard_tta.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_path, index=False)
    print(f"Results saved to: {results_path}")

    # Accuracy-vs-N plot
    fig_path = Path(args.figures_dir) / "tta" / f"{cfg.key}_accuracy_vs_n.png"
    accuracy_vs_n(df, save_path=fig_path,
                  title=f"{cfg.modality} — standard TTA: accuracy vs N")
    print(f"Plot saved to:    {fig_path}")

    # Summary verdict
    best_tta_acc = df[df["n_views"] > 1]["accuracy"].max()
    base_acc = base_metrics["accuracy"]
    print(f"\n{'='*70}")
    if best_tta_acc < base_acc:
        print(f"  VERDICT: standard TTA HURTS on {cfg.key}.")
        print(f"  Best TTA acc {best_tta_acc*100:.2f}% < no-TTA {base_acc*100:.2f}%")
        print(f"  → supports the 'TTA is not better' hypothesis for this dataset.")
    else:
        print(f"  VERDICT: standard TTA helps (or is neutral) on {cfg.key}.")
        print(f"  Best TTA acc {best_tta_acc*100:.2f}% >= no-TTA {base_acc*100:.2f}%")
    print(f"{'='*70}")
    print(f"\nPaste the rows from {results_path} into Sheet 2️⃣ Standard TTA.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
