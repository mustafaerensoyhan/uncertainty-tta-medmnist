"""
Train a baseline ResNet-18 on one MedMNIST dataset.

Usage from the repo root:
    python -m scripts.train_baseline --dataset pathmnist
    python -m scripts.train_baseline --dataset bloodmnist --epochs 30 --batch-size 64

Outputs:
    checkpoints/{dataset}_resnet18.pth                      — best-val checkpoint
    results/{dataset}_baseline.json                         — test metrics (Phase 1 deliverable)
    results/{dataset}_train_log.csv                         — per-epoch losses + val metrics
    figures/reliability/{dataset}_baseline.png              — calibration plot
    figures/curves/{dataset}_train_curves.png               — training/val loss + accuracy curves

The script also prints a copy-pasteable row for the 1️⃣  Baselines sheet
of the results tracker.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

# Allow running as `python -m scripts.train_baseline` from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_config, all_dataset_keys
from src.data import get_dataloaders
from src.model import build_resnet18, count_parameters
from src.train import fit, evaluate, predict_probs
from src.utils import (set_seed, get_device, save_json, print_tracker_row,
                       load_checkpoint)
from src.visualize import reliability_diagram, training_curves


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ResNet-18 baseline on one MedMNIST dataset.")
    p.add_argument("--dataset", required=True, choices=all_dataset_keys(),
                   help="Dataset key (e.g. pathmnist, bloodmnist).")
    p.add_argument("--epochs", type=int, default=30,
                   help="Training epochs (proposal: 30).")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--img-size", type=int, default=64, choices=[28, 64])
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-root", default="./data",
                   help="Folder where MedMNIST .npz files are cached.")
    p.add_argument("--checkpoints-dir", default="./checkpoints")
    p.add_argument("--results-dir", default="./results")
    p.add_argument("--figures-dir", default="./figures")
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    cfg = get_config(args.dataset)
    device = get_device(prefer_cuda=not args.cpu)

    print(f"\n{'='*70}")
    print(f"  Training baseline: {cfg.modality}")
    print(f"  Dataset key       : {cfg.key}")
    print(f"  Task              : {cfg.task} ({cfg.n_classes} classes)")
    print(f"  Channels (native) : {cfg.n_channels} → converted to 3 (ImageNet)")
    print(f"  Owner             : {cfg.student}")
    print(f"  Device            : {device}")
    print(f"  Epochs            : {args.epochs}")
    print(f"  Batch size        : {args.batch_size}")
    print(f"  LR                : {args.lr}")
    print(f"  Benchmark target  : {cfg.benchmark_acc*100:.1f}% (±2% tolerance)")
    print(f"{'='*70}\n")

    # Data
    print("Loading data (this triggers download on first run)...")
    train_loader, val_loader, test_loader, _ = get_dataloaders(
        args.dataset,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
        root=args.data_root,
    )
    print(f"  train batches: {len(train_loader)} | val: {len(val_loader)} | test: {len(test_loader)}")

    # Model
    model = build_resnet18(num_classes=cfg.n_classes, pretrained=True)
    print(f"  ResNet-18 parameters: {count_parameters(model):,}\n")

    # Train
    ckpt_path = Path(args.checkpoints_dir) / f"{cfg.key}_resnet18.pth"
    start = time.perf_counter()
    print(f"Training for {args.epochs} epochs (best-val checkpoint saved to {ckpt_path})\n")
    _, train_log = fit(model, train_loader, val_loader, cfg=cfg, device=device,
                       epochs=args.epochs, lr=args.lr, checkpoint_path=ckpt_path)
    elapsed = time.perf_counter() - start
    print(f"\nTraining done in {elapsed/60:.1f} min.")

    # Save per-epoch training log as CSV — useful for paper supplementary
    # and for diagnosing convergence issues across the team's 6 runs.
    log_path = Path(args.results_dir) / f"{cfg.key}_train_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train_log).to_csv(log_path, index=False)
    print(f"Training log saved to: {log_path}")

    # Reload best checkpoint and evaluate on the held-out test set
    print(f"\nLoading best checkpoint and evaluating on test split...")
    load_checkpoint(model, ckpt_path, device=device)
    model.to(device)
    # We need the raw probs/labels for the reliability diagram, so use
    # predict_probs once and derive both the metrics and the diagram from it.
    test_probs, test_labels = predict_probs(model, test_loader, device)
    from src.metrics import compute_all_metrics
    test_metrics = compute_all_metrics(test_probs, test_labels, task=cfg.task)

    # Generate figures — reliability diagram + training curves
    reliab_path = Path(args.figures_dir) / "reliability" / f"{cfg.key}_baseline.png"
    curves_path = Path(args.figures_dir) / "curves" / f"{cfg.key}_train_curves.png"
    reliability_diagram(
        test_probs, test_labels, n_bins=10,
        save_path=reliab_path,
        title=f"{cfg.modality} baseline — ECE={test_metrics['ece']:.3f}",
    )
    training_curves(
        train_log, save_path=curves_path,
        title=f"{cfg.modality} — ResNet-18 training",
    )
    print(f"Reliability diagram saved to: {reliab_path}")
    print(f"Training curves saved to:   {curves_path}")

    # Save metrics
    results_path = Path(args.results_dir) / f"{cfg.key}_baseline.json"
    save_json({
        "dataset": cfg.key,
        "modality": cfg.modality,
        "student": cfg.student,
        "task": cfg.task,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "img_size": args.img_size,
        "checkpoint": str(ckpt_path),
        "train_log_csv": str(log_path),
        "reliability_figure": str(reliab_path),
        "curves_figure": str(curves_path),
        "benchmark_acc": cfg.benchmark_acc,
        "test_metrics": test_metrics,
        "training_time_minutes": round(elapsed/60, 2),
    }, results_path)

    # Sanity check against the published benchmark — flag if we're outside ±2%
    test_acc = test_metrics["accuracy"]
    delta = test_acc - cfg.benchmark_acc
    inside_tolerance = abs(delta) <= 0.02
    flag = "✓ within ±2% tolerance" if inside_tolerance else "⚠ OUTSIDE ±2% tolerance — investigate"
    print(f"\nTest acc = {test_acc*100:.2f}% | benchmark = {cfg.benchmark_acc*100:.1f}% | Δ = {delta*100:+.2f}% | {flag}")

    print_tracker_row(cfg.key, cfg.student, test_metrics, str(ckpt_path))
    print(f"\nMetrics JSON saved to: {results_path}")

    return 0 if inside_tolerance else 1


if __name__ == "__main__":
    raise SystemExit(main())
