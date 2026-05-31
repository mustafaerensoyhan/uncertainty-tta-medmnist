"""
Run all TTA strategies on one MedMNIST dataset and record the full Phase 3
deliverable (addendum-aligned): metrics + inference time per strategy, the
fitted temperature T, and per-image prediction arrays for the Phase 4 stats.

Strategies (8): baseline, maxprob, entropy, variance, variance_inv,
mc_dropout, ts_only, ts_entropy.

Usage from the repo root:
    python -m scripts.run_weighted_tta --dataset bloodmnist
    python -m scripts.run_weighted_tta --dataset pathmnist --n-views 10 --no-time

Outputs:
    results/{dataset}_weighted_tta.csv     — one row per strategy (incl inf_ms, T)
    predictions/{dataset}_labels.npy       — ground-truth labels (once)
    predictions/{dataset}_{strategy}_preds.npy   — predicted class per image
    predictions/{dataset}_{strategy}_probs.npy   — full softmax per image
    (predictions/ is gitignored — push to the shared Drive, not git.)

Requires the Phase 1 checkpoint at checkpoints/{dataset}_resnet18.pth.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import all_dataset_keys, get_config
from src.evaluate import ALL_STRATEGIES, run_all_strategies
from src.model import build_resnet18
from src.tta import STRATEGY_LABELS
from src.utils import get_device, load_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run all weighted-TTA strategies on one dataset (Phase 3).")
    p.add_argument("--dataset", required=True, choices=all_dataset_keys())
    p.add_argument("--n-views", type=int, default=10)
    p.add_argument("--mc-T", type=int, default=20)
    p.add_argument("--mc-p", type=float, default=0.2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--img-size", type=int, default=64, choices=[28, 64])
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers. 0 (default) is safe on Windows; "
                        "use 2-4 on Linux/Kaggle for a speedup.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-root", default="./data")
    p.add_argument("--checkpoints-dir", default="./checkpoints")
    p.add_argument("--results-dir", default="./results")
    p.add_argument("--predictions-dir", default="./predictions")
    p.add_argument("--no-time", action="store_true",
                   help="Skip inference-time measurement (faster; leaves Inf.ms blank).")
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
        print(f"Run Phase 1 first:  python -m scripts.train_baseline --dataset {cfg.key}")
        return 1

    print(f"\n{'='*78}")
    print(f"  Weighted TTA (Phase 3): {cfg.modality}  ({cfg.key})")
    print(f"  Task    : {cfg.task} ({cfg.n_classes} classes)   Owner: {cfg.student}")
    print(f"  Device  : {device}   N views: {args.n_views}   MC: T={args.mc_T},p={args.mc_p}")
    print(f"  Timing  : {'OFF' if args.no_time else 'ON (warmup + cuda.synchronize)'}")
    print(f"{'='*78}\n")

    model = build_resnet18(num_classes=cfg.n_classes, pretrained=False)
    load_checkpoint(model, ckpt_path, device=device)
    model.to(device)

    print("Fitting temperature on val, computing per-view logits once, "
          "fusing all strategies...\n")
    results, fused, labels, T, n_test = run_all_strategies(
        model, args.dataset, device,
        n_views=args.n_views, seed=args.seed, batch_size=args.batch_size,
        img_size=args.img_size, num_workers=args.num_workers,
        data_root=args.data_root, mc_T=args.mc_T, mc_p=args.mc_p,
        measure_time=not args.no_time,
    )
    print(f"Fitted temperature T = {T:.3f}\n")

    base_acc = results["baseline"]["accuracy"]
    base_ece = results["baseline"]["ece"]

    def fmt(v, pct=False):
        if v is None:
            return f"{'-':>9}"
        return f"{v*100:9.2f}" if pct else f"{v:9.4f}"

    print(f"{'strategy':<14}{'acc%':>9}{'ECE':>9}{'NLL':>9}{'AUC':>9}"
          f"{'Inf.ms':>10}{'Δacc':>9}{'ΔECE':>10}")
    print("-" * 80)
    rows = []
    for strat in ALL_STRATEGIES:
        m = results[strat]
        d_acc = (m["accuracy"] - base_acc) * 100
        d_ece = m["ece"] - base_ece
        infms = f"{m['inf_ms']:10.3f}" if m["inf_ms"] is not None else f"{'-':>10}"
        tag = "" if strat == "baseline" else f"{d_acc:+9.2f}{d_ece:+10.4f}"
        print(f"{strat:<14}{fmt(m['accuracy'], pct=True)}"
              f"{fmt(m['ece'])}{fmt(m['nll'])}{fmt(m['auc_roc'])}{infms}{tag}")
        rows.append({
            "dataset": cfg.key, "student": cfg.student, "strategy": strat,
            "n_views": args.n_views if strat not in ("mc_dropout", "ts_only") else
                       (f"T={args.mc_T}" if strat == "mc_dropout" else 1),
            "accuracy": m["accuracy"], "auc_roc": m["auc_roc"],
            "ece": m["ece"], "nll": m["nll"], "inf_ms": m["inf_ms"],
            "temperature": round(T, 4),
        })

    out = Path(args.results_dir) / f"{cfg.key}_weighted_tta.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nResults saved to: {out}   ({n_test} test images, T={T:.3f})")

    # ── Per-image prediction arrays (addendum Addition 5) ──
    pdir = Path(args.predictions_dir)
    pdir.mkdir(parents=True, exist_ok=True)
    np.save(pdir / f"{cfg.key}_labels.npy", labels)
    for strat in ALL_STRATEGIES:
        probs = fused[strat]
        np.save(pdir / f"{cfg.key}_{strat}_probs.npy", probs.astype(np.float32))
        np.save(pdir / f"{cfg.key}_{strat}_preds.npy", probs.argmax(axis=1).astype(np.int64))
    print(f"Saved per-image preds/probs for {len(ALL_STRATEGIES)} strategies + labels "
          f"to {pdir}/ (push to shared Drive, NOT git).")

    weighted = {s: results[s]["accuracy"] for s in ALL_STRATEGIES if s != "baseline"}
    best = max(weighted, key=weighted.get)
    print(f"\n{'='*78}")
    print(f"  Best non-baseline on {cfg.key}: {STRATEGY_LABELS[best]} "
          f"({weighted[best]*100:.2f}% vs {base_acc*100:.2f}%, "
          f"Δ {(weighted[best]-base_acc)*100:+.2f}pp)")
    print(f"{'='*78}")
    print("Paste the rows into Sheet 3 (TS Only / TS+Entropy / Inf.ms / variance_inv now included).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
