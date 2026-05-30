"""
Run all five TTA fusion strategies on one MedMNIST dataset at a fixed N and
record the full metric set — the Phase 3 deliverable that fills Sheet 3 of the
results tracker.

Strategies (proposal §3.2):
    baseline   — equal weight w=1/N (the Phase 2 method, included as reference)
    maxprob    — w_i = max(p_i)
    entropy    — w_i = exp(-H(p_i))
    variance   — w_i = 1/(var(p_i)+eps)   [see semantics caveat in src/tta.py]
    mc_dropout — T stochastic dropout passes (no augmentation; epistemic proxy)

Usage from the repo root:
    python -m scripts.run_weighted_tta --dataset bloodmnist
    python -m scripts.run_weighted_tta --dataset pathmnist --n-views 10 --mc-T 20

Outputs:
    results/{dataset}_weighted_tta.csv   — one row per strategy (5 rows)
    (merge every dataset's CSV into results/full_matrix.csv with
     `python -m scripts.build_full_matrix` once the team's runs are in.)

Requires the Phase 1 checkpoint at checkpoints/{dataset}_resnet18.pth.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import all_dataset_keys, get_config
from src.evaluate import ALL_STRATEGIES, run_all_strategies
from src.model import build_resnet18
from src.tta import STRATEGY_LABELS
from src.utils import get_device, load_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run all 5 weighted-TTA strategies on one dataset (Phase 3)."
    )
    p.add_argument("--dataset", required=True, choices=all_dataset_keys())
    p.add_argument("--n-views", type=int, default=10,
                   help="Number of TTA views for the 4 fusion strategies (proposal: 10).")
    p.add_argument("--mc-T", type=int, default=20,
                   help="MC Dropout stochastic passes (proposal: 20).")
    p.add_argument("--mc-p", type=float, default=0.2,
                   help="MC Dropout probability on the FC input (proposal: 0.2).")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--img-size", type=int, default=64, choices=[28, 64])
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-root", default="./data")
    p.add_argument("--checkpoints-dir", default="./checkpoints")
    p.add_argument("--results-dir", default="./results")
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

    print(f"\n{'='*72}")
    print(f"  Weighted TTA (Phase 3): {cfg.modality}  ({cfg.key})")
    print(f"  Task          : {cfg.task} ({cfg.n_classes} classes)")
    print(f"  Owner         : {cfg.student}")
    print(f"  Device        : {device}")
    print(f"  N views       : {args.n_views}   |   MC Dropout: T={args.mc_T}, p={args.mc_p}")
    print(f"  Checkpoint    : {ckpt_path}")
    print(f"{'='*72}\n")

    model = build_resnet18(num_classes=cfg.n_classes, pretrained=False)
    load_checkpoint(model, ckpt_path, device=device)
    model.to(device)

    print("Computing per-view probabilities once, then fusing 5 ways "
          "(MC Dropout runs its own passes)...\n")
    results, n_test = run_all_strategies(
        model, args.dataset, device,
        n_views=args.n_views, seed=args.seed, batch_size=args.batch_size,
        img_size=args.img_size, num_workers=args.num_workers,
        data_root=args.data_root, mc_T=args.mc_T, mc_p=args.mc_p,
    )

    base_acc = results["baseline"]["accuracy"]
    base_ece = results["baseline"]["ece"]

    def fmt(v, pct=False):
        if v is None:
            return "  N/A "
        return f"{v*100:6.2f}" if pct else f"{v:6.4f}"

    print(f"{'strategy':<26} {'acc%':>7} {'ECE':>8} {'NLL':>8} {'AUC':>8} "
          f"{'Δacc':>8} {'ΔECE':>8}")
    print("-" * 80)
    rows = []
    for strat in ALL_STRATEGIES:
        m = results[strat]
        d_acc = (m["accuracy"] - base_acc) * 100
        d_ece = m["ece"] - base_ece
        tag = "" if strat == "baseline" else f"{d_acc:+7.2f} {d_ece:+8.4f}"
        print(f"{STRATEGY_LABELS[strat]:<26} {fmt(m['accuracy'], pct=True)} "
              f"{fmt(m['ece'])} {fmt(m['nll'])} {fmt(m['auc_roc'])} {tag}")
        rows.append({
            "dataset": cfg.key,
            "student": cfg.student,
            "strategy": strat,
            "n_views": args.n_views if strat != "mc_dropout" else f"T={args.mc_T}",
            "accuracy": m["accuracy"],
            "auc_roc": m["auc_roc"],
            "ece": m["ece"],
            "nll": m["nll"],
        })

    out = Path(args.results_dir) / f"{cfg.key}_weighted_tta.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nResults saved to: {out}   ({n_test} test images)")

    # Quick verdict: which weighted strategy most improves accuracy vs baseline?
    weighted = {s: results[s]["accuracy"] for s in ALL_STRATEGIES if s != "baseline"}
    best = max(weighted, key=weighted.get)
    delta = (weighted[best] - base_acc) * 100
    print(f"\n{'='*72}")
    print(f"  Best weighted strategy on {cfg.key}: {STRATEGY_LABELS[best]}")
    print(f"  acc {weighted[best]*100:.2f}% vs baseline {base_acc*100:.2f}%  "
          f"(Δ {delta:+.2f}pp)")
    print(f"{'='*72}")
    print(f"\nPaste these 5 rows into Sheet 3️⃣ Weighted TTA.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
