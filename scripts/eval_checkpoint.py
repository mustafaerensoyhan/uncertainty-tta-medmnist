"""
Re-evaluate an existing checkpoint on the test split and regenerate
results/{dataset}_baseline.json — WITHOUT retraining.

Use this to reconcile a baseline JSON that no longer matches the checkpoint on
disk (e.g. PathMNIST: the committed JSON says 91.77% but the current checkpoint
produces ~94%, because the JSON was written by an older checkpoint). This runs
the exact same no-TTA test evaluation Phase 1 used (single forward pass,
normalized test loader), so the regenerated numbers are authoritative for
whatever .pth is currently in checkpoints/.

It preserves the training metadata already in the JSON (epochs, lr, training
time, etc.) and only refreshes `test_metrics`, the checkpoint path, and a
re-eval note. It also regenerates the reliability diagram so the figure matches
the corrected ECE.

Usage from the repo root:
    python -m scripts.eval_checkpoint --dataset pathmnist --compare        # preview only
    python -m scripts.eval_checkpoint --dataset pathmnist                  # write JSON
    python -m scripts.eval_checkpoint --all --compare                      # check all 6

Requires checkpoints/{dataset}_resnet18.pth.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import all_dataset_keys, get_config
from src.data import get_dataloaders
from src.metrics import compute_all_metrics
from src.model import build_resnet18
from src.train import predict_probs
from src.utils import get_device, load_checkpoint, save_json, set_seed
from src.visualize import reliability_diagram


def _fmt(m):
    return (f"acc={m['accuracy']*100:.2f}%  auc={m['auc_roc']}  "
            f"ece={m['ece']:.4f}  nll={m['nll']:.4f}")


def eval_one(dataset_name: str, args, device) -> int:
    cfg = get_config(dataset_name)
    ckpt = Path(args.checkpoints_dir) / f"{cfg.key}_resnet18.pth"
    if not ckpt.exists():
        print(f"[skip] {cfg.key}: checkpoint not found at {ckpt}")
        return 1

    json_path = Path(args.results_dir) / f"{cfg.key}_baseline.json"
    old = json.load(open(json_path)) if json_path.exists() else {}
    old_metrics = old.get("test_metrics")

    model = build_resnet18(num_classes=cfg.n_classes, pretrained=False)
    load_checkpoint(model, ckpt, device=device)
    model.to(device)

    _, _, test_loader, _ = get_dataloaders(
        cfg.key, batch_size=args.batch_size, img_size=args.img_size,
        num_workers=args.num_workers, root=args.data_root)
    probs, labels = predict_probs(model, test_loader, device)
    new_metrics = compute_all_metrics(probs, labels, task=cfg.task)

    print(f"\n{cfg.key} ({cfg.modality}) — checkpoint {ckpt}")
    if old_metrics:
        print(f"  old JSON : {_fmt(old_metrics)}")
    print(f"  re-eval  : {_fmt(new_metrics)}")
    if old_metrics:
        d_acc = (new_metrics["accuracy"] - old_metrics["accuracy"]) * 100
        d_ece = new_metrics["ece"] - old_metrics["ece"]
        print(f"  delta    : acc {d_acc:+.2f}pp, ece {d_ece:+.4f}"
              + ("   <-- JSON was stale" if abs(d_acc) > 0.5 else ""))
    delta_bm = new_metrics["accuracy"] - cfg.benchmark_acc
    flag = "within ±2%" if abs(delta_bm) <= 0.02 else "OUTSIDE ±2% — investigate"
    print(f"  vs published benchmark {cfg.benchmark_acc*100:.1f}%: "
          f"{delta_bm*100:+.2f}pp ({flag})")

    if args.compare or args.dry_run:
        print("  (preview only — not written; drop --compare/--dry-run to save)")
        return 0

    # Regenerate the reliability diagram to match the corrected ECE.
    reliab_path = Path(args.figures_dir) / "reliability" / f"{cfg.key}_baseline.png"
    if not args.no_figure:
        reliability_diagram(probs, labels, n_bins=10, save_path=reliab_path,
                            title=f"{cfg.modality} baseline — ECE={new_metrics['ece']:.3f}")

    # Preserve existing metadata; refresh metrics + provenance.
    payload = dict(old) if old else {
        "dataset": cfg.key, "modality": cfg.modality, "student": cfg.student,
        "task": cfg.task, "img_size": args.img_size,
    }
    payload.update({
        "checkpoint": str(ckpt),
        "benchmark_acc": cfg.benchmark_acc,
        "test_metrics": new_metrics,
        "reliability_figure": str(reliab_path),
        "re_evaluated": True,
        "re_evaluated_on": _dt.date.today().isoformat(),
        "re_eval_note": "test_metrics regenerated from the checkpoint on disk "
                        "(no retraining); training metadata preserved from prior JSON.",
    })
    save_json(payload, json_path)
    print(f"  written  -> {json_path}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Regenerate baseline JSON from a checkpoint (no retrain).")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--dataset", choices=all_dataset_keys())
    g.add_argument("--all", action="store_true", help="Re-evaluate all 6 datasets.")
    p.add_argument("--compare", action="store_true", help="Preview old vs new, don't write.")
    p.add_argument("--dry-run", action="store_true", help="Alias for --compare.")
    p.add_argument("--no-figure", action="store_true", help="Skip reliability diagram.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--img-size", type=int, default=64, choices=[28, 64])
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-root", default="./data")
    p.add_argument("--checkpoints-dir", default="./checkpoints")
    p.add_argument("--results-dir", default="./results")
    p.add_argument("--figures-dir", default="./figures")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device(prefer_cuda=not args.cpu)
    datasets = all_dataset_keys() if args.all else [args.dataset]
    rc = 0
    for ds in datasets:
        rc |= eval_one(ds, args, device)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
