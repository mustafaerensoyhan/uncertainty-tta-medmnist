"""
Inference-time benchmark (VMV plan — "add inference time wherever it makes
sense so we can compare GPU runtime with future additions").

The Phase-3 weighted-TTA run reports one latency per strategy at the working N.
This script complements it with a clean, fully-comparable latency surface across
the dimensions that actually drive cost — backbone × N (views) × method — on the
SAME machine, so any future addition (a new backbone, a new fusion, a different
N) drops straight into the same table and is directly comparable.

Key facts it makes visible:
  - All augmentation-based strategies at a given N (baseline, maxprob, entropy,
    variance, ts_entropy, and every Top-K) share ONE set of N forward passes —
    fusion is microseconds — so they cost the same. Top-K's headline claim
    ("same cost as full TTA") is shown here, not asserted.
  - ts_only / single-pass is the N=1 cost; MC Dropout costs ~T single passes.

Measurement follows perf.measure_ms_per_image (warmup + torch.cuda.synchronize,
time.perf_counter), exactly as Sheet 6 requires.

Usage from the repo root:
    python -m scripts.benchmark_inference --datasets pathmnist --arch resnet18
    python -m scripts.benchmark_inference --datasets pathmnist bloodmnist \
        --arch effb0 --n-views 1 5 10 20 50 --mc-T 20

Outputs:
    results/<stem>_inference_benchmark.csv     per (arch, dataset)
    results/inference_benchmark.csv            combined across everything present
        columns: arch, dataset, method, n_views, ms_per_image
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.augmentations import get_augmentation_pipeline
from src.config import all_dataset_keys, get_config
from src.data import get_dataloaders, get_tta_test_loader
from src.mc_dropout import mc_dropout_per_pass_probs
from src.model import build_model, ARCH_LABELS, ARCHITECTURES
from src.perf import measure_ms_per_image
from src.train import predict_probs
from src.tta import tta_per_view_logits
from src.utils import (get_device, load_checkpoint, set_seed,
                       checkpoint_filename, result_stem)


def bench_dataset(ds: str, args, device) -> list[dict]:
    cfg = get_config(ds)
    ckpt = Path(args.checkpoints_dir) / checkpoint_filename(cfg.key, args.arch)
    if not ckpt.exists():
        print(f"  [skip] {cfg.key} ({args.arch}): checkpoint not found at {ckpt}")
        return []

    model = build_model(args.arch, num_classes=cfg.n_classes, pretrained=False)
    load_checkpoint(model, ckpt, device=device)
    model.to(device)

    # Normalized eval loader (single-pass + MC Dropout) and un-normalized TTA loader.
    _, _, test_loader, _ = get_dataloaders(cfg.key, batch_size=args.batch_size,
                                           img_size=args.img_size,
                                           num_workers=args.num_workers, root=args.data_root)
    tta_loader, _ = get_tta_test_loader(cfg.key, batch_size=args.batch_size,
                                        img_size=args.img_size,
                                        num_workers=args.num_workers, root=args.data_root)
    # n_test from one quick label pass.
    import numpy as np
    n_test = sum(len(np.asarray(lbl).reshape(-1)) for _, lbl in test_loader)

    rows = []
    print(f"\n  {ARCH_LABELS[args.arch]} / {cfg.key}  ({n_test} test images)")
    print(f"  {'method':<22}{'N':>4}{'ms/image':>12}")
    print("  " + "-" * 38)

    def record(method, n, ms):
        rows.append({"arch": args.arch, "dataset": cfg.key, "method": method,
                     "n_views": n, "ms_per_image": round(ms, 4)})
        print(f"  {method:<22}{n:>4}{ms:>12.4f}")

    for n in args.n_views:
        if n == 1:
            # Single un-augmented forward (== ts_only / baseline N=1 cost).
            ms = measure_ms_per_image(lambda: predict_probs(model, test_loader, device),
                                      n_test, device, warmup=args.warmup)
            record("single_pass", 1, ms)
        else:
            set_seed(args.seed)
            augs = get_augmentation_pipeline(n_views=n, seed=args.seed, include_original=True)
            ms = measure_ms_per_image(
                lambda: tta_per_view_logits(model, tta_loader, device, augs),
                n_test, device, warmup=args.warmup)
            # Shared by every aug-based strategy + Top-K at this N.
            record("tta_all_strategies", n, ms)

    if args.mc_T > 0:
        ms = measure_ms_per_image(
            lambda: mc_dropout_per_pass_probs(model, test_loader, device, T=args.mc_T, p=args.mc_p),
            n_test, device, warmup=args.warmup)
        record(f"mc_dropout_T{args.mc_T}", args.mc_T, ms)

    return rows


def main() -> int:
    p = argparse.ArgumentParser(description="Same-machine inference-time benchmark (arch x N x method).")
    p.add_argument("--datasets", nargs="+", default=all_dataset_keys(), choices=all_dataset_keys())
    p.add_argument("--arch", default="resnet18", choices=list(ARCHITECTURES))
    p.add_argument("--n-views", type=int, nargs="+", default=[1, 5, 10, 20, 50])
    p.add_argument("--mc-T", type=int, default=20, help="MC Dropout passes to time (0 = skip).")
    p.add_argument("--mc-p", type=float, default=0.2)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--img-size", type=int, default=64, choices=[28, 64])
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-root", default="./data")
    p.add_argument("--checkpoints-dir", default="./checkpoints")
    p.add_argument("--results-dir", default="./results")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device = get_device(prefer_cuda=not args.cpu)
    if device.type != "cuda":
        print("WARNING: timing on CPU — numbers are NOT comparable to the GPU figures "
              "in Sheet 6. Use a GPU machine for the reported latencies.")

    rdir = Path(args.results_dir)
    rdir.mkdir(parents=True, exist_ok=True)
    print(f"Inference benchmark on {device} | {ARCH_LABELS[args.arch]} | "
          f"N={args.n_views} | datasets: {', '.join(args.datasets)}")

    all_rows = []
    for ds in args.datasets:
        rows = bench_dataset(ds, args, device)
        if rows:
            stem = result_stem(ds, args.arch)
            pd.DataFrame(rows).to_csv(rdir / f"{stem}_inference_benchmark.csv", index=False)
            all_rows.extend(rows)

    if not all_rows:
        print("\nNo checkpoints found — nothing benchmarked.")
        return 1

    # Rebuild the combined table from every per-(arch,dataset) benchmark present.
    parts = [pd.read_csv(f) for f in sorted(rdir.glob("*_inference_benchmark.csv"))
             if f.name != "inference_benchmark.csv"]
    combined = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(all_rows)
    combined.to_csv(rdir / "inference_benchmark.csv", index=False)
    print(f"\nWrote per-dataset files + combined results/inference_benchmark.csv "
          f"({combined['arch'].nunique()} backbone(s), {combined['dataset'].nunique()} dataset(s)).")
    print("Drop these into Sheet 6 (Inference Time) — Top-K and every soft strategy "
          "share the tta_all_strategies row at each N.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
