"""
Aggregate the multi-seed stability study (addendum Addition 4, Tool 3) — Phase 4.

The multi-seed runs are produced by training and evaluating with a checkpoint
tag, e.g. for PathMNIST seeds 0/42/123:

    python -m scripts.train_baseline   --dataset pathmnist --seed 0   --ckpt-tag _seed0
    python -m scripts.run_weighted_tta --dataset pathmnist --ckpt-tag _seed0 --no-time
    # ... repeat for _seed42, _seed123

That leaves results/pathmnist_seed0_weighted_tta.csv, _seed42_, _seed123_. This
script reads all of them per dataset and reports mean ± std per (strategy,
metric), so the paper can show error bars instead of single numbers.

Usage:
    python -m scripts.aggregate_seeds
    python -m scripts.aggregate_seeds --datasets pathmnist pneumoniamnist breastmnist

Output:
    results/seed_stability.csv   (dataset, strategy, <metric>_mean, <metric>_std, n_seeds)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import all_dataset_keys
from src.model import ARCHITECTURES
from src.utils import result_stem

_METRICS = ["accuracy", "auc_roc", "ece", "nll"]
_SEED_RE = re.compile(r"_seed\d+_weighted_tta\.csv$")


def main() -> int:
    ap = argparse.ArgumentParser(description="Aggregate multi-seed runs into mean +/- std.")
    ap.add_argument("--datasets", nargs="+", default=all_dataset_keys(),
                    choices=all_dataset_keys())
    ap.add_argument("--arch", default="resnet18", choices=list(ARCHITECTURES),
                    help="Backbone whose seed runs to aggregate (resnet18 default).")
    ap.add_argument("--results-dir", default="./results")
    args = ap.parse_args()
    rdir = Path(args.results_dir)
    arch = args.arch
    # ResNet-18 keeps archless combined name (seed_stability.csv); effb0 ->
    # effb0_seed_stability.csv. Per-dataset files mirror result_stem.
    combined_name = "seed_stability.csv" if arch == "resnet18" else f"{arch}_seed_stability.csv"
    per_ds_suffix = "_seed_stability.csv" if arch == "resnet18" else f"_{arch}_seed_stability.csv"

    all_rows = []
    for ds in args.datasets:
        # Glob the seed stems for THIS backbone, e.g. pathmnist_seed*  (resnet18)
        # or pathmnist_effb0_seed*  (effb0) — result_stem encodes the namespace.
        glob_stem = result_stem(ds, arch, "_seed*")
        seed_files = sorted(f for f in rdir.glob(f"{glob_stem}_weighted_tta.csv")
                            if _SEED_RE.search(f.name))
        if len(seed_files) < 2:
            if seed_files:
                print(f"[skip] {ds} ({arch}): only {len(seed_files)} seed file(s) — need >=2 for std.")
            continue
        df = pd.concat([pd.read_csv(f) for f in seed_files], ignore_index=True)
        n_seeds = len(seed_files)
        for strat, g in df.groupby("strategy", sort=False):
            row = {"dataset": ds, "arch": arch, "strategy": strat, "n_seeds": n_seeds}
            for m in _METRICS:
                if m in g:
                    row[f"{m}_mean"] = round(float(g[m].mean()), 4)
                    row[f"{m}_std"] = round(float(g[m].std(ddof=1)), 4)
            all_rows.append(row)
        print(f"[ok]   {ds} ({arch}): aggregated {n_seeds} seeds.")

    if not all_rows:
        print(f"\nNo multi-seed result files found for arch={arch}. Run train_baseline + "
              f"run_weighted_tta with --arch {arch} --ckpt-tag _seed<N> for >=2 seeds first.")
        return 1

    # Write ONE file per dataset so three students running this never touch the
    # same file — the shared combined file was merge-hostile and silently dropped
    # rows. Each owner commits only their own {ds}{suffix} via PR.
    out = pd.DataFrame(all_rows)
    for ds, g in out.groupby("dataset", sort=False):
        g.to_csv(rdir / f"{ds}{per_ds_suffix}", index=False)
        print(f"       wrote results/{ds}{per_ds_suffix}")

    # Rebuild the combined view from EVERY per-dataset file present for THIS arch.
    parts = [pd.read_csv(f) for f in sorted(rdir.glob(f"*{per_ds_suffix}"))
             if f.name != combined_name]
    combined = pd.concat(parts, ignore_index=True) if parts else out
    combined.to_csv(rdir / combined_name, index=False)

    print(f"\nPer-dataset files + combined results/{combined_name} "
          f"({combined['dataset'].nunique()} dataset(s)) written.")
    # Pretty-print accuracy ± std for a quick read.
    print(f"\n{'dataset':<16}{'strategy':<14}{'acc mean±std':>18}{'ece mean±std':>18}")
    print("-" * 66)
    for r in all_rows:
        a = f"{r.get('accuracy_mean','?')*100:.2f} ± {r.get('accuracy_std',0)*100:.2f}"
        e = f"{r.get('ece_mean','?'):.4f} ± {r.get('ece_std',0):.4f}"
        print(f"{r['dataset']:<16}{r['strategy']:<14}{a:>18}{e:>18}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
