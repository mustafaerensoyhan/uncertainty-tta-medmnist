"""
Bootstrap ECE 95% confidence intervals (VMV Implementer-2 deliverable 1).

For every dataset x strategy with saved per-image probability arrays, this
computes a bootstrap CI for the Expected Calibration Error (2000 resamples,
10 equal-width bins) and flags whether each strategy's CI is disjoint from the
baseline CI on the same dataset (a conservative, non-parametric "significant
vs baseline" test).

Inputs (per dataset, looked up in this order so both the flat and the
multi-seed naming conventions work):
    predictions/{dataset}_labels.npy            +  predictions/{dataset}_{strategy}_probs.npy
    predictions/{dataset}_seed{SEED}_labels.npy +  predictions/{dataset}_seed{SEED}_{strategy}_probs.npy

These arrays are written by scripts.run_weighted_tta. If a dataset has no
arrays on disk, it is skipped with a clear message naming the exact file looked
for and the command that generates it.

Usage from the repo root:
    python -m scripts.bootstrap_ece_ci
    python -m scripts.bootstrap_ece_ci --datasets dermamnist --seed 42
    python -m scripts.bootstrap_ece_ci --n-boot 2000 --n-bins 10

Output:
    results/bootstrap_ece_ci.csv
    columns: dataset, strategy, ece, ci_low, ci_high,
             baseline_ci_low, baseline_ci_high, significant_vs_baseline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import all_dataset_keys
from src.metrics import bootstrap_ece_ci

try:
    # Canonical source of truth (pulls in torch via src.evaluate).
    from src.evaluate import ALL_STRATEGIES
except ImportError:
    # torch-free fallback (mirrors src.evaluate.ALL_STRATEGIES) so the CSV can
    # be regenerated, and the script imported, without the full ML stack.
    ALL_STRATEGIES = ["baseline", "maxprob", "entropy", "variance",
                      "variance_inv", "mc_dropout", "ts_only", "ts_entropy"]


def resolve_stem(pdir: Path, ds: str, seed: int) -> str | None:
    """
    Return the on-disk stem for a dataset's prediction arrays, or None if no
    labels file is found. Tries the flat name first, then the seeded name.
    """
    if (pdir / f"{ds}_labels.npy").exists():
        return ds
    if (pdir / f"{ds}_seed{seed}_labels.npy").exists():
        return f"{ds}_seed{seed}"
    return None


def ci_disjoint(a_low: float, a_high: float, b_low: float, b_high: float) -> bool:
    """True iff intervals [a_low, a_high] and [b_low, b_high] do not overlap."""
    return a_high < b_low or b_high < a_low


def main() -> int:
    ap = argparse.ArgumentParser(description="Bootstrap ECE 95% CIs (VMV deliverable 1).")
    ap.add_argument("--datasets", nargs="+", default=all_dataset_keys(),
                    choices=all_dataset_keys())
    ap.add_argument("--strategies", nargs="+", default=ALL_STRATEGIES,
                    help="Strategies to evaluate (default: all 8 core strategies).")
    ap.add_argument("--baseline", default="baseline",
                    help="Strategy used as the calibration reference (default: baseline).")
    ap.add_argument("--seed", type=int, default=42,
                    help="Seed tag to use when only multi-seed arrays exist "
                         "(predictions/{ds}_seed{SEED}_*.npy). Default: 42.")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--ci", type=float, default=0.95)
    ap.add_argument("--boot-seed", type=int, default=0,
                    help="RNG seed for the bootstrap resampling (reproducibility).")
    ap.add_argument("--predictions-dir", default="./predictions")
    ap.add_argument("--results-dir", default="./results")
    args = ap.parse_args()

    pdir = Path(args.predictions_dir)
    rdir = Path(args.results_dir)
    rdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for ds in args.datasets:
        stem = resolve_stem(pdir, ds, args.seed)
        if stem is None:
            print(f"[skip] {ds}: no prediction arrays found "
                  f"(looked for {pdir}/{ds}_labels.npy or {pdir}/{ds}_seed{args.seed}_labels.npy).\n"
                  f"        Generate with: python -m scripts.run_weighted_tta "
                  f"--dataset {ds} --ckpt-tag _seed{args.seed} --seed {args.seed}")
            continue

        labels = np.load(pdir / f"{stem}_labels.npy").ravel()

        # Baseline first — needed as the reference for every other strategy.
        base_path = pdir / f"{stem}_{args.baseline}_probs.npy"
        if not base_path.exists():
            print(f"[skip] {ds}: baseline probs missing ({base_path}); "
                  f"cannot compute significant_vs_baseline. "
                  f"Re-run scripts.run_weighted_tta for this dataset.")
            continue
        base_probs = np.load(base_path)
        b_ece, b_low, b_high = bootstrap_ece_ci(
            base_probs, labels, n_bins=args.n_bins, n_boot=args.n_boot,
            ci=args.ci, seed=args.boot_seed)

        made_for_ds = 0
        for strat in args.strategies:
            probs_path = pdir / f"{stem}_{strat}_probs.npy"
            if not probs_path.exists():
                continue
            probs = np.load(probs_path)
            ece, lo, hi = bootstrap_ece_ci(
                probs, labels, n_bins=args.n_bins, n_boot=args.n_boot,
                ci=args.ci, seed=args.boot_seed)
            is_baseline = strat == args.baseline
            sig = False if is_baseline else ci_disjoint(lo, hi, b_low, b_high)
            rows.append({
                "dataset": ds, "strategy": strat,
                "ece": round(ece, 6),
                "ci_low": round(lo, 6), "ci_high": round(hi, 6),
                "baseline_ci_low": round(b_low, 6),
                "baseline_ci_high": round(b_high, 6),
                "significant_vs_baseline": bool(sig),
            })
            made_for_ds += 1
        print(f"[ok]   {ds}: {made_for_ds} strateg(ies) from stem '{stem}'")

    if not rows:
        print("\nNo prediction arrays were found for any requested dataset. "
              "Run scripts.run_weighted_tta first (it writes predictions/).")
        return 1

    df = pd.DataFrame(rows, columns=[
        "dataset", "strategy", "ece", "ci_low", "ci_high",
        "baseline_ci_low", "baseline_ci_high", "significant_vs_baseline"])
    out = rdir / "bootstrap_ece_ci.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved -> {out}  ({len(df)} rows, "
          f"{df['dataset'].nunique()} dataset(s), {args.n_boot} resamples, "
          f"{args.n_bins} bins)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
