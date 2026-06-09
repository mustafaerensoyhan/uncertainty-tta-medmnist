"""
Statistical significance testing (addendum Addition 4) — Phase 4.

Two complementary tests, both consuming the per-image prediction arrays saved by
run_weighted_tta (predictions/{ds}_{strategy}_preds.npy + {ds}_labels.npy):

  McNemar (per dataset): did the weighted strategy beat baseline TTA on THIS
      test set, accounting for the paired image-by-image agreement? Uses the
      exact binomial test when the discordant count is small, else the
      chi-square approximation with continuity correction. No statsmodels
      dependency — implemented directly.

  Wilcoxon signed-rank (across datasets): does the weighted strategy CONSISTENTLY
      beat baseline across the datasets, or is the win driven by one or two? Run
      on the paired per-dataset accuracies (and, separately, ECE) from the
      weighted-TTA CSVs.

Usage from the repo root:
    python -m scripts.significance                          # entropy vs baseline
    python -m scripts.significance --strategy variance      # any strategy vs baseline

Outputs:
    results/significance.csv            — per-dataset McNemar rows
    results/significance_wilcoxon.csv   — across-dataset Wilcoxon (acc + ECE)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scipy.stats import binomtest, chi2, wilcoxon

from src.config import all_dataset_keys, get_config


# Phase-3 winning strategy per dataset (matches the tracker's "Best Strategy"
# column). Used by --best-per-dataset so each dataset is tested with the strategy
# it actually won with, instead of one strategy applied uniformly.
BEST_STRATEGY = {
    "pathmnist": "entropy", "dermamnist": "entropy", "bloodmnist": "entropy",
    "pneumoniamnist": "maxprob", "breastmnist": "maxprob", "organamnist": "variance",
}


def mcnemar(b01: int, b10: int) -> float:
    """
    McNemar p-value from the two discordant counts.
    b01 = baseline right & strategy wrong; b10 = baseline wrong & strategy right.
    """
    n = b01 + b10
    if n == 0:
        return 1.0
    if n < 25:  # small-sample: exact binomial on the discordant pairs
        return float(binomtest(min(b01, b10), n, 0.5).pvalue)
    stat = (abs(b01 - b10) - 1) ** 2 / n  # chi-square w/ continuity correction
    return float(chi2.sf(stat, df=1))


def _load(pdir: Path, ds: str, name: str):
    return np.load(pdir / f"{ds}_{name}.npy")


def main() -> int:
    ap = argparse.ArgumentParser(description="McNemar + Wilcoxon significance tests (Phase 4).")
    ap.add_argument("--strategy", default="entropy",
                    help="Weighted strategy to test against baseline (default: entropy).")
    ap.add_argument("--best-per-dataset", action="store_true",
                    help="Test each dataset with its Phase-3 winning strategy "
                         "(entropy/maxprob/variance) instead of one strategy for all.")
    ap.add_argument("--baseline", default="baseline")
    ap.add_argument("--predictions-dir", default="./predictions")
    ap.add_argument("--results-dir", default="./results")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    pdir = Path(args.predictions_dir)
    rdir = Path(args.results_dir)

    # ── McNemar per dataset (needs the per-image arrays) ──
    rows = []
    for ds in all_dataset_keys():
        strat = BEST_STRATEGY.get(ds, args.strategy) if args.best_per_dataset else args.strategy
        need = [pdir / f"{ds}_labels.npy",
                pdir / f"{ds}_{args.baseline}_preds.npy",
                pdir / f"{ds}_{strat}_preds.npy"]
        if not all(p.exists() for p in need):
            continue
        y = _load(pdir, ds, "labels").ravel()
        base_right = _load(pdir, ds, f"{args.baseline}_preds").ravel() == y
        strat_right = _load(pdir, ds, f"{strat}_preds").ravel() == y
        b01 = int((base_right & ~strat_right).sum())   # baseline right, strat wrong
        b10 = int((~base_right & strat_right).sum())   # baseline wrong, strat right
        p = mcnemar(b01, b10)
        rows.append({
            "dataset": ds, "student": get_config(ds).student,
            "strategy": strat, "baseline": args.baseline,
            "base_only_right": b01, "strat_only_right": b10,
            "net_gain_images": b10 - b01, "p_value": round(p, 4),
            "significant": bool(p < args.alpha),
        })

    if not rows:
        print(f"No per-image arrays found for '{args.strategy}' vs '{args.baseline}'. "
              f"Run scripts.run_weighted_tta first (it saves predictions/).")
        return 1

    mc = pd.DataFrame(rows)
    out = rdir / "significance.csv"
    mc.to_csv(out, index=False)

    _label = "best-per-dataset" if args.best_per_dataset else args.strategy
    print(f"\nMcNemar — {_label} vs {args.baseline} (per dataset)")
    print("-" * 78)
    print(f"{'dataset':<16}{'strategy':<10}{'base✓/strat✗':>13}{'base✗/strat✓':>13}"
          f"{'net':>6}{'p':>9}  sig")
    for r in rows:
        print(f"{r['dataset']:<16}{r['strategy']:<10}{r['base_only_right']:>13}{r['strat_only_right']:>13}"
              f"{r['net_gain_images']:>6}{r['p_value']:>9.4f}  {'YES' if r['significant'] else 'no'}")
    print(f"\nSaved -> {out}")

    # ── Wilcoxon across datasets (paired acc + ECE from the weighted CSVs) ──
    acc_s, acc_b, ece_s, ece_b, used = [], [], [], [], []
    for ds in all_dataset_keys():
        f = rdir / f"{ds}_weighted_tta.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f).set_index("strategy")
        strat = BEST_STRATEGY.get(ds, args.strategy) if args.best_per_dataset else args.strategy
        if strat in df.index and args.baseline in df.index:
            acc_s.append(df.loc[strat, "accuracy"]); acc_b.append(df.loc[args.baseline, "accuracy"])
            ece_s.append(df.loc[strat, "ece"]);       ece_b.append(df.loc[args.baseline, "ece"])
            used.append(ds)

    wrows = []
    n = len(used)
    print(f"\nWilcoxon signed-rank across {n} dataset(s): {', '.join(used) or '(none)'}")
    if n >= 2:
        def _safe_wilcoxon(x, y):
            x, y = np.asarray(x), np.asarray(y)
            if np.allclose(x, y):
                return 1.0  # no differences
            try:
                return float(wilcoxon(x, y).pvalue)
            except ValueError:
                return float("nan")
        p_acc = _safe_wilcoxon(acc_s, acc_b)
        p_ece = _safe_wilcoxon(ece_b, ece_s)  # lower ECE is better -> baseline minus strat
        wrows = [
            {"metric": "accuracy", "strategy": args.strategy, "n_datasets": n,
             "mean_delta": round(float(np.mean(acc_s) - np.mean(acc_b)), 4),
             "p_value": round(p_acc, 4), "significant": bool(p_acc < args.alpha)},
            {"metric": "ece", "strategy": args.strategy, "n_datasets": n,
             "mean_delta": round(float(np.mean(ece_s) - np.mean(ece_b)), 4),
             "p_value": round(p_ece, 4), "significant": bool(p_ece < args.alpha)},
        ]
        pd.DataFrame(wrows).to_csv(rdir / "significance_wilcoxon.csv", index=False)
        print(f"  accuracy: mean Δ {wrows[0]['mean_delta']:+.4f}, p={p_acc:.4f} "
              f"({'sig' if wrows[0]['significant'] else 'n.s.'})")
        print(f"  ECE     : mean Δ {wrows[1]['mean_delta']:+.4f}, p={p_ece:.4f} "
              f"({'sig' if wrows[1]['significant'] else 'n.s.'})")
        if n < 6:
            print(f"  NOTE: with {n} datasets Wilcoxon has low power (min achievable p "
                  f"≈ {0.5**n:.3f}); treat as indicative until all 6 are in.")
        print(f"  Saved -> {rdir / 'significance_wilcoxon.csv'}")
    else:
        print("  need >=2 datasets' weighted CSVs for the across-dataset test — skipped.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
