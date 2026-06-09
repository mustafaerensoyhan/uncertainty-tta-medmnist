"""
verify_strategies.py — sanity check behind the McNemar p-values.

A McNemar p≈1 means the strategy and baseline agree on (almost) every image.
That is the correct result for a tiny/balanced effect, but it *would* also be
what a fusion/save bug looks like (strategy silently == baseline). This script
distinguishes the two: it loads the saved per-image arrays and reports, per
dataset's best strategy, how many predictions differ from baseline AND the mean
absolute difference in the probability vectors.

  preds_diff > 0                  -> strategy flips some predictions (clearly active)
  preds_diff == 0, |Δprob| > 1e-6 -> strategy re-weights but never flips a class
                                     (p≈1 is REAL: no accuracy effect, not a bug)
  preds_diff == 0, |Δprob| ~ 0    -> strategy == baseline at the prob level (BUG)

Usage:  python -m scripts.verify_strategies
"""
import argparse
from pathlib import Path
import numpy as np

from src.config import all_dataset_keys

# Phase-3 winning strategy per dataset (mirror of significance.py).
BEST_STRATEGY = {
    "pathmnist": "entropy", "dermamnist": "entropy", "bloodmnist": "entropy",
    "pneumoniamnist": "maxprob", "breastmnist": "maxprob", "organamnist": "variance",
}
EPS = 1e-6


def main() -> int:
    ap = argparse.ArgumentParser(description="Confirm each strategy differs from baseline.")
    ap.add_argument("--predictions-dir", default="./predictions")
    ap.add_argument("--baseline", default="baseline")
    args = ap.parse_args()
    pdir = Path(args.predictions_dir)

    print(f"{'dataset':<16}{'strategy':<10}{'preds_diff':>11}{'mean|Δprob|':>13}  verdict")
    print("-" * 78)
    any_bug = False
    for ds in all_dataset_keys():
        strat = BEST_STRATEGY.get(ds, "entropy")
        try:
            bp = np.load(pdir / f"{ds}_{args.baseline}_preds.npy").ravel()
            sp = np.load(pdir / f"{ds}_{strat}_preds.npy").ravel()
            bpr = np.load(pdir / f"{ds}_{args.baseline}_probs.npy")
            spr = np.load(pdir / f"{ds}_{strat}_probs.npy")
        except FileNotFoundError:
            print(f"{ds:<16}{strat:<10}{'(arrays missing)':>24}")
            continue
        n_diff = int((bp != sp).sum())
        prob_diff = float(np.abs(bpr - spr).mean())
        if prob_diff < EPS:
            verdict = "IDENTICAL to baseline -> BUG"
            any_bug = True
        elif n_diff == 0:
            verdict = "active; reweights, no flips (p~1 is real)"
        else:
            verdict = "active; flips predictions"
        print(f"{ds:<16}{strat:<10}{n_diff:>11}{prob_diff:>13.6f}  {verdict}")

    print("\n" + ("All strategies are active (no identical-to-baseline arrays)."
                  if not any_bug else
                  "WARNING: a strategy is identical to baseline at the probability "
                  "level — investigate the fusion/save path for that dataset."))
    return 1 if any_bug else 0


if __name__ == "__main__":
    raise SystemExit(main())
