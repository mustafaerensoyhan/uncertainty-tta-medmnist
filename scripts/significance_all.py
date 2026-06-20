"""
Arch-aware McNemar + Wilcoxon significance tests (all backbones).

The stock scripts/significance.py reads flat ResNet-18 prediction names
({ds}_{strategy}_preds.npy) and flat {ds}_weighted_tta.csv, so it only ever
covers ResNet-18. This version adds --arch and --seed and reads the
per-backbone, per-seed files that run_weighted_tta / aggregate_seeds actually
write:

  predictions/{stem}_{strategy}_preds.npy   (McNemar, per-image hard preds)
  results/{global}seed_stability.csv         (Wilcoxon, 3-seed means)

where  stem      = {ds}_seed{S}            (resnet18, archless)
                 = {ds}_{arch}_seed{S}      (effb0, deit_tiny)
       global    = ""        -> seed_stability.csv          (resnet18)
                 = "{arch}_" -> {arch}_seed_stability.csv    (others)

McNemar is run on ONE canonical seed per backbone (no pooling across seeds,
which would be pseudo-replication). Wilcoxon uses the 3-seed (or 2-seed) means
from seed_stability, which is the right level for "consistent across datasets".

Tests mirror scripts/significance.py exactly: exact binomial on the discordant
pairs when their count < 25, else chi-square with continuity correction;
Wilcoxon signed-rank across datasets on paired accuracy and ECE.

Run from the repo root:
    python significance_all.py --arch resnet18  --seed 42 --best-per-dataset
    python significance_all.py --arch effb0     --seed 42 --best-per-dataset
    python significance_all.py --arch deit_tiny --seed 0  --best-per-dataset
Outputs: results/significance[_{arch}].csv , results/significance_wilcoxon[_{arch}].csv
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2, wilcoxon

DEFAULT_ARCH = "resnet18"
DATASETS = ["pathmnist", "dermamnist", "pneumoniamnist",
            "breastmnist", "bloodmnist", "organamnist"]
BEST_STRATEGY = {
    "pathmnist": "entropy", "dermamnist": "entropy", "bloodmnist": "entropy",
    "pneumoniamnist": "maxprob", "breastmnist": "maxprob", "organamnist": "variance",
}


def mcnemar(b01: int, b10: int) -> float:
    n = b01 + b10
    if n == 0:
        return 1.0
    if n < 25:
        return float(binomtest(min(b01, b10), n, 0.5).pvalue)
    stat = (abs(b01 - b10) - 1) ** 2 / n
    return float(chi2.sf(stat, df=1))


def stem(ds, arch, seed):
    return f"{ds}_seed{seed}" if arch == DEFAULT_ARCH else f"{ds}_{arch}_seed{seed}"


def seed_stability_path(rdir, arch):
    return rdir / ("seed_stability.csv" if arch == DEFAULT_ARCH else f"{arch}_seed_stability.csv")


def _safe_wilcoxon(x, y):
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    if x.shape != y.shape or x.size == 0:
        return float("nan")
    if np.allclose(x, y):
        return 1.0
    try:
        return float(np.asarray(wilcoxon(x, y).pvalue).ravel()[0])
    except (ValueError, TypeError):
        return float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default=DEFAULT_ARCH, choices=["resnet18", "effb0", "deit_tiny"])
    ap.add_argument("--seed", type=int, default=42, help="Canonical seed for McNemar per-image arrays.")
    ap.add_argument("--strategy", default="entropy")
    ap.add_argument("--best-per-dataset", action="store_true")
    ap.add_argument("--baseline", default="baseline")
    ap.add_argument("--predictions-dir", default="./predictions")
    ap.add_argument("--results-dir", default="./results")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()
    pdir, rdir = Path(args.predictions_dir), Path(args.results_dir)
    suffix = "" if args.arch == DEFAULT_ARCH else f"_{args.arch}"

    def pick(ds):
        return BEST_STRATEGY.get(ds, args.strategy) if args.best_per_dataset else args.strategy

    # -- McNemar per dataset (single canonical seed) --
    rows = []
    for ds in DATASETS:
        st = stem(ds, args.arch, args.seed)
        strat = pick(ds)
        need = [pdir / f"{st}_labels.npy", pdir / f"{st}_{args.baseline}_preds.npy",
                pdir / f"{st}_{strat}_preds.npy"]
        if not all(p.exists() for p in need):
            print(f"[skip McNemar] {ds}: missing {[str(p) for p in need if not p.exists()]}")
            continue
        y = np.load(pdir / f"{st}_labels.npy").ravel()
        br = np.load(pdir / f"{st}_{args.baseline}_preds.npy").ravel() == y
        sr = np.load(pdir / f"{st}_{strat}_preds.npy").ravel() == y
        b01 = int((br & ~sr).sum()); b10 = int((~br & sr).sum())
        p = mcnemar(b01, b10)
        rows.append({"arch": args.arch, "dataset": ds, "seed": args.seed, "strategy": strat,
                     "baseline": args.baseline, "base_only_right": b01, "strat_only_right": b10,
                     "net_gain_images": b10 - b01, "p_value": round(p, 4),
                     "significant": bool(p < args.alpha)})

    if rows:
        out = rdir / f"significance{suffix}.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        lbl = "best-per-dataset" if args.best_per_dataset else args.strategy
        print(f"\nMcNemar [{args.arch}, seed {args.seed}] - {lbl} vs {args.baseline}")
        print("-" * 70)
        print(f"{'dataset':<16}{'strat':<9}{'base-only':>10}{'strat-only':>11}{'net':>6}{'p':>9}  sig")
        for r in rows:
            print(f"{r['dataset']:<16}{r['strategy']:<9}{r['base_only_right']:>10}{r['strat_only_right']:>11}"
                  f"{r['net_gain_images']:>6}{r['p_value']:>9.4f}  {'YES' if r['significant'] else 'no'}")
        print(f"Saved -> {out}")
    else:
        print(f"\nNo per-image arrays found for arch={args.arch}, seed={args.seed}. "
              f"Re-run scripts.run_weighted_tta for this backbone first (it writes predictions/).")

    # -- Wilcoxon across datasets from seed_stability means --
    ssp = seed_stability_path(rdir, args.arch)
    if not ssp.exists():
        print(f"\n[skip Wilcoxon] {ssp} not found. Run: python -m scripts.aggregate_seeds --arch {args.arch}")
        return 0
    ss = pd.read_csv(ssp)
    acc_s, acc_b, ece_s, ece_b, used = [], [], [], [], []
    for ds in DATASETS:
        sub = ss[ss["dataset"] == ds].set_index("strategy")
        strat = pick(ds)
        if strat in sub.index and args.baseline in sub.index:
            acc_s.append(sub.loc[strat, "accuracy_mean"]); acc_b.append(sub.loc[args.baseline, "accuracy_mean"])
            ece_s.append(sub.loc[strat, "ece_mean"]);       ece_b.append(sub.loc[args.baseline, "ece_mean"])
            used.append(ds)
    n = len(used)
    print(f"\nWilcoxon signed-rank [{args.arch}] across {n} dataset(s): {', '.join(used) or '(none)'}")
    if n >= 2:
        p_acc = _safe_wilcoxon(acc_s, acc_b)
        p_ece = _safe_wilcoxon(ece_b, ece_s)  # lower ECE better -> baseline minus strat
        wrows = [
            {"arch": args.arch, "metric": "accuracy", "n_datasets": n,
             "mean_delta": round(float(np.mean(acc_s) - np.mean(acc_b)), 4),
             "p_value": round(p_acc, 4), "significant": bool(p_acc < args.alpha)},
            {"arch": args.arch, "metric": "ece", "n_datasets": n,
             "mean_delta": round(float(np.mean(ece_s) - np.mean(ece_b)), 4),
             "p_value": round(p_ece, 4), "significant": bool(p_ece < args.alpha)},
        ]
        pd.DataFrame(wrows).to_csv(rdir / f"significance_wilcoxon{suffix}.csv", index=False)
        print(f"  accuracy: mean Δ {wrows[0]['mean_delta']:+.4f}, p={p_acc:.4f} "
              f"({'sig' if wrows[0]['significant'] else 'n.s.'})")
        print(f"  ECE     : mean Δ {wrows[1]['mean_delta']:+.4f}, p={p_ece:.4f} "
              f"({'sig' if wrows[1]['significant'] else 'n.s.'})")
        if n < 6:
            print(f"  NOTE: with {n} datasets min achievable p ≈ {0.5**n:.3f}; low power.")
        print(f"  Saved -> {rdir / f'significance_wilcoxon{suffix}.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
