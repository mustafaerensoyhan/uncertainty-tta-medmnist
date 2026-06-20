"""
Paired bootstrap for the ECE reduction of one strategy vs baseline, per backbone.

Why paired: the stock bootstrap_ece_ci.py resamples baseline and the compared
strategy with INDEPENDENT draws and checks whether two wide intervals overlap, a
weak test. Here we resample the SAME test indices for both, form
delta = ECE(strategy) - ECE(baseline) per replicate, and take the percentile
interval of delta. If the whole interval is below 0, the reduction is significant.
This is immune to a seed whose baseline LEVEL is off, because only the within-seed
difference matters.

ECE binning is copied verbatim from src/metrics._ece_from_conf so point estimates
match the seed_stability tables / the repo exactly.

--arch makes it work for every backbone by selecting the right prediction stem:
  resnet18  -> predictions/{ds}_seed{S}_{strategy}_probs.npy   (archless)
  effb0     -> predictions/{ds}_effb0_seed{S}_{strategy}_probs.npy
  deit_tiny -> predictions/{ds}_deit_tiny_seed{S}_{strategy}_probs.npy

Run from the repo root (predictions/ must be present):
    python bootstrap_ece_paired.py --arch resnet18  --seeds 0 42 123
    python bootstrap_ece_paired.py --arch effb0     --seeds 0 42 123
    python bootstrap_ece_paired.py --arch deit_tiny --seeds 0 42
Output: results/bootstrap_ece_paired[_{arch}].csv
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

DEFAULT_ARCH = "resnet18"


def ece_from_conf(confidences, correct, n_bins=10):
    """Identical to src/metrics._ece_from_conf (equal-width, exclusive lower)."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(confidences)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        c = int(mask.sum())
        if c == 0:
            continue
        ece += (c / n) * abs(correct[mask].mean() - confidences[mask].mean())
    return float(ece)


def conf_correct(probs, labels):
    return probs.max(axis=1), (probs.argmax(axis=1) == labels).astype(np.float64)


def stem(ds, arch, seed):
    return f"{ds}_seed{seed}" if arch == DEFAULT_ARCH else f"{ds}_{arch}_seed{seed}"


def load(pdir, ds, arch, s, strat):
    st = stem(ds, arch, s)
    labels = np.load(pdir / f"{st}_labels.npy").ravel()
    probs = np.load(pdir / f"{st}_{strat}_probs.npy")
    return probs, labels


def paired_bootstrap(cb, rb, ce, re, n_bins, n_boot, ci, seed):
    point_b = ece_from_conf(cb, rb, n_bins)
    point_e = ece_from_conf(ce, re, n_bins)
    n = len(rb)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)            # SAME indices for both = paired
        deltas[b] = ece_from_conf(ce[idx], re[idx], n_bins) - ece_from_conf(cb[idx], rb[idx], n_bins)
    lo = float(np.percentile(deltas, (1 - ci) / 2 * 100))
    hi = float(np.percentile(deltas, (1 + ci) / 2 * 100))
    return point_b, point_e, float(point_e - point_b), lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default=DEFAULT_ARCH, choices=["resnet18", "effb0", "deit_tiny"])
    ap.add_argument("--datasets", nargs="+", default=["pathmnist", "bloodmnist"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 42, 123])
    ap.add_argument("--baseline", default="baseline")
    ap.add_argument("--compare", default="entropy")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--ci", type=float, default=0.95)
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--predictions-dir", default="./predictions")
    ap.add_argument("--results-dir", default="./results")
    args = ap.parse_args()

    pdir = Path(args.predictions_dir)
    rows = []
    print(f"Paired bootstrap [{args.arch}]: {args.compare} - {args.baseline}  "
          f"({args.n_boot} resamples, {args.n_bins} bins, {int(args.ci*100)}% CI)\n")
    for ds in args.datasets:
        pool = {"cb": [], "rb": [], "ce": [], "re": []}
        print(f"== {ds} ==")
        for s in args.seeds:
            try:
                pb, lb = load(pdir, ds, args.arch, s, args.baseline)
                pe, le = load(pdir, ds, args.arch, s, args.compare)
            except FileNotFoundError as e:
                print(f"  seed {s}: MISSING ({e.filename}); skipped")
                continue
            cb, rb = conf_correct(pb, lb)
            ce, re = conf_correct(pe, le)
            pool["cb"].append(cb); pool["rb"].append(rb); pool["ce"].append(ce); pool["re"].append(re)
            b, e, d, lo, hi = paired_bootstrap(cb, rb, ce, re, args.n_bins, args.n_boot, args.ci, args.boot_seed)
            print(f"  seed {s:3d}: baseline {b:.4f}  {args.compare} {e:.4f}  "
                  f"reduction {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  {'SIGNIFICANT' if hi < 0 else 'n.s.'}")
            rows.append([args.arch, ds, s, b, e, d, lo, hi, hi < 0])
        if pool["cb"]:
            cb = np.concatenate(pool["cb"]); rb = np.concatenate(pool["rb"])
            ce = np.concatenate(pool["ce"]); re = np.concatenate(pool["re"])
            b, e, d, lo, hi = paired_bootstrap(cb, rb, ce, re, args.n_bins, args.n_boot, args.ci, args.boot_seed)
            print(f"  POOLED : baseline {b:.4f}  {args.compare} {e:.4f}  "
                  f"reduction {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  {'SIGNIFICANT' if hi < 0 else 'n.s.'}\n")
            rows.append([args.arch, ds, "pooled", b, e, d, lo, hi, hi < 0])

    suffix = "" if args.arch == DEFAULT_ARCH else f"_{args.arch}"
    out = Path(args.results_dir) / f"bootstrap_ece_paired{suffix}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("arch,dataset,seed,baseline_ece,compare_ece,reduction,ci_low,ci_high,significant\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
