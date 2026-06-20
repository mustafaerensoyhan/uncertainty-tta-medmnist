"""
#9 step 2: paired bootstrap of the SIGN FLIP, single-pass (no TTA) vs equal-weight TTA.

For each dataset x backbone, pools the per-image predictions across seeds and bootstraps
the paired ECE difference  diff = ECE(equal-weight TTA) - ECE(single pass).
  diff > 0 and CI excludes 0  ->  TTA HURTS calibration (well-calibrated case)
  diff < 0 and CI excludes 0  ->  TTA HELPS calibration (overconfident case)
Uses the paper's own ECE (src.metrics.expected_calibration_error) for exact parity.

Run AFTER make_singlepass_probs.py:
    python -m scripts.bootstrap_signflip --arch resnet18  --seeds 0 42 123
    python -m scripts.bootstrap_signflip --arch effb0     --seeds 0 42 123
    python -m scripts.bootstrap_signflip --arch deit_tiny --seeds 0 42 123
Writes results/signflip_bootstrap_<arch>.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from src.config import all_dataset_keys, get_config
from src.model import ARCHITECTURES
from src.utils import result_stem
from src.metrics import expected_calibration_error as ece  # ece(probs, labels, n_bins=10)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--arch", required=True, choices=list(ARCHITECTURES))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 42, 123])
    p.add_argument("--tag-template", default="_seed{s}")
    p.add_argument("--datasets", nargs="+", default=all_dataset_keys())
    p.add_argument("--predictions-dir", default="./predictions")
    p.add_argument("--results-dir", default="./results")
    p.add_argument("--n-bins", type=int, default=10)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)
    pdir = Path(a.predictions_dir)
    rows = []
    for ds in a.datasets:
        cfg = get_config(ds)
        sp, tta, lab = [], [], []
        for s in a.seeds:
            tag = a.tag_template.format(s=s) if a.tag_template else ""
            stem = result_stem(cfg.key, a.arch, tag)
            f_sp = pdir / f"{stem}_singlepass_probs.npy"
            f_tta = pdir / f"{stem}_baseline_probs.npy"
            f_lab = pdir / f"{stem}_labels.npy"
            if not (f_sp.exists() and f_tta.exists() and f_lab.exists()):
                print(f"  [skip] {stem}: missing singlepass/baseline/labels")
                continue
            sp.append(np.load(f_sp)); tta.append(np.load(f_tta)); lab.append(np.load(f_lab))
        if not sp:
            print(f"  [skip] {cfg.key}: no usable seeds"); continue
        sp = np.concatenate(sp); tta = np.concatenate(tta); lab = np.concatenate(lab)
        n = len(lab)
        e_sp = ece(sp, lab, a.n_bins); e_tta = ece(tta, lab, a.n_bins)
        obs = e_tta - e_sp
        boots = np.empty(a.n_boot)
        for b in range(a.n_boot):
            idx = rng.integers(0, n, n)
            boots[b] = ece(tta[idx], lab[idx], a.n_bins) - ece(sp[idx], lab[idx], a.n_bins)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        sig = (lo > 0) or (hi < 0)
        direction = "TTA hurts" if obs > 0 else "TTA helps"
        rows.append(dict(arch=a.arch, dataset=cfg.key, n=n,
                         ece_single=round(e_sp, 4), ece_tta=round(e_tta, 4),
                         diff=round(obs, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                         significant=bool(sig), direction=direction))
        print(f"  {cfg.key:14s} single={e_sp:.4f} tta={e_tta:.4f} "
              f"diff={obs:+.4f} CI[{lo:+.4f},{hi:+.4f}] "
              f"{'SIG' if sig else 'n.s.'} ({direction})")
    out = Path(a.results_dir) / f"signflip_bootstrap_{a.arch}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nsaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
