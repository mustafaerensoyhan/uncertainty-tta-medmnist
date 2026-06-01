"""
Generate reliability (calibration) diagrams for every dataset x strategy
(proposal Phase 4, Metric 6) — Phase 4.

Reads the per-image probability arrays saved by run_weighted_tta
(predictions/{ds}_{strategy}_probs.npy + {ds}_labels.npy), so it needs no model
forward passes and runs in seconds. Each diagram plots predicted confidence
(x) vs actual accuracy (y); the diagonal is perfect calibration.

Usage:
    python -m scripts.make_reliability_diagrams
    python -m scripts.make_reliability_diagrams --datasets pathmnist --strategies baseline entropy

Output: figures/reliability/{dataset}_{strategy}.pdf  (consistent 0..1 axes)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import all_dataset_keys
from src.evaluate import ALL_STRATEGIES
from src.metrics import expected_calibration_error
from src.visualize import reliability_diagram


def main() -> int:
    ap = argparse.ArgumentParser(description="Reliability diagrams from saved probs (Phase 4).")
    ap.add_argument("--datasets", nargs="+", default=all_dataset_keys(), choices=all_dataset_keys())
    ap.add_argument("--strategies", nargs="+", default=ALL_STRATEGIES, choices=ALL_STRATEGIES)
    ap.add_argument("--predictions-dir", default="./predictions")
    ap.add_argument("--figures-dir", default="./figures")
    ap.add_argument("--n-bins", type=int, default=10)
    args = ap.parse_args()

    pdir = Path(args.predictions_dir)
    outdir = Path(args.figures_dir) / "reliability"

    made = 0
    for ds in args.datasets:
        lab_path = pdir / f"{ds}_labels.npy"
        if not lab_path.exists():
            print(f"[skip] {ds}: no predictions/{ds}_labels.npy (run run_weighted_tta first)")
            continue
        labels = np.load(lab_path).ravel()
        for strat in args.strategies:
            probs_path = pdir / f"{ds}_{strat}_probs.npy"
            if not probs_path.exists():
                continue
            probs = np.load(probs_path)
            ece = expected_calibration_error(probs, labels, n_bins=args.n_bins)
            save_path = outdir / f"{ds}_{strat}.pdf"
            reliability_diagram(probs, labels, n_bins=args.n_bins, save_path=save_path,
                                title=f"{ds} — {strat} (ECE={ece:.3f})")
            made += 1
        print(f"[ok]   {ds}: reliability diagrams written")

    print(f"\nDone. {made} reliability diagram(s) in {outdir}. "
          f"(supplementary material; the calibration story for the Results section.)")
    return 0 if made else 1


if __name__ == "__main__":
    raise SystemExit(main())
