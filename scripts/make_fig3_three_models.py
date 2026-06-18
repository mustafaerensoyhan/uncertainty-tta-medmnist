"""
Combined reliability figure across the three backbones (3x2 grid).

Rows = models (ResNet-18, EfficientNet-B0, DeiT-Tiny);
columns = datasets (PathMNIST, BloodMNIST);
each panel = the *entropy* (weighted-TTA) reliability curve.

Two outputs are written:
    figures/fig3_reliability_3models.pdf         (labelled: titles, row/col labels, ECE)
    figures/fig3_reliability_3models_nolabel.pdf (clean: bars + diagonal + ticks only)

Inputs (predictions/ is gitignored): for each (model, dataset) it needs the
seed42 entropy probs + labels, resolving any of these stems:
    {ds}{sfx}_seed42_entropy_probs.npy / _labels.npy
    {ds}{sfx}_entropy_probs.npy        / {ds}{sfx}_labels.npy
where sfx = "" (resnet18), "_effb0", "_deit_tiny". Missing model rows are
drawn as a labelled blank row rather than failing.

Usage:
    python -m scripts.make_fig3_three_models
    python -m scripts.make_fig3_three_models --strategy entropy --seed 42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import expected_calibration_error
from scripts.make_vmv_figures import _reliability_panel, _arch_suffix, SHORT_NAME

DPI = 300
MODELS = [("resnet18", "ResNet-18"), ("effb0", "EfficientNet-B0"),
          ("deit_tiny", "DeiT-Tiny")]
DATASETS = ["pathmnist", "bloodmnist"]


def _resolve(pdir: Path, ds: str, arch: str, strat: str, seed: int):
    """Return (labels_path, probs_path) for one (ds, arch, strat) or None."""
    sfx = _arch_suffix(arch)
    for stem in (f"{ds}{sfx}_seed{seed}", f"{ds}{sfx}"):
        probs = pdir / f"{stem}_{strat}_probs.npy"
        labels = pdir / f"{stem}_labels.npy"
        if probs.exists() and labels.exists():
            return labels, probs
    return None


def build(predictions_dir="./predictions", figures_dir="./figures",
          strategy="entropy", seed=42, n_bins=10):
    pdir, fdir = Path(predictions_dir), Path(figures_dir)
    nrow, ncol = len(MODELS), len(DATASETS)

    # Resolve everything up front; report any missing (model, dataset) cells.
    resolved: dict[tuple[str, str], tuple[Path, Path]] = {}
    missing = []
    for arch, _ in MODELS:
        for ds in DATASETS:
            got = _resolve(pdir, ds, arch, strategy, seed)
            if got is None:
                missing.append((arch, ds))
            else:
                resolved[(arch, ds)] = got
    if missing:
        print("[fig3-3m] missing prediction arrays (rows drawn blank):")
        for arch, ds in missing:
            sfx = _arch_suffix(arch)
            print(f"   {pdir}/{ds}{sfx}_seed{seed}_{strategy}_probs.npy "
                  f"(+ {ds}{sfx}_seed{seed}_labels.npy)")

    def render(labelled: bool):
        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3.4, nrow * 3.3),
                                 squeeze=False)
        for r, (arch, model_name) in enumerate(MODELS):
            for c, ds in enumerate(DATASETS):
                ax = axes[r][c]
                got = resolved.get((arch, ds))
                if got is None:
                    ax.axis("off")
                    if labelled:
                        ax.text(0.5, 0.5, f"{model_name}\n{SHORT_NAME.get(ds, ds)}\n"
                                "(no predictions)", ha="center", va="center",
                                fontsize=9, color="grey")
                    continue
                labels = np.load(got[0]).ravel()
                probs = np.load(got[1])
                ece = expected_calibration_error(probs, labels, n_bins=n_bins)
                title = (f"{SHORT_NAME.get(ds, ds)} — {strategy} (ECE={ece:.3f})"
                         if labelled else "")
                _reliability_panel(ax, probs, labels, n_bins, title)
                if not labelled:
                    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_title("")
                else:
                    if c == 0:
                        ax.text(-0.34, 0.5, model_name, transform=ax.transAxes,
                                rotation=90, va="center", ha="center",
                                fontsize=12, fontweight="bold")
        if labelled:
            fig.suptitle(f"Reliability diagrams — {strategy} TTA ({n_bins} bins)",
                         fontsize=13, fontweight="bold")
            fig.tight_layout(rect=(0.03, 0, 1, 0.97))
            out = fdir / "fig3_reliability_3models.pdf"
        else:
            fig.tight_layout()
            out = fdir / "fig3_reliability_3models_nolabel.pdf"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"[fig3-3m] wrote {out}")
        return out

    return [render(True), render(False)]


def main() -> int:
    ap = argparse.ArgumentParser(description="3-model reliability figure (3x2).")
    ap.add_argument("--predictions-dir", default="./predictions")
    ap.add_argument("--figures-dir", default="./figures")
    ap.add_argument("--strategy", default="entropy")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-bins", type=int, default=10)
    args = ap.parse_args()
    build(args.predictions_dir, args.figures_dir, args.strategy, args.seed, args.n_bins)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
