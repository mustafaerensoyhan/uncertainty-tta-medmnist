"""
Sign-flip figure (S2): paired-bootstrap delta ECE (equal-weight TTA - single-pass)
with 95% CIs, per dataset and backbone.

This ONLY draws results/signflip_scatter_18pts.csv (already computed + verified by
the paired bootstrap in scripts.significance_all); it does not recompute anything.

CSV columns: arch, dataset, modality, single_pass_ece, tta_ece, delta_ece,
             ci_lo, ci_hi, significant, direction.
delta_ece = ECE_tta - ECE_single. delta > 0  => TTA HURTS calibration (red zone);
delta < 0  => TTA HELPS (green zone). Filled marker = 95% CI excludes zero.

Outputs (title-free + labelled, to match the rest of the figure set):
    figures/fig_signflip.pdf            (labelled)
    figures/fig_signflip_nolabel.pdf    (no title / no annotations)

Usage:
    python -m scripts.make_fig_signflip
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DPI = 300
SHORT = {"bloodmnist": "Blood", "breastmnist": "Breast", "dermamnist": "Derma",
         "organamnist": "OrganA", "pathmnist": "Path", "pneumoniamnist": "Pneumonia"}
# datasets ordered so the consistent "hurts" case (OrganA) sits at one end.
DS_ORDER = ["dermamnist", "pneumoniamnist", "pathmnist", "bloodmnist",
            "breastmnist", "organamnist"]
ARCH = [("resnet18", "ResNet-18", "o"), ("effb0", "EfficientNet-B0", "s"),
        ("deit_tiny", "DeiT-Tiny", "^")]
DODGE = {"resnet18": -0.24, "effb0": 0.0, "deit_tiny": 0.24}


def build(results_dir="./results", figures_dir="./figures"):
    rdir, fdir = Path(results_dir), Path(figures_dir)
    df = pd.read_csv(rdir / "signflip_scatter_18pts.csv")
    df["significant"] = df["significant"].astype(str).str.lower().eq("true")
    df = df.set_index(["arch", "dataset"])
    datasets = [d for d in DS_ORDER if d in df.index.get_level_values("dataset")]

    def render(labelled: bool):
        fig, ax = plt.subplots(figsize=(8.6, 4.8))
        ax.axhspan(0, 1, color="#d62728", alpha=0.06)
        ax.axhspan(-1, 0, color="#2ca02c", alpha=0.06)
        ax.axhline(0, color="black", lw=1.0)
        ymin, ymax = 0.0, 0.0
        for arch, _alabel, marker in ARCH:
            for xi, ds in enumerate(datasets):
                if (arch, ds) not in df.index:
                    continue
                r = df.loc[(arch, ds)]
                x = xi + DODGE[arch]
                d = float(r["delta_ece"]); lo = float(r["ci_lo"]); hi = float(r["ci_hi"])
                sig = bool(r["significant"])
                color = "#d62728" if d > 0 else "#2ca02c"
                ax.errorbar(x, d, yerr=[[d - lo], [hi - d]], fmt=marker, ms=8,
                            color=color, ecolor=color, elinewidth=1.5, capsize=3,
                            mfc=(color if sig else "white"), mew=1.5, zorder=3)
                ymin, ymax = min(ymin, lo), max(ymax, hi)
        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels([SHORT[d] for d in datasets])
        ax.set_xlim(-0.6, len(datasets) - 0.4)
        pad = 0.1 * (ymax - ymin)
        ax.set_ylim(ymin - pad, ymax + pad)
        if labelled:
            ax.set_ylabel(r"$\Delta$ECE = ECE$_\mathrm{TTA}$ $-$ ECE$_\mathrm{single}$")
            ax.set_title("Sign-flip test: does equal-weight TTA help or hurt calibration?\n"
                         "paired bootstrap, 95% CI (filled marker = CI excludes 0)")
            ax.text(0.012, 0.97, "TTA HURTS", transform=ax.transAxes, va="top",
                    ha="left", color="#d62728", fontweight="bold", fontsize=10)
            ax.text(0.012, 0.03, "TTA HELPS", transform=ax.transAxes, va="bottom",
                    ha="left", color="#2ca02c", fontweight="bold", fontsize=10)
            handles = [Line2D([0], [0], marker=m, color="grey", ls="", ms=8, label=lab)
                       for _, lab, m in ARCH]
            ax.legend(handles=handles, loc="center right", frameon=True,
                      title="Backbone", fontsize=8, title_fontsize=8)
        fig.tight_layout()
        out = fdir / ("fig_signflip.pdf" if labelled else "fig_signflip_nolabel.pdf")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"[signflip] wrote {out}")

    render(True)
    render(False)


if __name__ == "__main__":
    build()
