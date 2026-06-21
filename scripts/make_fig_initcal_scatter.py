"""
Sign-flip vs initial calibration scatter (the comment-#10 figure).

x = single-pass (no-TTA) ECE   -> how (mis)calibrated the model is BEFORE TTA
y = delta_ece = ECE_tta - ECE_single   -> what equal-weight TTA does to it

Points below y=0 (green): TTA HELPS. Above (red): TTA HURTS. The story: TTA helps
exactly where the single-pass model is poorly calibrated (high x), and is neutral/
slightly harmful where it is already well calibrated (low x). Marker filled = the
paired-bootstrap 95% CI excludes zero (significant); hollow = not significant.
Shape encodes backbone. Optional vertical CI whiskers from ci_lo/ci_hi.

Pure plot of results/signflip_scatter_18pts.csv — nothing recomputed.

Outputs:
    figures/fig_signflip_scatter.pdf           (labelled)
    figures/fig_signflip_scatter_nolabel.pdf   (no title/annotations)

Usage:
    python -m scripts.make_fig_initcal_scatter
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
DS_ORDER = ["organamnist", "bloodmnist", "breastmnist", "pathmnist",
            "pneumoniamnist", "dermamnist"]
ARCH = [("resnet18", "ResNet-18", "o"), ("effb0", "EfficientNet-B0", "s"),
        ("deit_tiny", "DeiT-Tiny", "^")]


def build(results_dir="./results", figures_dir="./figures", whiskers=True):
    rdir, fdir = Path(results_dir), Path(figures_dir)
    df = pd.read_csv(rdir / "signflip_scatter_18pts.csv")
    df["significant"] = df["significant"].astype(str).str.lower().eq("true")
    markers = dict((a, m) for a, _, m in ARCH)
    # dataset palette deliberately avoids red/green so marker colour never clashes
    # with the red "hurts" / green "helps" zone shading.
    dcolor = {"organamnist": "#1f77b4", "bloodmnist": "#ff7f0e",
              "breastmnist": "#9467bd", "pathmnist": "#8c564b",
              "pneumoniamnist": "#e377c2", "dermamnist": "#17becf"}

    # y-limits from the actual data (incl CI whiskers) so the spread is visible.
    lo = float(min(df["delta_ece"].min(), df["ci_lo"].min()))
    hi = float(max(df["delta_ece"].max(), df["ci_hi"].max()))
    pad = 0.12 * (hi - lo)
    ylo, yhi = lo - pad, hi + pad
    xmax = float(df["single_pass_ece"].max()) * 1.12

    def render(labelled: bool):
        fig, ax = plt.subplots(figsize=(7.8, 5.2))
        ax.set_xlim(0, xmax)
        ax.set_ylim(ylo, yhi)
        # help/hurt shading spanning the visible region (not data 0..1)
        ax.axhspan(0, yhi, color="#d62728", alpha=0.05)
        ax.axhspan(ylo, 0, color="#2ca02c", alpha=0.05)
        ax.axhline(0, color="black", lw=1.0)
        for _, r in df.iterrows():
            x, y = float(r["single_pass_ece"]), float(r["delta_ece"])
            sig = bool(r["significant"])
            c = dcolor.get(r["dataset"], "grey")
            m = markers.get(r["arch"], "o")
            if whiskers:
                ax.plot([x, x], [r["ci_lo"], r["ci_hi"]], color=c, lw=1.0,
                        alpha=0.45, zorder=1)
            ax.scatter([x], [y], marker=m, s=80, zorder=3,
                       facecolor=(c if sig else "white"),
                       edgecolor=c, linewidths=1.8)
        if labelled:
            ax.set_xlabel("Single-pass ECE (initial calibration)")
            ax.set_ylabel(r"$\Delta$ECE = ECE$_\mathrm{TTA}$ $-$ ECE$_\mathrm{single}$")
            ax.set_title("Sign-flip vs initial calibration: TTA helps where the model "
                         "starts poorly calibrated\n(fill = 95% CI excludes 0)")
            ax.text(0.985, 0.97, "TTA HURTS", transform=ax.transAxes, va="top",
                    ha="right", color="#d62728", fontweight="bold", fontsize=11)
            ax.text(0.985, 0.03, "TTA HELPS", transform=ax.transAxes, va="bottom",
                    ha="right", color="#2ca02c", fontweight="bold", fontsize=11)
            ds_handles = [Line2D([0], [0], marker="o", color=dcolor[d], ls="", ms=9,
                                 label=SHORT[d]) for d in DS_ORDER]
            leg1 = ax.legend(handles=ds_handles, loc="lower left", frameon=True,
                             fontsize=8, title="Dataset", title_fontsize=8, ncol=2)
            ax.add_artist(leg1)
            style_handles = [Line2D([0], [0], marker=m, color="grey", ls="", ms=9, label=lab)
                             for _, lab, m in ARCH]
            style_handles += [
                Line2D([0], [0], marker="o", color="grey", mfc="grey", ls="", ms=9,
                       label="significant"),
                Line2D([0], [0], marker="o", color="grey", mfc="white", ls="", ms=9,
                       label="n.s.")]
            ax.legend(handles=style_handles, loc="upper left", frameon=True, fontsize=8,
                      title="Backbone / fill", title_fontsize=8)
        fig.tight_layout()
        out = fdir / ("fig_signflip_scatter.pdf" if labelled
                      else "fig_signflip_scatter_nolabel.pdf")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"[signflip-scatter] wrote {out}")

    render(True)
    render(False)


if __name__ == "__main__":
    build()
