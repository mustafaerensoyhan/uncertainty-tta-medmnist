"""One-off: export all paper figures as high-quality, title-free JPEGs.

- fig1/2/4/5 are regenerated from source data with titles suppressed
  (matplotlib set_title/suptitle monkeypatched to no-ops) and savefig
  redirected to JPEG at 300 DPI.
- fig3 (reliability) has no prediction arrays to regenerate from, so its
  existing PDF is rasterised at 300 DPI and the title band cropped off.

Output: figures/jpeg_no_title/*.jpg   (tracked PDFs are left untouched)
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import fitz
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "figures" / "jpeg_no_title"
OUT.mkdir(parents=True, exist_ok=True)
DPI = 300

# 1) strip every title.
Axes.set_title = lambda self, *a, **k: self
Figure.suptitle = lambda self, *a, **k: None

# 2) redirect every savefig to a JPEG in OUT/, forcing high DPI + white bg.
_orig_savefig = Figure.savefig


def _savefig(self, fname, *a, **k):
    stem = Path(str(fname)).stem
    k.pop("format", None)
    k.setdefault("bbox_inches", "tight")
    k["dpi"] = DPI
    k["facecolor"] = "white"
    out = OUT / f"{stem}.jpg"
    _orig_savefig(self, str(out), format="jpeg", **k)
    w, h = Image.open(out).size
    print(f"  wrote {out.relative_to(ROOT)}  ({w}x{h})")
    return None


Figure.savefig = _savefig

from scripts.make_vmv_figures import fig2_ece, fig4_heatmap, fig5_mechanism
from scripts.make_fig1_composite import assemble_from_strips

DATASETS = ["pathmnist", "dermamnist", "pneumoniamnist", "bloodmnist"]


def regen():
    print("fig1 (from strips):")
    for arch in ("resnet18", "effb0"):
        assemble_from_strips("figures/strip", "figures", DATASETS, arch, 3,
                             mode="all", dpi=DPI)
    print("fig2 ECE:")
    for arch in ("resnet18", "effb0"):
        fig2_ece("./results", "./figures", arch=arch)
    print("fig4 heatmap:")
    for arch in ("resnet18", "effb0"):
        fig4_heatmap("./results", "./figures", arch=arch)
    print("fig5 mechanism:")
    fig5_mechanism("./results", "./figures")


def crop_fig3():
    """Rasterise fig3_reliability.pdf and crop the top title band."""
    src = ROOT / "figures" / "fig3_reliability.pdf"
    if not src.exists():
        print("[fig3] PDF missing — skipped")
        return
    pix = fitz.open(src)[0].get_pixmap(dpi=DPI, alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    arr = np.asarray(img)
    # row has "ink" if any pixel is clearly non-white.
    ink = (arr.min(axis=2) < 230).any(axis=1)
    rows = np.where(ink)[0]
    if rows.size:
        top = rows[0]
        # walk through the title's ink rows, then the whitespace gap below it,
        # and crop where panel content resumes.
        i = top
        while i < len(ink) and ink[i]:
            i += 1               # end of title ink
        while i < len(ink) and not ink[i]:
            i += 1               # end of whitespace gap -> panels start
        img = img.crop((0, i, img.width, img.height))
    out = OUT / "fig3_reliability.jpg"
    img.save(out, "JPEG", quality=95, dpi=(DPI, DPI))
    print(f"  wrote {out.relative_to(ROOT)}  ({img.size[0]}x{img.size[1]})")


def combine_ece():
    """Stack the title-free ResNet/EffB0 ECE JPEGs into one combined JPEG."""
    parts = [OUT / "fig2_ece.jpg", OUT / "fig2_ece_effb0.jpg"]
    if not all(p.exists() for p in parts):
        print("[combined] missing an ECE panel — skipped")
        return
    imgs = [Image.open(p).convert("RGB") for p in parts]
    w = max(i.width for i in imgs)
    h = sum(i.height for i in imgs)
    canvas = Image.new("RGB", (w, h), "white")
    y = 0
    for im in imgs:
        canvas.paste(im, ((w - im.width) // 2, y))
        y += im.height
    out = OUT / "fig2_ece_combined.jpg"
    canvas.save(out, "JPEG", quality=95, dpi=(DPI, DPI))
    print(f"  wrote {out.relative_to(ROOT)}  ({canvas.size[0]}x{canvas.size[1]})")


if __name__ == "__main__":
    regen()
    print("fig3 reliability (crop title from PDF):")
    crop_fig3()
    print("combined ECE:")
    combine_ece()
    print(f"\nAll JPEGs in {OUT.relative_to(ROOT)}/")
