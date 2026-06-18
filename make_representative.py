#!/usr/bin/env python3
"""
Build the VMV representative image (>=1500x1200 JPEG) from existing figure PDFs.

Layout:  ResNet-18 ECE (top-left)  |  mechanism scatter (right, full height)
         EfficientNet-B0 ECE (bottom-left)

Run from the repo root (so the figures/ paths resolve), or edit the paths below.

    python make_representative.py
    python make_representative.py --figures-dir figures --out representative_image.jpg

Requires: pdf2image OR pdftoppm on PATH (poppler), Pillow.
"""
import argparse, subprocess, tempfile, os, sys
from pathlib import Path
from PIL import Image


def render_pdf(pdf: Path, dpi: int = 300) -> Image.Image:
    """Render page 1 of a PDF to a PIL image via poppler's pdftoppm."""
    if not pdf.exists():
        return None
    with tempfile.TemporaryDirectory() as td:
        stem = os.path.join(td, "p")
        subprocess.run(["pdftoppm", "-png", "-r", str(dpi), "-f", "1", "-l", "1",
                        str(pdf), stem], check=True)
        pngs = sorted(Path(td).glob("p*.png"))
        if not pngs:
            return None
        return Image.open(pngs[0]).convert("RGB")


def scale_to_w(im: Image.Image, w: int) -> Image.Image:
    h = round(im.height * w / im.width)
    return im.resize((w, h), Image.LANCZOS)


def scale_to_h(im: Image.Image, h: int) -> Image.Image:
    w = round(im.width * h / im.height)
    return im.resize((w, h), Image.LANCZOS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures-dir", default="figures")
    ap.add_argument("--resnet-ece", default=None, help="default: <figures>/fig2_ece.pdf")
    ap.add_argument("--effb0-ece",  default=None, help="default: <figures>/fig2_ece_effb0.pdf")
    ap.add_argument("--mechanism",  default=None, help="default: <figures>/fig5_mechanism.pdf")
    ap.add_argument("--out", default="representative_image.jpg")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--pad", type=int, default=50)
    args = ap.parse_args()

    fdir = Path(args.figures_dir)
    p_res = Path(args.resnet_ece) if args.resnet_ece else fdir / "fig2_ece.pdf"
    p_eff = Path(args.effb0_ece)  if args.effb0_ece  else fdir / "fig2_ece_effb0.pdf"
    p_mec = Path(args.mechanism)  if args.mechanism  else fdir / "fig5_mechanism.pdf"

    res = render_pdf(p_res, args.dpi)
    eff = render_pdf(p_eff, args.dpi)
    mec = render_pdf(p_mec, args.dpi)

    if res is None:
        sys.exit(f"ERROR: ResNet ECE figure not found at {p_res}")
    if mec is None:
        sys.exit(f"ERROR: mechanism figure not found at {p_mec}")

    pad = args.pad
    # Left column = the two ECE charts stacked (top resnet, bottom effb0).
    LEFT_W = 2400
    res_s = scale_to_w(res, LEFT_W)
    if eff is not None:
        eff_s = scale_to_w(eff, LEFT_W)
        left_h = res_s.height + pad + eff_s.height
    else:
        print("WARNING: EfficientNet ECE figure not found "
              f"({p_eff}); building with ResNet panel only on the left.")
        eff_s = None
        left_h = res_s.height

    # Right column = mechanism scatter, scaled to the full left-column height.
    mec_s = scale_to_h(mec, left_h)

    total_w = pad + LEFT_W + pad + mec_s.width + pad
    total_h = pad + left_h + pad
    canvas = Image.new("RGB", (total_w, total_h), "white")

    # paste left column
    y = pad
    canvas.paste(res_s, (pad, y))
    if eff_s is not None:
        y += res_s.height + pad
        canvas.paste(eff_s, (pad, y))
    # paste right column, vertically centered
    rx = pad + LEFT_W + pad
    ry = pad + (left_h - mec_s.height) // 2
    canvas.paste(mec_s, (rx, ry))

    canvas.save(args.out, "JPEG", quality=92, dpi=(args.dpi, args.dpi))
    ok = canvas.size[0] >= 1500 and canvas.size[1] >= 1200
    print(f"wrote {args.out}  {canvas.size}  >=1500x1200: {ok}")
    if not ok:
        sys.exit("ERROR: image below 1500x1200 — increase LEFT_W or DPI.")


if __name__ == "__main__":
    main()
