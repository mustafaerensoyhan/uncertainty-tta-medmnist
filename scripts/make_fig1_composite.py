"""
Fig 1 composite — Augmentation Confidence Strips, 4 modalities x 3 samples.

The VMV anchor figure (proposal §3.5 / Addition 3) is owned by Implementer 1;
`scripts.make_confidence_strips` already produces the individual per-strip PDFs.
This assembler tiles them into the single paper figure the plan specifies
("Final layout = 4 rows x 3 columns") so there is one ready-to-drop-in file:

    figures/fig1_confidence_strips.pdf   (300 DPI)

It reuses Implementer 1's view-weight / image-selection helpers and the shared
`confidence_strip_panel` renderer, so the numbers are identical to the per-strip
script — this only changes the layout, not the content.

Like the other VMV scripts, it fails clearly: a modality whose checkpoint or
data is missing is left as a blank row with a printed message naming the exact
file and the command that produces it. It exits non-zero only if NOTHING could
be rendered.

Usage from the repo root:
    python -m scripts.make_fig1_composite
    python -m scripts.make_fig1_composite --arch effb0
    python -m scripts.make_fig1_composite --datasets pathmnist bloodmnist --n-images 3

Note: rendering requires each modality's checkpoint
(checkpoints/{ds}_{arch}.pth) and the MedMNIST data, exactly like
make_confidence_strips. With only some checkpoints present, only those rows fill.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import all_dataset_keys, get_config
from src.model import ARCHITECTURES
from src.utils import get_device, load_checkpoint, checkpoint_filename
from src.visualize import confidence_strip_panel

# Reuse Implementer 1's helpers verbatim (no duplicated model logic).
from scripts.make_confidence_strips import (
    _view_weights_for_image, _pick_random_correct, _pick_spread)

DPI = 300
DEFAULT_DATASETS = ["pathmnist", "dermamnist", "pneumoniamnist", "bloodmnist"]
SHORT_NAME = {
    "pathmnist": "Path", "dermamnist": "Derma", "pneumoniamnist": "Pneumonia",
    "bloodmnist": "Blood", "breastmnist": "Breast", "organamnist": "Organ",
}


def _load_model(ds: str, arch: str, ckpt_dir: Path, device):
    """Return a ready model for `ds`, or None if its checkpoint is absent."""
    from src.model import build_model
    cfg = get_config(ds)
    ckpt = ckpt_dir / checkpoint_filename(cfg.key, arch)
    if not ckpt.exists():
        return None
    model = build_model(arch, num_classes=cfg.n_classes, pretrained=False)
    load_checkpoint(model, ckpt, device=device)
    model.to(device)
    return model


def make_composite(args, device) -> Path | None:
    from src.augmentations import get_augmentation_pipeline
    from src.data import get_dataset

    datasets = args.datasets
    n_cols = args.n_images
    augs = get_augmentation_pipeline(n_views=args.n_views, seed=args.seed,
                                     include_original=not args.include_all_augs)
    arch_flag = "" if args.arch == "resnet18" else f" --arch {args.arch}"

    fig = plt.figure(figsize=(n_cols * 3.4, len(datasets) * 2.6))
    outer = fig.add_gridspec(len(datasets), n_cols, hspace=0.45, wspace=0.12)

    rendered_rows = 0
    for r, ds in enumerate(datasets):
        cfg = get_config(ds)
        model = _load_model(ds, args.arch, Path(args.checkpoints_dir), device)
        if model is None:
            ckpt = Path(args.checkpoints_dir) / checkpoint_filename(cfg.key, args.arch)
            print(f"[fig1] [skip row] {ds}: checkpoint not found at {ckpt}\n"
                  f"        generate strips with: python -m scripts.make_confidence_strips "
                  f"--datasets {ds}{arch_flag}")
            ax = fig.add_subplot(outer[r, :]); ax.axis("off")
            ax.text(0.5, 0.5, f"{SHORT_NAME.get(ds, ds)}: checkpoint missing",
                    ha="center", va="center", fontsize=9, color="grey")
            continue

        test_ds = get_dataset(cfg, "test", img_size=args.img_size,
                              root=args.data_root, normalize=False)
        for c in range(n_cols):
            if args.select == "spread":
                idx, img = _pick_spread(model, test_ds, device, augs,
                                        args.max_scan, args.n_candidates, args.min_conf)
            else:
                seed = args.seeds[c] if c < len(args.seeds) else c
                idx, img = _pick_random_correct(model, test_ds, device, seed)
            if img is None:
                ax = fig.add_subplot(outer[r, c]); ax.axis("off")
                ax.text(0.5, 0.5, "no correct\nimage", ha="center", va="center",
                        fontsize=8, color="grey")
                continue
            views, names, weights, per_view = _view_weights_for_image(
                model, img, augs, device)
            gold = None
            if args.gold_k and args.gold_k > 0:
                from src.tta import top_k_keep_indices
                gold = top_k_keep_indices(per_view, args.gold_k)[:, 0].tolist()
            confidence_strip_panel(
                fig, outer[r, c], views, weights, names,
                highlight_idx=gold, show_names=(r == 0),
                ylabel=(SHORT_NAME.get(ds, ds) if c == 0 else None))
            rendered_rows += 1
            if args.select == "spread":
                break

    if rendered_rows == 0:
        plt.close(fig)
        print("\n[fig1] no rows could be rendered — no checkpoints/data found. "
              "Generate the strips first with scripts.make_confidence_strips.")
        return None

    arch_label = {"resnet18": "ResNet-18", "effb0": "EfficientNet-B0"}.get(args.arch, args.arch)
    fig.suptitle(f"Figure 1 — Augmentation Confidence Strips ({arch_label})",
                 fontsize=13, fontweight="bold")
    if args.gold_k:
        fig.text(0.5, 0.005,
                 f"gold outline = Top-{args.gold_k} kept (lowest-entropy) views",
                 ha="center", fontsize=8, color="darkgoldenrod")

    sfx = "" if args.arch == "resnet18" else f"_{args.arch}"
    out = Path(args.figures_dir) / f"fig1_confidence_strips{sfx}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig1] wrote {out}  ({len(datasets)} modalities x {n_cols} samples)")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Fig 1 composite confidence-strip assembler (VMV).")
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                   choices=all_dataset_keys())
    p.add_argument("--arch", default="resnet18", choices=list(ARCHITECTURES))
    p.add_argument("--n-images", type=int, default=3, help="Samples per modality (columns).")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--select", choices=["random", "spread"], default="random")
    p.add_argument("--gold-k", type=int, default=5,
                   help="Gold-outline the Top-K lowest-entropy views. 0 disables.")
    p.add_argument("--n-views", type=int, default=10)
    p.add_argument("--include-all-augs", action="store_true")
    p.add_argument("--max-scan", type=int, default=400)
    p.add_argument("--n-candidates", type=int, default=40)
    p.add_argument("--min-conf", type=float, default=0.60)
    p.add_argument("--img-size", type=int, default=64, choices=[28, 64])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-root", default="./data")
    p.add_argument("--checkpoints-dir", default="./checkpoints")
    p.add_argument("--figures-dir", default="./figures")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device = get_device(prefer_cuda=not args.cpu)
    print(f"Fig 1 composite on {device} | modalities: {', '.join(args.datasets)}\n")
    out = make_composite(args, device)
    return 0 if out is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
