"""
Generate the Augmentation Confidence Strip figures — the VMV visual
contribution (proposal §3.5, Figure 1 of the paper).

For each requested dataset, this:
  1. picks one correctly-classified, confident test image,
  2. generates N augmented views of it (named),
  3. computes the entropy weight w_i = exp(-H(p_i)) per view (normalized),
  4. draws the two-row strip (thumbnails over coloured weight bars).

Default datasets are the four the proposal calls for (one per modality):
PathMNIST, DermaMNIST, PneumoniaMNIST, BloodMNIST.

Usage from the repo root:
    python -m scripts.make_confidence_strips
    python -m scripts.make_confidence_strips --datasets pathmnist bloodmnist
    python -m scripts.make_confidence_strips --n-views 10 --include-all-augs

Outputs:
    figures/strip/{dataset}_strip.pdf   (one per dataset)

Requires each dataset's Phase 1 checkpoint at checkpoints/{dataset}_resnet18.pth.

Note on N and which augmentations appear: with --include-original (default),
N=10 yields original + the first 9 of the 10 base augmentations, so
`elastic` is not shown. Pass --include-all-augs to drop the original and show
all 10 base augmentation types instead (useful if you want every type in the
strip, matching the §3.3 policy table one-for-one).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.augmentations import get_augmentation_pipeline
from src.config import all_dataset_keys, get_config
from src.data import get_dataset, normalize_imagenet
from src.model import build_resnet18
from src.tta import entropy_weights
from src.utils import get_device, load_checkpoint, set_seed
from src.visualize import confidence_strip

DEFAULT_DATASETS = ["pathmnist", "dermamnist", "pneumoniamnist", "bloodmnist"]


def _view_weights_for_image(model, img, augs, device):
    """
    Run the N augmented views of one [0,1] image through the model and return
    (views, names, weights) where weights are the normalized entropy weights.
    """
    views, names, probs = [], [], []
    model.eval()
    with torch.no_grad():
        for aug_fn, aug_name in augs:
            aug_img = aug_fn(img).clamp(0, 1)
            logits = model(normalize_imagenet(aug_img).unsqueeze(0).to(device))
            probs.append(F.softmax(logits, dim=1).squeeze(0).cpu().numpy())
            views.append(aug_img)
            names.append(aug_name)
    per_view = np.stack(probs, axis=0)[:, None, :]          # (N, 1, C)
    weights = entropy_weights(per_view)[:, 0]               # (N,)
    return views, names, weights


def _pick_illustrative_image(model, dataset, device, augs, *, max_scan: int = 400,
                             n_candidates: int = 40, min_conf: float = 0.60):
    """
    Two-pass selection of the most *illustrative* test image for the strip.

    The point of Figure 1 is to SHOW contrast — some augmented views trusted,
    some rejected. Picking the single most-confident image (the old behaviour)
    tends to produce a uniformly-confident image whose entropy weights are all
    near-equal, so the bars look flat and prove nothing. Instead:

      Pass 1 — among the first `max_scan` test images, keep those that are
               correctly classified with baseline (original-view) confidence
               >= min_conf. This honours §3.5's "confident, correctly classified"
               requirement.
      Pass 2 — for the top `n_candidates` of those (by confidence), compute the
               entropy-weight spread (max - min) across the N augmented views and
               return the image whose weights vary MOST.

    This is a deliberate choice of a clean, illustrative example (which §3.5 asks
    for) — NOT a quantitative claim that every image behaves this way; the strip
    is a qualitative figure and its caption should say so.

    Returns (img, base_conf, views, names, weights) or (None, 0.0, ...) if no
    confident correct image is found.
    """
    model.eval()
    candidates = []  # (conf, img)
    with torch.no_grad():
        for idx in range(min(len(dataset), max_scan)):
            img, label = dataset[idx]
            label = int(np.asarray(label).ravel()[0])
            logits = model(normalize_imagenet(img).unsqueeze(0).to(device))
            prob = F.softmax(logits, dim=1).squeeze(0)
            conf, pred = prob.max(0)
            if int(pred) == label and float(conf) >= min_conf:
                candidates.append((float(conf), img))

    if not candidates:
        return None, 0.0, None, None, None

    candidates.sort(key=lambda c: -c[0])  # most confident first
    best = None  # (spread, conf, img, views, names, weights)
    for conf, img in candidates[:n_candidates]:
        views, names, weights = _view_weights_for_image(model, img, augs, device)
        spread = float(weights.max() - weights.min())
        if best is None or spread > best[0]:
            best = (spread, conf, img, views, names, weights)

    spread, conf, img, views, names, weights = best
    return img, conf, views, names, weights


def make_strip(dataset_name: str, args, device) -> bool:
    cfg = get_config(dataset_name)
    ckpt = Path(args.checkpoints_dir) / f"{cfg.key}_resnet18.pth"
    if not ckpt.exists():
        print(f"  [skip] {cfg.key}: checkpoint not found at {ckpt}")
        return False

    model = build_resnet18(num_classes=cfg.n_classes, pretrained=False)
    load_checkpoint(model, ckpt, device=device)
    model.to(device)

    # Un-normalized [0,1] test set (augmentations act on [0,1], then we normalize).
    test_ds = get_dataset(cfg, "test", img_size=args.img_size, root=args.data_root,
                          normalize=False)

    augs = get_augmentation_pipeline(n_views=args.n_views, seed=args.seed,
                                     include_original=not args.include_all_augs)

    set_seed(args.seed)
    img, conf, views, names, weights = _pick_illustrative_image(
        model, test_ds, device, augs,
        max_scan=args.max_scan, n_candidates=args.n_candidates, min_conf=args.min_conf)
    if img is None:
        print(f"  [skip] {cfg.key}: no confident correct image (conf>={args.min_conf}) "
              f"in first {args.max_scan} test images")
        return False

    save_path = Path(args.figures_dir) / "strip" / f"{cfg.key}_strip.pdf"
    confidence_strip(views, weights, names,
                     dataset_name=f"{cfg.medmnist_class} ({cfg.modality})",
                     save_path=save_path)
    spread = float(weights.max() - weights.min())
    print(f"  [ok]   {cfg.key}: base conf={conf:.3f}, weight spread={spread:.3f} "
          f"(w {weights.min():.3f}..{weights.max():.3f}) -> {save_path}")
    return True


def main() -> int:
    p = argparse.ArgumentParser(description="Generate Augmentation Confidence Strips (§3.5).")
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                   choices=all_dataset_keys())
    p.add_argument("--n-views", type=int, default=10)
    p.add_argument("--include-all-augs", action="store_true",
                   help="Drop the original view and show all 10 base augmentation types.")
    p.add_argument("--max-scan", type=int, default=400,
                   help="How many test images to scan when picking the example.")
    p.add_argument("--n-candidates", type=int, default=40,
                   help="Top-confidence images to rank by weight spread (the most "
                        "illustrative one wins).")
    p.add_argument("--min-conf", type=float, default=0.60,
                   help="Minimum baseline confidence for a candidate image.")
    p.add_argument("--img-size", type=int, default=64, choices=[28, 64])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-root", default="./data")
    p.add_argument("--checkpoints-dir", default="./checkpoints")
    p.add_argument("--figures-dir", default="./figures")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device = get_device(prefer_cuda=not args.cpu)
    print(f"Generating confidence strips on {device} for: {', '.join(args.datasets)}\n")

    made = 0
    for ds in args.datasets:
        if make_strip(ds, args, device):
            made += 1

    print(f"\nDone. {made}/{len(args.datasets)} strips written to "
          f"{Path(args.figures_dir) / 'strip'}.")
    print("These four strips stack into Figure 1 of the paper.")
    return 0 if made > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
