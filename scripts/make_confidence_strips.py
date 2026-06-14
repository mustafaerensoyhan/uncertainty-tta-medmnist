"""
Generate Augmentation Confidence Strips — the VMV visual contribution
(proposal §3.5, expanded by addendum Addition 3 to 3 images per modality).

For each dataset and each of n_images seeds, this:
  1. picks a correctly-classified test image (see --select below),
  2. generates N augmented views of it (named),
  3. computes the entropy weight w_i = exp(-H(p_i)) per view (normalized),
  4. draws the two-row strip (thumbnails over coloured weight bars).

Default datasets are the four the proposal/addendum call for (one per modality):
PathMNIST, DermaMNIST, PneumoniaMNIST, BloodMNIST. Default is 3 images each with
seeds 0,1,2 → 12 strips total, arranged 4×3 in the paper.

Selection (--select):
  random (default, addendum) : first correctly-classified image in a seeded
                               permutation. Reproducible and not cherry-picked —
                               showing 3 demonstrates the pattern is stable.
  spread                     : among confident correct images, the one whose
                               entropy weights vary MOST (most illustrative single
                               example). Useful for a single hero strip.

Usage:
    python -m scripts.make_confidence_strips
    python -m scripts.make_confidence_strips --datasets bloodmnist --n-images 3
    python -m scripts.make_confidence_strips --select spread --n-images 1

Outputs: figures/strip/{dataset}_sample{k}.pdf
Requires each dataset's Phase 1 checkpoint at checkpoints/{dataset}_resnet18.pth.
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
from src.model import build_model, ARCH_LABELS, ARCHITECTURES
from src.tta import entropy_weights, top_k_keep_indices
from src.utils import get_device, load_checkpoint, set_seed, checkpoint_filename
from src.visualize import confidence_strip

DEFAULT_DATASETS = ["pathmnist", "dermamnist", "pneumoniamnist", "bloodmnist"]


def _view_weights_for_image(model, img, augs, device):
    """Run the N augmented views of one [0,1] image; return (views, names, weights, per_view)."""
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
    return views, names, weights, per_view


def _pick_random_correct(model, dataset, device, seed):
    """First correctly-classified image in a seeded permutation (addendum)."""
    model.eval()
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(dataset))
    with torch.no_grad():
        for idx in order:
            img, label = dataset[int(idx)]
            label = int(np.asarray(label).ravel()[0])
            logits = model(normalize_imagenet(img).unsqueeze(0).to(device))
            if int(logits.argmax(1)) == label:
                return int(idx), img
    return None, None


def _pick_spread(model, dataset, device, augs, max_scan, n_candidates, min_conf):
    """Among confident correct images, the one whose entropy weights vary most."""
    model.eval()
    candidates = []
    with torch.no_grad():
        for idx in range(min(len(dataset), max_scan)):
            img, label = dataset[idx]
            label = int(np.asarray(label).ravel()[0])
            prob = F.softmax(model(normalize_imagenet(img).unsqueeze(0).to(device)), 1).squeeze(0)
            conf, pred = prob.max(0)
            if int(pred) == label and float(conf) >= min_conf:
                candidates.append((float(conf), idx, img))
    if not candidates:
        return None, None
    candidates.sort(key=lambda c: -c[0])
    best = None
    for _conf, idx, img in candidates[:n_candidates]:
        _, _, w, _ = _view_weights_for_image(model, img, augs, device)
        spread = float(w.max() - w.min())
        if best is None or spread > best[0]:
            best = (spread, idx, img)
    return best[1], best[2]


def make_strips(dataset_name, args, device) -> int:
    cfg = get_config(dataset_name)
    ckpt = Path(args.checkpoints_dir) / checkpoint_filename(cfg.key, args.arch)
    if not ckpt.exists():
        print(f"  [skip] {cfg.key}: checkpoint not found at {ckpt}")
        return 0

    model = build_model(args.arch, num_classes=cfg.n_classes, pretrained=False)
    load_checkpoint(model, ckpt, device=device)
    model.to(device)

    test_ds = get_dataset(cfg, "test", img_size=args.img_size, root=args.data_root,
                          normalize=False)
    augs = get_augmentation_pipeline(n_views=args.n_views, seed=args.seed,
                                     include_original=not args.include_all_augs)

    made = 0
    for k in range(args.n_images):
        if args.select == "spread":
            set_seed(args.seed)
            idx, img = _pick_spread(model, test_ds, device, augs,
                                    args.max_scan, args.n_candidates, args.min_conf)
        else:  # random (addendum): one image per seed in args.seeds
            seed = args.seeds[k] if k < len(args.seeds) else k
            idx, img = _pick_random_correct(model, test_ds, device, seed)

        if img is None:
            print(f"  [skip] {cfg.key} sample {k+1}: no correct image found")
            continue

        views, names, weights, per_view = _view_weights_for_image(model, img, augs, device)
        # Gold-outline the Top-K (lowest-entropy) views Top-K TTA would keep.
        gold = (top_k_keep_indices(per_view, args.gold_k)[:, 0].tolist()
                if args.gold_k and args.gold_k > 0 else None)
        arch_sfx = "" if args.arch == "resnet18" else f"_{args.arch}"
        save_path = Path(args.figures_dir) / "strip" / f"{cfg.key}{arch_sfx}_sample{k+1}.pdf"
        confidence_strip(views, weights, names,
                         dataset_name=f"{cfg.medmnist_class} ({cfg.modality}) "
                                      f"- Sample {k+1} (idx={idx})",
                         save_path=save_path, highlight_idx=gold)
        spread = float(weights.max() - weights.min())
        gtag = f", gold Top-{args.gold_k}={gold}" if gold else ""
        print(f"  [ok]   {cfg.key} sample {k+1}: idx={idx}, "
              f"weight spread={spread:.3f}{gtag} -> {save_path}")
        made += 1
        if args.select == "spread":
            break  # spread mode produces a single hero strip
    return made


def main() -> int:
    p = argparse.ArgumentParser(description="Generate Augmentation Confidence Strips (§3.5 / Addition 3).")
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                   choices=all_dataset_keys())
    p.add_argument("--arch", default="resnet18", choices=list(ARCHITECTURES),
                   help="Backbone: resnet18 (default) or effb0.")
    p.add_argument("--gold-k", type=int, default=5,
                   help="Gold-outline the Top-K lowest-entropy (kept) bars to tie "
                        "Top-K TTA to Fig 1 (VMV plan). 0 disables. Default 5.")
    p.add_argument("--n-images", type=int, default=3,
                   help="Images per modality (addendum: 3).")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                   help="Seeds for random selection, one per image (addendum: 0 1 2).")
    p.add_argument("--select", choices=["random", "spread"], default="random",
                   help="random=seeded correct image (addendum); spread=most illustrative.")
    p.add_argument("--n-views", type=int, default=10)
    p.add_argument("--include-all-augs", action="store_true",
                   help="Drop the original view and show all 10 base augmentation types.")
    p.add_argument("--max-scan", type=int, default=400, help="(spread mode) images to scan.")
    p.add_argument("--n-candidates", type=int, default=40, help="(spread mode) top-conf pool.")
    p.add_argument("--min-conf", type=float, default=0.60, help="(spread mode) min confidence.")
    p.add_argument("--img-size", type=int, default=64, choices=[28, 64])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-root", default="./data")
    p.add_argument("--checkpoints-dir", default="./checkpoints")
    p.add_argument("--figures-dir", default="./figures")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device = get_device(prefer_cuda=not args.cpu)
    mode = "spread (1 hero)" if args.select == "spread" else f"random seeds {args.seeds[:args.n_images]}"
    print(f"Confidence strips on {device} | datasets: {', '.join(args.datasets)} | {mode}\n")

    total = 0
    for ds in args.datasets:
        total += make_strips(ds, args, device)
    print(f"\nDone. {total} strip(s) in {Path(args.figures_dir) / 'strip'}. "
          f"Paper figure: arrange 4 modalities x {args.n_images} images.")
    return 0 if total > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
