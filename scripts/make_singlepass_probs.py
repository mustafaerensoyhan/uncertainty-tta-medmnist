"""
#9 step 1: generate single-pass (no-TTA) per-image softmax probabilities.

run_weighted_tta already saved predictions/<stem>_baseline_probs.npy (equal-weight
TTA) and <stem>_labels.npy. The only missing piece for the sign-flip bootstrap is the
single-pass prediction (the plain model forward on un-augmented images). This writes
predictions/<stem>_singlepass_probs.npy with a stem that matches the baseline file.

Place in scripts/ and run (matches your seed/tag convention; adjust if different):
    python -m scripts.make_singlepass_probs --arch resnet18  --seeds 0 42 123
    python -m scripts.make_singlepass_probs --arch effb0     --seeds 0 42 123
    python -m scripts.make_singlepass_probs --arch deit_tiny --seeds 0 42 123
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from src.config import all_dataset_keys, get_config
from src.model import build_model, ARCHITECTURES
from src.data import get_dataloaders
from src.train import predict_probs
from src.utils import (get_device, load_checkpoint, set_seed,
                       checkpoint_filename, result_stem)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--arch", required=True, choices=list(ARCHITECTURES))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 42, 123])
    p.add_argument("--tag-template", default="_seed{s}",
                   help="how checkpoints/predictions are tagged per seed; "
                        "use '' if you stored a single tagless run")
    p.add_argument("--datasets", nargs="+", default=all_dataset_keys())
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--img-size", type=int, default=64, choices=[28, 64])
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--checkpoints-dir", default="./checkpoints")
    p.add_argument("--predictions-dir", default="./predictions")
    p.add_argument("--data-root", default="./data")
    a = p.parse_args()

    device = get_device()
    pdir = Path(a.predictions_dir)
    pdir.mkdir(parents=True, exist_ok=True)

    for ds in a.datasets:
        cfg = get_config(ds)
        for s in a.seeds:
            tag = a.tag_template.format(s=s) if a.tag_template else ""
            ckpt = Path(a.checkpoints_dir) / checkpoint_filename(cfg.key, a.arch, tag)
            if not ckpt.exists():
                print(f"  [skip] {cfg.key} {a.arch} {tag!r}: no checkpoint at {ckpt}")
                continue
            set_seed(s)
            model = build_model(a.arch, num_classes=cfg.n_classes, pretrained=False)
            load_checkpoint(model, ckpt, device=device)
            model.to(device)
            _, _, test_loader, _ = get_dataloaders(cfg.key, batch_size=a.batch_size,
                                                   img_size=a.img_size,
                                                   num_workers=a.num_workers,
                                                   root=a.data_root)
            probs, labels = predict_probs(model, test_loader, device)
            stem = result_stem(cfg.key, a.arch, tag)
            np.save(pdir / f"{stem}_singlepass_probs.npy", probs.astype(np.float32))
            lab_path = pdir / f"{stem}_labels.npy"
            if not lab_path.exists():
                np.save(lab_path, labels.astype(np.int64))
            print(f"  [ok] {stem}: single-pass probs {probs.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
