"""
#5 fix: clean forward-only single-pass latency (no dataloader/augmentation noise).

Times PURE model forward passes on one fixed batch with heavy warmup and
torch.cuda.synchronize(), so the per-image number reflects GPU compute, not
Python/dataloader overhead. This is the stable replacement for the noisy
single-pass numbers (DeiT measured 1.74 ms on Organ but 4.25 ms on Path for
the identical model).

Place in scripts/ and run:
    python -m scripts.benchmark_forward_only --arch resnet18 --dataset pathmnist
    python -m scripts.benchmark_forward_only --arch effb0    --dataset pathmnist
    python -m scripts.benchmark_forward_only --arch deit_tiny --dataset pathmnist
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
import torch
from src.config import get_config
from src.model import build_model, ARCHITECTURES, ARCH_LABELS
from src.data import get_dataloaders
from src.utils import get_device, load_checkpoint, checkpoint_filename


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--arch", required=True, choices=list(ARCHITECTURES))
    p.add_argument("--dataset", default="pathmnist")
    p.add_argument("--tag", default="", help="checkpoint tag, e.g. _seed0; default tagless canonical")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--img-size", type=int, default=64, choices=[28, 64])
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--checkpoints-dir", default="./checkpoints")
    p.add_argument("--data-root", default="./data")
    a = p.parse_args()

    device = get_device()
    cfg = get_config(a.dataset)
    ckpt = Path(a.checkpoints_dir) / checkpoint_filename(cfg.key, a.arch, a.tag)
    model = build_model(a.arch, num_classes=cfg.n_classes, pretrained=False)
    load_checkpoint(model, ckpt, device=device)
    model.to(device).eval()

    _, _, test_loader, _ = get_dataloaders(cfg.key, batch_size=a.batch_size,
                                           img_size=a.img_size, num_workers=0,
                                           root=a.data_root)
    x, _ = next(iter(test_loader))
    x = x.to(device)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(a.warmup):
            model(x)
        sync()
        t0 = time.perf_counter()
        for _ in range(a.iters):
            model(x)
        sync()
        dt = time.perf_counter() - t0

    ms_per_image = dt / (a.iters * x.shape[0]) * 1000.0
    print(f"{ARCH_LABELS[a.arch]:14s} {cfg.key}: forward-only "
          f"{ms_per_image:.3f} ms/image  (batch={x.shape[0]}, iters={a.iters}, warmup={a.warmup})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
