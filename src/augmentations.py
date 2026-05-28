"""
TTA augmentation pipeline — the 10 augmentation types from proposal Section 3.3.

Each augmentation is a function that takes a [0, 1] RGB image tensor (C, H, W)
and returns an augmented [0, 1] tensor of the same shape. The TTA loop
(src/tta.py) applies these, then normalizes each augmented view with ImageNet
statistics before the forward pass.

`get_augmentation_pipeline(n_views, seed)` returns a list of (fn, name) pairs.
For n_views <= number of base types, returns distinct types; for larger n_views
it cycles through the pool, relying on the stochastic augmentations (rotation,
brightness, noise, etc.) to produce different views on each repeat.

The first view is always the identity ("original"), matching standard TTA
convention where the un-augmented image is one of the ensemble members.
"""

from __future__ import annotations

import random
from typing import Callable, List, Tuple

import torch
import torchvision.transforms.functional as TF

# An augmentation is a callable: (C,H,W) float tensor in [0,1] -> same shape/range.
AugFn = Callable[[torch.Tensor], torch.Tensor]


def _clamp01(t: torch.Tensor) -> torch.Tensor:
    return t.clamp(0.0, 1.0)


# ── The 10 augmentation types (proposal Section 3.3) ───────────────────────
# Each takes a [0,1] tensor and returns a [0,1] tensor. Stochastic ones sample
# their parameter fresh on each call so repeated use produces different views.

def aug_identity(img: torch.Tensor) -> torch.Tensor:
    """No-op — the original image. Always view 0 of the ensemble."""
    return img


def aug_hflip(img: torch.Tensor) -> torch.Tensor:
    """Horizontal flip (deterministic)."""
    return TF.hflip(img)


def aug_vflip(img: torch.Tensor) -> torch.Tensor:
    """Vertical flip (deterministic)."""
    return TF.vflip(img)


def aug_rotate(img: torch.Tensor) -> torch.Tensor:
    """Rotation by a random angle in [-10, +10] degrees."""
    angle = random.uniform(-10.0, 10.0)
    return TF.rotate(img, angle)


def aug_brightness(img: torch.Tensor) -> torch.Tensor:
    """Brightness scaling by a random factor in [0.85, 1.15] (±15%)."""
    factor = random.uniform(0.85, 1.15)
    return _clamp01(TF.adjust_brightness(img, factor))


def aug_gaussian_noise(img: torch.Tensor) -> torch.Tensor:
    """Additive Gaussian noise, sigma = 0.01."""
    return _clamp01(img + torch.randn_like(img) * 0.01)


def aug_center_crop(img: torch.Tensor) -> torch.Tensor:
    """Centre crop to 90% then resize back to original size (deterministic)."""
    _, h, w = img.shape
    ch, cw = int(round(h * 0.9)), int(round(w * 0.9))
    cropped = TF.center_crop(img, [ch, cw])
    return TF.resize(cropped, [h, w], antialias=True)


def aug_contrast(img: torch.Tensor) -> torch.Tensor:
    """Contrast scaling by a random factor in [0.9, 1.1] (±10%)."""
    factor = random.uniform(0.9, 1.1)
    return _clamp01(TF.adjust_contrast(img, factor))


def aug_color_jitter(img: torch.Tensor) -> torch.Tensor:
    """Combined brightness/contrast/saturation/hue jitter."""
    img = TF.adjust_brightness(img, random.uniform(0.9, 1.1))
    img = TF.adjust_contrast(img, random.uniform(0.9, 1.1))
    img = TF.adjust_saturation(img, random.uniform(0.9, 1.1))
    img = TF.adjust_hue(img, random.uniform(-0.05, 0.05))
    return _clamp01(img)


def aug_elastic(img: torch.Tensor) -> torch.Tensor:
    """
    Light elastic deformation. Uses torchvision's ElasticTransform if available;
    falls back to a small random affine shear otherwise (keeps the pipeline
    working across torchvision versions).
    """
    try:
        from torchvision.transforms import ElasticTransform
        t = ElasticTransform(alpha=20.0, sigma=4.0)
        return _clamp01(t(img))
    except Exception:
        shear = random.uniform(-5.0, 5.0)
        return _clamp01(TF.affine(img, angle=0, translate=[0, 0], scale=1.0, shear=[shear, 0.0]))


def aug_sharpness(img: torch.Tensor) -> torch.Tensor:
    """Sharpness scaling by a random factor in [0.8, 1.2] (±20%)."""
    factor = random.uniform(0.8, 1.2)
    return _clamp01(TF.adjust_sharpness(img, factor))


# Ordered pool of (name, fn). The order matters: get_augmentation_pipeline takes
# the first n_views of [identity, *pool] for small N, so the most "standard"
# augmentations come first.
BASE_AUGMENTATIONS: List[Tuple[str, AugFn]] = [
    ("hflip", aug_hflip),
    ("rotate", aug_rotate),
    ("brightness", aug_brightness),
    ("gaussian_noise", aug_gaussian_noise),
    ("center_crop", aug_center_crop),
    ("contrast", aug_contrast),
    ("sharpness", aug_sharpness),
    ("vflip", aug_vflip),
    ("color_jitter", aug_color_jitter),
    ("elastic", aug_elastic),
]


def get_augmentation_pipeline(n_views: int = 10, seed: int | None = None,
                              include_original: bool = True
                              ) -> List[Tuple[AugFn, str]]:
    """
    Build a list of (aug_fn, aug_name) pairs of length n_views.

    Args:
        n_views: number of views in the ensemble (e.g. 5, 10, 20, 50)
        seed: if set, makes the *selection* of augmentations deterministic
              (individual stochastic augmentations still vary per call unless
              their internal RNG is also seeded by the caller)
        include_original: if True, view 0 is the identity (un-augmented image)

    Returns:
        list of (fn, name) tuples, length n_views.

    For n_views beyond the base pool size, the pool is cycled — repeated
    stochastic augmentations (rotate, brightness, noise, ...) produce different
    views each time, while deterministic ones (hflip, vflip, center_crop)
    repeat identically (which simply gives them slightly more ensemble weight).
    """
    if n_views < 1:
        raise ValueError("n_views must be >= 1")

    rng = random.Random(seed) if seed is not None else random

    pipeline: List[Tuple[AugFn, str]] = []
    if include_original:
        pipeline.append((aug_identity, "original"))

    pool = list(BASE_AUGMENTATIONS)
    # For reproducible selection when a seed is given, shuffle a copy.
    if seed is not None:
        rng.shuffle(pool)

    i = 0
    while len(pipeline) < n_views:
        name, fn = pool[i % len(pool)]
        pipeline.append((fn, name))
        i += 1

    return pipeline[:n_views]


def augmentation_names() -> List[str]:
    """Return the names of the 10 base augmentation types (no identity)."""
    return [name for name, _ in BASE_AUGMENTATIONS]
