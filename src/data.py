"""
MedMNIST data loading.

Key design choice: every dataset is converted to 3-channel RGB at the transform
stage by replicating the single channel. This keeps the rest of the codebase
(model, training, TTA) channel-agnostic and lets us use ImageNet-pretrained
weights without modifying ResNet-18's first conv layer.

Normalization uses ImageNet statistics regardless of original modality, because
the pretrained backbone was trained against that distribution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import medmnist
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from .config import DatasetConfig, get_config


# ImageNet normalisation — applied to all datasets after grayscale-to-RGB
# replication. We chose this over per-modality stats because the backbone is
# pretrained on ImageNet; matching its training distribution at evaluation
# gives the best transfer.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _grayscale_to_rgb(tensor: torch.Tensor) -> torch.Tensor:
    """If a tensor has 1 channel, replicate it across 3. No-op if already 3-channel."""
    if tensor.shape[0] == 1:
        return tensor.repeat(3, 1, 1)
    return tensor


def normalize_imagenet(tensor: torch.Tensor) -> torch.Tensor:
    """
    Apply ImageNet normalization to a [0, 1] tensor (or batch of tensors).

    Exposed so the TTA pipeline can normalize each augmented view AFTER applying
    photometric augmentations (which are defined on [0, 1] images, not on
    already-normalized tensors).
    """
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=tensor.device).view(-1, 1, 1)
    return (tensor - mean) / std


def build_transforms(img_size: int = 64, train: bool = False,
                     normalize: bool = True) -> transforms.Compose:
    """
    Build the standard train or eval transform pipeline.

    Args:
        img_size: target resolution (medmnist returns this already)
        train: if True, applies mild on-the-fly augmentation (flip + rotation)
        normalize: if True (default), applies ImageNet normalization. Set False
                   to get [0, 1] RGB tensors — used by the TTA pipeline, which
                   applies its own augmentations on [0, 1] images then normalizes
                   each view separately.

    NOTE: Training-time augmentation here is mild (random flip + small rotation).
    The aggressive 10-augmentation pipeline used for TTA lives separately in
    src/augmentations.py and is NOT applied here.
    """
    ops = [transforms.ToTensor()]
    if train:
        # Mild on-the-fly augmentation during baseline training.
        # We deliberately keep this conservative — heavy aug would conflate
        # the baseline with the TTA experiment.
        ops.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
        ])
    ops.append(transforms.Lambda(_grayscale_to_rgb))
    if normalize:
        ops.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return transforms.Compose(ops)


def get_dataset(cfg: DatasetConfig, split: str, img_size: int = 64,
                root: str | Path = "./data", download: bool = True,
                normalize: bool = True):
    """
    Build a MedMNIST dataset object for the given split.

    Args:
        cfg: DatasetConfig from src.config
        split: "train", "val", or "test"
        img_size: 28 or 64 (we use 64 throughout the project)
        root: folder where .npz files are cached
        download: whether to download if not present
        normalize: if False, returns un-normalized [0, 1] RGB tensors (for TTA)
    """
    if split not in {"train", "val", "test"}:
        raise ValueError(f"split must be train/val/test, got {split}")

    DatasetCls = getattr(medmnist, cfg.medmnist_class)
    is_train = split == "train"
    tf = build_transforms(img_size=img_size, train=is_train, normalize=normalize)

    Path(root).mkdir(parents=True, exist_ok=True)
    return DatasetCls(
        split=split,
        transform=tf,
        download=download,
        size=img_size,
        root=str(root),
    )


def get_dataloaders(dataset_name: str, batch_size: int = 64, img_size: int = 64,
                    num_workers: int = 2, root: str | Path = "./data"
                    ) -> Tuple[DataLoader, DataLoader, DataLoader, DatasetConfig]:
    """
    Build (train, val, test) DataLoaders for one dataset.

    Returns the DatasetConfig alongside so callers don't have to look it up twice.
    """
    cfg = get_config(dataset_name)

    train_ds = get_dataset(cfg, "train", img_size=img_size, root=root)
    val_ds = get_dataset(cfg, "val", img_size=img_size, root=root)
    test_ds = get_dataset(cfg, "test", img_size=img_size, root=root)

    # pin_memory only helps when copying to a CUDA device; enabling it on a
    # CPU-only machine just prints a warning and does nothing useful.
    pin = torch.cuda.is_available()
    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)

    return train_loader, val_loader, test_loader, cfg


def get_tta_test_loader(dataset_name: str, batch_size: int = 64, img_size: int = 64,
                        num_workers: int = 2, root: str | Path = "./data"
                        ) -> Tuple[DataLoader, DatasetConfig]:
    """
    Build a test-split DataLoader that yields UN-normalized [0, 1] RGB images,
    for use by the TTA pipeline. The TTA loop applies augmentations on these
    [0, 1] images and normalizes each augmented view itself.
    """
    cfg = get_config(dataset_name)
    test_ds = get_dataset(cfg, "test", img_size=img_size, root=root, normalize=False)
    pin = torch.cuda.is_available()
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin)
    return test_loader, cfg
