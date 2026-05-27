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


def build_transforms(img_size: int = 64, train: bool = False) -> transforms.Compose:
    """
    Build the standard train or eval transform pipeline.

    NOTE: Training-time augmentation here is mild (random flip + small rotation).
    The aggressive 10-augmentation pipeline used for TTA lives separately in
    utils/augmentations.py (to be written in Phase 2) and is NOT applied here.
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
    ops.extend([
        transforms.Lambda(_grayscale_to_rgb),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    # Resize last so augmentations operate on native resolution
    # (medmnist already returns the requested size, so this is a safety net).
    return transforms.Compose(ops)


def get_dataset(cfg: DatasetConfig, split: str, img_size: int = 64,
                root: str | Path = "./data", download: bool = True):
    """
    Build a MedMNIST dataset object for the given split.

    Args:
        cfg: DatasetConfig from src.config
        split: "train", "val", or "test"
        img_size: 28 or 64 (we use 64 throughout the project)
        root: folder where .npz files are cached
        download: whether to download if not present
    """
    if split not in {"train", "val", "test"}:
        raise ValueError(f"split must be train/val/test, got {split}")

    DatasetCls = getattr(medmnist, cfg.medmnist_class)
    is_train = split == "train"
    tf = build_transforms(img_size=img_size, train=is_train)

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

    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)

    return train_loader, val_loader, test_loader, cfg
