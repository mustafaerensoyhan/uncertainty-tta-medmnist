"""
Dataset configuration registry for the 6 MedMNIST datasets in this study.

Every per-dataset detail (channels, classes, task type, student owner, expected
baseline accuracy) lives here so the rest of the codebase can stay dataset-
agnostic. To support a new dataset, add an entry to DATASETS — nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class DatasetConfig:
    """Static metadata for one MedMNIST dataset."""

    key: str                 # medmnist string key (lower-case, e.g. "pathmnist")
    medmnist_class: str      # name of the class exported by `medmnist`
    modality: str            # human-readable modality label
    task: str                # "multi-class" or "binary-class"
    n_channels: int          # 1 (grayscale) or 3 (RGB)
    n_classes: int           # number of classes
    train_size: int          # documented train-split size
    student: str             # S1..S5
    benchmark_acc: float     # published MedMNIST ResNet-18 accuracy (decimal, e.g. 0.881)
    notes: str = ""          # any caveats


# Six MedMNIST datasets selected for this study. Train sizes and benchmark
# accuracies are sourced from the MedMNIST v2 paper and the project proposal.
DATASETS: Dict[str, DatasetConfig] = {
    "pathmnist": DatasetConfig(
        key="pathmnist",
        medmnist_class="PathMNIST",
        modality="Colon Pathology (Histology)",
        task="multi-class",
        n_channels=3,
        n_classes=9,
        train_size=89_996,
        student="S1",
        benchmark_acc=0.881,
    ),
    "dermamnist": DatasetConfig(
        key="dermamnist",
        medmnist_class="DermaMNIST",
        modality="Dermatoscope",
        task="multi-class",
        n_channels=3,
        n_classes=7,
        train_size=7_007,
        student="S2",
        benchmark_acc=0.734,
        notes="Class-imbalanced; consider weighted loss in ablations.",
    ),
    "pneumoniamnist": DatasetConfig(
        key="pneumoniamnist",
        medmnist_class="PneumoniaMNIST",
        modality="Chest X-Ray",
        task="binary-class",
        n_channels=1,
        n_classes=2,
        train_size=4_708,
        student="S5",  # reassigned to Sudha
        benchmark_acc=0.856,
    ),
    "breastmnist": DatasetConfig(
        key="breastmnist",
        medmnist_class="BreastMNIST",
        modality="Breast Ultrasound",
        task="binary-class",
        n_channels=1,
        n_classes=2,
        train_size=546,
        student="S3",  # reassigned to Trang
        benchmark_acc=0.864,
        notes="Smallest dataset; expect higher variance; use longer warmup.",
    ),
    "bloodmnist": DatasetConfig(
        key="bloodmnist",
        medmnist_class="BloodMNIST",
        modality="Blood Cell Microscopy",
        task="multi-class",
        n_channels=3,
        n_classes=8,
        train_size=11_959,
        student="S1",  # taken by Mustafa as second dataset
        benchmark_acc=0.969,
    ),
    "organamnist": DatasetConfig(
        key="organamnist",
        medmnist_class="OrganAMNIST",
        modality="Abdominal CT",
        task="multi-class",
        n_channels=1,
        n_classes=11,
        train_size=34_561,
        student="S4",  # assigned to Vaidehi (third largest, multi-class radiology)
        benchmark_acc=0.778,
    ),
}


def get_config(name: str) -> DatasetConfig:
    """Look up a dataset config by key, case-insensitively."""
    key = name.lower()
    if key not in DATASETS:
        valid = ", ".join(DATASETS.keys())
        raise ValueError(f"Unknown dataset '{name}'. Valid: {valid}")
    return DATASETS[key]


def all_dataset_keys() -> list[str]:
    return list(DATASETS.keys())
