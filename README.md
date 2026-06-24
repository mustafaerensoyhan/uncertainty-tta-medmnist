# Uncertainty-Weighted Test-Time Augmentation for Robust Medical Image Classification

A calibration-focused, modality-stratified study of test-time augmentation (TTA) and uncertainty-aware fusion for medical image classification, evaluated across six MedMNIST v2 datasets and three backbones. This repository holds all source code, experiments, figures, and results behind the project.

**Course:** Real-Time Data Analytics for IoT, Final Project
**Team:** IoT Thunders, Ontario Tech University
**Supervisor:** Professor Khalid Elgazzar
**Companion paper:** *When Does Test-Time Augmentation Help Calibration? A Visual, Modality-Stratified Study for Medical Image Classification*, submitted to Vision, Modeling, and Visualization (VMV) 2026 and currently under review. See [Project documents](#project-documents) below for the report, slides, and poster.

---

## Overview

Medical image classifiers are usually deployed with a single forward pass per image, and those predictions are frequently overconfident: the stated confidence does not match how often the model is actually right. In clinical decision support this is a safety risk. Test-Time Augmentation (TTA) runs several augmented copies of an image through a frozen model and fuses their predictions without retraining, but standard equal-weight TTA gives every view the same weight, which lets distorted views mislead the result.

This project studies a simple alternative: weight each augmented view by how confident the model is about it, so trusted views count more and uncertain views count less. The goal is to characterize when this helps or hurts calibration across modalities, rather than to propose a new algorithm.

### Key findings

- **Calibration sign flip.** Equal-weight TTA improves calibration on overconfident modalities (histology, dermoscopy, chest X-ray) but worsens it on the one well-calibrated modality (abdominal CT). The direction is governed by the model's initial calibration state.
- **Uncertainty weighting helps where the signal is informative and trustworthy.** Entropy weighting cuts Expected Calibration Error (ECE) by up to 41% on multi-class microscopy, with accuracy left statistically unchanged (Wilcoxon p = 0.17).
- **Temperature-scaled entropy (TS+Entropy) is the safest default**, attaining the best or near-best NLL and ECE on five of six datasets.
- **Augmentation Confidence Strips:** a visualization that makes per-augmentation trust directly inspectable.
- **A practitioner selection guide** for choosing a fusion strategy per modality.

### Relevance to real-time IoT analytics

The pipeline is inference-only and retraining-free, so it applies to an already-trained, frozen model at the edge. Calibrated TTA adds roughly 10 ms per image (N=10 views) on a single GPU, with the raw forward pass at 0.05 to 0.13 ms per image, making confidence-calibrated inference practical for real-time, resource-constrained medical IoT and point-of-care devices, where trustworthy confidence matters as much as raw accuracy. Reported metrics include accuracy, AUC-ROC, ECE, NLL, and per-image latency.

---

## Project documents

- **Final report (IEEE two-column):** [`docs/report.pdf`](docs/report.pdf)
- **Presentation slides:** [`docs/presentation.pdf`](docs/presentation.pdf) (editable source: `docs/presentation.pptx`)
- **Poster:** [`docs/poster.pdf`](docs/poster.pdf) (editable source: `docs/poster.pptx`)
- **Original proposal:** [`docs/Research_Proposal_UncertaintyTTA_v2.pdf`](docs/Research_Proposal_UncertaintyTTA_v2.pdf)

---

## Repository structure

```text
uncertainty-tta-medmnist/
├── src/                              # Core library (imported by every script)
│   ├── config.py                     # Per-dataset configuration, single source of truth
│   ├── data.py                       # MedMNIST loading and transforms
│   ├── model.py                      # Backbone builders: ResNet-18, EfficientNet-B0, DeiT-Tiny
│   ├── augmentations.py              # The 10 test-time augmentation types
│   ├── tta.py                        # Per-view probabilities and fusion strategies
│   ├── mc_dropout.py                 # MC-Dropout epistemic baseline (no retraining)
│   ├── temperature.py                # Temperature scaling: fit T, TS-only, TS+Entropy
│   ├── metrics.py                    # Accuracy, AUC-ROC, ECE, NLL
│   ├── perf.py                       # Inference-time measurement (warm-up + synchronize)
│   ├── evaluate.py                   # Runs all strategies and collects metrics
│   ├── visualize.py                  # Reliability diagrams, curves, confidence strips
│   └── utils.py                      # Seeding, device, checkpoint I/O
├── scripts/                          # Entry points, run as: python -m scripts.<name>
│   ├── train_baseline.py             # Train a baseline and log metrics
│   ├── run_standard_tta.py           # Standard equal-weight TTA
│   ├── run_weighted_tta.py           # 8 weighted strategies + temperature scaling
│   ├── make_confidence_strips.py     # Augmentation Confidence Strips (lead figure)
│   ├── make_reliability_diagrams.py  # Calibration diagrams from saved probabilities
│   ├── significance.py               # McNemar + Wilcoxon + bootstrap CIs
│   ├── build_full_matrix.py          # Merge per-dataset CSVs into full_matrix.csv
│   ├── analysis_figures.py           # Modality bar charts and latency tradeoff
│   ├── ablate_n.py                   # Accuracy and ECE vs number of views N
│   ├── ablate_augmentations.py       # Leave-one-out per augmentation
│   └── aggregate_seeds.py            # Multi-seed mean and standard deviation
├── tests/                            # Unit tests: metrics, fusion, MC-Dropout, temperature, significance
├── notebooks/                        # kaggle_baseline.ipynb for free-GPU runs
├── results/                          # Metric tables and CSVs (committed, small)
├── figures/                          # Publication figures (strips, reliability, heatmaps, bars)
├── docs/                             # Report, presentation, poster, and proposal
│   ├── report.pdf                    # Final report (IEEE two-column)
│   ├── presentation.pdf              # Project presentation slides
│   ├── poster.pdf                    # Project poster
│   └── Research_Proposal_UncertaintyTTA_v2.pdf
├── checkpoints/                      # Trained .pth files (gitignored, large)
├── predictions/                      # Per-image .npy arrays (gitignored, large)
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Installation

Requires Python 3.10+ and PyTorch 2.x.

```bash
git clone https://github.com/mustafaerensoyhan/uncertainty-tta-medmnist.git
cd uncertainty-tta-medmnist
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If you have an NVIDIA GPU, install the CUDA build of PyTorch first so the GPU is actually used (the default PyPI wheel is CPU-only on Windows):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

MedMNIST data downloads automatically on first run and is cached under `./data/`, so no manual dataset download is needed.

---

## Datasets

All data is from **MedMNIST v2** (Yang et al., 2023): pre-split, standardized to 64x64, and pip-installable, so no manual download is required. Six datasets span five modalities and both binary and multi-class tasks.

| Dataset | Modality | Task | Classes | Test size |
|---------|----------|------|---------|-----------|
| PathMNIST | Microscopy (histology) | Multi-class | 9 | 7,180 |
| DermaMNIST | Dermatoscopy | Multi-class | 7 | 2,005 |
| PneumoniaMNIST | Chest X-ray | Binary | 2 | 624 |
| BreastMNIST | Breast ultrasound | Binary | 2 | 156 |
| BloodMNIST | Microscopy (blood) | Multi-class | 8 | 3,421 |
| OrganAMNIST | Abdominal CT | Multi-class | 11 | 17,778 |

Dataset homepage and license: https://medmnist.com

---

## Running the project

All scripts run as modules from the repo root and take a `--dataset` flag. Valid datasets: `pathmnist`, `dermamnist`, `pneumoniamnist`, `breastmnist`, `bloodmnist`, `organamnist`.

```bash
# 1. Train a baseline (ResNet-18, ImageNet-pretrained) -> checkpoint + metrics
python -m scripts.train_baseline --dataset pathmnist

# 2. Standard equal-weight TTA (the failure-mode baseline)
python -m scripts.run_standard_tta --dataset pathmnist

# 3. Uncertainty-weighted TTA: all eight fusion strategies plus temperature scaling
python -m scripts.run_weighted_tta --dataset pathmnist

# 4. Visualizations, statistics, and the full results matrix
python -m scripts.make_confidence_strips        # Augmentation Confidence Strips
python -m scripts.make_reliability_diagrams     # calibration diagrams
python -m scripts.significance                   # McNemar + Wilcoxon + bootstrap CIs
python -m scripts.build_full_matrix              # merge per-dataset CSVs -> full_matrix.csv
python -m scripts.aggregate_seeds                # multi-seed mean and std
```

Useful flags: `--epochs`, `--batch-size`, `--lr`, `--seed`, `--num-workers`, and `--cpu` (force CPU). On Windows, keep `--num-workers 0`. Metrics are written to `results/`, figures to `figures/`, per-image predictions to `predictions/`, and checkpoints to `checkpoints/`.

**Experimental protocol.** ImageNet-pretrained models; Adam (lr 1e-4, 30 epochs, batch 64, 64x64 input); temperature fit on the validation split by L-BFGS; three seeds (0, 42, 123); N=10 TTA views with per-view logits computed once and reused by every fusion strategy.

---

## Results at a glance

| Item | Result |
|------|--------|
| ECE reduction, entropy vs equal-weight | up to 41% (BloodMNIST), 33% (PathMNIST) |
| Best overall strategy | TS+Entropy, best or near-best NLL on 5 of 6 datasets |
| Accuracy change | none significant (Wilcoxon p = 0.17) |
| Inference latency | about 10 ms per image at N=10; 0.05 to 0.13 ms single forward pass |
| Backbones | ResNet-18, EfficientNet-B0, DeiT-Tiny |
| Fusion configurations | 11 (8 soft strategies plus Top-K at K = 3, 5, 7) |

Full tables and figures are in `results/` and `figures/`; the complete analysis is in the report under `docs/`.

---

## Team and contributions

**IoT Thunders, Ontario Tech University.** Supervised by Professor Khalid Elgazzar.

| Member | Contributions |
|--------|---------------|
| **Mustafa Eren Soyhan** | Repository admin and pipeline owner; core TTA and fusion code; EfficientNet-B0 and DeiT-Tiny backbones; Top-K strategies; unified runner and inference benchmarking; report and conference review. |
| **Mohamed Ahmed** | DermaMNIST owner; publication tables, figures, and bootstrap confidence intervals; Augmentation Confidence Strips composite; pipeline debugging; report and conference review. |
| **Thuy Trang Cao** | BreastMNIST owner; four-phase pipeline (baselines, standard and weighted TTA, ablations, three-seed stability); results, discussion, conclusion, and selection guide. |
| **Vaidehi Patel** | OrganAMNIST owner; baseline and weighted-TTA experiments; results tables and reliability diagrams; methodology, dataset, and experimental-setup writing. |
| **Sudha Rajendran** | PneumoniaMNIST owner; full pipeline (baselines, eight weighted-TTA strategies, multi-seed ablations); introduction, related work, and references. |
| **Mohamed Hafez** | Project proposal and mentorship; foundational proposal, guidance, and feedback. |

---

## License

Released under the MIT License. See [`LICENSE`](LICENSE).

## Acknowledgements

Built on [MedMNIST v2](https://medmnist.com) and PyTorch. The calibration question was motivated by prior work from Medeiros (2026).
