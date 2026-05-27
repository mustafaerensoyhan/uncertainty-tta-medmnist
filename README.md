# Uncertainty-Weighted TTA for Medical Image Classification

Codebase for the 17-day research project on uncertainty-weighted Test-Time
Augmentation across 6 MedMNIST datasets.

**Proposal:** [`Research_Proposal_UncertaintyTTA_v2.pdf`](./docs/Research_Proposal_UncertaintyTTA_v2.pdf)
**Results tracker:** [`TTA_Results_Tracker.xlsx`](./docs/TTA_Results_Tracker.xlsx)

---

## Team & dataset assignments

| Student | Name | Dataset(s) | Modality | Notes |
|---|---|---|---|---|
| S1 | Mustafa Eren Soyhan | PathMNIST, BloodMNIST | Colon Pathology + Blood Microscopy | Also owns the pipeline / repo |
| S2 | Mohamed Abdel Sattar | DermaMNIST | Dermatoscope | |
| S3 | Trang | BreastMNIST | Breast Ultrasound (binary) | |
| S4 | Vaidehi | OrganAMNIST | Abdominal CT | |
| S5 | Sudha | PneumoniaMNIST | Chest X-Ray (binary) | |

---

## Quick start (one-time setup, ~5 minutes)

### 1. Clone the repo

```bash
git clone https://github.com/mustafaerensoyhan/uncertainty-tta-medmnist.git
cd uncertainty-tta-medmnist
```

### 2. Create a virtual environment

**On Mac / Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
```

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This installs PyTorch, torchvision, MedMNIST, scikit-learn, matplotlib, and a few
others. Total install is ~2 GB (most of it is PyTorch).

### 4. Verify the install

```bash
python -c "import torch, medmnist; print('torch', torch.__version__, '| medmnist', medmnist.__version__)"
```

If both versions print, you're ready.

---

## Running your baseline

Each student trains their dataset(s) with a single command from the repo root:

```bash
# S1 — Mustafa
python -m scripts.train_baseline --dataset pathmnist
python -m scripts.train_baseline --dataset bloodmnist

# S2 — Mohamed
python -m scripts.train_baseline --dataset dermamnist

# S3 — Trang
python -m scripts.train_baseline --dataset breastmnist

# S4 — Vaidehi
python -m scripts.train_baseline --dataset organamnist

# S5 — Sudha
python -m scripts.train_baseline --dataset pneumoniamnist
```

The first run for each dataset will **download the data automatically** from
the MedMNIST Zenodo repository (~50 MB per dataset). Subsequent runs read from
the local cache in `./data/`.

### What the script does

1. Downloads + loads the dataset (cached in `./data/`)
2. Builds a ResNet-18 with ImageNet-pretrained weights, FC head sized to the dataset's classes
3. Trains for 30 epochs with Adam (lr=1e-4), saving the best-val checkpoint
4. Evaluates the best checkpoint on the test split
5. Computes **Accuracy, AUC-ROC, ECE, NLL**
6. Saves results to `results/{dataset}_baseline.json`
7. Saves the checkpoint to `checkpoints/{dataset}_resnet18.pth`
8. Prints a copy-pasteable row for the **1️⃣  Baselines** sheet of the tracker

### Useful flags

```bash
python -m scripts.train_baseline --dataset bloodmnist \
    --epochs 30 \
    --batch-size 64 \
    --lr 1e-4 \
    --num-workers 4 \
    --seed 42
```

`--cpu` forces CPU even if CUDA is available (slow — only for debugging).

---

## No local GPU? Use the Kaggle notebook

Open `notebooks/kaggle_baseline.ipynb` in Kaggle (Settings → Accelerator → GPU T4)
or Google Colab. The notebook clones this repo, installs deps, and runs the same
`train_baseline` script. Free GPU access for everyone.

---

## After your baseline is done

1. Open the **1️⃣  Baselines** sheet of `TTA_Results_Tracker.xlsx`
2. Paste the numbers printed by the script into your row
3. Commit your `results/{dataset}_baseline.json` file:
   ```bash
   git add results/{dataset}_baseline.json
   git commit -m "S2: DermaMNIST baseline — 73.6% test acc"
   git push
   ```
4. **Do not** commit `.pth` checkpoint files (gitignored — they're large and we
   share them via Drive instead). Upload your checkpoint to the shared Drive
   folder when done.

---

## Validating against published benchmarks

The proposal requires every baseline to match the published MedMNIST benchmark
within **±2 %**. The training script automatically checks this and flags
out-of-tolerance results.

| Dataset | Published benchmark | Tolerance |
|---|---|---|
| PathMNIST | 88.1 % | 86.1–90.1 % |
| DermaMNIST | 73.4 % | 71.4–75.4 % |
| PneumoniaMNIST | 85.6 % | 83.6–87.6 % |
| BreastMNIST | 86.4 % | 84.4–88.4 % |
| BloodMNIST | 96.9 % | 94.9–98.9 % |
| OrganAMNIST | 77.8 % | 75.8–79.8 % |

If yours is off, common causes are:
- Wrong split (using val instead of test)
- Forgot to load ImageNet pretrained weights
- Wrong normalization (we use ImageNet stats — grayscale is replicated to 3-channel)
- Random seed unluckiness — try `--seed 0` or `--seed 7`

---

## Repository layout

```
uncertainty-tta-medmnist/
├── src/                   # Core library — DO NOT modify without team review
│   ├── config.py          # Per-dataset configs (single source of truth)
│   ├── data.py            # MedMNIST loading + transforms
│   ├── model.py           # ResNet-18 builder
│   ├── metrics.py         # Accuracy, AUC, ECE, NLL
│   ├── train.py           # Training & evaluation loops
│   └── utils.py           # Seeding, device, checkpoint I/O
├── scripts/
│   ├── train_baseline.py  # ← run this for Phase 1
│   └── smoke_test.py      # plumbing check, no data needed
├── notebooks/
│   └── kaggle_baseline.ipynb  # Kaggle/Colab runner
├── checkpoints/           # gitignored — saved .pth files
├── results/               # commit JSON metrics here
├── figures/               # for Phase 3+ outputs
├── requirements.txt
└── README.md              # this file
```

---

## Phases of the project (overview)

| Phase | Days | Deliverable | Code lives in |
|---|---|---|---|
| 1 — Baselines | 1–3 | 6 trained checkpoints + Sheet 1 filled | `scripts/train_baseline.py` (**this is current**) |
| 2 — Standard TTA | 4–7 | `utils/tta.py`, Sheet 2 filled | (to be written) |
| 3 — Weighted TTA | 8–11 | `utils/fusion.py`, Sheet 3 filled, 4 confidence strips | (to be written) |
| 4 — Ablations | 12–14 | Sheets 4–7 + reliability diagrams | (to be written) |
| 5 — Paper | 15–17 | Manuscript | — |

Each phase will add new modules under `src/` and `scripts/`. The existing files
will not be modified — they are the stable contract Phase 2+ builds on.

---

## Troubleshooting

**`HTTPError: 403 Forbidden` when downloading data**
The MedMNIST Zenodo URL was throttled. Wait 5 min and retry, or download
manually from `https://zenodo.org/records/10519652/` and place the `.npz` file
under `./data/`.

**`CUDA out of memory`**
Lower the batch size: `--batch-size 32`. ResNet-18 at 64×64 should fit
comfortably in 4 GB.

**`ValueError: y_true contains only one class` for AUC**
Happens on tiny eval batches in early debug runs. Ignore — the test set is
always large enough.

**Slow training on CPU**
30 epochs of PathMNIST on CPU takes ~6 hours. Use the Kaggle notebook
(`notebooks/kaggle_baseline.ipynb`) for free GPU access.

**Windows: `BrokenPipeError` or freeze during DataLoader iteration**
Re-run with `--num-workers 0`. The default of 2 works on most Windows setups
but a minority of Python installs have multiprocessing quirks.

---

## Contact

Proposal author: Mohamed Hafez
Repo maintainer: Mustafa Eren Soyhan (S1)
