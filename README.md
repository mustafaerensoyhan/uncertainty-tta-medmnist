# Uncertainty-Weighted TTA for Medical Image Classification

Codebase for the 17-day research project on uncertainty-weighted Test-Time
Augmentation across 6 MedMNIST datasets.

**Proposal:** [`Research_Proposal_UncertaintyTTA_v2.pdf`](./docs/Research_Proposal_UncertaintyTTA_v2.pdf)
**Results tracker:** [`TTA_Results_Tracker.xlsx`](./docs/TTA_Results_Tracker.xlsx)

---

## Team & dataset assignments

| Student | Name | Dataset(s) | Modality | Notes |
|---|---|---|---|---|
| S1 | Mustafa Eren Soyhan | PathMNIST, BloodMNIST | Colon Pathology + Blood Microscopy | Repo admin / pipeline owner |
| S2 | Mohamed Ahmed | DermaMNIST | Dermatoscope | |
| S3 | Trang | BreastMNIST | Breast Ultrasound (binary) | |
| S4 | Vaidehi | OrganAMNIST | Abdominal CT | |
| S5 | Sudha | PneumoniaMNIST | Chest X-Ray (binary) | |

**Supervisor:** Mohamed Hafez (proposal author)

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

**If you have an NVIDIA GPU**, install PyTorch with CUDA support *first*. PyPI's
default torch wheel is CPU-only on Windows, which would silently leave your GPU
unused. Run `nvidia-smi` to find your driver's CUDA version, then pick the
matching index URL:

```bash
# Most modern NVIDIA laptop GPUs (RTX 30/40 series with current drivers):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Older drivers (CUDA 11.8–12.3):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Then install the rest** (this is also the single step for CPU-only / Kaggle / Colab users):

```bash
pip install -r requirements.txt
```

`pip` will skip torch and torchvision if you already installed them above.

### 4. Verify the install

```bash
python -c "import torch, medmnist; print('torch', torch.__version__, '| medmnist', medmnist.__version__, '| CUDA:', torch.cuda.is_available())"
```

If you have an NVIDIA GPU and `CUDA: False`, you got the CPU-only torch build —
go back to step 3 and reinstall torch from the CUDA index URL.

---

## Workflow & access

`main` is a protected branch. Only @mustafaerensoyhan (S1, repo admin) can push
to `main` directly. Everyone else contributes through Pull Requests, which S1
reviews and merges. This is enforced by GitHub — direct pushes to `main` from
anyone else are rejected automatically.

### Standard flow for S2–S5

```bash
# 1. Always start by syncing main with the latest
git checkout main
git pull

# 2. Create a branch for your work, named <yourId>-<short-description>
git checkout -b s2-dermamnist-baseline    # example for S2; adjust to your id

# 3. Do your work, then stage + commit + push the BRANCH (not main)
git add results/dermamnist_baseline.json
git commit -m "S2: DermaMNIST baseline — XX.X% test acc"
git push -u origin s2-dermamnist-baseline
```

Then go to the repo on GitHub. A yellow banner will appear: **"Compare & pull
request"**. Click it, write a one-line description of what you did, then click
**Create pull request**.

@mustafaerensoyhan will review and either merge or comment with requested
changes. You'll get a GitHub notification either way.

### What you may modify

**Phase 1 (S2–S5):** only commit your own `results/<your_dataset>_baseline.json`.
Don't touch `src/`, `scripts/`, `tests/`, or anything else. If you spot a bug in
the code, message @mustafaerensoyhan in the team chat and S1 will fix it on `main`.

**Phase 2 onward:** code contributions will be allocated explicitly in the
group chat. You'll still use the branch + PR workflow for those.

### After your PR is merged

```bash
git checkout main
git pull
git branch -d <your-branch-name>   # clean up local branch
```

---

## Running your baseline

Each student trains their assigned dataset(s) with a single command from the repo root:

```bash
# S1 — Mustafa
python -m scripts.train_baseline --dataset pathmnist
python -m scripts.train_baseline --dataset bloodmnist

# S2 — Mohamed Ahmed
python -m scripts.train_baseline --dataset dermamnist

# S3 — Trang
python -m scripts.train_baseline --dataset breastmnist

# S4 — Vaidehi
python -m scripts.train_baseline --dataset organamnist

# S5 — Sudha
python -m scripts.train_baseline --dataset pneumoniamnist
```

The first run for each dataset will **download the data automatically** from
the MedMNIST Zenodo repository (~20–80 MB per dataset). Subsequent runs read from
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
or Google Colab (Runtime → Change runtime type → T4 GPU). The notebook clones
this repo, installs deps, and runs the same `train_baseline` script. Free GPU
access for everyone.

Realistic training times (30 epochs, batch=64) for context:

| Dataset | Train size | CPU (local) | GPU (RTX 40-series / Kaggle T4) |
|---|---|---|---|
| BreastMNIST | 546 | ~5 min | ~1 min |
| PneumoniaMNIST | 4,708 | ~40 min | ~3 min |
| DermaMNIST | 7,007 | ~50 min | ~4 min |
| BloodMNIST | 11,959 | ~90 min | ~8 min |
| OrganAMNIST | 34,561 | ~4–5 hrs | ~20 min |
| PathMNIST | 89,996 | ~10–12 hrs | ~50 min |

If CPU is too slow for your dataset, switch to Kaggle.

---

## After your baseline is done

1. Open the **1️⃣  Baselines** sheet of `TTA_Results_Tracker.xlsx` (shared Google Sheet version)
2. Paste the numbers printed by the script into your row
3. Commit your `results/{your_dataset}_baseline.json` file via the **branch + PR workflow** above:
   ```bash
   git checkout -b <yourId>-<dataset>-baseline
   git add results/<dataset>_baseline.json
   git commit -m "<yourId>: <dataset> baseline — XX.X% test acc"
   git push -u origin <yourId>-<dataset>-baseline
   ```
   Then open a PR on GitHub. S1 will review and merge.
4. **Do not commit `.pth` checkpoint files** — they're gitignored. Upload your
   checkpoint to the shared Drive folder when done; the repo references them
   only by path.

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

If yours is off, common causes:
- Wrong split (using val instead of test)
- Forgot to load ImageNet pretrained weights
- Wrong normalization (we use ImageNet stats — grayscale is replicated to 3-channel)
- Random seed unluckiness — try `--seed 0` or `--seed 7`

---

## Repository layout

```
uncertainty-tta-medmnist/
├── .github/
│   └── CODEOWNERS         # PR approval rules (S1 reviews everything)
├── src/                   # Core library — modify only via PR, S1 approval
│   ├── config.py          # Per-dataset configs (single source of truth)
│   ├── data.py            # MedMNIST loading + transforms
│   ├── model.py           # ResNet-18 builder
│   ├── metrics.py         # Accuracy, AUC, ECE, NLL
│   ├── train.py           # Training & evaluation loops
│   └── utils.py           # Seeding, device, checkpoint I/O
├── scripts/
│   └── train_baseline.py  # ← run this for Phase 1
├── tests/
│   └── test_metrics.py    # pytest unit tests for the metrics module
├── notebooks/
│   └── kaggle_baseline.ipynb  # Kaggle/Colab runner
├── docs/                  # Proposal PDF + results tracker XLSX
├── checkpoints/           # gitignored — saved .pth files (~45 MB each)
├── results/               # commit JSON metrics here (small, in-repo)
├── figures/               # for Phase 3+ outputs
├── requirements.txt
└── README.md              # this file
```

---

## Phases of the project (overview)

| Phase | Days | Deliverable | Code lives in |
|---|---|---|---|
| 1 — Baselines | 1–3 | 6 trained checkpoints + Sheet 1 filled | `scripts/train_baseline.py` (**current**) |
| 2 — Standard TTA | 4–7 | `utils/tta.py`, Sheet 2 filled | (to be written) |
| 3 — Weighted TTA | 8–11 | `utils/fusion.py`, Sheet 3 filled, 4 confidence strips | (to be written) |
| 4 — Ablations | 12–14 | Sheets 4–7 + reliability diagrams | (to be written) |
| 5 — Paper | 15–17 | Manuscript | — |

Each phase adds new modules under `src/` and `scripts/`. Existing files are
the stable contract Phase 2+ builds on — they should not change without an
explicit team decision.

---

## Troubleshooting

**`remote rejected ... protected branch` when pushing**
You tried to push to `main`. That's blocked for everyone except S1. Create a
branch and open a PR instead — see the **Workflow & access** section above.

**`fatal: Authentication failed for https://github.com/...`**
GitHub disabled password authentication for git in 2021. You need either the
GitHub CLI (`gh auth login`) or a Personal Access Token. Generate a token at
https://github.com/settings/tokens → "Generate new token (classic)" → tick the
`repo` scope. Paste the token as the password when git prompts.

**`HTTPError: 403 Forbidden` when downloading data**
The MedMNIST Zenodo URL was throttled. Wait 5 min and retry, or download
manually from `https://zenodo.org/records/10519652/` and place the `.npz` file
under `./data/`.

**`CUDA out of memory`**
Lower the batch size: `--batch-size 32`. ResNet-18 at 64×64 should fit
comfortably in 4 GB.

**`torch.cuda.is_available()` returns False but I have an NVIDIA GPU**
You got the CPU-only PyTorch build. Reinstall:
```bash
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

**`ValueError: y_true contains only one class` for AUC**
Happens on tiny eval batches in early debug runs. Ignore — the full test set
is large enough to always contain both classes.

**Slow training on CPU**
30 epochs of PathMNIST on CPU takes ~10+ hours. Use the Kaggle notebook
(`notebooks/kaggle_baseline.ipynb`) for free GPU access.

**Windows: `BrokenPipeError` or freeze during DataLoader iteration**
Re-run with `--num-workers 0`. The default of 2 works on most Windows setups
but a minority of Python installs have multiprocessing quirks.

**My PR says "Review required" — what do I do?**
Nothing. Wait for @mustafaerensoyhan to review it. CODEOWNERS makes S1's
approval mandatory before any PR can merge.

**I committed the wrong files to my branch**
If your PR shows changes outside `results/`, reset and re-commit:
```bash
git reset --soft main           # keep changes staged, undo commits
git restore --staged .          # unstage everything
git add results/<your_file>.json
git commit -m "<yourId>: <dataset> baseline"
git push --force-with-lease     # safe force-push to your branch only
```

---

## Contact

**Proposal author / supervisor:** Mohamed Hafez
**Repo admin / pipeline owner:** Mustafa Eren Soyhan (S1) — @mustafaerensoyhan

Team chat for day-to-day questions. Open a GitHub issue for anything that
needs a written record (bugs, design discussions, decisions about file layout).
