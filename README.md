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

## Phase 2 — Standard TTA (Days 4–7)

Once your baseline checkpoint exists (Phase 1), run standard equal-weight TTA to
measure how accuracy/ECE/NLL change as you add more augmented views. This
replicates the "TTA is not better" finding.

```bash
# S1 — Mustafa
python -m scripts.run_standard_tta --dataset pathmnist
python -m scripts.run_standard_tta --dataset bloodmnist

# S2 — Mohamed Ahmed
python -m scripts.run_standard_tta --dataset dermamnist

# S3 — Trang
python -m scripts.run_standard_tta --dataset breastmnist

# S4 — Vaidehi
python -m scripts.run_standard_tta --dataset organamnist

# S5 — Sudha
python -m scripts.run_standard_tta --dataset pneumoniamnist
```

By default it evaluates N = 5, 10, 20, 50 views plus an N=1 no-TTA baseline.
Override with `--n-views 5 10 20 50` if you want a different sweep.

### What it does

1. Loads your Phase 1 checkpoint from `checkpoints/{dataset}_resnet18.pth`
2. For each N, generates N augmented views per test image (10 augmentation
   types from the proposal's Section 3.3, applied on un-normalized images then
   normalized per view), runs them through the model, and averages the softmax
   outputs with equal weight 1/N
3. Computes Accuracy, AUC, ECE, NLL at each N
4. Saves `results/{dataset}_standard_tta.csv` (one row per N)
5. Saves `figures/tta/{dataset}_accuracy_vs_n.png` (accuracy + ECE vs N)
6. Prints a verdict: does TTA help or hurt on this dataset?

### Deliverable

Paste your CSV rows into **Sheet 2️⃣ Standard TTA** of the tracker, and note in
the group chat whether TTA helped or hurt your dataset. The team needs at least
3 of 6 datasets to show degradation to confirm the hypothesis.

> **Important:** Phase 2 only adds new files (`src/tta.py`, `src/augmentations.py`,
> `scripts/run_standard_tta.py`) and one new method in `src/data.py`. It does NOT
> change the Phase 1 training pipeline — your existing baselines and checkpoints
> remain valid.

---

## Phase 3 — Uncertainty-Weighted TTA + addendum additions (Days 8–11)

The core phase. Builds the main results matrix (Sheet 3) across **8 strategies**,
adds temperature scaling, inference-time measurement, per-image prediction
arrays, and the 3-image confidence strips. Read `docs/Addendum_Phase3_4_Additions.pdf`
alongside the main proposal — its additions are **not optional**.

The eight strategies (proposal §3.2 + addendum Addition 2):

| Strategy | Weight / method | Notes |
|---|---|---|
| `baseline` | w = 1/N | Equal weight (the Phase 2 method) |
| `maxprob` | w_i = max(p_i) | Confident views dominate |
| `entropy` | w_i = exp(−H(p_i)) | Low-entropy views dominate — usually the winner |
| `variance` | w_i = var(p_i) | **Confidence-aligned** (sharp views up) — supervisor-approved direction |
| `variance_inv` | w_i = 1/(var(p_i)+ε) | Literal proposal formula — reported as a **negative-finding ablation** |
| `mc_dropout` | mean of T dropout passes | Epistemic proxy (no augmentation) |
| `ts_only` | softmax(logits/T) | Temperature scaling, single pass, no TTA |
| `ts_entropy` | entropy TTA with logits/T | TS + entropy — best calibration |

Run all eight on your dataset(s) with one command:

```bash
# S1 — Mustafa
python -m scripts.run_weighted_tta --dataset pathmnist
python -m scripts.run_weighted_tta --dataset bloodmnist
# S2 dermamnist · S3 breastmnist · S4 organamnist · S5 pneumoniamnist
```

For each dataset this:
1. **Fits temperature T** on the validation split (LBFGS on NLL) — reported in the CSV.
2. Computes per-view **logits** once and reuses them for every augmentation-based
   strategy (plain and temperature-scaled).
3. Runs MC Dropout and TS-only on their own passes.
4. **Measures inference time** (ms/image) per strategy — warmup + `torch.cuda.synchronize()`
   so GPU numbers are real (Addition 1).
5. Writes `results/{dataset}_weighted_tta.csv` — one row per strategy with
   accuracy, AUC (macro for multiclass / binary for the two binary sets), ECE, NLL,
   **inf_ms**, and the fitted **temperature**.
6. Saves **per-image prediction arrays** (Addition 5) to `predictions/`
   (`{dataset}_labels.npy`, `{dataset}_{strategy}_preds.npy`,
   `{dataset}_{strategy}_probs.npy`) — needed for the Phase 4 McNemar/Wilcoxon
   tests. `predictions/` is **gitignored**; push it to the shared Drive, not git.

`--no-time` skips the latency pass (faster debug runs). `--mc-T`, `--mc-p`,
`--n-views` tune the rest.

Standard TTA (`run_standard_tta.py`) now also fills **inf_ms** per N, so Sheet 2's
Inference-time column gets populated on the same pass.

### Building the full matrix

```bash
python -m scripts.build_full_matrix      # merges results/{dataset}_weighted_tta.csv
```

Each student commits only their own `{dataset}_weighted_tta.csv` (branch + PR);
S1 runs the merge on `main`.

### Figure 1 — Augmentation Confidence Strips (×3 images, Addition 3)

```bash
python -m scripts.make_confidence_strips                  # 4 modalities x 3 images = 12 strips
python -m scripts.make_confidence_strips --select spread  # single hero strip per dataset
```

Default selection is **random, seeded (0,1,2), correctly-classified** — three
images per modality so the pattern is demonstrably stable, not cherry-picked
(addendum). Output: `figures/strip/{dataset}_sample{1,2,3}.pdf`, arranged 4×3 in
the paper. The `spread` mode (picks the most illustrative single image by weight
variance) is kept for a one-off hero figure.

### Reconciling a stale baseline JSON (no retrain)

If a `results/{dataset}_baseline.json` no longer matches the checkpoint on disk
(e.g. a better retrain replaced the .pth but the JSON wasn't refreshed),
regenerate it from the existing checkpoint — no retraining:

```bash
python -m scripts.eval_checkpoint --dataset pathmnist --compare   # preview old vs new
python -m scripts.eval_checkpoint --dataset pathmnist             # write it
python -m scripts.eval_checkpoint --all --compare                 # audit all 6
```

This runs the same no-TTA test evaluation Phase 1 used, preserves the training
metadata, and refreshes `test_metrics` + the reliability diagram.

### Variance direction (resolved)

The proposal's variance *formula* (`1/(var+ε)`) contradicts its *intuition*
("confident views dominate"). Per supervisor decision (M. Hafez), the headline
`variance` strategy is the **confidence-aligned** `w = var` (sharp views up), and
the literal `1/(var+ε)` is shipped as `variance_inv` and reported as the
"naive variance weighting is unstable" negative finding. ECE uses **10 bins**
everywhere (proposal §3.4) — generate all official numbers through
`run_weighted_tta` so every student's row is computed identically.

## Phase 4 — Ablations, Significance & Figures (Days 12–14)

Phase 4 is **analysis**. Most of it needs **no retraining** — it reads the saved
`predictions/*.npy` and `results/*_weighted_tta.csv`. The one exception is the
multi-seed stability study, which retrains.

### Who runs what

Phase 4 splits differently from Phase 3: each student owns **one dataset**, but
the cross-dataset analysis needs *everyone's* results, so it is run **once by S1**
after all six datasets are committed.

| Task | Who | Retrain? |
|---|---|---|
| Ablations (N-views + augmentation leave-one-out) on your dataset | **each student** | no |
| Multi-seed stability (3 seeds) | **S1 (Path), S5 (Pneumonia), S3 (Breast)** only | **yes** |
| Significance (McNemar + Wilcoxon) | **S1**, after all 6 datasets in | no |
| Reliability diagrams + analysis figures | **S1**, after all 6 datasets in | no |

So S2 (Derma) and S4 (OrganA): you run **only the two ablation commands** below.
S3 and S5: ablations **plus** the seed study for your dataset. S1: ablations for
Path + Blood, the Path seed study, and all the cross-dataset analysis at the end.

### Each student — ablations on your dataset (no retrain)

Run the ablation with the **strategy that won Phase 3 for your dataset** (the
"Best Strategy" column in the Sheet-4 ablation tab), via `--strategy`:

```bash
# S2 — DermaMNIST (entropy)
python -m scripts.ablate_n            --dataset dermamnist --strategy entropy
python -m scripts.ablate_augmentations --dataset dermamnist --strategy entropy
# S5 — PneumoniaMNIST (max-prob)
python -m scripts.ablate_n            --dataset pneumoniamnist --strategy maxprob
python -m scripts.ablate_augmentations --dataset pneumoniamnist --strategy maxprob
# S3 — BreastMNIST (max-prob)
python -m scripts.ablate_n            --dataset breastmnist --strategy maxprob
python -m scripts.ablate_augmentations --dataset breastmnist --strategy maxprob
# S4 — OrganAMNIST (variance)
python -m scripts.ablate_n            --dataset organamnist --strategy variance
python -m scripts.ablate_augmentations --dataset organamnist --strategy variance
# S1 — PathMNIST + BloodMNIST (entropy)
python -m scripts.ablate_n            --dataset pathmnist  --strategy entropy
python -m scripts.ablate_augmentations --dataset pathmnist  --strategy entropy
python -m scripts.ablate_n            --dataset bloodmnist --strategy entropy
python -m scripts.ablate_augmentations --dataset bloodmnist --strategy entropy
```

Default `--strategy` is `entropy`; pass the one that matches your Sheet-4 row.
Outputs (fill the ablation sheet directly from these):
  - `results/{dataset}_ablation_N.csv` — one row per N (acc/ECE/NLL) -> Sheet 4 (N-views)
  - `results/{dataset}_ablation_aug.csv` — leave-one-out per augmentation -> Sheet 5
  - `figures/ablation_N/`, `figures/ablation_aug/` — the plots

`ablate_augmentations` flags augmentations that are *harmful* (accuracy rises
when removed) — the per-modality safety story.

Commit your two ablation CSVs + the figures via PR.

### S3 / S5 / S1 — multi-seed stability (the ONLY retraining)

Train your dataset with 3 seeds into **tagged** checkpoints (canonical one is
never overwritten), evaluate each, then aggregate. Example for PathMNIST:

```bash
python -m scripts.train_baseline   --dataset pathmnist --seed 0   --ckpt-tag _seed0
python -m scripts.run_weighted_tta --dataset pathmnist --ckpt-tag _seed0 --no-time
python -m scripts.train_baseline   --dataset pathmnist --seed 42  --ckpt-tag _seed42
python -m scripts.run_weighted_tta --dataset pathmnist --ckpt-tag _seed42 --no-time
python -m scripts.train_baseline   --dataset pathmnist --seed 123 --ckpt-tag _seed123
python -m scripts.run_weighted_tta --dataset pathmnist --ckpt-tag _seed123 --no-time
python -m scripts.aggregate_seeds  --datasets pathmnist     # -> results/seed_stability.csv (mean ± std)
```

`--ckpt-tag` writes `checkpoints/{ds}_resnet18_seed0.pth` and
`results/{ds}_seed0_weighted_tta.csv`; the canonical (no-tag) files are untouched,
so the headline Phase 1–3 numbers stay valid and the seeds only add error bars.
Commit the aggregated `seed_stability.csv` (the per-seed weighted CSVs and
`predictions/` arrays stay out of git — Drive only).

### S1 only — cross-dataset analysis (after all 6 datasets are committed)

These need every dataset's results/predictions, so run them last, once:

```bash
python -m scripts.build_full_matrix              # merge per-dataset CSVs
python -m scripts.significance                   # McNemar (per ds) + Wilcoxon (across ds)
python -m scripts.make_reliability_diagrams      # figures/reliability/{ds}_{strategy}.pdf
python -m scripts.analysis_figures               # modality bars + latency tradeoff
```

McNemar -> `results/significance.csv`; Wilcoxon -> `results/significance_wilcoxon.csv`
(needs all 6 datasets for real power). Don't commit these until all 6 are in —
partial runs are misleading.

## Phase 5 additions — second backbone, Top-K TTA, one-command runs

Phase 5 adds an EfficientNet-B0 backbone, a Top-K TTA fusion family, an inference-time
benchmark, a runnable variance sanity check, and a unified runner — all without changing
the published ResNet-18 numbers. Full write-up: [`docs/PHASE5_ADDITIONS.md`](docs/PHASE5_ADDITIONS.md).
The short version:

```bash
# Every training/eval script now takes --arch {resnet18,effb0} (default resnet18,
# so all the commands above are unchanged). EfficientNet uses the same recipe.
python -m scripts.train_baseline   --dataset pathmnist --arch effb0 --ckpt-tag _seed42
python -m scripts.run_weighted_tta --dataset pathmnist --arch effb0 --ckpt-tag _seed42

# Top-K TTA (hard entropy filter: keep K lowest-entropy views, average) runs by
# default in run_weighted_tta as top3/top5/top7. Disable with --no-top-k.
python -m scripts.run_weighted_tta --dataset bloodmnist --top-k 5

# Prove the variance-weighting direction (VMV Task 2); exits 0 if code matches.
python -m scripts.variance_sanity_check

# Same-machine latency surface across arch x N x method -> results/inference_benchmark.csv
python -m scripts.benchmark_inference --datasets pathmnist --arch resnet18

# ONE command for the whole pipeline across backbones x datasets x seeds. Always
# preview with --dry-run first (no GPU/data needed); writes results/run_manifest.csv.
python -m scripts.run_all --models resnet18 effb0 --seeds 42 0 123 --phases all --dry-run
python -m scripts.run_all --models effb0 --seeds 0 42 123 --phases train weighted aggregate
```

Naming policy: ResNet-18 at the canonical seed keeps the exact original filenames; EfficientNet
and seed-tagged runs get a parallel `{ds}_effb0_*` / `{ds}_seedN_*` namespace so nothing collides
(this also retired the old "seed run clobbered the baseline JSON" gotcha). The confidence strips
(`make_confidence_strips --gold-k 5`) now gold-outline the Top-K kept views, tying Figure 1 to the
Top-K strategy.

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
│   ├── model.py           # ResNet-18 builder (optional dropout head)
│   ├── metrics.py         # Accuracy, AUC, ECE, NLL
│   ├── augmentations.py   # 10 TTA augmentation types (§3.3)
│   ├── tta.py             # per-view probs + fusion strategies (equal/maxprob/entropy/variance)
│   ├── mc_dropout.py      # MC Dropout baseline (forward-hook dropout, no retrain)
│   ├── temperature.py     # Temperature scaling: fit T on val, ts_only/ts_entropy
│   ├── perf.py            # Inference-time measurement (warmup + cuda.synchronize)
│   ├── evaluate.py        # tta_evaluate() + run_all_strategies() across all 5
│   ├── train.py           # Training & evaluation loops
│   ├── visualize.py       # reliability / curves / aug grid / confidence strip
│   └── utils.py           # Seeding, device, checkpoint I/O
├── scripts/
│   ├── train_baseline.py       # ← Phase 1
│   ├── run_standard_tta.py     # ← Phase 2 (+ inf_ms for Sheet 2)
│   ├── run_weighted_tta.py     # ← Phase 3: 8 strategies + TS + inf_ms + per-image preds
│   ├── make_confidence_strips.py  # ← Phase 3: Figure 1 strips (×3 images/modality)
│   ├── eval_checkpoint.py      # ← reconcile baseline JSON from a checkpoint (no retrain)
│   ├── build_full_matrix.py    # ← Phase 3: merge per-dataset CSVs -> full_matrix.csv
│   ├── significance.py         # ← Phase 4: McNemar + Wilcoxon (reads predictions/)
│   ├── make_reliability_diagrams.py  # ← Phase 4: calibration diagrams from saved probs
│   ├── analysis_figures.py     # ← Phase 4: modality bars + latency tradeoff
│   ├── ablate_n.py             # ← Phase 4: accuracy/ECE vs N
│   ├── ablate_augmentations.py # ← Phase 4: leave-one-out per augmentation
│   └── aggregate_seeds.py      # ← Phase 4: multi-seed mean ± std
├── tests/
│   ├── test_metrics.py    # metrics unit tests
│   ├── test_tta.py        # augmentation + equal-weight fusion tests
│   ├── test_fusion.py     # weighted fusion strategy tests (Phase 3)
│   ├── test_mc_dropout.py # MC Dropout tests (Phase 3)
│   ├── test_temperature.py # temperature scaling tests (Phase 3)
│   └── test_significance.py # McNemar test (Phase 4)
├── notebooks/
│   └── kaggle_baseline.ipynb  # Kaggle/Colab runner (Phases 1–3)
├── docs/                  # Proposal PDF + results tracker XLSX
├── checkpoints/           # gitignored — saved .pth files (~45 MB each)
├── results/               # commit JSON/CSV metrics here (small, in-repo)
├── predictions/           # gitignored — per-image .npy arrays (push to Drive)
├── figures/               # TTA curves, reliability diagrams, strips
├── requirements.txt
└── README.md              # this file
```

---

## Phases of the project (overview)

| Phase | Days | Deliverable | Code lives in |
|---|---|---|---|
| 1 — Baselines | 1–3 | 6 trained checkpoints + Sheet 1 filled | `scripts/train_baseline.py` |
| 2 — Standard TTA | 4–7 | `src/tta.py` + `src/augmentations.py`, Sheet 2 filled | `scripts/run_standard_tta.py` |
| 3 — Weighted TTA (+addendum) | 8–11 | 8 strategies + temperature scaling + inference time + per-image preds, Sheet 3 + `full_matrix.csv`, 12 confidence strips | `scripts/run_weighted_tta.py`, `scripts/make_confidence_strips.py` (**current**) |
| 4 — Ablations + stats | 12–14 | reliability diagrams, McNemar/Wilcoxon, ablations, multi-seed mean±std | `significance.py`, `analysis_figures.py`, `ablate_*.py`, `aggregate_seeds.py` (**current**) |
| 5 — Paper | 15–17 | Manuscript | — |

Each phase adds new modules under `src/` and `scripts/`. Existing files are
the stable contract Phase 2+ builds on — they should not change without an
explicit team decision.

---

## Troubleshooting

**Scripts hang for ages on "Fitting temperature / fusing..." (Windows)**
On Windows, DataLoader workers use *spawn*, so each worker reboots Python and
re-loads torch's CUDA DLLs — with several loaders open (val + test + TTA + MC
dropout + the timing re-runs), `run_weighted_tta` can stall for many minutes.
The scripts now default to `--num-workers 0` (single-process), which avoids it.
If you ever override it, keep workers at 0 on Windows; on Linux/Kaggle you can
use `--num-workers 2`–`4` for a speedup.

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

**EfficientNet `mc_dropout` shows ZERO spread in a quick test (no trained checkpoint)**
Expected, not a bug. An *untrained* EfficientNet-B0 (`pretrained=False`) outputs
all-zero penultimate features in eval mode (random init + eval-mode BatchNorm
with default running stats), so dropping them does nothing and MC Dropout has no
variation. ResNet-18's untrained features are non-zero, so it never shows this.
With a real trained checkpoint (or after a few train-mode forwards to populate
BatchNorm stats) the features are non-zero and MC Dropout works normally. Don't
debug a "broken" hook on an effb0 model that was never trained —
`tests/test_model_arch.py` pins this behaviour both ways.

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
