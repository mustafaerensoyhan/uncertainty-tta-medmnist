# Kaggle Setup Guide — Baseline Training + TTA (Phases 1–3)

For teammates running their assigned baseline on Kaggle's free GPU instead of locally. Total time from zero to results: **~15 minutes**, most of which is the training itself running unattended.

---

## Prerequisites

- A Kaggle account (free — sign up at <https://www.kaggle.com>)
- **Phone verification on Kaggle.** Required before you can enable GPU. Go to <https://www.kaggle.com/settings> → scroll to "Phone Verification" → enter your number → enter the SMS code. Takes 2 minutes.
- You've accepted the GitHub collaborator invite from @mustafaerensoyhan (check your `@ontariotechu.net` inbox)

---

## Step 1 — Get the repo onto your machine

You need a local copy of the repo before you start. Two ways — pick whichever:

### Option A — Download the ZIP from GitHub (easiest, no git needed)

1. Go to <https://github.com/mustafaerensoyhan/uncertainty-tta-medmnist>
2. Click the green **`<> Code`** button (top right of the file list)
3. At the bottom of the dropdown, click **Download ZIP**
4. Save the file — it'll be named `uncertainty-tta-medmnist-main.zip` (~5 MB)

This requires you to be logged into GitHub *in the browser* (the repo is private), but no extra auth setup. Keep the zip file somewhere you can find it — you'll upload it to Kaggle in Step 4.

### Option B — Git clone (if you'll also work locally)

```bash
git clone https://github.com/mustafaerensoyhan/uncertainty-tta-medmnist.git
```

If git asks for credentials, see the README's "Authentication failed" troubleshooting entry.

---

## Step 2 — Create a new notebook on Kaggle

1. Go to <https://www.kaggle.com/code>
2. Click the blue **+ New Notebook** button (top right)
3. A blank notebook opens. Delete the default cell — we won't use it.

---

## Step 3 — Turn on the GPU

This is the most-forgotten step. If you skip it, training takes hours instead of minutes.

1. On the right sidebar, find **Session options** (click the `>` arrow if collapsed)
2. Under **Accelerator**, click the dropdown (default is "None")
3. Select **GPU T4 x2** (or **GPU P100** if T4 isn't available — both work)
4. Click **Turn on GPU** if Kaggle prompts you to confirm

Kaggle gives you **30 hours of free GPU per week**. One training run uses 5–50 minutes depending on dataset, so you have plenty of headroom.

---

## Step 4 — Upload the repo zip to your Kaggle notebook

This is how the code gets into the Kaggle environment. Since the repo is private, plain `git clone` won't work from Kaggle — Kaggle doesn't have your GitHub credentials. The zip upload sidesteps that entirely.

1. On the right sidebar of the notebook, click **+ Add Input** (sometimes labeled "Add Data")
2. In the modal that opens, click the **Upload** tab (top)
3. Drag your `uncertainty-tta-medmnist-main.zip` from Step 1 into the drop zone (or click and browse)
4. Give it a title like `uncertainty-tta-repo` and click **Create**
5. After upload completes, click **Add** to attach it to your notebook

You should now see your dataset listed under **Input** in the right sidebar. It lives at `/kaggle/input/uncertainty-tta-repo/` (or whatever you titled it).

---

## Step 5 — Import the team notebook from the zip

Inside the zip you just uploaded is `notebooks/kaggle_baseline.ipynb`. We need to load it as the active notebook.

1. In your Kaggle notebook, paste this into a code cell and run it (Shift+Enter):

   ```python
   !find /kaggle/input -name "*.zip"
   ```

   This prints the full path to your uploaded zip. Copy that path.

2. In a new cell, paste this and replace `YOUR_ZIP_PATH` with the path you just copied:

   ```python
   import zipfile, shutil, os
   ZIP_PATH = "YOUR_ZIP_PATH"   # e.g. /kaggle/input/uncertainty-tta-repo/uncertainty-tta-medmnist-main.zip

   # Extract the repo into /kaggle/working
   with zipfile.ZipFile(ZIP_PATH) as z:
       z.extractall("/kaggle/working")

   # GitHub names the folder <repo>-<branch>; rename for convenience
   if os.path.isdir("/kaggle/working/uncertainty-tta-medmnist-main"):
       if os.path.isdir("/kaggle/working/uncertainty-tta-medmnist"):
           shutil.rmtree("/kaggle/working/uncertainty-tta-medmnist")
       shutil.move("/kaggle/working/uncertainty-tta-medmnist-main",
                   "/kaggle/working/uncertainty-tta-medmnist")
   print("✓ Repo extracted to /kaggle/working/uncertainty-tta-medmnist")
   !ls /kaggle/working/uncertainty-tta-medmnist
   ```

3. Run the cell. You should see `checkpoints docs figures notebooks ...` printed — confirming the repo is now in `/kaggle/working/`.

4. Now load the actual training notebook. Click **File → Import Notebook → Upload tab** and select `notebooks/kaggle_baseline.ipynb` from the local zip you downloaded in Step 1 (you may need to unzip it locally first to extract the `.ipynb` file). Click **Import**.

You should now see the full team notebook loaded with multiple cells (Verify GPU, Clone repo, Train, etc.).

> **Note for whoever modifies the notebook later:** The notebook's Cell 3 contains `git clone`, which fails for the private repo. Either delete Cell 3 or replace its contents with the extraction snippet from Step 5.2 above. You can also modify the line to point at your already-extracted folder with `%cd /kaggle/working/uncertainty-tta-medmnist`.

---

## Step 6 — Set your dataset in Cell 1

Find this cell near the top of the notebook:

```python
DATASET    = "dermamnist"   # ← change me
EPOCHS     = 30
BATCH_SIZE = 64
LR         = 1e-4
SEED       = 42
```

Change the `DATASET` value to your assignment:

| Student | DATASET value |
|---|---|
| Mohamed Ahmed (S2) | `"dermamnist"` |
| Trang (S3) | `"breastmnist"` |
| Vaidehi (S4) | `"organamnist"` |
| Sudha (S5) | `"pneumoniamnist"` |

Leave everything else (`EPOCHS`, `BATCH_SIZE`, `LR`, `SEED`) at the defaults — the proposal locked those values and the team needs apples-to-apples comparisons across the six datasets.

---

## Step 7 — Run the cells in order

Three ways to run cells in Kaggle:
- Click the **▶** triangle to the left of each cell, one at a time
- Press **Shift+Enter** with a cell selected (runs current cell, moves to next)
- **Run All** button at the top — runs every cell top-to-bottom (recommended for first run)

For your first run, go one cell at a time so you can spot issues early. Here's what each does:

### Cell 1 — CONFIG
Just sets variables. No output, runs instantly.

### Cell 2 — Verify GPU
Prints PyTorch version and CUDA status. If it says `⚠ No GPU detected`, you forgot Step 3 — enable the accelerator, restart the session (right sidebar → ⋮ → "Restart session"), and re-run.

### Cell 3 — Clone the repo and install dependencies
**This will fail on the private repo.** Replace its contents with the extraction snippet from Step 5.2, plus the `medmnist` install:

```python
import os
%cd /kaggle/working/uncertainty-tta-medmnist
!pip install -q medmnist==3.0.2
```

(If you've already run Step 5.2 in a separate cell, this just changes directory and installs medmnist. Takes ~30 seconds.)

### Cell 4 — Visual sanity check (optional but recommended)
Loads 6 training images and shows them with their class labels. Takes ~15 seconds because it downloads the dataset on first run. If the images look like noise or all-black, something's wrong — message Mustafa. If they look like reasonable medical images, you're good.

### Cell 5 — Train the baseline (this is the long one)
Runs the full 30-epoch training. Expected times by dataset on Kaggle T4:

| Dataset | Approx. time |
|---|---|
| BreastMNIST | ~1 minute |
| PneumoniaMNIST | ~3 minutes |
| DermaMNIST | ~4 minutes |
| BloodMNIST | ~8 minutes |
| OrganAMNIST | ~20 minutes |
| PathMNIST | ~50 minutes |

You'll see one line per epoch showing `train_loss`, `val_loss`, `val_acc`, `val_ece`. After epoch 30, the script loads the best checkpoint, evaluates on the test set, and prints a `📊 Tracker row` box. **Copy the numbers from that box** — they go into the shared spreadsheet.

The script also prints either `✓ within ±2% tolerance` or `⚠ OUTSIDE ±2% tolerance` based on the published MedMNIST benchmark. If it's the second one, re-run with `SEED = 7` or `SEED = 0` in Cell 1 — sometimes the seed is just unlucky. If three different seeds all fail, message Mustafa.

### Cells 6, 7, 8, 9 — Display saved artifacts
Quick cells that show you the JSON metrics, the training curves plot, the reliability diagram, and the per-epoch CSV inline. The training curves should show `train_loss` going down and `val_acc` going up over the 30 epochs. If `val_acc` plateaus very early or both curves are flat, training failed.

### Cell 10 — Download cell (Colab only)
Leave commented out if you're on Kaggle. On Kaggle, your output files are accessible from the right sidebar's **Output** tab.

---

## Step 8 — Download your artifacts

Kaggle saves everything under `/kaggle/working/uncertainty-tta-medmnist/`. To download:

1. On the right sidebar of the notebook, click the **Output** tab (folder icon)
2. Navigate into `uncertainty-tta-medmnist/`
3. Download these **five files**:
   - `results/{dataset}_baseline.json` — your test metrics
   - `results/{dataset}_train_log.csv` — per-epoch training history
   - `figures/reliability/{dataset}_baseline.png` — calibration plot
   - `figures/curves/{dataset}_train_curves.png` — training curves
   - `checkpoints/{dataset}_resnet18.pth` — the trained model (~45 MB)
4. For each: click the file → three-dot menu → **Download**

Replace `{dataset}` with your dataset key, e.g. `dermamnist_baseline.json`.

---

## Step 9 — Submit your results

Three places things go:

### A) Spreadsheet
Open the shared `TTA_Results_Tracker.xlsx` (Google Sheets version), go to **Sheet 1️⃣ Baselines**, find your row, paste the values from the printed tracker box:
- Accuracy (%)
- AUC-ROC
- ECE
- NLL
- Checkpoint path: `checkpoints/{dataset}_resnet18.pth`

### B) Shared Drive
Upload your `.pth` checkpoint file to the team's shared Drive folder (Mustafa will share the link in chat). Checkpoints are gitignored — they're too large for the repo.

### C) GitHub PR
This is the Phase 1 deliverable on GitHub. **You need a local clone of the repo to push** — Kaggle's filesystem doesn't have git auth configured. On your local PC:

```bash
# In your local clone of the repo
git checkout main
git pull
git checkout -b s2-dermamnist-baseline   # adjust to your S-id + dataset
```

Copy the two files you downloaded (`{dataset}_baseline.json` and `{dataset}_train_log.csv`) into the `results/` folder of your local clone, then:

```bash
git add results/dermamnist_baseline.json results/dermamnist_train_log.csv
git commit -m "S2: dermamnist baseline — XX.X% test acc"
git push -u origin s2-dermamnist-baseline
```

Then on GitHub, click **Compare & pull request**, add a one-line description (mention your test accuracy and whether it passed the ±2% benchmark), click **Create pull request**. Mustafa will review and merge.

**If you don't have a local git setup**: send Mustafa your `.json` and `.csv` files via the team chat — he'll push them on your behalf for the first time. Set up local git properly before Phase 2 starts.

---

## Running Phase 2 and Phase 3 in the same notebook

The notebook continues past the Phase 1 baseline into the TTA phases — same
`DATASET` config, no extra setup. After your baseline cells finish, just keep
running the cells top to bottom:

- **Sections 9–11 (Phase 2 — Standard TTA):** runs equal-weight TTA at
  N=5,10,20,50 and shows the accuracy-vs-N plot. Writes
  `results/{DATASET}_standard_tta.csv`. → paste into **Sheet 2️⃣**.
- **Sections 12–13 (Phase 3 — Weighted TTA):** runs all five strategies
  (baseline, max-prob, entropy, variance, MC Dropout) and prints the five-row
  table. Writes `results/{DATASET}_weighted_tta.csv`. → paste into **Sheet 3️⃣**.
- **Section 14 (Figure 1 strip):** for `pathmnist`, `dermamnist`,
  `pneumoniamnist`, or `bloodmnist` only, generates
  `figures/strip/{DATASET}_strip.pdf`. Other datasets skip this automatically.
- **Section 15 (Download):** the Colab download cell now includes the Phase 2
  and Phase 3 outputs too.

All TTA phases reuse your Phase 1 checkpoint at
`checkpoints/{DATASET}_resnet18.pth` — if you restarted the Kaggle session and
lost it, re-run the training cell (or upload the checkpoint from the shared
Drive) before running the TTA sections. None of this needs a GPU reconfigure;
TTA inference is much faster than training.

**Submitting Phase 3:** commit your `results/{DATASET}_weighted_tta.csv` via the
branch + PR workflow (same as Phase 1). Do **not** commit a shared
`full_matrix.csv` — S1 builds that on `main` with
`python -m scripts.build_full_matrix` after everyone's per-dataset CSV is merged,
so your runs never collide on one file.

---

## Common gotchas

**"My session disconnected mid-training"**
Kaggle disconnects free sessions after about 9 hours of inactivity in the browser. Keep the tab open, don't close your laptop lid. If it disconnects mid-training, the checkpoint is gone — you'll need to re-run. For datasets under 10 minutes this is rarely a problem.

**"Can I close my browser and come back?"**
Kaggle has "Save & Run All (Commit)" which runs the notebook in the background — but it requires the notebook to be in a Saved state and consumes your GPU quota the whole time. For Phase 1, it's simpler to just keep the tab open and watch.

**"I want to re-run with a different seed"**
Change `SEED = 42` to `SEED = 0` or `SEED = 7` in Cell 1, then **Run All** again from the top. Or use **Restart & Run All** to fully reset the session.

**"My accuracy is way below the benchmark band"**
Most common cause: GPU wasn't actually on (check Cell 2's output). Second: the dataset download was interrupted — delete `/kaggle/working/uncertainty-tta-medmnist/data/` and re-run Cell 5. Third: bad luck on the random seed — try 0, 7, or 13.

**"Kaggle says I've used up my GPU quota"**
30 hours per week, resets every Saturday. If you're out, switch to Google Colab (the same notebook works there — Runtime → Change runtime type → T4 GPU).

**"Git clone in Cell 3 fails with authentication error"**
Expected — the repo is private. Replace Cell 3's contents with the extraction snippet from Step 5.2.

**"The PR step is confusing because I trained on Kaggle"**
Correct — you still need a local clone of the repo to push. Either set up git locally (one-time), or send the files to Mustafa via chat for the first round.

---

## TL;DR for sharing in chat

> 1. Download repo zip from GitHub: **`<> Code`** button → **Download ZIP** (you must be logged in — repo is private)
> 2. Sign in to Kaggle, verify phone for GPU access
> 3. **New Notebook** → right sidebar → enable **GPU T4 x2**
> 4. **+ Add Input** → **Upload** tab → drag the repo zip → **Create** → **Add**
> 5. **File → Import Notebook → Upload** → choose `notebooks/kaggle_baseline.ipynb` from the unzipped repo locally
> 6. Replace Cell 3 with the extraction snippet from the setup guide's Step 5.2
> 7. In Cell 1, set `DATASET = "..."` to your assignment (`dermamnist` / `breastmnist` / `organamnist` / `pneumoniamnist`)
> 8. **Run All** — wait 1 to 20 minutes depending on dataset
> 9. Copy the `📊 Tracker row` numbers into Sheet 1️⃣ Baselines
> 10. Download `results/{dataset}_baseline.json`, `results/{dataset}_train_log.csv`, and the `.pth` checkpoint from the Output panel
> 11. Upload the `.pth` to shared Drive; PR the two `results/` files via your local clone (or send them to Mustafa to push)
