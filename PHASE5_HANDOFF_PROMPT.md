# Project Handoff — Uncertainty-Weighted TTA for Medical Image Classification

Paste this whole document into the new chat as your opening message. It brings a fresh
assistant fully up to speed on what's done. After it confirms it understands, I'll give it
the new additions and tasks.

> **Update (Phase 5 additions folded in).** Section 11 below records the EfficientNet-B0 +
> Top-K + tooling work that has now been *implemented in code* (Mustafa / VMV "Implementer 1").
> The runs themselves still need a GPU pass. Read §11 last — §§1–10 are the original handoff and
> still describe the published ResNet-18 results that the paper's numbers come from.

---

## 1. Who I am and what this project is

I'm Mustafa Eren Soyhan (student "S1" and the GitHub repo admin). I'm working on a 5-student
research project supervised by **Mohamed Hafez**, who also proposed it. We're writing a paper
for **VMV** (a visual computing venue), so the framing matters: the contribution has to read as
a *visual computing* result, not just an ML benchmark.

**Topic:** Uncertainty-weighted Test-Time Augmentation (TTA) for medical image classification,
evaluated across 6 MedMNIST v2 datasets. It builds on the April 2026 paper
*"I Can't Believe TTA Is Not Better"*, which showed that standard equal-weight TTA often *hurts*
medical classifiers. Our angle: weight each augmented view by the model's confidence in it
(entropy / max-prob / variance weighting), plus temperature scaling, and measure both accuracy
and **calibration (ECE)**.

**THE HEADLINE FINDING (this drives everything in the paper):**
Uncertainty weighting does **not** produce statistically significant accuracy gains. The 3-seed
study confirms baseline accuracy wobbles ±0.89pp (Path) to ±0.98pp (Breast) across seeds, so the
~0.1–0.2pp TTA "gains" are inside the seed noise. **The paper's claim rests on CALIBRATION, not
accuracy** — uncertainty weighting + temperature scaling act as a calibration safety net that
consistently lowers ECE without the accuracy risk standard TTA carries. Across-dataset Wilcoxon
(best strategy vs standard TTA): accuracy p=0.1250, ECE p=0.6875 — both non-significant, which is
itself an honest, defensible result.

---

## 2. My setup

- **OS / shell:** Windows 11, PowerShell, Python venv (`.venv`)
- **GPU:** RTX 4080 Mobile (CUDA). Some teammates run on Kaggle/Colab (filesystem wipes between sessions there).
- **Local repo:** `C:\Users\mustafaerensoyhan\Downloads\uncertainty-tta-medmnist`
- **GitHub (private):** `https://github.com/mustafaerensoyhan/uncertainty-tta-medmnist`
  I push to `main` directly with admin bypass (commits show "Bypassed rule violations").
- **Stack:** PyTorch 2.x, torchvision, medmnist, scikit-learn, matplotlib, pandas, numpy
- **Model / training:** ResNet-18 (ImageNet-pretrained), 64×64 input, lr=1e-4, 30 epochs, batch=64, Adam.

---

## 3. Team and dataset assignment (config.py is the source of truth)

- **S1 = me (Mustafa):** PathMNIST + BloodMNIST
- **S2 = Mohamed Abdel Sattar:** DermaMNIST
- **S3 = Trang:** BreastMNIST
- **S4 = Vaidehi:** OrganAMNIST
- **S5 = Sudha:** PneumoniaMNIST

(The original proposal PDF lists a slightly different assignment, but `config.py` is what the code
uses. In practice I ended up running the Blood and Breast seed studies myself.)

The 6 datasets: PathMNIST (colon pathology, 9-class), DermaMNIST (dermatoscope, 7-class),
PneumoniaMNIST (chest X-ray, binary), BreastMNIST (breast ultrasound, binary, tiny — 546 train /
156 test), BloodMNIST (blood cell microscopy, 8-class), OrganAMNIST (abdominal CT, 11-class).

---

## 4. Repo structure and key code

**`src/`:** `config.py` (assignments, dataset keys), `data.py`, `model.py`, `metrics.py`
(`compute_all_metrics` → accuracy/auc_roc/ece/nll; ECE uses 10 bins), `train.py`,
`augmentations.py` (`get_augmentation_pipeline(n_views, seed, include_original=True)`;
N=2 = [original, vflip], N=5 adds gaussian_noise/brightness/color_jitter), `tta.py` (fusion
functions), `mc_dropout.py`, `temperature.py` (LBFGS temperature fit on val), `perf.py`
(warmup + cuda.synchronize timing), `evaluate.py` (`run_all_strategies`), `visualize.py`.

**8 core strategies (+ Top-K):** `baseline` (equal weight 1/N), `maxprob` (w=max p_i),
`entropy` (w=exp(−H)), `variance` (w=var(p_i) across class axis — confidence-aligned),
`variance_inv` (w=1/(var+ε) — the literal proposal formula, kept as a negative-finding ablation),
`mc_dropout` (T=20 stochastic passes), `ts_only` (temperature-scaled softmax), `ts_entropy`
(TS + entropy weighting). **Phase 5 adds `top3`/`top5`/`top7`** — a HARD entropy filter: keep only
the K lowest-entropy views and average them (still on the simplex). They reuse the cached per-view
probabilities, so they cost the same as full TTA at the same N (see §11).

**Important on `fuse_variance`:** it's `per_view_probs.var(axis=2)` — variance across the CLASS
axis, so confident/peaked views get MORE weight. This is the confidence-aligned direction Hafez
approved; the literal `1/(var+ε)` is `variance_inv`, reported as the "naive variance is unstable"
negative finding. The formula has been verified correct (not a bug — see §8), and Phase 5 adds a
runnable `scripts/variance_sanity_check.py` that prints the proof and guards it as a regression.

**Scripts (`scripts/`):** `train_baseline`, `run_standard_tta`, `run_weighted_tta`,
`build_full_matrix`, `eval_checkpoint`, `significance` (has `--best-per-dataset` mode),
`aggregate_seeds`, `make_reliability_diagrams`, `make_confidence_strips`, `analysis_figures`,
`ablate_n`, `ablate_augmentations`, `verify_strategies`. **Phase 5 adds** `variance_sanity_check`,
`benchmark_inference` (latency surface across arch×N×method), and `run_all` (one command that runs
every phase for every backbone×dataset×seed and writes a runtime manifest).
All scripts default to `--num-workers 0` (Windows spawn fix). `--no-time` skips timing.
`--ckpt-tag _seedN` selects a tagged checkpoint. **New: `--arch {resnet18,effb0}`** on every
training/eval script (default `resnet18`, so all existing commands are unchanged).

---

## 5. What's DONE (Phases 1–4 complete)

- **Phase 1–2:** All 6 baselines trained and validated against MedMNIST benchmarks. Standard
  (equal-weight) TTA run at N=5/10/20/50 — confirmed it degrades/doesn't help, replicating prior work.
- **Phase 3:** All 8 strategies implemented and run on all 6 datasets at N=10. Full results matrix done.
- **Phase 4:** Ablations (effect of N; leave-one-out augmentation), reliability diagrams (30),
  inference-time analysis (same-machine), statistical significance (McNemar exact per dataset +
  across-dataset Wilcoxon), and the **3-seed stability study** (the big recent task).
- **VMV visual figure** (Augmentation Confidence Strip, the paper's Figure 1) — spec exists; this
  is the visual-computing anchor for VMV.

**Seed study (just finished, all 6 datasets × 3 seeds = 0/42/123):**
`results/seed_stability.csv` has 48 rows (6 datasets × 8 strategies) with mean±std for
accuracy/auc/ece/nll. Per-dataset `{ds}_seed_stability.csv` files also committed.

Current mean±std headline numbers (baseline → best non-baseline accuracy; lowest ECE strategy):

| Dataset | Baseline acc | Best acc strategy | Baseline ECE | Lowest ECE |
|---|---|---|---|---|
| PathMNIST | 92.00 ± 0.89 | ts_entropy 92.12 ± 0.85 | 0.0251 | ts_entropy 0.0154 |
| DermaMNIST | 83.24 ± 1.08 | entropy 83.28 ± 0.97 | 0.0739 | variance_inv 0.0582 |
| PneumoniaMNIST | 85.26 ± 0.70 | ts_only 86.49 ± 0.81 | 0.0957 | variance_inv 0.0876 |
| BreastMNIST | 88.68 ± 0.98 | variance 89.53 ± 0.37 | 0.0376 | baseline 0.0376 |
| BloodMNIST | 98.25 ± 0.05 | entropy 98.33 ± 0.03 | 0.0098 | entropy 0.0060 |
| OrganAMNIST | 95.21 ± 0.36 | maxprob 95.23 ± 0.37 | 0.0577 | ts_only 0.0290 |

Note the accuracy "wins" are all within the ± noise — that's the point. ECE improvements
(e.g. Path 0.0251→0.0154, Organa 0.0577→0.0290, Blood 0.0098→0.0060) are the real, consistent story.

---

## 6. The Excel tracker (Google Sheet) — current state

The working file is **`TTA_Results_Tracker_meanstd_FINAL.xlsx`** (I'll share it with you). Sheets:

- 📋 Guide
- 1️⃣ Baselines
- 2️⃣ Standard TTA
- ⭐ **Weighted TTA (mean±std)** — the NEW primary results sheet: all 6 datasets × 8 strategies,
  acc/auc/ece/nll shown as "mean ± std", best non-baseline accuracy bolded per dataset.
- 3️⃣ **Weighted TTA (seed 42)** — the old single-seed sheet, renamed. This is the "separate sheet"
  Hafez asked to keep the raw single-seed numbers in.
- 4️⃣ Ablation N
- 5️⃣ Ablation Augments
- 6️⃣ Inference Time (same-machine timing; N=1 ~0.2ms → N=50 ~44ms; augmentation generation
  dominates latency)
- 7️⃣ Statistical Tests — best-per-dataset: Best Strategy, Test Used (McNemar exact), p-values,
  interpretations. The B13 summary note cites the seed-noise error bars backing the calibration claim.
- 📊 Paper Summary

**Hafez's restructure request was:** "replace all results with mean±std as the one to be used,
leave the single-seed results in a separate sheet." This is DONE for the Weighted TTA results
(primary mean±std sheet + renamed single-seed sheet).

**STILL OPEN — mean±std on Sheets 2, 4, 5:** Hafez said "every column except inference and the
stat-tests sheet," which includes Standard TTA + both ablation sheets. **The data for these does
NOT exist yet** — the seed study only re-ran Phase 3 weighted TTA, not Phase 2 standard TTA or the
ablations. To fill them would require re-running `run_standard_tta` / `ablate_n` /
`ablate_augmentations` per seed (~50 inference runs; checkpoints already exist, no retraining) AND
extending `aggregate_seeds` to aggregate those CSV types. My recommendation: Sheet 5 (Ablation
Augments) is the highest value to seed-average, because the leave-one-out deltas (~±0.1pp) are
likely within seed noise — averaging would reveal which "harmful augmentation" findings are real.
This is a pending decision with Hafez.

---

## 7. What's in GitHub vs not

**Pushed to GitHub (recent commits):**
- `86f8502` — significance `--best-per-dataset` mode
- `31c3c98` — "Complete seed study: all 6 datasets (3 seeds each)" — all per-dataset
  `*_seed_stability.csv` + combined `seed_stability.csv`
- earlier: same-machine timing CSVs, `verify_strategies.py`

**NOT in git (kept on shared Google Drive instead):** per-image predictions/probabilities
(`predictions/` folder), and model checkpoints (too large for the repo).

**Git hygiene gotchas (recurring):**
- `.gitignore` has broad `results/*.csv` and `figures/*.png` rules; we added `!` allow-rules for
  result/figure files. If a needed file is blocked, use `git add -f`.
- The shared combined `seed_stability.csv` repeatedly hit merge collisions — resolve with
  `git checkout --theirs results/seed_stability.csv` then regenerate from the per-dataset files via
  `aggregate_seeds`.
- `train_baseline --seed N` USED to overwrite the untagged `{ds}_baseline.json` — **Phase 5 fixed
  this**: tagged/effb0 runs now write namespaced stems (`{ds}_seedN_*`, `{ds}_effb0_*`) via the new
  `result_stem` helper, so seed/effb0 runs no longer touch the canonical files. No more `git restore`
  dance after seed runs.
- Leave these byproducts untracked: `figures/curves/*_train_curves.png`,
  `figures/reliability/*_baseline.png`.
- The `--ckpt-tag` trick for seeds: the canonical untagged ResNet-18 checkpoint must be copied to a
  `_seed42` tag and re-evaluated so `aggregate_seeds` counts it as the 3rd seed (it only globs
  `*_seedN_weighted_tta.csv`). For EfficientNet there is no untagged canonical — every effb0 seed
  (including 42) is tagged, and `aggregate_seeds --arch effb0` writes `effb0_seed_stability.csv`.

---

## 8. Hafez's review comments — all resolved

1. **p=1 in significance tests** — not a bug; McNemar correctly gives p≈1 for tiny/balanced
   discordant pairs (verified from arrays; some strategies reweight without flipping predictions).
2. **best-per-dataset significance mode** — added to `significance.py`.
3. **Derma 84.49 vs 83.34 discrepancy** — was a cross-machine/checkpoint issue (S2 ran phases on
   different setups), not a code bug; resolved, both now 83.34 at N=10.
4. **Blood Phase-2 NLL==ECE** — fixed in tracker.
5. **OrganA N=2 (72%) and Path N=2 ECE anomalies** — real deterministic small-N artifacts (N=2 =
   original + one vertical flip; vflip wrecks CT). Report ablations from N=5 up; footnote N=2.
6. **mean±std for all 6 datasets** — done (the seed study above).
7. **"Next phase": redo the whole procedure on 2 extra models + more datasets + 1–2 more metrics**
   — NOT started. This is part of what I'll task you with.

**Variance-on-Breast investigation (just done):** Hafez thought Breast variance looked wrong.
Verified `fuse_variance` is correct. The oddity is a binary + tiny-test-set artifact: for 2 classes,
`var([p,1−p]) = (p−0.5)²` exactly, so variance weighting becomes *squared-confidence* weighting —
far more extreme than max-prob. On 156 test images the +0.85pp "win" is ~1–2 images, and variance
actually makes ECE worse (0.061 vs 0.038). Not an environment issue (I ran all 3 seeds on my
machine in one session). Plan: footnote variance as unstable on binary tasks.

---

## 9. Where we are now: Phase 5 (paper writing)

Phase 4 is complete. We're moving into **Phase 5 — writing and submitting the manuscript** (VMV,
~8 pages excluding references, 15–25 BibTeX refs). Proposed section ownership:
Abstract/Intro/Conclusion = Supervisor; Related Work = S1 (me); Methodology = S2; Datasets & Setup
= S3; Results = S4 + S5; Discussion = S5.

**The reframing that should run through the whole paper:** standard TTA is unpredictable and our
uncertainty-weighted accuracy gains are not statistically significant — so we lead with
**calibration**. Uncertainty weighting + temperature scaling is a calibration safety net that
reliably lowers ECE without standard TTA's accuracy risk. The VMV visual hook is the Augmentation
Confidence Strip (Figure 1), which visually shows which augmented views the model trusts per modality.

---

## 10. What I'm sharing with you + what I'll ask next

Alongside this message I'm sharing:
- `TTA_Results_Tracker_meanstd_FINAL.xlsx` — the full results tracker
- `seed_stability.csv` — the combined 3-seed mean±std table (all 6 datasets)
- (and I can paste any `src/` code file or the research proposal PDF on request)

**Please confirm you've understood the project state above.** After that I'll give you the new
additions and the specific tasks I want done for Phase 5 (and possibly Hafez's "next phase" with
extra models/datasets). Don't start work until I give those — for now just read in and tell me
back, in your own words, (a) the headline finding, (b) what's done vs pending, and (c) anything in
here that looks inconsistent or that you'd want clarified before we start.

---

## 11. Phase 5 additions now IMPLEMENTED in code (Mustafa / VMV "Implementer 1")

Everything in this section is written, smoke-tested, and unit-tested (95 tests pass), but the
*runs* still need a GPU pass — there were no trained EfficientNet checkpoints and no GPU when this
was implemented. None of it changes the published ResNet-18 numbers in §§5–6.

**(1) EfficientNet-B0 as a second backbone.** `src/model.py` now has `build_efficientnet_b0(...)`
and a `build_model(arch, ...)` factory (`ARCHITECTURES = {resnet18, effb0}`). Every training/eval
script takes `--arch` (default `resnet18`). Same hyperparameters as ResNet-18 (lr=1e-4, 30 epochs,
64×64, Adam). To produce the EfficientNet results for all 6 datasets × 3 seeds:
`python -m scripts.run_all --models effb0 --seeds 0 42 123 --phases train weighted aggregate`.

**(2) Backbone-aware, collision-free file naming** (`src/utils.py`): `checkpoint_filename`,
`result_stem`, `default_ckpt_tag`. Policy: **ResNet-18 at the canonical seed keeps the exact
original filenames** (so nothing in Phases 1–4 breaks); EfficientNet and seed-tagged runs live in a
parallel namespace (`{ds}_effb0_*`, `{ds}_seedN_*`). This is also what removed the §7 baseline-JSON
clobber gotcha.

**(3) Top-K TTA fusion** (`src/tta.py`): `top3`/`top5`/`top7` keep only the K lowest-entropy views
and average them. `run_weighted_tta` runs them by default (`--top-k 3 5 7`, or `--no-top-k`); they
reuse the cached per-view probs so they're ~free on top of full TTA. The CSV gains an `arch` column.

**(4) Variance sanity check** (`scripts/variance_sanity_check.py`, VMV Task 2): prints the proof
that the literal `w=1/(var+ε)` is backward (rewards the *uncertain* view) and confirms the repo's
`w=var` is the confidence-aligned choice — then regression-guards the real `fuse_variance` /
`fuse_variance_inv`. Runs in <1s on CPU, exit 0 = code matches the decision. **No code change was
needed** — this just makes the §8 finding runnable for Writer 3 (Methodology/Discussion).

**(5) Confidence-strip gold outline** (`scripts/make_confidence_strips.py --gold-k 5`): the Top-K
kept (lowest-entropy) bars in Figure 1 now get a gold outline, visually tying Top-K TTA to the
paper's hero figure at zero extra cost.

**(6) Inference-time everywhere** (`scripts/benchmark_inference.py` + the existing `inf_ms` in
`run_weighted_tta`): a clean latency surface across backbone × N × method on one machine, written to
`results/inference_benchmark.csv` (cols: arch, dataset, method, n_views, ms_per_image). Every soft
strategy AND every Top-K share one `tta_all_strategies` row per N (fusion is microseconds), so the
table stays comparable as future additions land. **For the tracker:** the mean±std sheet only needs
ONE inference column sourced from a single timed run (timing is deterministic — no need to
seed-average it); paste the benchmark rows into Sheet 6.

**(7) Unified runner** (`scripts/run_all.py`): `for model: for dataset: for seed:` runs train →
weighted (per cell), standard/ablate (canonical seed), aggregate/strips/benchmark (per backbone),
and matrix/significance/reliability/analysis (once). It drives the existing scripts as subprocesses
and writes `results/run_manifest.csv` with the wall-clock of every step — the GPU-runtime comparison
the plan asked for. Always preview with `--dry-run` first; `--skip-existing` skips training when a
checkpoint exists; `--continue-on-error` pushes through failures.

**TESTING GOTCHA worth knowing (cost me an afternoon, not a bug):** an *untrained*
EfficientNet-B0 (`pretrained=False`) outputs **all-zero penultimate features** in eval mode (random
init + eval-mode BatchNorm with default running stats), so MC Dropout on its head shows ZERO
across-pass variation. That looks exactly like a broken MC-Dropout hook but isn't — ResNet-18's
untrained features are non-zero so it never shows this. On a real trained checkpoint (or after a few
train-mode forwards to populate BatchNorm stats) the features are non-zero and MC Dropout produces
real spread. So: **don't panic if an effb0 smoke test without a checkpoint shows mc_dropout ≈ 0** —
it's expected. There's a regression test (`tests/test_model_arch.py`) that pins both the gotcha and
the working-with-features behaviour.

**What still needs doing (not code — compute + decisions):**
- Run EfficientNet-B0 on all 6 datasets × 3 seeds on the GPU; paste into the tracker's effb0 block.
- Run `benchmark_inference` on the GPU for both backbones; fill Sheet 6.
- The §6 open item (mean±std on Sheets 2/4/5) is unchanged — still a decision with Hafez.
- Reconcile two small inconsistencies a fresh reader will spot: the README team table calls S2
  "Mohamed Ahmed" while this handoff calls S2 "Mohamed Abdel Sattar"; and `config.py` is the source
  of truth for dataset ownership (S1 = path+blood) over the proposal PDF.
