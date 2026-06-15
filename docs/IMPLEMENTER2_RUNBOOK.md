# Implementer 2 Runbook (VMV) — Bootstrap CIs & Paper Figures

Owner: Mohamed (Implementer 2). Scope: the calibration-focused statistics and
figures from `VMV_Team_Plan_FINAL.pdf`. **Headline = calibration (ECE), not
accuracy.** Nothing here adds datasets, methods, or claims; it only summarises
results already in the repo.

## What was added / changed

| File | Status | Purpose |
|------|--------|---------|
| `src/metrics.py` | extended | Added `bootstrap_ece_ci()` and `_ece_from_conf()`. Also made the `torch` import in `_to_numpy()` optional so the metric runs without the full ML stack. Existing functions unchanged. |
| `scripts/bootstrap_ece_ci.py` | **new** | Deliverable 1 — bootstrap ECE 95% CIs → `results/bootstrap_ece_ci.csv`. |
| `scripts/make_vmv_figures.py` | **new** | Deliverables 2–5 — Fig 2 (ECE bars), Fig 3 (reliability), Fig 4 (aug heatmap), Fig 5 (mechanism scatter). |
| `tests/test_vmv_implementer2.py` | **new** | Deliverable 7 — synthetic bootstrap test + import/helper tests; no MedMNIST data needed. |
| `docs/IMPLEMENTER2_RUNBOOK.md` | **new** | This file. |

Existing scripts (`make_reliability_diagrams.py`, `significance.py`,
`aggregate_seeds.py`, …) were **not** modified.

## Required input files

These are produced by the existing pipeline (`train_baseline.py`,
`run_weighted_tta.py`, `aggregate_seeds.py`, `ablate_augmentations.py`).

| Output | Needs |
|--------|-------|
| `results/bootstrap_ece_ci.csv` | `predictions/{ds}_labels.npy` + `predictions/{ds}_{strategy}_probs.npy` — or the multi-seed form `predictions/{ds}_seed{SEED}_*.npy` (the script tries the flat name first, then `--seed`, default 42). |
| `figures/fig2_ece.pdf` | `results/seed_stability.csv`; optional `results/{ds}_baseline.json` (no-TTA dashed line). |
| `figures/fig5_mechanism.pdf` | `results/seed_stability.csv`; optional `results/effb0_seed_stability.csv` (hollow EfficientNet-B0 overlay). |
| `figures/fig4_heatmap.pdf` | `results/{ds}_ablation_aug.csv` for all 6 datasets. |
| `figures/fig3_reliability.pdf` | `predictions/{path,blood}[_seed{SEED}]_{baseline,entropy}_probs.npy` + `_labels.npy`. |

If an input is missing, every script prints the exact file it looked for and
the command that generates it, rather than inventing numbers.

## Exact commands

Run from the repo root.

```bash
# 1. Bootstrap ECE CIs (2000 resamples, 10 bins) -> results/bootstrap_ece_ci.csv
python -m scripts.bootstrap_ece_ci
#    options: --datasets dermamnist --seed 42 --n-boot 2000 --n-bins 10

# 2-5. All four figures at 300 DPI (ResNet-18 by default)
python -m scripts.make_vmv_figures --figure all
#    EfficientNet-B0 variants (Fig 2/3/4); Fig 5 always overlays both backbones:
python -m scripts.make_vmv_figures --figure all --arch effb0 --allow-partial
#    both backbones in one go:
python -m scripts.make_vmv_figures --figure all --arch both --allow-partial
#    or one figure at a time:
python -m scripts.make_vmv_figures --figure 2 --arch effb0   # effb0 Fig 2 includes the top5 bar
python -m scripts.make_vmv_figures --figure 5
python -m scripts.make_vmv_figures --figure 4               # fails clearly if 6x10 incomplete
python -m scripts.make_vmv_figures --figure 4 --allow-partial  # render available rows, blanks for missing
python -m scripts.make_vmv_figures --figure 3 --seed 42

# 7. Tests (no MedMNIST data / checkpoints required)
pytest tests/test_vmv_implementer2.py -v
```

## Per-backbone figures

Fig 2/3/4 are produced per backbone via `--arch {resnet18,effb0,both}`.
ResNet-18 keeps the archless filename; EfficientNet-B0 gets an `_effb0` suffix
(`fig2_ece_effb0.pdf`, `fig4_heatmap_effb0.pdf`, …). Fig 5 already overlays both
backbones and is arch-independent.

**Top-K(5):** the plan's `top5` bar appears in **EfficientNet-B0** Fig 2
(top3/top5/top7 exist in `effb0_seed_stability.csv`). ResNet-18 was **never run
with Top-K**, so its `seed_stability.csv` has only the 8 core strategies and Fig 2
omits `top5` — adding it there requires re-running `run_weighted_tta` with the
Top-K columns enabled for all ResNet-18 seeds, not a figure change.

## Expected outputs

- `results/bootstrap_ece_ci.csv` — columns: `dataset, strategy, ece, ci_low,
  ci_high, baseline_ci_low, baseline_ci_high, significant_vs_baseline`.
  `significant_vs_baseline` is `True` iff the strategy CI and the baseline CI on
  that dataset do **not** overlap.
- `figures/fig2_ece.pdf` (+ `_effb0`) — 6 dataset groups (5 for effb0; no Organ),
  bars per strategy (baseline, entropy, maxprob, variance, ts_entropy; plus
  `top5` for effb0), error bars = 3-seed std, dashed no-TTA line per group.
- `figures/fig5_mechanism.pdf` — x = #classes, y = baseline ECE − entropy ECE
  (positive = entropy helped). DermaMNIST is annotated as the color-augmentation
  exception; EfficientNet-B0 overlaid as hollow markers where available.
- `figures/fig4_heatmap.pdf` — datasets × 10 augmentations, value = full_acc −
  remove_X_acc (shown in percentage points).
- `figures/fig3_reliability.pdf` — 2×2: PathMNIST/BloodMNIST × baseline/entropy,
  10 bins, y=x diagonal, ECE in each panel title.

## Current data-availability notes (state of the repo)

These reflect what is on disk now; they are **not** script bugs — they are the
"generate it first" path firing.

- **Bootstrap CSV:** only **dermamnist** prediction arrays exist (as
  `dermamnist_seed{0,42,123}_*`), so the CSV currently has dermamnist rows only.
  The other 5 datasets print a `[skip]` with the `run_weighted_tta` command to
  regenerate their arrays. (Only the dermamnist resnet18 checkpoints are present
  and there is no local `data/` dir, so they cannot be regenerated on this
  machine.)
- **Fig 2 / Fig 5:** fully reproduce from `seed_stability.csv` /
  `effb0_seed_stability.csv`. The Fig 5 plan values are embedded as `PLAN_FIG5`
  for cross-checking — they match `seed_stability.csv` exactly. `top5` is absent
  from the ResNet-18 `seed_stability.csv` (never run), present for EfficientNet-B0.
  EfficientNet-B0 covers 5 datasets (no organamnist in `effb0_seed_stability.csv`).
- **Fig 4:** ResNet-18 is now **complete** (6×10) — `dermamnist_ablation_aug.csv`
  was restored from git history (commit `9659646`; it had been generated then
  dropped by the old `.gitignore`). EfficientNet-B0 (`--arch effb0`) is missing
  only OrganAMNIST, so it needs `--allow-partial` (Organ row blank, no numbers
  invented).
- **Fig 3:** PathMNIST/BloodMNIST prediction arrays are not on disk (neither
  backbone), so this figure cannot be built here yet. The script prints the exact
  missing files and the `run_weighted_tta` commands; once those arrays exist the
  2×2 renders.

## Regenerating the missing inputs

```bash
# Prediction arrays for a dataset (writes predictions/{ds}_seed42_*.npy):
python -m scripts.run_weighted_tta --dataset pathmnist  --ckpt-tag _seed42 --seed 42
python -m scripts.run_weighted_tta --dataset bloodmnist --ckpt-tag _seed42 --seed 42

# OrganAMNIST EfficientNet-B0 augmentation ablation (for the effb0 Fig 4 row):
python -m scripts.ablate_augmentations --dataset organamnist --arch effb0
```
