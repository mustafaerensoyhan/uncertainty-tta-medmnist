# Phase 3 — Completion Checklist

Read alongside the main proposal and `Addendum_Phase3_4_Additions.pdf`. This maps
every Phase 3 deliverable to the exact command that produces it, so we can
confirm Phase 3 is **done for all six datasets** before starting Phase 4.

Everything here runs on the **existing Phase 1 checkpoints — no retraining.**
(Retraining only appears in Phase 4, for the optional multi-seed study.)

## Who runs what

| Student | Dataset(s) | Checkpoint group |
|---|---|---|
| S1 — Mustafa | `pathmnist`, `bloodmnist` | A / C |
| S2 — Mohamed Abdel Sattar | `dermamnist` | A |
| S3 — Trang | `breastmnist` | B |
| S4 — Vaidehi | `organamnist` | C |
| S5 — Sudha | `pneumoniamnist` | B |

## Run order (per dataset)

Replace `<ds>` with your dataset key. Run from the repo root (or use the Kaggle
notebook, which chains the same steps).

```bash
# 0. Make sure your baseline JSON matches your checkpoint (no retrain)
python -m scripts.eval_checkpoint --dataset <ds> --compare      # preview
python -m scripts.eval_checkpoint --dataset <ds>                # write if stale

# 1. Standard TTA — fills Sheet 2 incl. inference time
python -m scripts.run_standard_tta --dataset <ds>

# 2. Weighted TTA — 8 strategies + temperature + inf_ms + per-image preds (Sheet 3)
python -m scripts.run_weighted_tta --dataset <ds>

# 3. Confidence strips (only the 4 Figure-1 modalities: path/derma/pneumonia/blood)
python -m scripts.make_confidence_strips --datasets <ds>
```

S1, after everyone's `results/<ds>_weighted_tta.csv` is merged via PR:

```bash
python -m scripts.build_full_matrix        # -> results/full_matrix.csv
```

## What "done" means — tick each per dataset

- [ ] **Baseline reconciled** — `eval_checkpoint --compare` shows the JSON matches the checkpoint (or it was rewritten). Accuracy within ±2% of the published MedMNIST benchmark.
- [ ] **Sheet 2 (Standard TTA)** filled for N=5,10,20,50 incl. the **Inf.ms** column. (Earlier mis-pastes fixed: BloodMNIST ECE ~0.01 not ~0.99; Pneumonia/OrganA Inf.ms are real latencies, not AUC.)
- [ ] **Sheet 3 (Weighted TTA)** has all 8 strategy rows: `baseline, maxprob, entropy, variance, variance_inv, mc_dropout, ts_only, ts_entropy`, each with accuracy, AUC (macro for multiclass / binary for the 2 binary sets), ECE, NLL, **Inf.ms**, and the fitted **temperature T**.
- [ ] **ECE everywhere uses 10 bins** (proposal §3.4) — guaranteed if numbers come from `run_weighted_tta`, not a personal notebook.
- [ ] **Per-image arrays** saved to `predictions/<ds>_{labels,<strategy>_preds,<strategy>_probs}.npy` and pushed to the **shared Drive** (gitignored — not git). Needed for Phase 4 stats.
- [ ] **Confidence strips** (the 4 Figure-1 datasets only): `figures/strip/<ds>_sample{1,2,3}.pdf` generated and eyeballed.
- [ ] **Results CSV committed** via branch + PR: `results/<ds>_weighted_tta.csv` (and `<ds>_standard_tta.csv`).

S1 / supervisor, once all six are in:

- [ ] `results/full_matrix.csv` built and present for all 6 datasets × 8 strategies.
- [ ] 12 confidence strips (4 modalities × 3 images) reviewed.
- [ ] Supervisor has reviewed the matrix for sanity (no NaN, plausible ranges, ECE decreasing roughly baseline → entropy → TS-only → TS+entropy).

## Settled decisions (so numbers are comparable)

- **Variance direction:** headline `variance` is confidence-aligned (`w = var`, sharp views up); the literal `1/(var+ε)` ships as `variance_inv` and is reported as the "naive variance weighting is unstable" negative finding. (Supervisor-approved.)
- **Single source of truth:** all official numbers come from `run_weighted_tta` so every student's row is computed with identical formulas, bins, and CSV schema.
- **Inference time:** measured with warmup + `torch.cuda.synchronize()` (so GPU numbers are real), same machine per dataset.

## Explicitly deferred to Phase 4 (do NOT block Phase 3 on these)

- McNemar (per-dataset) + Wilcoxon (across datasets) significance tests — they consume the per-image `.npy` arrays saved above.
- Multi-seed stability study (retrain with seeds 0/42/123 on ≥3 datasets for mean ± std). This is the **only** task that needs retraining; it's optional-but-recommended and needs per-seed checkpoint naming + an aggregation script first.
- Per-strategy reliability diagrams, N-ablation, latency/accuracy tradeoff plots.
