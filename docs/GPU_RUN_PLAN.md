# GPU run-plan — trust-strip (Fig 1) regeneration

Everything else in the paper's figure/table set has been regenerated on CPU from the
new run (sign-flip scatter, fig2–5 for all 3 backbones, the 3-model reliability combo,
and all tables). The **one** piece that needs model inference — and therefore a GPU to
be quick — is the **trust-strip / confidence-strip figure (Fig 1)**, the paper's
signature visual.

## Why this needs a (re)run
- The committed strips in `figures/strip/` are the **old** ones: only **4 modalities**
  (Path, Derma, Pneumonia, Blood) × **3 samples**, and they predate the
  "phantom PathMNIST checkpoint" fix, so PathMNIST is stale.
- Reviewer ask: *"show several images per modality so it doesn't look cherry-picked"* —
  we want **6 samples per modality across all 6 modalities**, from the **new canonical
  checkpoints**.
- Interim state already committed (CPU): a 3-sample × 4-modality composite, so there is
  a working multi-image figure in the meantime.

## Prerequisites on the GPU box
- Repo at this commit; `pip install -r requirements.txt` plus `timm` (needed for DeiT-Tiny).
- `checkpoints/{dataset}_{arch}.pth` present for all 6 datasets × {resnet18, effb0,
  deit_tiny} — these are the canonical (no-seed-tag) checkpoints from `All seeds final.zip`.
- MedMNIST data auto-downloads to `./data` (or copy the existing `data/*_64.npz`).
- Drop `--cpu` from every command below to use the GPU.

## Step 1 — generate strips: 6 samples × 6 modalities × 3 backbones
```bash
DSETS="pathmnist dermamnist pneumoniamnist bloodmnist breastmnist organamnist"
for ARCH in resnet18 effb0 deit_tiny; do
  python -m scripts.make_confidence_strips \
      --arch $ARCH --datasets $DSETS \
      --n-images 6 --seeds 0 1 2 3 4 5 --select random
done
# writes figures/strip/{ds}{_arch}_sample{1..6}.pdf   (108 strips total)
```
Notes:
- `--select random` picks seeded *correctly-classified* images (non-cherry-picked, reproducible).
  Use `--select spread` instead if you prefer the most-illustrative high-confidence picks
  (`--max-scan 400` controls how many it scans — that's the only slow knob).
- `--gold-k 5` (default) gold-outlines the kept Top-5 lowest-entropy views.

## Step 2 — assemble composites (CPU, fast; can run anywhere)
```bash
DSETS="pathmnist dermamnist pneumoniamnist bloodmnist breastmnist organamnist"
for ARCH in resnet18 effb0 deit_tiny; do
  python -m scripts.make_fig1_composite \
      --from-strips figures/strip --arch $ARCH --datasets $DSETS \
      --n-images 6 --mode all
done
# writes figures/fig1_{ds}{_arch}.pdf and figures/fig1_confidence_strips{_arch}.pdf
```

## Step 3 — title-free JPEGs (CPU, fast)
```bash
python -m scripts._export_jpeg_no_title
# refreshes figures/jpeg_no_title/*.jpg for every figure, including the new Fig 1
```

## What to send back
- `figures/strip/*.pdf` (the 108 new strips), and/or
- `figures/fig1_*.pdf` + `figures/jpeg_no_title/fig1_*.jpg`

Then the labelled + title-free Fig 1 is final for all three backbones across all six
modalities. Commit those and the figure set is complete.
