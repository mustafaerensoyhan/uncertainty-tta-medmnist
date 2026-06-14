# Phase 5 additions — EfficientNet-B0, Top-K TTA, and pipeline tooling

This document records the code added in Phase 5 for the VMV submission, assigned to **Mustafa
(S1, "Implementer 1")** in the team plan. Everything here is implemented, smoke-tested, and covered
by unit tests (the suite is 95 tests and passes), but the *runs* still need a GPU pass — there were
no trained EfficientNet checkpoints and no GPU available when this was written. **None of it changes
the published ResNet-18 results**; the design rule throughout was that ResNet-18 at the canonical
seed keeps its exact original filenames and behaviour, and everything new lives in a parallel
namespace so Phases 1–4 cannot break.

## A second backbone: EfficientNet-B0

`src/model.py` gained `build_efficientnet_b0(num_classes, pretrained, dropout_p)` alongside the
existing `build_resnet18`, plus an `ARCHITECTURES` registry and a `build_model(arch, ...)` factory
that the scripts call. EfficientNet-B0 uses the same training recipe as ResNet-18 (Adam, lr=1e-4,
30 epochs, 64×64 input) so the two backbones are directly comparable. Every training and evaluation
script now accepts `--arch {resnet18,effb0}` with `resnet18` as the default, which means every
command in the existing README continues to work unchanged. To produce the full EfficientNet result
set, the one-liner is `python -m scripts.run_all --models effb0 --seeds 0 42 123 --phases train
weighted aggregate`.

`src/mc_dropout.py` was generalised so MC Dropout works on either backbone: the head-discovery
helper now finds ResNet's bare `.fc` Linear and EfficientNet's `.classifier` Sequential, and applies
the same forward-pre-hook dropout to the pooled feature vector in both cases. See the testing gotcha
at the end of this document before trusting any MC-Dropout smoke test on an untrained EfficientNet.

## Backbone-aware, collision-free file naming

The naming policy lives in three small helpers in `src/utils.py`: `checkpoint_filename(dataset,
arch, tag)`, `result_stem(dataset, arch, tag)`, and `default_ckpt_tag(arch, seed, canonical_seed)`.
ResNet-18 at the canonical seed (42) resolves to the original archless, untagged stems — for example
`checkpoints/pathmnist_resnet18.pth` and `results/pathmnist_weighted_tta.csv` — so all existing
checkpoints, result CSVs, figures, and the globbing in `aggregate_seeds` keep working exactly as
before. EfficientNet and seed-tagged runs get a namespaced infix instead (`pathmnist_effb0_*`,
`pathmnist_seed0_*`), so they never collide with or overwrite the canonical files. A useful
side-effect: this removed the recurring "`train_baseline --seed N` clobbered the baseline JSON, run
`git restore`" gotcha that the handoff's §7 used to warn about, because tagged runs no longer touch
the canonical artifacts.

## Top-K TTA fusion

`src/tta.py` gained a hard-filter fusion family: `top3`, `top5`, and `top7`. Where the soft
strategies *weight* every view, Top-K *keeps* only the K lowest-entropy (most confident) views and
averages those, discarding the rest; the result still lies on the probability simplex. The
selection and fusion are `top_k_keep_indices` and `fuse_top_k`, registered into `FUSION_FNS` via
`functools.partial`. Crucially, Top-K reuses the per-view probabilities that `run_all_strategies`
already computes once, so adding the three columns costs essentially nothing beyond the full-TTA
forward passes that were happening anyway. `run_weighted_tta` runs Top-K by default; use
`--top-k 5` to pick specific values or `--no-top-k` to skip. The weighted-TTA CSV now also carries
an `arch` column, and `build_full_matrix` orders the Top-K columns immediately after the eight core
strategies.

## Variance-weighting sanity check

`scripts/variance_sanity_check.py` answers the supervisor's Task-2 question — is the proposal's
`w = 1/(var(p_i)+ε)` correct? — with a runnable proof rather than prose. It prints the weights both
ways on a confident vs an uncertain probability vector, showing that the literal `1/(var+ε)` formula
is **backward** (because variance is taken across the class axis, a peaked confident vector has high
class-variance, so `1/var` actually upweights the *uncertain* views), and that the repo's
confidence-aligned `w = var(p_i)` is the right choice. It then regression-guards the real
`fuse_variance` and `fuse_variance_inv` on a peaked-vs-flat example so a future edit that silently
flips the direction fails loudly. It runs in under a second on CPU and exits 0 when the code matches
the decision. The upshot is unchanged from the original investigation: **no code change was needed**;
the repo already does the right thing, and the Methodology section should keep the variance row as
`w = var(p_i)`.

## Confidence-strip gold outline (Figure 1)

`scripts/make_confidence_strips.py` gained `--gold-k` (default 5): the Top-K kept (lowest-entropy)
bars in the Augmentation Confidence Strip are now drawn with a gold outline and a small legend cue.
This ties the paper's hero figure directly to the Top-K TTA strategy at no extra computation — the
same per-view entropies that set the bar heights also pick the gold bars.

## Inference time, made comparable across additions

The plan asked for inference time "wherever it makes sense, so we can compare GPU runtime with
future additions." `run_weighted_tta` already reported a per-strategy `inf_ms`; Phase 5 adds
`scripts/benchmark_inference.py`, which sweeps backbone × N (views) × method on a single machine
using the existing warmup-plus-`cuda.synchronize` timing, and writes `results/inference_benchmark.csv`
with columns `arch, dataset, method, n_views, ms_per_image`. Because every soft strategy and every
Top-K share one set of N forward passes, they share a single `tta_all_strategies` latency row per N
— so the table stays small and directly comparable as new backbones or fusions are added later. A
practical note for the tracker: latency is deterministic, so the mean±std sheet only needs **one**
inference column sourced from a single timed run; there is no need to seed-average timing.

## One command to run everything: `run_all.py`

`scripts/run_all.py` is the unified orchestrator the plan asked for — "for model, for dataset, run
all phases in one go." It expands a request over backbones, datasets, seeds, and phases into an
ordered plan and drives the existing phase scripts as subprocesses (so every flag, fix, and Windows
guard they already have is reused verbatim). Per cell it runs `train` then `weighted`; on the
canonical seed it adds `standard` and the two ablations; once per backbone it runs `aggregate`,
`strips`, and `benchmark`; and once overall it runs `matrix`, `significance`, `reliability`, and
`analysis`. It records the wall-clock of every step to `results/run_manifest.csv` and prints a
per-phase runtime rollup, which is exactly the GPU-runtime comparison the plan wanted. Always
preview a plan with `--dry-run` (it needs no GPU or data); `--skip-existing` skips training when a
checkpoint is already present, `--no-time` skips the latency pass for faster seed runs, and
`--continue-on-error` pushes through a failing step instead of stopping. For example, a full dry-run
preview of both backbones across two datasets and two seeds is `python -m scripts.run_all --models
resnet18 effb0 --datasets pathmnist bloodmnist --seeds 42 0 --phases all --dry-run`.

## Tests

Four new test files accompany the additions and run on CPU with no data download:
`tests/test_topk.py` (keep-index selection, simplex property, the Top-N == equal-weight identity,
and that Top-K sharpens toward the confident class), `tests/test_naming.py` (the canonical-stays-
unchanged / parallel-namespace policy and that the resnet seed glob can't match an effb0 stem),
`tests/test_model_arch.py` (the factory, both backbones' forward shapes, head discovery, and MC
Dropout on EfficientNet — including the untrained-features gotcha), and `tests/test_run_all_plan.py`
(the orchestrator's planning logic: tag policy, canonical-only gating, phase counts, and ordering).

## Testing gotcha: untrained EfficientNet shows zero MC-Dropout spread

This is documented because it looks alarming and is not a bug. An untrained EfficientNet-B0
(`pretrained=False`) produces **all-zero penultimate features** in eval mode — a consequence of
random initialisation combined with eval-mode BatchNorm using its default running statistics. With
zero features feeding the classifier, masking them does nothing, so MC Dropout shows zero
across-pass variation and looks exactly like a broken hook. ResNet-18's untrained features are
non-zero, so it never exhibits this. On a real trained checkpoint — or after a few train-mode
forwards to populate BatchNorm's running stats — the features are non-zero and MC Dropout produces
genuine per-pass spread (verified at across-pass std ≈ 0.007 on a BatchNorm-warmed EfficientNet).
The practical rule: do not trust an EfficientNet MC-Dropout smoke test that has no trained
checkpoint behind it; the regression test in `tests/test_model_arch.py` pins both the gotcha and the
correct behaviour once features exist.
