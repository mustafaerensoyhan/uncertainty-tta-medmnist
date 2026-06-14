"""
run_all.py — one command to run the whole pipeline across backbones, datasets,
and seeds (VMV plan: "for model: for dataset: run all phases").

This drives the existing, already-tested phase scripts as subprocesses (so every
flag, fix, and Windows guard they have is reused verbatim) and records the
wall-clock time of every step to results/run_manifest.csv — giving the
GPU-runtime comparison the plan asks for, including how much each ADDITION
(EfficientNet, Top-K, extra seeds) costs relative to the ResNet-18 baseline.

What runs, in order:

  PER (backbone × dataset × seed):
    train      -> scripts.train_baseline        (skip with --skip-existing if ckpt present)
    weighted   -> scripts.run_weighted_tta       (8 strategies + Top-K, per-image preds)

  PER (backbone × dataset), CANONICAL seed only (these use the untagged checkpoint):
    standard   -> scripts.run_standard_tta        (N sweep, fills Sheet 2 inf_ms)
    ablate     -> scripts.ablate_n + ablate_augmentations  (best strategy per dataset)

  PER backbone, ONCE (after the cells):
    aggregate  -> scripts.aggregate_seeds         (mean ± std, needs >=2 seeds)
    strips     -> scripts.make_confidence_strips  (Fig 1, gold Top-K outline)
    benchmark  -> scripts.benchmark_inference     (latency surface for Sheet 6)

  GLOBAL, ONCE (ResNet-18 canonical artifacts):
    matrix         -> scripts.build_full_matrix
    significance   -> scripts.significance --best-per-dataset
    reliability    -> scripts.make_reliability_diagrams
    analysis       -> scripts.analysis_figures

Seed/tag policy (src.utils.default_ckpt_tag): ResNet-18 at the canonical seed
(42) writes the untagged canonical files so the headline numbers stay put; every
other (backbone, seed) is seed-tagged and lives in a parallel namespace.

Examples:
    # Plan only — print every command without running anything (great first step)
    python -m scripts.run_all --models resnet18 effb0 --seeds 42 0 123 --phases all --dry-run

    # The VMV "Implementer 1" job: EfficientNet on all 6, 3 seeds, core phases
    python -m scripts.run_all --models effb0 --seeds 0 42 123 --phases train weighted aggregate

    # Reproduce the whole ResNet-18 pipeline end to end on one GPU
    python -m scripts.run_all --models resnet18 --seeds 42 --phases all

    # Just (re)build the cross-dataset analysis once everyone's CSVs are merged
    python -m scripts.run_all --phases matrix significance reliability analysis

Use --skip-existing to skip training when the checkpoint already exists, and
--no-time to skip the latency pass inside run_weighted_tta for faster seed runs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import all_dataset_keys
from src.model import ARCHITECTURES
from src.utils import checkpoint_filename, default_ckpt_tag, result_stem

REPO_ROOT = Path(__file__).resolve().parents[1]

# Best Phase-3 strategy per dataset (mirror of scripts/significance.py's
# BEST_STRATEGY) — used to pick the ablation --strategy per dataset.
BEST_STRATEGY = {
    "pathmnist": "entropy", "dermamnist": "entropy", "bloodmnist": "entropy",
    "pneumoniamnist": "maxprob", "breastmnist": "maxprob", "organamnist": "variance",
}

# Phase groups. "cell" runs per (arch, dataset, seed); "canon" per (arch,
# dataset) on the canonical seed only; "per_arch" once per backbone; "global"
# once overall.
CELL_PHASES = ["train", "weighted"]
CANON_PHASES = ["standard", "ablate"]
PER_ARCH_PHASES = ["aggregate", "strips", "benchmark"]
GLOBAL_PHASES = ["matrix", "significance", "reliability", "analysis"]
ALL_PHASES = CELL_PHASES + CANON_PHASES + PER_ARCH_PHASES + GLOBAL_PHASES

STRIP_MODALITIES = ["pathmnist", "dermamnist", "pneumoniamnist", "bloodmnist"]


def _canon_steps(ds: str, arch: str, phases, args):
    """Yield (key, cmd) for the canonical-seed phases (standard, ablate) of one dataset."""
    if "standard" in phases:
        cmd = _py("scripts.run_standard_tta", "--dataset", ds, "--arch", arch) + _common(args)
        if args.no_time:
            cmd.append("--no-time")
        yield f"{arch}/{ds}/standard", cmd
    if "ablate" in phases:
        strat = BEST_STRATEGY.get(ds, "entropy")
        for tool in ("ablate_n", "ablate_augmentations"):
            cmd = _py(f"scripts.{tool}", "--dataset", ds, "--arch", arch,
                      "--strategy", strat) + _common(args)
            yield f"{arch}/{ds}/{tool}", cmd


# Canonical phases that read an untagged {ds}_{arch}.pth checkpoint.
_CANONICAL_CKPT_PHASES = {"standard", "ablate", "strips", "benchmark"}


def _ensure_canonical_checkpoints(args, phases) -> None:
    """
    Make sure each canonical-phase checkpoint exists before the run loop.

    ResNet-18 writes an untagged canonical checkpoint, but EfficientNet (and any
    backbone with no untagged seed) only has seed-tagged files. The canonical
    phases (standard/ablate/strips/benchmark) load the untagged name, so here we
    copy {ds}_{arch}_seed{canonical}.pth -> {ds}_{arch}.pth when needed. This
    automates the old manual Copy-Item and covers the multi-dataset strips/
    benchmark commands too. No-op for resnet18 and when the untagged file exists.
    """
    import shutil

    active = ALL_PHASES if "all" in phases else phases
    if not (_CANONICAL_CKPT_PHASES & set(active)):
        return
    ckdir = REPO_ROOT / args.checkpoints_dir
    for arch in args.models:
        if arch == "resnet18":
            continue  # has a real untagged canonical checkpoint
        canon_tag = default_ckpt_tag(arch, args.canonical_seed, args.canonical_seed - 1)
        for ds in args.datasets:
            dst = ckdir / checkpoint_filename(ds, arch, "")
            if dst.exists():
                continue
            src = ckdir / checkpoint_filename(ds, arch, canon_tag)
            if src.exists():
                shutil.copy2(src, dst)
                print(f"  [prep] {arch}/{ds}: copied {src.name} -> {dst.name} "
                      f"(canonical checkpoint for standard/ablate/strips/benchmark)")
            else:
                print(f"  [prep] {arch}/{ds}: WARNING no canonical checkpoint "
                      f"({src.name} missing) — its canonical phases will skip.")


def _py(*mod_args: str) -> list[str]:
    """Build a `python -m scripts.<x> ...` command."""
    return [sys.executable, "-m", *mod_args]


def _common(args) -> list[str]:
    """Flags every sub-script accepts that we forward from run_all."""
    flags = ["--num-workers", str(args.num_workers),
             "--batch-size", str(args.batch_size),
             "--img-size", str(args.img_size),
             "--data-root", args.data_root]
    if args.cpu:
        flags.append("--cpu")
    return flags


def build_plan(args) -> list[dict]:
    """Return an ordered list of steps; each is {key, cmd, skip_if}."""
    phases = ALL_PHASES if "all" in args.phases else args.phases
    steps: list[dict] = []

    # ── per (arch, dataset, seed) ──
    for arch in args.models:
        for seed in args.seeds:
            tag = default_ckpt_tag(arch, seed, args.canonical_seed)
            is_canon = (tag == "" if arch == "resnet18" else seed == args.canonical_seed)
            for ds in args.datasets:
                if "train" in phases:
                    ckpt = REPO_ROOT / args.checkpoints_dir / checkpoint_filename(ds, arch, tag)
                    cmd = _py("scripts.train_baseline", "--dataset", ds, "--arch", arch,
                              "--seed", str(seed), "--ckpt-tag", tag,
                              "--epochs", str(args.epochs)) + _common(args)
                    steps.append({"key": f"{arch}/{ds}/seed{seed}/train", "cmd": cmd,
                                  "skip_if": ckpt if args.skip_existing else None})
                if "weighted" in phases:
                    cmd = _py("scripts.run_weighted_tta", "--dataset", ds, "--arch", arch,
                              "--ckpt-tag", tag, "--seed", str(seed)) + _common(args)
                    if args.no_time:
                        cmd.append("--no-time")
                    if args.no_top_k:
                        cmd.append("--no-top-k")
                    steps.append({"key": f"{arch}/{ds}/seed{seed}/weighted", "cmd": cmd, "skip_if": None})

            # canonical-seed-only phases. ResNet-18 has an untagged canonical
            # checkpoint; effb0 does not (every seed is tagged), so for effb0 we
            # point these phases at a {ds}_effb0.pth copied from the canonical
            # seed's tagged file just-in-time (see _ensure_canonical in the run
            # loop). This removes the manual Copy-Item step.
            # Canonical-seed-only phases. The untagged checkpoint they need is
            # guaranteed to exist by _ensure_canonical_checkpoints() before the
            # run loop (effb0 has no untagged canonical, so it's copied from the
            # canonical seed's tagged file — automating the old manual Copy-Item).
            if is_canon:
                for ds in args.datasets:
                    for key, cmd in _canon_steps(ds, arch, phases, args):
                        steps.append({"key": key, "cmd": cmd, "skip_if": None})

    # ── per backbone, once ──
    for arch in args.models:
        if "aggregate" in phases:
            cmd = _py("scripts.aggregate_seeds", "--arch", arch,
                      "--datasets", *args.datasets, "--results-dir", args.results_dir)
            steps.append({"key": f"{arch}/aggregate", "cmd": cmd, "skip_if": None})
        if "strips" in phases:
            strip_ds = [d for d in STRIP_MODALITIES if d in args.datasets] or STRIP_MODALITIES
            cmd = _py("scripts.make_confidence_strips", "--arch", arch,
                      "--datasets", *strip_ds, "--data-root", args.data_root)
            if args.cpu:
                cmd.append("--cpu")
            steps.append({"key": f"{arch}/strips", "cmd": cmd, "skip_if": None})
        if "benchmark" in phases:
            cmd = _py("scripts.benchmark_inference", "--arch", arch,
                      "--datasets", *args.datasets) + _common(args)
            steps.append({"key": f"{arch}/benchmark", "cmd": cmd, "skip_if": None})

    # ── global, once ──
    if "matrix" in phases:
        steps.append({"key": "global/matrix", "cmd": _py("scripts.build_full_matrix"), "skip_if": None})
    if "significance" in phases:
        steps.append({"key": "global/significance",
                      "cmd": _py("scripts.significance", "--best-per-dataset"), "skip_if": None})
    if "reliability" in phases:
        steps.append({"key": "global/reliability",
                      "cmd": _py("scripts.make_reliability_diagrams"), "skip_if": None})
    if "analysis" in phases:
        steps.append({"key": "global/analysis", "cmd": _py("scripts.analysis_figures"), "skip_if": None})

    return steps


def main() -> int:
    p = argparse.ArgumentParser(
        description="Run the full pipeline across backbones × datasets × seeds (one command).")
    p.add_argument("--models", nargs="+", default=["resnet18"], choices=list(ARCHITECTURES),
                   help="Backbones to run (e.g. resnet18 effb0).")
    p.add_argument("--datasets", nargs="+", default=all_dataset_keys(), choices=all_dataset_keys())
    p.add_argument("--seeds", type=int, nargs="+", default=[42],
                   help="Seeds. ResNet-18 @ canonical seed writes untagged files; others are seed-tagged.")
    p.add_argument("--phases", nargs="+", default=["train", "weighted"],
                   help=f"Phases to run, or 'all'. Available: {', '.join(ALL_PHASES)}.")
    p.add_argument("--canonical-seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip training when the target checkpoint already exists.")
    p.add_argument("--no-time", action="store_true", help="Skip latency pass in run_weighted_tta/standard.")
    p.add_argument("--no-top-k", action="store_true", help="Skip Top-K columns in run_weighted_tta.")
    p.add_argument("--dry-run", action="store_true", help="Print the plan; run nothing.")
    p.add_argument("--continue-on-error", action="store_true",
                   help="Keep going if a step fails (default: stop).")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--img-size", type=int, default=64, choices=[28, 64])
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--data-root", default="./data")
    p.add_argument("--checkpoints-dir", default="./checkpoints")
    p.add_argument("--results-dir", default="./results")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    # Validate phase names early.
    bad = [ph for ph in args.phases if ph not in ALL_PHASES + ["all"]]
    if bad:
        print(f"Unknown phase(s): {', '.join(bad)}. Valid: {', '.join(ALL_PHASES)}, all")
        return 2

    steps = build_plan(args)
    print(f"\n{'='*78}")
    print(f"  run_all — {len(steps)} step(s)")
    print(f"  models : {', '.join(args.models)}")
    print(f"  data   : {', '.join(args.datasets)}")
    print(f"  seeds  : {', '.join(map(str, args.seeds))}")
    print(f"  phases : {', '.join(args.phases)}")
    print(f"  mode   : {'DRY-RUN (nothing executed)' if args.dry_run else 'EXECUTE'}")
    print(f"{'='*78}\n")

    try:
        for i, s in enumerate(steps, 1):
            print(f"  [{i:>3}/{len(steps)}] {s['key']}")
            print(f"        {' '.join(s['cmd'][2:])}")
    except BrokenPipeError:
        return 0  # piped to head/less and closed early — fine

    if args.dry_run:
        print("\nDry run complete — re-run without --dry-run to execute.")
        return 0

    # Ensure effb0 (or any tagged-canonical backbone) has the untagged checkpoint
    # the canonical phases load — copies it from the canonical seed if missing.
    _ensure_canonical_checkpoints(args, args.phases)

    manifest = []
    rc_final = 0
    for i, s in enumerate(steps, 1):
        if s["skip_if"] is not None and Path(s["skip_if"]).exists():
            print(f"\n[{i}/{len(steps)}] SKIP {s['key']} (exists: {s['skip_if']})")
            manifest.append({**_row(s, "skipped", 0.0)})
            continue
        print(f"\n{'='*78}\n[{i}/{len(steps)}] RUN  {s['key']}\n{'='*78}")
        t0 = time.perf_counter()
        proc = subprocess.run(s["cmd"], cwd=str(REPO_ROOT))
        dt = time.perf_counter() - t0
        status = "ok" if proc.returncode == 0 else f"FAIL(rc={proc.returncode})"
        print(f"\n--> {s['key']}: {status} in {dt:.1f}s")
        manifest.append({**_row(s, status, dt)})
        if proc.returncode != 0:
            rc_final = 1
            if not args.continue_on_error:
                print("Stopping (use --continue-on-error to push through failures).")
                break

    # Write the runtime manifest — the GPU-runtime comparison artifact.
    mdf = pd.DataFrame(manifest)
    mpath = REPO_ROOT / args.results_dir / "run_manifest.csv"
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mdf.to_csv(mpath, index=False)

    total = sum(m["seconds"] for m in manifest)
    print(f"\n{'='*78}")
    print(f"  run_all finished: {len(manifest)} step(s), total {total/60:.1f} min")
    print(f"  manifest -> {mpath}")
    print(f"{'='*78}")
    # Quick per-phase rollup so additions are easy to compare.
    if not mdf.empty:
        roll = mdf.groupby("phase")["seconds"].sum().sort_values(ascending=False)
        print(f"\n  {'phase':<14}{'total s':>10}")
        print("  " + "-" * 24)
        for ph, sec in roll.items():
            print(f"  {ph:<14}{sec:>10.1f}")
    return rc_final


def _row(step, status, seconds) -> dict:
    parts = step["key"].split("/")
    phase = parts[-1]
    return {"key": step["key"], "phase": phase, "status": status,
            "seconds": round(float(seconds), 2), "command": " ".join(step["cmd"][2:])}


if __name__ == "__main__":
    raise SystemExit(main())
