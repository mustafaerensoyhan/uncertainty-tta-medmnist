"""
Unit tests for the run_all orchestrator's planning logic (scripts/run_all.py).

These exercise build_plan() — the pure function that turns
(models × datasets × seeds × phases) into an ordered command list — without
running any subprocess, so they're fast and need no GPU/data.

Run with:  pytest tests/test_run_all_plan.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_all import build_plan, ALL_PHASES, BEST_STRATEGY


def _args(**over):
    base = dict(
        models=["resnet18"], datasets=["pathmnist"], seeds=[42],
        phases=["train", "weighted"], canonical_seed=42, epochs=30,
        skip_existing=False, no_time=False, no_top_k=False,
        batch_size=64, img_size=64, num_workers=0,
        data_root="./data", checkpoints_dir="./checkpoints", results_dir="./results",
        cpu=False,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _keys(steps):
    return [s["key"] for s in steps]


def _cmd_str(step):
    return " ".join(step["cmd"][2:])


# ── tag / canonical policy ─────────────────────────────────────────────────

def test_resnet_canonical_seed_is_untagged():
    steps = build_plan(_args(phases=["train"]))
    train = next(s for s in steps if s["key"].endswith("/train"))
    # untagged: the ckpt-tag value after the flag is empty
    assert "--ckpt-tag  " in _cmd_str(train) or _cmd_str(train).rstrip().endswith("--ckpt-tag")
    assert "--arch resnet18" in _cmd_str(train)


def test_effb0_seed_is_tagged():
    steps = build_plan(_args(models=["effb0"], seeds=[42], phases=["train"]))
    train = next(s for s in steps if s["key"].endswith("/train"))
    assert "--ckpt-tag _seed42" in _cmd_str(train)
    assert "--arch effb0" in _cmd_str(train)


def test_noncanonical_resnet_seed_tagged():
    steps = build_plan(_args(seeds=[0], phases=["train"]))
    train = next(s for s in steps if s["key"].endswith("/train"))
    assert "--ckpt-tag _seed0" in _cmd_str(train)


# ── phase gating ───────────────────────────────────────────────────────────

def test_standard_and_ablate_only_on_canonical_seed():
    # two seeds; standard/ablate must appear once (canonical only), not per seed.
    steps = build_plan(_args(seeds=[42, 0], phases=["standard", "ablate"]))
    standard = [k for k in _keys(steps) if k.endswith("/standard")]
    assert len(standard) == 1
    assert standard[0] == "resnet18/pathmnist/standard"


def test_ablate_uses_best_strategy_per_dataset():
    steps = build_plan(_args(datasets=["organamnist"], phases=["ablate"]))
    abl = next(s for s in steps if "ablate_n" in s["key"])
    assert f"--strategy {BEST_STRATEGY['organamnist']}" in _cmd_str(abl)


def test_weighted_no_top_k_flag_forwarded():
    steps = build_plan(_args(phases=["weighted"], no_top_k=True))
    w = next(s for s in steps if s["key"].endswith("/weighted"))
    assert "--no-top-k" in _cmd_str(w)


def test_no_time_forwarded_to_weighted():
    steps = build_plan(_args(phases=["weighted"], no_time=True))
    w = next(s for s in steps if s["key"].endswith("/weighted"))
    assert "--no-time" in _cmd_str(w)


# ── per-arch & global phases ───────────────────────────────────────────────

def test_global_phases_appear_once_each():
    steps = build_plan(_args(models=["resnet18", "effb0"], seeds=[42, 0], phases=ALL_PHASES))
    for g in ("global/matrix", "global/significance", "global/reliability", "global/analysis"):
        assert _keys(steps).count(g) == 1


def test_per_arch_phases_run_once_per_backbone():
    steps = build_plan(_args(models=["resnet18", "effb0"], phases=["aggregate", "benchmark"]))
    assert "resnet18/aggregate" in _keys(steps)
    assert "effb0/aggregate" in _keys(steps)
    assert _keys(steps).count("resnet18/benchmark") == 1


def test_skip_existing_sets_skip_if():
    steps = build_plan(_args(phases=["train"], skip_existing=True))
    train = next(s for s in steps if s["key"].endswith("/train"))
    assert train["skip_if"] is not None
    steps2 = build_plan(_args(phases=["train"], skip_existing=False))
    assert next(s for s in steps2 if s["key"].endswith("/train"))["skip_if"] is None


def test_step_count_scales_with_cells():
    # 2 models x 3 datasets x 2 seeds, train+weighted only = 2*3*2*2 = 24 cell steps
    steps = build_plan(_args(models=["resnet18", "effb0"],
                             datasets=["pathmnist", "bloodmnist", "organamnist"],
                             seeds=[42, 0], phases=["train", "weighted"]))
    assert len(steps) == 24


def test_ordering_models_outermost_then_seed_then_dataset():
    keys = _keys(build_plan(_args(models=["resnet18", "effb0"],
                                  datasets=["pathmnist"], seeds=[42],
                                  phases=["train"])))
    assert keys.index("resnet18/pathmnist/seed42/train") < keys.index("effb0/pathmnist/seed42/train")


# ── canonical-checkpoint auto-copy (effb0 has no untagged seed) ──────────────

def test_ensure_canonical_copies_effb0_seed42(tmp_path):
    import scripts.run_all as ra
    from types import SimpleNamespace
    ck = tmp_path / "checkpoints"; ck.mkdir()
    (ck / "pathmnist_effb0_seed42.pth").write_text("x")  # tagged canonical exists
    orig_root = ra.REPO_ROOT
    try:
        ra.REPO_ROOT = tmp_path
        args = SimpleNamespace(models=["effb0"], datasets=["pathmnist"],
                               canonical_seed=42, checkpoints_dir="checkpoints")
        ra._ensure_canonical_checkpoints(args, ["standard", "ablate", "strips", "benchmark"])
        assert (ck / "pathmnist_effb0.pth").exists()  # untagged copy created
    finally:
        ra.REPO_ROOT = orig_root


def test_ensure_canonical_noop_for_resnet18(tmp_path):
    import scripts.run_all as ra
    from types import SimpleNamespace
    ck = tmp_path / "checkpoints"; ck.mkdir()
    orig_root = ra.REPO_ROOT
    try:
        ra.REPO_ROOT = tmp_path
        args = SimpleNamespace(models=["resnet18"], datasets=["pathmnist"],
                               canonical_seed=42, checkpoints_dir="checkpoints")
        ra._ensure_canonical_checkpoints(args, ["standard"])
        assert not (ck / "pathmnist_resnet18.pth").exists()  # never fabricates resnet ckpt
    finally:
        ra.REPO_ROOT = orig_root


def test_ensure_canonical_noop_when_no_canonical_phase(tmp_path):
    import scripts.run_all as ra
    from types import SimpleNamespace
    ck = tmp_path / "checkpoints"; ck.mkdir()
    (ck / "pathmnist_effb0_seed42.pth").write_text("x")
    orig_root = ra.REPO_ROOT
    try:
        ra.REPO_ROOT = tmp_path
        args = SimpleNamespace(models=["effb0"], datasets=["pathmnist"],
                               canonical_seed=42, checkpoints_dir="checkpoints")
        ra._ensure_canonical_checkpoints(args, ["train", "weighted", "aggregate"])
        assert not (ck / "pathmnist_effb0.pth").exists()  # no copy without a canonical phase
    finally:
        ra.REPO_ROOT = orig_root
