"""
VMV paper figures (Implementer-2 deliverables 2-5).

Generates, at 300 DPI:
    figures/fig2_ece.pdf          ECE bar chart, 6 datasets x strategies (mean +/- std over 3 seeds)
    figures/fig3_reliability.pdf  2x2 reliability diagrams (Path/Blood, baseline/entropy)
    figures/fig4_heatmap.pdf      augmentation-importance heatmap (datasets x augmentations)
    figures/fig5_mechanism.pdf    ECE improvement from entropy weighting vs number of classes

The headline of this project is CALIBRATION, not accuracy. Fig 5 plots the
calibration mechanism (entropy weighting helps more as the number of classes
grows); DermaMNIST is the documented color-augmentation exception and is
annotated, never hidden.

Data sources (all already in the repo):
    Fig 2/5 : results/seed_stability.csv (resnet18, 3-seed mean+/-std)
              results/effb0_seed_stability.csv (EfficientNet-B0 overlay, where available)
              results/{ds}_baseline.json (no-TTA reference line)
    Fig 4   : results/{ds}_ablation_aug.csv  (value = full_acc - remove_X_acc)
    Fig 3   : predictions/{ds}[_seed{SEED}]_{strategy}_probs.npy + _labels.npy

Usage from the repo root:
    python -m scripts.make_vmv_figures --figure all
    python -m scripts.make_vmv_figures --figure 2
    python -m scripts.make_vmv_figures --figure 4 --allow-partial
    python -m scripts.make_vmv_figures --figure 3 --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless / reproducible
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import all_dataset_keys, get_config
from src.metrics import expected_calibration_error

DPI = 300

# Short display names for annotations / axis labels.
SHORT_NAME = {
    "pathmnist": "Path", "dermamnist": "Derma", "pneumoniamnist": "Pneumonia",
    "breastmnist": "Breast", "bloodmnist": "Blood", "organamnist": "Organ",
}

# Strategies shown in Fig 2, in order, restricted to those actually present.
FIG2_STRATEGIES = ["baseline", "entropy", "maxprob", "variance", "ts_entropy", "top5"]

# The 10 ablated augmentations (Fig 4 columns), in the repo's canonical order.
AUG_ORDER = ["hflip", "rotate", "brightness", "gaussian_noise", "center_crop",
             "contrast", "sharpness", "vflip", "color_jitter", "elastic"]

# Verified VMV-plan values (baseline ECE, entropy ECE) for Fig 5. Used only as a
# fallback / cross-check; the live numbers are read from seed_stability.csv,
# which these match exactly.
PLAN_FIG5 = {
    "pneumoniamnist": (2, 0.0957, 0.0975),
    "breastmnist":    (2, 0.0376, 0.0518),
    "dermamnist":     (7, 0.0739, 0.0817),
    "bloodmnist":     (8, 0.0098, 0.0060),
    "pathmnist":      (9, 0.0251, 0.0159),
    "organamnist":    (11, 0.0577, 0.0490),
}


# ──────────────────────────────────────────────────────────────────────────
#   shared helpers
# ──────────────────────────────────────────────────────────────────────────
def _load_stability(path: Path) -> pd.DataFrame | None:
    """Load a *_seed_stability.csv (dataset, strategy, ece_mean, ece_std, ...)."""
    if not path.exists():
        return None
    return pd.read_csv(path)


def _stability_for_arch(rdir: Path, arch: str) -> pd.DataFrame | None:
    """
    Per-backbone seed-stability rows, indexed by (dataset, strategy).

    Handles both layouts:
      * new combined results/seed_stability.csv with an `arch` column
        (resnet18 + effb0 in one file) — filtered to `arch`;
      * legacy archless results/seed_stability.csv (resnet18 only) plus a
        separate results/{arch}_seed_stability.csv for other backbones.
    Returns None if no rows for this backbone are found.
    """
    frames = []
    for path in (rdir / "seed_stability.csv", rdir / f"{arch}_seed_stability.csv"):
        df = _load_stability(path)
        if df is None:
            continue
        if "arch" in df.columns:
            df = df[df["arch"] == arch]
        elif arch != "resnet18":
            # Archless file is resnet18 only; skip it for other backbones.
            continue
        if not df.empty:
            frames.append(df[["dataset", "strategy", "ece_mean", "ece_std"]])
    if not frames:
        return None
    out = pd.concat(frames).drop_duplicates(["dataset", "strategy"])
    return out.set_index(["dataset", "strategy"])


def _arch_suffix(arch: str) -> str:
    """File/figure suffix for a backbone (resnet18 keeps the archless name)."""
    return "" if arch == "resnet18" else f"_{arch}"


def _ablation_path(rdir: Path, ds: str, arch: str) -> Path:
    return rdir / (f"{ds}_ablation_aug.csv" if arch == "resnet18"
                   else f"{ds}_{arch}_ablation_aug.csv")


def _baseline_no_tta_ece(rdir: Path, ds: str, arch: str = "resnet18") -> float | None:
    """
    No-TTA test ECE from the baseline JSON, or None if absent.
    resnet18: {ds}_baseline.json ; other backbones: {ds}_{arch}_seed0_baseline.json.
    """
    f = (rdir / f"{ds}_baseline.json" if arch == "resnet18"
         else rdir / f"{ds}_{arch}_seed0_baseline.json")
    if not f.exists():
        return None
    try:
        return float(json.loads(f.read_text())["test_metrics"]["ece"])
    except (KeyError, ValueError, json.JSONDecodeError):
        return None


def _aug_importance(df: pd.DataFrame) -> dict[str, float]:
    """
    From one dataset's ablation_aug dataframe, return {aug: full_acc - remove_acc}.
    Positive = removing that augmentation hurt accuracy (it was important).
    """
    full_rows = df[df["removed"] == "(none/full)"]
    if full_rows.empty:
        raise ValueError("ablation_aug CSV has no '(none/full)' row")
    full_acc = float(full_rows["accuracy"].iloc[0])
    out: dict[str, float] = {}
    for _, r in df.iterrows():
        aug = r["removed"]
        if aug == "(none/full)":
            continue
        out[aug] = full_acc - float(r["accuracy"])
    return out


def _resolve_stem(pdir: Path, ds: str, seed: int, arch: str = "resnet18") -> str | None:
    base = ds + _arch_suffix(arch)             # e.g. "pathmnist" or "pathmnist_effb0"
    if (pdir / f"{base}_labels.npy").exists():
        return base
    if (pdir / f"{base}_seed{seed}_labels.npy").exists():
        return f"{base}_seed{seed}"
    return None


# ──────────────────────────────────────────────────────────────────────────
#   Fig 2 — ECE bar chart
# ──────────────────────────────────────────────────────────────────────────
def fig2_ece(results_dir="./results", figures_dir="./figures",
             arch="resnet18") -> Path | None:
    rdir, fdir = Path(results_dir), Path(figures_dir)
    idx = _stability_for_arch(rdir, arch)
    if idx is None:
        agg = "aggregate_seeds" + ("" if arch == "resnet18" else f" --arch {arch}")
        print(f"[fig2] no {arch} rows in results/seed_stability.csv — generate with "
              f"`python -m scripts.{agg}`. Skipping Fig 2 ({arch}).")
        return None

    present_ds = idx.index.get_level_values("dataset")
    present_st = idx.index.get_level_values("strategy")
    datasets = [d for d in all_dataset_keys() if d in set(present_ds)]
    strategies = [s for s in FIG2_STRATEGIES if s in set(present_st)]
    missing_strats = [s for s in FIG2_STRATEGIES if s not in strategies]
    if missing_strats:
        print(f"[fig2/{arch}] strategies not available (omitted): {missing_strats}")

    n_ds, n_st = len(datasets), len(strategies)
    width = 0.8 / n_st
    x = np.arange(n_ds)

    fig, ax = plt.subplots(figsize=(1.7 * n_ds + 2, 5))
    cmap = plt.get_cmap("tab10")
    for j, strat in enumerate(strategies):
        means, stds = [], []
        for ds in datasets:
            if (ds, strat) in idx.index:
                means.append(float(idx.loc[(ds, strat), "ece_mean"]))
                stds.append(float(idx.loc[(ds, strat), "ece_std"]))
            else:
                means.append(np.nan); stds.append(0.0)
        ax.bar(x + j * width, means, width, yerr=stds, capsize=2,
               label=strat, color=cmap(j), edgecolor="black", linewidth=0.4)

    # Dashed no-TTA reference line per group (from the baseline JSON).
    ref_label_used = False
    for i, ds in enumerate(datasets):
        ref = _baseline_no_tta_ece(rdir, ds, arch)
        if ref is None:
            continue
        ax.hlines(ref, x[i] - 0.1, x[i] + 0.8, colors="grey", linestyles="--",
                  linewidth=1.2, zorder=5,
                  label=None if ref_label_used else "no-TTA baseline")
        ref_label_used = True

    arch_label = {"resnet18": "ResNet-18", "effb0": "EfficientNet-B0", "mnv3_small": "MobileNetV3-Small", "deit_tiny": "DeiT-Tiny"}.get(arch, arch)
    ax.set_xticks(x + 0.4 - width / 2)
    ax.set_xticklabels([SHORT_NAME.get(d, d) for d in datasets])
    ax.set_ylabel("Expected Calibration Error (ECE)")
    ax.set_title(f"ECE by dataset and TTA strategy — {arch_label} "
                 f"(mean ± std over seeds)")
    ax.legend(ncol=min(4, n_st + 1), fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out = fdir / f"fig2_ece{_arch_suffix(arch)}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig2] wrote {out}  ({arch}: {n_ds} datasets x {n_st} strategies"
          + (", incl. top5" if "top5" in strategies else "") + ")")
    return out


# ──────────────────────────────────────────────────────────────────────────
#   Fig 5 — mechanism scatter
# ──────────────────────────────────────────────────────────────────────────
def fig5_mechanism(results_dir="./results", figures_dir="./figures") -> Path | None:
    from matplotlib.lines import Line2D

    rdir, fdir = Path(results_dir), Path(figures_dir)

    def _gains(arch, fallback=None):
        """{ds: (n_classes, baseline_ece - entropy_ece)} for one architecture."""
        idx = _stability_for_arch(rdir, arch)
        out: dict[str, tuple[int, float]] = {}
        if idx is not None:
            for ds in all_dataset_keys():
                if (ds, "baseline") in idx.index and (ds, "entropy") in idx.index:
                    out[ds] = (get_config(ds).n_classes,
                               float(idx.loc[(ds, "baseline"), "ece_mean"])
                               - float(idx.loc[(ds, "entropy"), "ece_mean"]))
        if not out and fallback:
            out = {d: (v[0], v[1] - v[2]) for d, v in fallback.items()}
        return out

    res = _gains("resnet18", fallback=PLAN_FIG5)
    eff = _gains("effb0")
    deit = _gains("deit_tiny")
    for label, data, fname in (("EfficientNet-B0", eff, "effb0_seed_stability.csv"),
                               ("DeiT-Tiny", deit, "deit_tiny_seed_stability.csv")):
        if not data:
            print(f"[fig5] WARNING: results/{fname} not found — {label} omitted.")
        else:
            missing = [SHORT_NAME.get(d, d) for d in all_dataset_keys() if d not in data]
            print(f"[fig5] {label} overlay: {len(data)} dataset(s)"
                  + (f"; not available for {missing}" if missing else ""))

    # Colour encodes the dataset; marker shape encodes the model.
    datasets = sorted({*res, *eff, *deit}, key=lambda d: get_config(d).n_classes)
    cmap = plt.get_cmap("tab10")
    colors = {d: cmap(i % 10) for i, d in enumerate(datasets)}
    MARKER = {"resnet18": ("o", "ResNet-18"), "effb0": ("s", "EfficientNet-B0"),
              "deit_tiny": ("^", "DeiT-Tiny")}
    # Dodge the models apart on x so a per-dataset marker group stays visible
    # even when their gains nearly coincide (e.g. Organ, Path).
    DODGE = {"resnet18": -0.22, "effb0": 0.0, "deit_tiny": 0.22}

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.axhline(0.0, color="black", linewidth=0.9, zorder=1)

    for arch, data in (("resnet18", res), ("effb0", eff), ("deit_tiny", deit)):
        marker = MARKER[arch][0]
        for ds, (nc, gain) in data.items():
            ax.scatter(nc + DODGE[arch], gain, s=130, marker=marker,
                       color=colors[ds], edgecolor="black", linewidth=0.7,
                       alpha=0.9, zorder=3)

    # Keep only the narrative annotation (Derma is the color-aug exception).
    if "dermamnist" in res:
        nc, gain = res["dermamnist"]
        ax.annotate("color-aug\nexception", (nc + DODGE["resnet18"], gain),
                    textcoords="offset points", xytext=(-58, 0), fontsize=8.5,
                    color=colors["dermamnist"], va="center", ha="right")

    # Light "helps / hurts" zone cues on the left margin.
    ax.text(0.012, 0.99, "entropy weighting helps ↑", transform=ax.transAxes,
            fontsize=8.5, color="dimgrey", va="top")
    ax.text(0.012, 0.01, "entropy weighting hurts ↓", transform=ax.transAxes,
            fontsize=8.5, color="dimgrey", va="bottom")

    ax.set_xlabel("Number of classes")
    ax.set_ylabel("ECE improvement from entropy weighting\n(baseline ECE − entropy ECE)")
    ax.set_title("Calibration mechanism: entropy weighting helps more as classes grow")
    ax.grid(alpha=0.3)

    # Two legends, both outside the plot on the right.
    ds_handles = [Line2D([], [], marker="o", linestyle="none", markersize=9,
                         markerfacecolor=colors[d], markeredgecolor="black",
                         label=SHORT_NAME.get(d, d)) for d in datasets]
    model_handles = [Line2D([], [], marker=m, linestyle="none", markersize=9,
                            markerfacecolor="lightgrey", markeredgecolor="black",
                            label=name) for m, name in MARKER.values()]
    leg_ds = ax.legend(handles=ds_handles, title="Dataset", frameon=False,
                       loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.add_artist(leg_ds)
    ax.legend(handles=model_handles, title="Model", frameon=False,
              loc="upper left", bbox_to_anchor=(1.02, 0.42))

    out = fdir / "fig5_mechanism.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig5] wrote {out}")
    return out


# ──────────────────────────────────────────────────────────────────────────
#   Fig 4 — augmentation-importance heatmap
# ──────────────────────────────────────────────────────────────────────────
def fig4_heatmap(results_dir="./results", figures_dir="./figures",
                 allow_partial=False, arch="resnet18") -> Path | None:
    rdir, fdir = Path(results_dir), Path(figures_dir)
    datasets = all_dataset_keys()

    matrix = np.full((len(datasets), len(AUG_ORDER)), np.nan)
    missing_ds, missing_cells = [], []
    for i, ds in enumerate(datasets):
        f = _ablation_path(rdir, ds, arch)
        if not f.exists():
            missing_ds.append(ds)
            continue
        imp = _aug_importance(pd.read_csv(f))
        for j, aug in enumerate(AUG_ORDER):
            if aug in imp:
                matrix[i, j] = imp[aug]
            else:
                missing_cells.append(f"{ds}/{aug}")

    arch_flag = "" if arch == "resnet18" else f" --arch {arch}"
    if missing_ds or missing_cells:
        msg = [f"[fig4/{arch}] augmentation-importance matrix is incomplete:"]
        if missing_ds:
            cmds = "\n".join(
                f"           python -m scripts.ablate_augmentations --dataset {d}{arch_flag}"
                for d in missing_ds)
            msg.append(f"  missing datasets (no {_ablation_path(rdir, '{ds}', arch).name}): {missing_ds}\n"
                       f"        generate with:\n{cmds}")
        if missing_cells:
            msg.append(f"  missing cells: {missing_cells}")
        print("\n".join(msg))
        if not allow_partial:
            print("[fig4] refusing to render an incomplete heatmap. "
                  "Re-run with --allow-partial to render available rows "
                  "(missing cells left blank).")
            return None
        print("[fig4] --allow-partial set: rendering available rows; "
              "missing cells shown blank.")

    # Render in percentage points so colours and cell annotations share one scale.
    matrix_pp = matrix * 100.0
    fig, ax = plt.subplots(figsize=(1.0 * len(AUG_ORDER) + 2, 0.7 * len(datasets) + 2))
    vmax = np.nanmax(np.abs(matrix_pp)) if np.isfinite(matrix_pp).any() else 1.0
    im = ax.imshow(matrix_pp, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(AUG_ORDER)))
    ax.set_xticklabels(AUG_ORDER, rotation=45, ha="right")
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels([SHORT_NAME.get(d, d) for d in datasets])
    for i in range(len(datasets)):
        for j in range(len(AUG_ORDER)):
            v = matrix_pp[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        fontsize=7, color="black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Accuracy change when augmentation removed (pp; full − ablated)")
    arch_label = {"resnet18": "ResNet-18", "effb0": "EfficientNet-B0", "mnv3_small": "MobileNetV3-Small", "deit_tiny": "DeiT-Tiny"}.get(arch, arch)
    ax.set_title(f"Augmentation importance — {arch_label} "
                 f"(positive = augmentation helps accuracy)")
    fig.tight_layout()

    out = fdir / f"fig4_heatmap{_arch_suffix(arch)}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig4] wrote {out}")
    return out


# ──────────────────────────────────────────────────────────────────────────
#   Fig 3 — reliability diagrams (2x2)
# ──────────────────────────────────────────────────────────────────────────
def _reliability_panel(ax, probs, labels, n_bins, title):
    labels = np.asarray(labels).ravel()
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    mids = (edges[:-1] + edges[1:]) / 2
    accs = np.full(n_bins, np.nan)
    confs = np.full(n_bins, np.nan)
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum() == 0:
            continue
        accs[i] = correct[m].mean()
        confs[i] = conf[m].mean()
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.bar(mids, np.nan_to_num(accs), width=1.0 / n_bins, alpha=0.6,
           edgecolor="black", label="Observed accuracy")
    good = np.isfinite(confs)
    ax.scatter(confs[good], accs[good], color="red", s=20, zorder=3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title, fontsize=10)


def fig3_reliability(predictions_dir="./predictions", figures_dir="./figures",
                     seed=42, n_bins=10, arch="resnet18") -> Path | None:
    pdir, fdir = Path(predictions_dir), Path(figures_dir)
    panels = [("pathmnist", "baseline"), ("pathmnist", "entropy"),
              ("bloodmnist", "baseline"), ("bloodmnist", "entropy")]
    sfx = _arch_suffix(arch)
    arch_flag = "" if arch == "resnet18" else f" --arch {arch}"

    # Resolve and verify every required array up front.
    missing = []
    resolved = {}
    for ds, strat in panels:
        stem = _resolve_stem(pdir, ds, seed, arch)
        if stem is None:
            missing.append(f"{pdir}/{ds}{sfx}_labels.npy (or {ds}{sfx}_seed{seed}_labels.npy)")
            continue
        probs_f = pdir / f"{stem}_{strat}_probs.npy"
        if not probs_f.exists():
            missing.append(str(probs_f))
        else:
            resolved[(ds, strat)] = (pdir / f"{stem}_labels.npy", probs_f)

    if missing:
        print(f"[fig3/{arch}] cannot build reliability diagrams — missing prediction arrays:")
        for m in missing:
            print(f"         {m}")
        needed_ds = sorted({ds for ds, _ in panels})
        print("       Generate them with (per dataset):")
        for ds in needed_ds:
            print(f"         python -m scripts.run_weighted_tta "
                  f"--dataset {ds}{arch_flag} --ckpt-tag _seed{seed} --seed {seed}")
        return None

    arch_label = {"resnet18": "ResNet-18", "effb0": "EfficientNet-B0", "mnv3_small": "MobileNetV3-Small", "deit_tiny": "DeiT-Tiny"}.get(arch, arch)
    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    for ax, (ds, strat) in zip(axes.ravel(), panels):
        lab_f, probs_f = resolved[(ds, strat)]
        labels = np.load(lab_f).ravel()
        probs = np.load(probs_f)
        ece = expected_calibration_error(probs, labels, n_bins=n_bins)
        _reliability_panel(ax, probs, labels, n_bins,
                           f"{SHORT_NAME.get(ds, ds)} — {strat} (ECE={ece:.3f})")
    fig.suptitle(f"Reliability diagrams — {arch_label} (10 bins)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out = fdir / f"fig3_reliability{sfx}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig3] wrote {out}")
    return out


# ──────────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description="VMV paper figures (Implementer-2).")
    ap.add_argument("--figure", default="all",
                    choices=["2", "3", "4", "5", "all"])
    ap.add_argument("--results-dir", default="./results")
    ap.add_argument("--figures-dir", default="./figures")
    ap.add_argument("--predictions-dir", default="./predictions")
    ap.add_argument("--seed", type=int, default=42,
                    help="Seed tag for Fig 3 prediction arrays (default 42).")
    ap.add_argument("--arch", default="resnet18",
                    choices=["resnet18", "effb0", "mnv3_small", "deit_tiny", "both"],
                    help="Backbone for the per-arch figures (2/3/4). "
                         "'both' renders ResNet-18 and EfficientNet-B0. "
                         "Fig 5 always overlays both. Default: resnet18.")
    ap.add_argument("--allow-partial", action="store_true",
                    help="Fig 4: render with missing cells blank instead of failing.")
    args = ap.parse_args()

    want = {args.figure} if args.figure != "all" else {"2", "3", "4", "5"}
    archs = ["resnet18", "effb0"] if args.arch == "both" else [args.arch]
    produced = []
    if "5" in want:  # arch-independent (overlays both backbones)
        produced.append(fig5_mechanism(args.results_dir, args.figures_dir))
    for arch in archs:
        if "2" in want:
            produced.append(fig2_ece(args.results_dir, args.figures_dir, arch=arch))
        if "4" in want:
            produced.append(fig4_heatmap(args.results_dir, args.figures_dir,
                                         allow_partial=args.allow_partial, arch=arch))
        if "3" in want:
            produced.append(fig3_reliability(args.predictions_dir, args.figures_dir,
                                             seed=args.seed, arch=arch))

    ok = [p for p in produced if p is not None]
    print(f"\nDone. {len(ok)}/{len(produced)} figure(s) produced.")
    # Non-zero only if a specifically-requested single figure failed.
    if args.figure != "all" and not ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
