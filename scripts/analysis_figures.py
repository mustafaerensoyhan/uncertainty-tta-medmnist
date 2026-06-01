"""
Phase 4 analysis figures — modality comparison + latency/accuracy tradeoff.

Reads the merged results/full_matrix.csv (built by scripts.build_full_matrix;
falls back to concatenating results/{ds}_weighted_tta.csv if the matrix isn't
built yet) and produces:

  figures/analysis/accuracy_by_modality.pdf   grouped bars: accuracy per strategy,
                                               grouped by dataset/modality
  figures/analysis/ece_by_modality.pdf         same for ECE (lower is better)
  figures/analysis/latency_tradeoff.pdf        accuracy gain vs baseline (pp)
                                               against inference time (ms/image)

Usage:
    python -m scripts.analysis_figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import all_dataset_keys
from src.evaluate import ALL_STRATEGIES


def _load_matrix(rdir: Path) -> pd.DataFrame:
    fm = rdir / "full_matrix.csv"
    if fm.exists():
        return pd.read_csv(fm)
    frames = [pd.read_csv(f) for ds in all_dataset_keys()
              if (f := rdir / f"{ds}_weighted_tta.csv").exists()]
    if not frames:
        raise FileNotFoundError("No full_matrix.csv or per-dataset weighted CSVs in results/.")
    return pd.concat(frames, ignore_index=True)


def _grouped_bar(df, metric, title, ylabel, save_path):
    datasets = [d for d in all_dataset_keys() if d in set(df["dataset"])]
    strategies = [s for s in ALL_STRATEGIES if s in set(df["strategy"])]
    x = np.arange(len(datasets))
    w = 0.8 / max(1, len(strategies))
    fig, ax = plt.subplots(figsize=(max(8, len(datasets) * 1.6), 4.5))
    for j, strat in enumerate(strategies):
        vals = [df[(df.dataset == d) & (df.strategy == strat)][metric].mean() for d in datasets]
        ax.bar(x + j * w, vals, w, label=strat)
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels(datasets, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=7, ncol=4, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _latency_tradeoff(df, save_path):
    if "inf_ms" not in df or df["inf_ms"].isna().all():
        print("[skip] latency tradeoff: no inf_ms values (run with timing on).")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for ds in [d for d in all_dataset_keys() if d in set(df["dataset"])]:
        sub = df[df.dataset == ds]
        base = sub[sub.strategy == "baseline"]
        if base.empty:
            continue
        base_acc = base["accuracy"].iloc[0]
        for _, r in sub.iterrows():
            if r["strategy"] == "baseline" or pd.isna(r.get("inf_ms")):
                continue
            ax.scatter(r["inf_ms"], (r["accuracy"] - base_acc) * 100, s=30)
            ax.annotate(f"{ds[:4]}/{r['strategy'][:4]}", (r["inf_ms"], (r["accuracy"] - base_acc) * 100),
                        fontsize=6, alpha=0.7)
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("inference time (ms / image)")
    ax.set_ylabel("accuracy gain vs baseline (pp)")
    ax.set_title("Accuracy gain vs latency cost", fontweight="bold")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 4 modality + latency analysis figures.")
    ap.add_argument("--results-dir", default="./results")
    ap.add_argument("--figures-dir", default="./figures")
    args = ap.parse_args()

    df = _load_matrix(Path(args.results_dir))
    outdir = Path(args.figures_dir) / "analysis"

    _grouped_bar(df, "accuracy", "Accuracy by strategy and modality", "accuracy",
                 outdir / "accuracy_by_modality.pdf")
    _grouped_bar(df, "ece", "ECE by strategy and modality (lower better)", "ECE",
                 outdir / "ece_by_modality.pdf")
    _latency_tradeoff(df, outdir / "latency_tradeoff.pdf")

    n = df["dataset"].nunique()
    print(f"Analysis figures for {n} dataset(s) -> {outdir}")
    print("  accuracy_by_modality.pdf, ece_by_modality.pdf, latency_tradeoff.pdf")
    if n < 6:
        print(f"  ({n}/6 datasets present — rebuild once everyone's CSVs are merged.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
