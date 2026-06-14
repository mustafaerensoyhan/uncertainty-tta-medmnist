"""
Merge every results/{dataset}_weighted_tta.csv into the single
results/full_matrix.csv the proposal calls for (6 datasets × 5 strategies).

Kept as a SEPARATE step (rather than every student appending to one shared file)
so teammates' Phase 3 runs don't collide on the same file in git — exactly the
conflict pattern we hit in Phase 2. Each student commits only their own
{dataset}_weighted_tta.csv via PR; S1 runs this merge on main afterwards.

Usage from the repo root:
    python -m scripts.build_full_matrix

Output:
    results/full_matrix.csv   (rows ordered by dataset, then strategy)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import all_dataset_keys
from src.evaluate import strategies_in_order
from src.tta import TOP_K_VALUES


def main() -> int:
    p = argparse.ArgumentParser(description="Merge per-dataset weighted-TTA CSVs.")
    p.add_argument("--results-dir", default="./results")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    frames, missing = [], []
    for ds in all_dataset_keys():
        f = results_dir / f"{ds}_weighted_tta.csv"
        if f.exists():
            frames.append(pd.read_csv(f))
        else:
            missing.append(ds)

    if not frames:
        print("No per-dataset weighted-TTA CSVs found. Run "
              "scripts.run_weighted_tta for each dataset first.")
        return 1

    df = pd.concat(frames, ignore_index=True)

    # Stable ordering: by dataset (config order), then strategy (paper order,
    # core 8 then Top-K). Unknown strategies sort last.
    ds_order = {d: i for i, d in enumerate(all_dataset_keys())}
    full_order = strategies_in_order(TOP_K_VALUES)
    st_order = {s: i for i, s in enumerate(full_order)}
    df["_d"] = df["dataset"].map(ds_order)
    df["_s"] = df["strategy"].map(st_order)
    df = df.sort_values(["_d", "_s"]).drop(columns=["_d", "_s"]).reset_index(drop=True)

    out = results_dir / "full_matrix.csv"
    df.to_csv(out, index=False)

    have = df["dataset"].nunique()
    print(f"Merged {len(frames)} dataset file(s), {len(df)} rows -> {out}")
    print(f"Datasets present: {have}/6")
    if missing:
        print(f"Still missing: {', '.join(missing)} "
              f"(run scripts.run_weighted_tta for these).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
