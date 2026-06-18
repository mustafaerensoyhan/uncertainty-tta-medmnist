# ──────────────────────────────────────────────────────────────────────────
#   Fig 5 — mechanism scatter   (drop-in replacement for fig5_mechanism)
# ──────────────────────────────────────────────────────────────────────────
def fig5_mechanism(results_dir="./results", figures_dir="./figures") -> Path | None:
    rdir, fdir = Path(results_dir), Path(figures_dir)
    idx = _stability_for_arch(rdir, "resnet18")

    # (n_classes, baseline_ece, entropy_ece) per dataset.
    pts: dict[str, tuple[int, float, float]] = {}
    if idx is not None:
        for ds in all_dataset_keys():
            if (ds, "baseline") in idx.index and (ds, "entropy") in idx.index:
                pts[ds] = (get_config(ds).n_classes,
                           float(idx.loc[(ds, "baseline"), "ece_mean"]),
                           float(idx.loc[(ds, "entropy"), "ece_mean"]))
    if not pts:
        print("[fig5] seed_stability.csv missing/empty — falling back to verified "
              "VMV-plan values.")
        pts = dict(PLAN_FIG5)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0.0, color="black", linewidth=0.8)

    # ResNet-18 points (solid). Label ABOVE each point.
    for i, ds in enumerate(pts):
        nc, base, ent = pts[ds]
        gain = base - ent
        is_derma = ds == "dermamnist"
        ax.scatter(nc, gain, s=90, zorder=3,
                   color="tab:red" if is_derma else "tab:blue",
                   edgecolor="black", linewidth=0.6,
                   label="ResNet-18" if i == 0 else None)
        lbl = SHORT_NAME.get(ds, ds)
        if is_derma:
            lbl += "\n(color-aug exception)"
        ax.annotate(lbl, (nc, gain), textcoords="offset points",
                    xytext=(7, 7), fontsize=8,
                    color="tab:red" if is_derma else "black")

    # EfficientNet-B0 overlay (hollow green). Label BELOW each point so it
    # doesn't collide with the ResNet label on the same dataset.
    eidx = _stability_for_arch(rdir, "effb0")
    if eidx is not None:
        n_overlay = 0
        for ds in all_dataset_keys():
            if (ds, "baseline") in eidx.index and (ds, "entropy") in eidx.index:
                nc = get_config(ds).n_classes
                gain = (float(eidx.loc[(ds, "baseline"), "ece_mean"])
                        - float(eidx.loc[(ds, "entropy"), "ece_mean"]))
                ax.scatter(nc, gain, s=110, facecolors="none",
                           edgecolors="tab:green", linewidths=1.6, zorder=2,
                           label="EfficientNet-B0" if n_overlay == 0 else None)
                ax.annotate(SHORT_NAME.get(ds, ds), (nc, gain),
                            textcoords="offset points", xytext=(7, -13),
                            fontsize=8, color="tab:green")
                n_overlay += 1
        missing_eff = [SHORT_NAME.get(d, d) for d in all_dataset_keys()
                       if (d, "entropy") not in eidx.index]
        print(f"[fig5] EfficientNet-B0 overlay: {n_overlay} dataset(s)"
              + (f"; not available for {missing_eff}" if missing_eff else ""))
    else:
        print("[fig5] WARNING: results/effb0_seed_stability.csv not found — "
              "rendering ResNet-18 figure only.")

    ax.set_xlabel("Number of classes")
    ax.set_ylabel("ECE improvement from entropy weighting\n(baseline ECE − entropy ECE)")
    ax.set_title("Calibration mechanism: entropy weighting helps more as classes grow")
    ax.grid(alpha=0.3)
    # Legend OUTSIDE the plot (right side) so it never sits on the points.
    ax.legend(frameon=True, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              borderaxespad=0, fontsize=9)
    fig.subplots_adjust(right=0.78)   # reserve room for the external legend

    out = fdir / "fig5_mechanism.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig5] wrote {out}")
    return out
