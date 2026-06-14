"""
Variance-weighting sanity check (VMV plan — Implementer 1, Task 2).

The proposal's variance row lists w_i = 1/(var(p_i)+ε). That FORMULA contradicts
its stated INTUITION ("confident/stable views dominate"): because var is taken
across the class axis, a confident PEAKED vector has HIGH class-variance and a
flat UNCERTAIN vector has LOW class-variance, so 1/(var+ε) actually upweights the
UNCERTAIN views — it is backward. This 5-minute check prints the weights both
ways on a confident vs an uncertain vector and states the verdict the supervisor
asked for, then confirms the repo's code already matches it:

  - headline `variance`      = w_i = var(p_i)        (confidence-aligned, CORRECT)
  - ablation  `variance_inv` = w_i = 1/(var(p_i)+ε)  (literal proposal, BACKWARD)

It doubles as a regression guard: it runs the ACTUAL fuse_variance /
fuse_variance_inv from src.tta on a synthetic two-view example and asserts they
pull in opposite directions, so a future edit that silently flips the direction
fails here.

Usage:
    python -m scripts.variance_sanity_check
Exit code 0 = code matches the confidence-aligned decision; 1 = mismatch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tta import _normalize_weights, fuse_variance, fuse_variance_inv


def main() -> int:
    # The exact vectors from the VMV plan's Task-2 snippet (5-class).
    pc = np.array([0.95, 0.02, 0.01, 0.01, 0.01])   # confident / peaked
    pu = np.array([0.20, 0.20, 0.20, 0.20, 0.20])   # uncertain / flat
    eps = 1e-6

    var_c, var_u = float(pc.var()), float(pu.var())
    inv_c, inv_u = 1.0 / (var_c + eps), 1.0 / (var_u + eps)

    print("=" * 72)
    print("  Variance-weighting sanity check (VMV plan, Implementer 1 Task 2)")
    print("=" * 72)
    print(f"  confident p = {pc.tolist()}   var = {var_c:.5f}")
    print(f"  uncertain p = {pu.tolist()}        var = {var_u:.5f}\n")

    print("  Literal proposal formula  w = 1/(var+eps):")
    print(f"     confident w = {inv_c:10.2f}")
    print(f"     uncertain w = {inv_u:10.2f}")
    backward = inv_c < inv_u
    print(f"     -> {'BACKWARD: rewards the UNCERTAIN view' if backward else 'ok'}\n")

    print("  Confidence-aligned form   w = var(p_i):")
    print(f"     confident w = {var_c:10.5f}")
    print(f"     uncertain w = {var_u:10.5f}")
    print(f"     -> {'CORRECT: rewards the CONFIDENT view' if var_c > var_u else 'unexpected'}\n")

    # Regression guard against the real implementations: a 2-view example where
    # view 0 is peaked (class 0) and view 1 is flat. `variance` must lean to
    # class 0; `variance_inv` must lean away from it.
    peaked = [0.95, 0.02, 0.01, 0.01, 0.01]
    flat = [0.20, 0.20, 0.20, 0.20, 0.20]
    per_view = np.array([[peaked], [flat]])                  # (N=2, S=1, C=5)
    w_var = _normalize_weights(per_view.var(axis=2))[:, 0]
    fv = fuse_variance(per_view)[0]
    fvi = fuse_variance_inv(per_view)[0]

    print("  Repo code check (src.tta on a peaked-vs-flat 2-view example):")
    print(f"     normalized var-weights         = [{w_var[0]:.3f} peaked, {w_var[1]:.3f} flat]")
    print(f"     fuse_variance      P(class 0)   = {fv[0]:.3f}  (should be the higher one)")
    print(f"     fuse_variance_inv  P(class 0)   = {fvi[0]:.3f}  (should be the lower one)")

    code_ok = backward and (var_c > var_u) and (w_var[0] > w_var[1]) and (fv[0] > fvi[0])
    print("\n" + "=" * 72)
    if code_ok:
        print("  VERDICT: the literal 1/(var+eps) is BACKWARD on binary/peaked tasks.")
        print("  The repo already uses the confidence-aligned w=var for the headline")
        print("  `variance` strategy and keeps 1/(var+eps) as `variance_inv` (the")
        print("  reported negative finding). No code change needed — report this to")
        print("  the supervisor and keep the Methodology variance row as w=var(p_i).")
    else:
        print("  VERDICT: MISMATCH — fuse_variance / fuse_variance_inv no longer match")
        print("  the confidence-aligned decision. Investigate src/tta.py before writing")
        print("  the Discussion (Writer 3 is blocked on this answer).")
    print("=" * 72)
    return 0 if code_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
