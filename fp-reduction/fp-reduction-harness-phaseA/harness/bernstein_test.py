"""
bernstein_test.py — derive the tighter Bernstein tolerance and test whether it
STILL HOLDS against the spread data already collected (CPU and GPU).

Hoeffding tau uses the max rounding error u.  Bernstein tau uses the measured
variance sigma^2 = SIGMA2_OVER_U2 * u^2 (from variance.py), which is smaller, so
the bound is tighter.  Tightening only counts if the tighter tau is NOT exceeded
by the real measured spreads -> that is what this script checks.

It re-reads the existing result files (no new runs) and, per cell, reports:
  tau_hoeffding, tau_bernstein, how much tighter, and whether the real measured
  spread still stays below tau_bernstein.

Usage:
  python bernstein_test.py --cpu results/exceedance.jsonl --gpu results/gpu_exceedance.jsonl
"""

import argparse, json, math
import numpy as np

U = 2.0 ** -24
from bounds import SIGMA2_OVER_U2


from bounds import tau_probabilistic as tau_hoeffding, tau_bernstein, tau_freedman, SIGMA2_OVER_U2_UNIFORM, lambda_for_alpha


def load(path):
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def spread_of(rec):
    """Pull the measured spread quantile (CPU) or max-min (GPU) from a record."""
    for k in ("spread_q_at_1_minus_alpha", "empirical_q_at_1_minus_alpha",
              "spread_maxmin", "spread_max"):
        if k in rec and rec[k] is not None:
            return float(rec[k])
    return None


def analyze(path, label):
    recs = load(path)
    print(f"\n=== {label}  ({path}) ===")
    print(f"{'strategy':>13} {'n':>6} {'kappa':>9} {'spread':>10} "
          f"{'tau_H':>10} {'tau_B':>10} {'tau_Fa':>10} {'B?':>4} {'Fa?':>4}")
    print("-" * 86)
    held = 0; total = 0; tighter_vals = []; heldFa = 0
    for r in recs:
        n = r["n"]; S = r["sum_abs"]; alpha = r.get("alpha", 1e-3)
        kappa = r.get("kappa", 1)
        strat = r.get("strategy", "?")
        sp = spread_of(r)
        if sp is None:
            continue
        tH = tau_hoeffding(n, U, S, alpha)
        tB = tau_bernstein(n, U, S, alpha)
        tFa = tau_freedman(n, U, S, alpha, sigma2_over_u2=SIGMA2_OVER_U2_UNIFORM)  # a priori
        tighter = tH / tB if tB > 0 else float("nan")
        holds = sp <= tB; holdsFa = sp <= tFa
        if sp > 0:
            total += 1
            held += int(holds); heldFa += int(holdsFa)
            tighter_vals.append(tighter)
        print(f"{strat:>13} 2^{int(math.log2(n)):<4} {kappa:>9.2g} {sp:>10.2e} "
              f"{tH:>10.2e} {tB:>10.2e} {tFa:>10.2e} {'Y' if holds else 'N':>4} {'Y' if holdsFa else 'N':>4}")
    if total:
        print(f"\n  Bernstein / Freedman(measured) tau held in {held}/{total} nondeterministic cells")
        print(f"  Freedman(a priori, sigma2=u^2/3) tau held in {heldFa}/{total}")
        print(f"  median tightening vs Hoeffding: Bernstein {np.median(tighter_vals):.2f}x")
    return held, total, heldFa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", default=None)
    ap.add_argument("--gpu", default=None)
    args = ap.parse_args()

    print(f"Bernstein tau using measured sigma^2/u^2 = {SIGMA2_OVER_U2}")
    print(f"(sigma/u = {math.sqrt(SIGMA2_OVER_U2):.3f}; expected per-term tightening "
          f"~ {1/math.sqrt(SIGMA2_OVER_U2):.2f}x)")

    gh = gt = gfa = 0
    if args.cpu:
        h, t, fa = analyze(args.cpu, "CPU (permutation spread)")
        gh += h; gt += t; gfa += fa
    if args.gpu:
        h, t, fa = analyze(args.gpu, "GPU (real 4090 nondeterminism)")
        gh += h; gt += t; gfa += fa

    print(f"\n{'='*50}")
    if gt:
        if gh == gt:
            print(f"RESULT: Bernstein / Freedman(measured) tau HELD in ALL {gt} nondeterministic cells.")
            print(f"        Freedman(a priori, no measurement) held in {gfa}/{gt}.")
            print("        Tightening holds under mean-independence alone (Freedman), not just independence.")
        else:
            print(f"RESULT: Bernstein tau held in {gh}/{gt} cells — exceeded in {gt-gh}.")
            print("        Tightening too aggressive in those cells: the variance")
            print("        bound is violated there (a finding — conservatism was needed).")


if __name__ == "__main__":
    main()