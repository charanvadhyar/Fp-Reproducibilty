"""
decompose.py — explain WHY the derived tau is ~10x conservative at kappa=1.

The exceedance test found the tightening factor t = (real spread quantile)/tau
is ~0.096 at kappa=1, flat across n. That flatness means the gap is a constant
PRODUCT of conservative choices baked into tau. This script separates them so
"tau is loose by 10x" becomes "tau is loose by 10x BECAUSE ...".

tau = 2 * gamma_tilde(lambda(alpha/2)) * S        (bounds.tau_probabilistic)

The conservative factors, each measured or computed separately:

  F1  two-sided triangle factor:  tau uses 2*(single-run bound). Two independent
      reorder errors add like a random walk (~sqrt2), not linearly (2). So the
      '2' overcounts by 2/sqrt2 = sqrt2 ~ 1.41.   [analytic]

  F2  error-vs-spread:  tau bounds one run's error-from-truth (x2). We measure
      spread between two reordered runs. Compare the empirical spread quantile
      to the empirical single-run error quantile.   [measured from data]

  F3  bound-constant looseness (Hoeffding):  tau uses lambda(alpha/2)~4.07, the
      value that makes the Hoeffding TAIL BOUND hold. The real rounding-error
      sum is Gaussian-ish and lands well inside Hoeffding's envelope, so its real
      (1-alpha) quantile sits at fewer 'natural units' than 4.07.   [measured]

  F4  gamma_tilde vs linear:  exp(...)-1 and the n*u^2 term are upper bounds;
      negligible at these n (included for completeness).   [computed]

Check:  F1 * F2 * F3 * F4  should reproduce  1 / t  (~10.4 at kappa=1).

Usage:
  python decompose.py --sweep results/phaseA.jsonl --exc results/exceedance.jsonl
  (sweep optional: if given, F2 uses real single-run errors; else F2 from theory)
"""

import argparse, json, math
from collections import defaultdict
import numpy as np

from bounds import UNIT_ROUNDOFF, lambda_for_alpha


def load(path):
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exc", required=True, help="exceedance.jsonl")
    ap.add_argument("--sweep", default=None, help="phaseA.jsonl (for measured single-run error)")
    ap.add_argument("--kappa", type=float, default=1.0, help="which kappa_target to analyze")
    ap.add_argument("--alpha", type=float, default=1e-3)
    args = ap.parse_args()

    u = UNIT_ROUNDOFF["float32"]
    exc = [r for r in load(args.exc)
           if r["strategy"] == "recursive" and r["kappa_target"] == args.kappa]
    if not exc:
        print("no matching exceedance records"); return

    # dedupe by n (file may have repeated runs), keep last
    by_n = {}
    for r in exc:
        by_n[r["n"]] = r
    exc = [by_n[n] for n in sorted(by_n)]

    # ---- F1: two-sided triangle factor (analytic) ----
    F1 = 2.0 / math.sqrt(2.0)   # ~1.414

    # ---- F3: Hoeffding bound-constant looseness ----
    # tau's lambda for the spread (alpha/2 each side) :
    # per-cell n matters now (HM Thm 3.1 union bound): use each record's n
    lam_tau_by_n = {r['n']: lambda_for_alpha(args.alpha / 2.0, r['n'], u) for r in exc}
    lam_tau = float(np.median(list(lam_tau_by_n.values())))   # ~6.6 at n~2^20
    # The REAL spread, in 'natural units' u*sqrt(n)*S, at the (1-alpha) quantile:
    # spread_q ~ lam_real * u * sqrt(n) * S   (ignoring the sqrt2 of two runs, which is F1)
    # so lam_real = spread_q / (u*sqrt(n)*S) / sqrt2 ... we fold sqrt2 into F1, so:
    lam_real_list = []
    for r in exc:
        n = r["n"]; S = r["sum_abs"]; q = r["empirical_q_at_1_minus_alpha"]
        natural = u * math.sqrt(n) * S
        lam_real = q / natural if natural > 0 else float("nan")
        lam_real_list.append((n, lam_real))
    lam_real = float(np.median([v for _, v in lam_real_list if math.isfinite(v)]))
    # F3 is how much looser tau's lambda is than the real deviation multiple,
    # after removing the sqrt2 (F1): compare lam_tau (single-side, so /sqrt2 for 2 runs)
    F3 = (lam_tau * math.sqrt(2.0)) / lam_real if lam_real > 0 else float("nan")

    # ---- F4: gamma_tilde vs linear lambda*sqrt(n)*u ----
    # how much exp(...)-1 exceeds the linear term, at a representative n
    nrep = exc[len(exc)//2]["n"]
    lin = lam_tau * math.sqrt(nrep) * u
    gt = math.expm1((lam_tau * math.sqrt(nrep) * u + nrep * u * u) / (1 - u))
    F4 = gt / lin if lin > 0 else float("nan")

    # ---- F2: error-vs-spread (measured, if sweep provided) ----
    F2 = None
    if args.sweep:
        sw = [r for r in load(args.sweep)
              if r["strategy"] == "recursive" and r["kappa_target"] == args.kappa]
        err_by_n = defaultdict(list)
        for r in sw:
            err_by_n[r["n"]].append(r["abs_error"])
        # compare, per n, the single-run error quantile to the spread quantile.
        # Phase A has few seeds, so use the MAX single-run error as a rough high quantile.
        ratios = []
        for r in exc:
            n = r["n"]
            if err_by_n.get(n):
                err_q = max(err_by_n[n])              # rough high quantile of single-run error
                spread_q = r["empirical_q_at_1_minus_alpha"]
                if err_q > 0:
                    ratios.append((2 * err_q) / spread_q)  # tau's "2*error" vs real spread
        if ratios:
            F2 = float(np.median(ratios))

    # ---- report ----
    # IMPORTANT: do NOT read the stored tightening_factor from the jsonl -- it was
    # computed at sweep time with whatever tau was then in bounds.py. Recompute
    # t = (stored spread quantile) / tau_NOW(n) so the decomposition always
    # compares against the current bound. lambda varies with n (HM Thm 3.1
    # union bound: sqrt(2 ln(2n/beta))), so closure is checked PER CELL.
    from bounds import tau_probabilistic
    F1 = 2.0 / math.sqrt(2.0)
    print(f"\nDecomposition of tau conservatism at kappa={args.kappa:g}, alpha={args.alpha:g}")
    print("=" * 76)
    print("lambda now depends on n (union bound over n terms), so the accounting is per cell:")
    print(f"{'n':>6} {'lam_tau':>8} {'lam_real':>9} {'t=q/tau':>9} {'1/t':>7} {'F1*F3':>7} {'ratio':>6}")
    print("-" * 60)
    ratios=[]; ts=[]
    for r in exc:
        n=r["n"]; S=r["sum_abs"]; q=r["empirical_q_at_1_minus_alpha"]
        tau_now = tau_probabilistic(n, u, S, args.alpha)
        t = q / tau_now
        lam_tau_n = lambda_for_alpha(args.alpha/2.0, n, u)
        lam_real_n = q / (u*math.sqrt(n)*S)
        F3_n = (lam_tau_n*math.sqrt(2.0))/lam_real_n
        pred = F1*F3_n
        ratio = pred*t          # should be ~1.0 if pred == 1/t
        ratios.append(ratio); ts.append(t)
        print(f"2^{int(math.log2(n)):<4} {lam_tau_n:>8.3f} {lam_real_n:>9.3f} {t:>9.4f} {1/t:>7.2f} {pred:>7.2f} {ratio:>6.3f}")
    print("-" * 60)
    print(f"median closure ratio (F1*F3 * t) = {float(np.median(ratios)):.3f}   (1.000 = exact)")
    print(f"t ranges {min(ts):.4f} .. {max(ts):.4f}: mild n-dependence from sqrt(ln n) in lambda;")
    print(f"lam_real is the flat invariant (~{lam_real:.2f}).")
    print()
    t_measured = float(np.median(ts)); target = 1.0/t_measured
    F3 = (float(np.median([lambda_for_alpha(args.alpha/2.0, r['n'], u) for r in exc]))*math.sqrt(2.0))/lam_real
    product = F1*F3*F4
    print(f"POOLED (median): F1={F1:.3f} F3={F3:.2f} F4={F4:.3f} -> F1*F3*F4={product:.2f} vs 1/t={target:.2f}")
    print()
    print("DIAGNOSTIC (not in the product -- reported separately):")
    if F2 is not None:
        print(f"  F2  (2*single-run error)/spread   = {F2:.3f}")
        print(f"      spread is {'tighter' if F2>1 else 'looser'} than 2x single-run error")
        print(f"      by {F2:.2f}x -- consistent with random-walk addition of two errors")
    else:
        print(f"  F2  (2*error)/spread              = (need --sweep to measure)")
    print()
    print("Per-n real spread deviation multiple (flat => structural, not scaling):")
    for n, lr in lam_real_list:
        print(f"  n=2^{int(math.log2(n)):<2}  lam_real = {lr:.3f}")


if __name__ == "__main__":
    main()
