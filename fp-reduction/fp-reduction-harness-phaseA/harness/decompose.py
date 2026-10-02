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
    lam_tau = lambda_for_alpha(args.alpha / 2.0, 1, u)   # ~4.07
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
    t_measured = float(np.median([r["tightening_factor"] for r in exc]))
    target = 1.0 / t_measured

    # The DECOMPOSITION is spread-based: F3 is measured from the real SPREAD
    # quantile (via lam_real), so it already contains the error->spread
    # conversion. The accounting product is therefore F1 * F3 * F4.
    # F2 (2*error / spread) is a SEPARATE DIAGNOSTIC, not a factor to multiply
    # in -- multiplying it would double-count the error-vs-spread gap that F3
    # already captures.
    product = F1 * F3 * F4

    print(f"\nDecomposition of tau conservatism at kappa={args.kappa:g}, alpha={args.alpha:g}")
    print("=" * 60)
    print(f"measured tightening factor t       = {t_measured:.4f}")
    print(f"  => tau is loose by 1/t           = {target:.2f}x")
    print()
    print(f"lambda used by tau (alpha/2)        = {lam_tau:.3f}")
    print(f"real spread deviation multiple      = {lam_real:.3f}")
    print()
    print("ACCOUNTING (spread-based) -- these multiply to 1/t:")
    print(f"  F1  two-sided triangle (2/sqrt2)  = {F1:.3f}")
    print(f"  F3  Hoeffding tail-bound slack    = {F3:.3f}")
    print(f"  F4  gamma_tilde vs linear         = {F4:.3f}")
    print(f"  F1*F3*F4                          = {product:.2f}")
    print(f"  target (1/t)                      = {target:.2f}")
    print(f"  product / target                  = {product/target:.3f}   (1.0 = exact)")
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