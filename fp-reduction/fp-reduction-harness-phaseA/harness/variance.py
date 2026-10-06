"""
variance.py — measure the per-rounding error mean and variance (v2).

Why this exists
---------------
The Freedman (variance-aware) tolerance needs a bound on the CONDITIONAL
variance of each rounding error, E[delta_k^2 | past]. This script measures:

  (a) the marginal mean and variance of delta_k over many sums   (as before)
  (b) the variance BINNED by position in the sum (i.e. by the magnitude of the
      running partial sum), and reports the maximum binned variance — an
      empirical proxy for the conditional-variance bound a reviewer asks for.

v2 changes (review-2 fixes)
---------------------------
  * exact reference step via fractions.Fraction for EVERY dtype, so fp64 can
    be measured too (v1 used fp64 as the "exact" step, which is exact for fp32
    but trivially zero for fp64).
  * --binned: conditional-variance proxy (max over bins of var(delta_k)).
  * --kappa uses gensum when available, else a cancelling construction.
  * reports standard error of the mean, so "unbiased" has a number behind it.
  * NOTE on regime: for fp16, n must satisfy n*u < 1 (n < 2048) to be inside
    the bound's validity regime. Measuring at n = 50,000 (v1 default) puts
    fp16 in the stagnation regime, where errors are systematically negative.
    Run both to show the regime boundary.

Usage
-----
  python variance.py                                  # fp32, n=50000, 8 sums
  python variance.py --dtype float16 --n 1000 --sums 40
  python variance.py --dtype float16 --n 50000 --sums 8     # outside regime
  python variance.py --dtype float64 --n 50000 --sums 8
  python variance.py --binned --bins 20
"""

import argparse, math, sys
from fractions import Fraction
import numpy as np

U_BY_DTYPE = {"float32": 2.0**-24, "float16": 2.0**-11, "float64": 2.0**-53}


def collect_deltas(x, dtype="float32"):
    """
    Recursive sum in `dtype`; return delta_k = (fl(s+x_k) - (s+x_k)) / (s+x_k)
    with the exact step (s + x_k) evaluated in exact rational arithmetic.
    Returns (deltas, positions) with positions = index k of each addition.
    """
    npdt = getattr(np, dtype)
    deltas, pos = [], []
    s = npdt(0.0)
    for k, xk in enumerate(x):
        exact_step = Fraction(float(s)) + Fraction(float(xk))   # exact
        fl_step = npdt(s + xk)
        if not np.isfinite(fl_step):
            break                                                # overflow: stop
        if exact_step != 0:
            d = (Fraction(float(fl_step)) - exact_step) / exact_step
            deltas.append(float(d))
            pos.append(k)
        s = fl_step
    return np.array(deltas), np.array(pos)


def make_input(n, kappa, rng, npdt, seed):
    """Inputs with prescribed conditioning. Uses gensum if importable."""
    try:
        from gensum import gensum
        x, s_exact, k_ach = gensum(n, kappa, seed=seed)
        return x.astype(npdt)
    except Exception:
        if kappa <= 1:
            return (rng.random(n) + 0.5).astype(npdt)
        # cancelling construction: pairs +a, -a plus a small residual
        a = rng.random(n // 2) + 0.5
        x = np.concatenate([a, -a])
        x[-1] += np.abs(x).sum() / kappa
        rng.shuffle(x)
        return x.astype(npdt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--sums", type=int, default=8)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="float32", choices=list(U_BY_DTYPE))
    ap.add_argument("--binned", action="store_true",
                    help="report variance binned by position (conditional-variance proxy)")
    ap.add_argument("--bins", type=int, default=10)
    args = ap.parse_args()

    U = U_BY_DTYPE[args.dtype]
    npdt = getattr(np, args.dtype)
    rng = np.random.default_rng(args.seed)
    all_d, all_p = [], []
    for i in range(args.sums):
        x = make_input(args.n, args.kappa, rng, npdt, args.seed + i)
        d, p = collect_deltas(x, args.dtype)
        all_d.append(d); all_p.append(p)
    d = np.concatenate(all_d); p = np.concatenate(all_p)

    mean = float(d.mean()); var = float(d.var()); sd = math.sqrt(var)
    sem = sd / math.sqrt(len(d))
    print(f"dtype = {args.dtype}   u = {U:.3e}   n = {args.n}   n*u = {args.n*U:.3g}"
          f"   {'INSIDE' if args.n*U < 1 else 'OUTSIDE'} regime n*u<1")
    print(f"samples (delta_k): {len(d):,}   kappa requested = {args.kappa:g}")
    print(f"mean(delta)/u      = {mean/U:+.4f}   (SE of mean / u = {sem/U:.4f};"
          f" |mean| / SE = {abs(mean)/sem:.1f})")
    print(f"sigma^2 / u^2      = {var/(U*U):.4f}   (uniform model: 0.3333; worst case: 1.0)")
    print(f"max |delta| / u    = {np.abs(d).max()/U:.4f}   (must be <= 1)")

    if args.binned:
        edges = np.linspace(0, args.n, args.bins + 1)
        print(f"\nvariance by position bin (conditional-variance proxy), {args.bins} bins:")
        mx = 0.0
        for b in range(args.bins):
            m = (p >= edges[b]) & (p < edges[b + 1])
            if m.sum() < 100:
                continue
            v = float(d[m].var()) / (U * U); mu = float(d[m].mean()) / U
            mx = max(mx, v)
            print(f"  k in [{int(edges[b]):>7},{int(edges[b+1]):>7}):  "
                  f"mean/u = {mu:+.4f}   sigma^2/u^2 = {v:.4f}   (N={m.sum():,})")
        print(f"max binned sigma^2/u^2 = {mx:.4f}   "
              f"({'<= 1/3: uniform-model bound respected' if mx <= 1/3 else '> 1/3: uniform-model bound VIOLATED'})")


if __name__ == "__main__":
    main()
