"""
variance.py — measure the per-rounding error variance.

Bernstein's bound replaces Hoeffding's use of the MAX error (u) with the
VARIANCE of the error (sigma^2). The tightening factor hinges on how small
sigma^2 is relative to u^2:

  Hoeffding assumes each delta can be as large as u           -> uses u^2
  Bernstein uses the actual spread of delta                   -> uses sigma^2

For round-to-nearest, THEORY says delta is ~uniform on [-u, +u], giving
  sigma^2 = u^2 / 3      (variance of a uniform distribution on [-u,u])
so sigma^2 / u^2 ~ 0.333. That factor is what Bernstein exploits.

This script MEASURES sigma^2 directly, because the Phase A sweep stored only
final errors, not individual rounding errors. It:
  1. does recursive summation in fp32
  2. at each addition, computes the exact rounding error delta_k =
     (fl(s+x) - (s+x)) / (s+x)   using an fp64 "truth" for that single step
  3. collects all delta_k across many sums and reports their variance

Output: measured sigma^2 / u^2. If ~0.33, Bernstein's assumption holds and the
~sqrt(3)x-per-term tightening is justified. If much larger, the errors are
biased/heavy-tailed and Bernstein won't deliver the full gain -> a finding.

Usage:
  python variance.py
  python variance.py --n 100000 --sums 50 --kappa 1
"""

import argparse, math
import numpy as np

U = 2.0 ** -24  # fp32 unit roundoff


def collect_deltas(x):
    """
    Recursive fp32 sum; return the relative rounding error delta_k of each
    addition. delta_k = (fl(s+x_k) - (s+x_k)) / (s+x_k), with (s+x_k) the
    exact (fp64) value of that single step's operands.
    """
    deltas = []
    s = np.float32(0.0)
    for xk in x:
        exact_step = np.float64(s) + np.float64(xk)   # exact sum of THIS step
        fl_step = np.float32(s + xk)                   # rounded fp32 result
        if exact_step != 0.0:
            d = (np.float64(fl_step) - exact_step) / exact_step
            deltas.append(float(d))
        s = fl_step
    return np.array(deltas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--sums", type=int, default=40)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    all_d = []
    for i in range(args.sums):
        if args.kappa <= 1:
            x = (rng.random(args.n).astype(np.float32) + np.float32(0.5))
        else:
            # mixed-sign data to raise conditioning
            x = ((rng.random(args.n) * 2 - 1).astype(np.float32))
        all_d.append(collect_deltas(x))
    d = np.concatenate(all_d)

    mean = float(d.mean())
    var = float(d.var())
    sigma2_over_u2 = var / (U * U)
    # also the "effective" bound Bernstein would use vs Hoeffding
    #   Hoeffding per-term scale: u
    #   Bernstein per-term scale: sigma = sqrt(var)
    per_term_ratio = math.sqrt(var) / U  # sigma / u ; ~1/sqrt(3)=0.577 if uniform

    print(f"fp32 u = {U:.3e},  u^2 = {U*U:.3e}")
    print(f"samples (delta_k): {len(d):,}")
    print()
    print(f"mean(delta)        = {mean:+.3e}   (should be ~0 if unbiased)")
    print(f"var(delta)         = {var:.3e}")
    print(f"sigma^2 / u^2      = {sigma2_over_u2:.4f}   (uniform theory: 0.333)")
    print(f"sigma / u          = {per_term_ratio:.4f}   (uniform theory: 0.577)")
    print()
    print("Interpretation:")
    if abs(mean) < 0.1 * math.sqrt(var) and 0.2 < sigma2_over_u2 < 0.5:
        print("  mean~0 and sigma^2/u^2 ~ 1/3: round-to-nearest behaves ~uniform.")
        print("  Bernstein's variance assumption HOLDS -> tightening justified.")
        print(f"  Expected per-term tightening ~ u/sigma = {1/per_term_ratio:.2f}x")
    elif abs(mean) >= 0.1 * math.sqrt(var):
        print("  mean is NOT ~0: errors are BIASED (swamping/structure).")
        print("  Bernstein assumes mean-zero -> gain reduced; this is a finding.")
    else:
        print(f"  sigma^2/u^2 = {sigma2_over_u2:.3f}, off the uniform 1/3.")
        print("  Bernstein still helps but by a different factor than sqrt(3).")


if __name__ == "__main__":
    main()
