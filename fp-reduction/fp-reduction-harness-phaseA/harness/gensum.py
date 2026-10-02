"""
gensum.py — generate summation inputs with a PRESCRIBED condition number.

Condition number of summation:  kappa = Σ|x_i| / |Σ x_i|
  kappa = 1   : all same sign, no cancellation
  kappa >> 1  : terms nearly cancel; every error bound scales linearly in kappa

Sampling from one distribution gives ONE point on the kappa-curve. To learn
anything general, kappa must be swept deliberately. This is the standard way
to do it, after Ogita, Rump & Oishi, "Accurate Sum and Dot Product",
SIAM J. Sci. Comput. 26(6), 2005, Algorithm 6.1 (GenDot), adapted to sums.

Idea: the first half gets random exponents in [0, b/2] where b = log2(kappa),
so |x_i| spans a wide range. The second half is chosen to CANCEL the exact
running sum, so the final sum is tiny relative to Σ|x_i| — i.e. large kappa.
A final random permutation removes any structure in the order.

The exact sum is recomputed with fsum at the end, and the ACHIEVED kappa is
reported — never assume the target was hit exactly (rounding each x_i to
fp32 perturbs it). Use kappa_achieved in every analysis.
"""

import math
import numpy as np
from reference import exact_sum, sum_abs


def gensum(n: int, kappa_target: float, seed: int = 0):
    """
    Returns (x: float32[n], s_exact: float, kappa_achieved: float).
    kappa_target >= 1. For kappa_target == 1 returns all-positive data.
    """
    rng = np.random.default_rng(seed)
    if kappa_target <= 1.0:
        x = rng.random(n).astype(np.float32) + np.float32(0.5)  # all positive
        s = exact_sum(x)
        return x, s, sum_abs(x) / abs(s)

    b = math.log2(kappa_target)
    n2 = max(n // 2, 1)
    x = np.zeros(n, dtype=np.float64)

    # --- first half: random exponents in [0, b/2], endpoints forced ---
    e = np.round(rng.random(n2) * b / 2).astype(int)
    e[0] = int(round(b / 2)) + 1
    e[-1] = 0
    x[:n2] = (2 * rng.random(n2) - 1) * (2.0 ** e)

    # --- second half: exponents decreasing b/2 -> 0, each chosen to cancel ---
    # We fill n-1 elements this way, then use the LAST element to set the
    # residual sum to ~ sum|x| / kappa_target, which pins kappa near target.
    m = n - 1
    e2 = np.round(np.linspace(b / 2, 0, m - n2)).astype(int)
    running = math.fsum(x[:n2].astype(np.float32).astype(np.float64).tolist())
    for j, i in enumerate(range(n2, m)):
        target = (2 * rng.random() - 1) * (2.0 ** e2[j])
        xi = np.float32(target - running)          # cancel the running exact sum
        x[i] = float(xi)
        running = math.fsum([running, float(xi)])   # keep the running sum exact

    # --- last element: set residual so that sum|x| / |sum| ~= kappa_target ---
    A = math.fsum(np.abs(x[:m]).astype(np.float32).astype(np.float64).tolist())
    x[m] = float(np.float32(A / kappa_target - running))

    # --- random permutation, cast, and measure what we actually got ---
    x = x[rng.permutation(n)].astype(np.float32)
    s = exact_sum(x)
    kappa = sum_abs(x) / abs(s) if s != 0 else float("inf")
    return x, s, kappa


if __name__ == "__main__":
    for kt in [1, 1e2, 1e4, 1e6]:
        x, s, k = gensum(4096, kt, seed=1)
        print(f"target kappa={kt:>8g}  achieved={k:>12.4g}  exact_sum={s:+.6e}  sum|x|={sum_abs(x):.4e}")
