"""
reference.py — the trusted 'exact' sum.

Spread (two computed results disagreeing) needs no ground truth.
ERROR (distance from the exact sum) does. This module supplies it.

Reference: math.fsum. For float inputs it returns the exact real-number sum
correctly rounded to float64 (Shewchuk's algorithm). fp32 inputs are exactly
representable in fp64, so the reference carries relative error <= 2^-53 —
about 10^9x finer than the fp32 errors (~2^-24 * n) we measure. Adequate.

NOT used as reference: fp64 on the GPU. That is still a parallel reduction
and inherits the same ordering problem it would be refereeing.

validate() checks fsum against fractions.Fraction (truly exact, very slow)
at small n so the fast path is trusted before it is used at large n.
"""

from fractions import Fraction
import math
import numpy as np


def exact_sum(x: np.ndarray) -> float:
    """Correctly-rounded fp64 sum of the (exact) fp32 inputs."""
    return math.fsum(np.asarray(x, dtype=np.float32).astype(np.float64).tolist())


def exact_sum_fraction(x: np.ndarray) -> Fraction:
    """Truly exact rational sum. O(n) big-integer arithmetic — small n only."""
    total = Fraction(0)
    for v in np.asarray(x, dtype=np.float32).tolist():
        total += Fraction(v)  # every float is an exact dyadic rational
    return total


def sum_abs(x: np.ndarray) -> float:
    """Σ|x_i| in fp64 via fsum — the quantity every bound scales with."""
    return math.fsum(np.abs(np.asarray(x, dtype=np.float32)).astype(np.float64).tolist())


def validate(n_values=(16, 256, 4096), trials=5, seed=0) -> bool:
    """fsum must match Fraction to within one fp64 rounding."""
    rng = np.random.default_rng(seed)
    ok = True
    for n in n_values:
        for _ in range(trials):
            # mixed magnitudes and signs so cancellation actually happens
            e = rng.integers(-10, 10, size=n)
            x = ((rng.random(n) * 2 - 1) * (2.0 ** e)).astype(np.float32)
            fast = exact_sum(x)
            true = exact_sum_fraction(x)
            # fsum is correctly rounded: |fast - true| <= 0.5 ulp(fast)
            tol = abs(np.spacing(fast)) if fast != 0 else 5e-324
            if abs(Fraction(fast) - true) > Fraction(tol):
                print(f"  MISMATCH n={n}: fsum={fast!r} fraction={float(true)!r}")
                ok = False
    print("reference validation:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    validate()
