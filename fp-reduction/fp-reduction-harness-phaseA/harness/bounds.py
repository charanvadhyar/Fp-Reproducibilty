"""
bounds.py — forward error bounds the measurements are checked against.

Notation (Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed.):
  u        unit roundoff. fp32: 2^-24. fp64: 2^-53. bf16: 2^-8. fp16: 2^-11.
  gamma_k  = k*u / (1 - k*u)         (Higham Lemma 3.1)
  S        = sum_i |x_i|

DETERMINISTIC (worst-case) forward error bounds, all of the form
    |s_hat - s| <= gamma_k * S
with k = the length of the longest rounding chain:
    recursive : k = n - 1                    (Higham eq. 4.4)
    pairwise  : k = ceil(log2 n)             (Higham Sec. 4.2)
    kahan     : ~ 2u * S + O(n u^2) S        (Higham eq. 4.9, Neumaier-type)

These are GUARANTEED but pessimistic: they assume every rounding error is
maximal and every one reinforces the others. Real error sits far below them.

PROBABILISTIC bound (Higham & Mary 2019, SIAM J. Sci. Comput. 41(5)):
    treats rounding errors as mean-independent, mean-zero -> error grows
    like ~ lambda * sqrt(n) * u instead of n * u.
    ---> tau_probabilistic() below is LEFT FOR YOU TO DERIVE AND FILL IN.
    The formula in its docstring is from memory; verify every constant
    against the paper before it goes anywhere. Provenance: it is a
    'source' entry only once you have checked it.
"""

import math

UNIT_ROUNDOFF = {
    "float64": 2.0 ** -53,
    "float32": 2.0 ** -24,
    "tf32":    2.0 ** -11,
    "float16": 2.0 ** -11,
    "bfloat16": 2.0 ** -8,
}



def gamma(k: float, u: float) -> float:
    """Higham's gamma_k = k*u / (1 - k*u). Requires k*u < 1."""
    ku = k * u
    if ku >= 1.0:
        return math.inf  # bound is vacuous: worst case exceeds the answer itself
    return ku / (1.0 - ku)


def deterministic_bound(strategy: str, n: int, u: float, S: float) -> float:
    """Worst-case |s_hat - s| for a fixed-order strategy. S = sum|x_i|."""
    if strategy == "recursive":
        return gamma(n - 1, u) * S
    if strategy in ("pairwise", "numpy_sum"):
        return gamma(math.ceil(math.log2(n)), u) * S
    if strategy == "kahan":
        # Neumaier/Kahan: leading term 2u, plus O(n u^2). (Higham eq. 4.9)
        return (2 * u + n * u * u) * S
    raise ValueError(strategy)


def kappa_ceiling(strategy: str, n: int, u: float) -> float:
    """
    The conditioning beyond which the strategy's worst-case relative error
    reaches 1 (no correct digits). Derived from  gamma_k * kappa >= 1.
    Useful for choosing kappa levels that still produce meaningful data.
    """
    from strategies import chain_length
    k = chain_length(strategy, n)
    g = gamma(k, u)
    return math.inf if g == 0 else 1.0 / g


# ---------------------------------------------------------------- YOURS
def lambda_for_alpha(alpha: float, n: int, u: float) -> float:
    """
    Higham & Mary (2019), Theorem 2.4.

    P(lambda) = 1 - 2 exp(-lambda^2 / 2).

    Choosing P(lambda) >= 1 - alpha gives

        lambda = sqrt(2 * log(2 / alpha)).

    The probability is independent of n. The sqrt(n) dependence
    appears in gamma_tilde, not in lambda.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must satisfy 0 < alpha < 1")

    return math.sqrt(2.0 * math.log(2.0 / alpha))


def tau_probabilistic(n: int, u: float, S: float, alpha: float) -> float:
    """
    Two independent computations, each with failure probability alpha/2.

    Higham & Mary Theorem 2.4:

        gamma_tilde_n(lambda)
          = exp((lambda*sqrt(n)*u + n*u^2) / (1-u)) - 1

    and

        P(gamma_tilde bound holds)
          >= 1 - 2 exp(-lambda^2/2).

    Applying the union bound to the two computed results gives
    overall failure probability <= alpha.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must satisfy 0 < alpha < 1")
    if not (0.0 <= u < 1.0):
        raise ValueError("u must satisfy 0 <= u < 1")
    if S < 0.0:
        raise ValueError("S must be nonnegative")

    lam = lambda_for_alpha(alpha / 2.0, n, u)

    exponent = (lam * math.sqrt(n) * u + n * u * u) / (1.0 - u)

    gamma_tilde = math.expm1(exponent)

    return 2.0 * gamma_tilde * S

if __name__ == "__main__":
    u = UNIT_ROUNDOFF["float32"]
    got = tau_probabilistic(2**20, u, 1.0, 1e-3)
    expected = 4.972e-4
    assert abs(got - expected) < 1e-6, f"tau/S changed: got {got:.6e}, expected {expected:.6e}"
    print(f"assert passed: tau/S = {got:.6e}")
    print("kappa ceiling at n=2^20, fp32 (relative error hits 1):")
    for s in ["recursive", "pairwise", "kahan"]:
        print(f"  {s:>10}: kappa_max ~ {kappa_ceiling(s, 2**20, u):.3g}")
