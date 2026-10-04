"""
bounds.py — forward error bounds and verification tolerances.

Three layers, loosest to tightest:
  1. deterministic worst-case  (gamma_{n-1}, Higham 2002)      -> grows ~ n*u
  2. probabilistic / Hoeffding (tau_probabilistic)             -> grows ~ sqrt(n)*u
  3. variance-aware / Bernstein (tau_bernstein)                -> ~2.3x tighter still

Notation (Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed.):
  u        unit roundoff. fp32: 2^-24, fp64: 2^-53, bf16: 2^-8, fp16/tf32: 2^-11.
  gamma_k  = k*u / (1 - k*u)         (Higham Lemma 3.1)
  S        = sum_i |x_i|

Probabilistic bound: Higham & Mary 2019, SIAM J. Sci. Comput. 41(5), Thm 2.4.
  |s_hat - s| <= gamma_tilde_n(lambda) * S   with prob >= 1 - 2 exp(-lambda^2/2),
  gamma_tilde_n(lambda) = exp((lambda*sqrt(n)*u + n*u^2)/(1-u)) - 1.
  [VERIFY the (1-u) placement and the prefactor against the paper.]

Bernstein bound: uses the measured rounding-error variance sigma^2 instead of
the worst-case u^2. sigma^2/u^2 measured ~0.18 on fp32 (variance.py), stable
across kappa, errors mean-zero -> Bernstein's assumptions hold.
"""

import math

UNIT_ROUNDOFF = {
    "float64": 2.0 ** -53,
    "float32": 2.0 ** -24,
    "tf32":    2.0 ** -11,
    "float16": 2.0 ** -11,
    "bfloat16": 2.0 ** -8,
}

# Measured per-rounding relative-error variance as a fraction of u^2.
# variance.py: 0.186 at kappa=1, 0.179 at kappa=1e6 (fp32). Mean ~0 (unbiased).
# Uniform-on-[-u,u] theory would give 1/3; real round-to-nearest concentrates
# nearer zero, so the measured value is smaller (tighter).
SIGMA2_OVER_U2 = 0.18


# ----------------------------------------------------------- deterministic
def gamma(k: float, u: float) -> float:
    """Higham's gamma_k = k*u / (1 - k*u). Requires k*u < 1."""
    ku = k * u
    if ku >= 1.0:
        return math.inf
    return ku / (1.0 - ku)


def deterministic_bound(strategy: str, n: int, u: float, S: float) -> float:
    """Worst-case |s_hat - s| for a fixed-order strategy. S = sum|x_i|."""
    if strategy == "recursive":
        return gamma(n - 1, u) * S
    if strategy in ("pairwise", "numpy_sum"):
        return gamma(math.ceil(math.log2(n)), u) * S
    if strategy == "kahan":
        return (2 * u + n * u * u) * S
    raise ValueError(strategy)


def kappa_ceiling(strategy: str, n: int, u: float) -> float:
    """Conditioning beyond which worst-case relative error reaches 1."""
    from strategies import chain_length
    k = chain_length(strategy, n)
    g = gamma(k, u)
    return math.inf if g == 0 else 1.0 / g


# ----------------------------------------------------------- probabilistic (Hoeffding)
def lambda_for_alpha(alpha: float, n: int, u: float) -> float:
    """
    Higham & Mary (2019). Thm 2.4 gives, for ONE product of (1+delta) factors,
        P(lambda) = 1 - 2 exp(-lambda^2 (1-u)^2 / 2).
    Thm 3.1 (inner products / sums of n terms) applies it to all n terms via a
    union bound, so the SUM bound holds with probability at least
        Q(lambda, n) = 1 - n (1 - P(lambda)) = 1 - 2n exp(-lambda^2 (1-u)^2 / 2).
    Setting the failure probability n(1-P) = alpha and solving:
        lambda(alpha, n) = sqrt(2 ln(2n/alpha)) / (1-u).
    The n DOES appear (the union bound over n terms). An earlier version of
    this file used Thm 2.4's P(lambda) directly and omitted the n, which made
    lambda too small by ~1.6x at n=2^20. Corrected per HM Thm 3.1, eq (3.1).
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must satisfy 0 < alpha < 1")
    if n < 1:
        raise ValueError("n must be >= 1")
    if not (0.0 <= u < 1.0):
        raise ValueError("u must satisfy 0 <= u < 1")
    return math.sqrt(2.0 * math.log(2.0 * n / alpha)) / (1.0 - u)


def tau_probabilistic(n: int, u: float, S: float, alpha: float) -> float:
    """
    Two-run tolerance from the Hoeffding/probabilistic bound.
    Each run gets failure prob alpha/2 (union bound over two runs).
    tau = 2 * gamma_tilde(lambda(alpha/2)) * S.
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
    return 2.0 * math.expm1(exponent) * S


# ----------------------------------------------------------- variance-aware (Bernstein)
def tau_bernstein(n: int, u: float, S: float, alpha: float,
                  sigma2_over_u2: float = SIGMA2_OVER_U2) -> float:
    """
    Tighter two-run tolerance using Bernstein's inequality, which uses the
    measured rounding-error variance sigma^2 = sigma2_over_u2 * u^2 rather than
    the worst-case u^2 that tau_probabilistic assumes.

    Two-sided Bernstein per term, union-bounded over the n terms (matching the
    structure of HM Thm 3.1), with beta = alpha/2 per run (union over two runs):
        P(|sum delta| >= t) <= 2 exp(-t^2 / (2(n sigma^2 + u t/3)))
    Setting 2n exp(...) = beta and solving for t gives the quadratic
        t^2 - (2 u L / 3) t - 2 n sigma^2 L = 0,   L = ln(2n/beta),
    whose positive root bounds one run's relative error; two runs differ by
    at most 2*t*S.

    Assumptions (state these in the paper): mean-independent, mean-zero
    rounding errors with |delta| <= u and variance sigma^2 -- the HM Model 2.1
    plus a variance parameter. sigma2_over_u2 is MEASURED (variance.py), so
    tau_bernstein is a bound under a calibrated error model, not purely a
    priori. Recovers the variance-attributable part of tau_probabilistic's
    conservatism (~2.35x at the measured sigma; the ratio is invariant to the
    n-factor correction since both bounds scale as sqrt(L)). Valid only where
    the errors are unbiased: fp32/fp64 (verified), NOT fp16 (biased).
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must satisfy 0 < alpha < 1")
    if not (0.0 <= u < 1.0):
        raise ValueError("u must satisfy 0 <= u < 1")
    if S < 0.0:
        raise ValueError("S must be nonnegative")
    beta = alpha / 2.0
    L = math.log(2.0 * n / beta)
    sigma2 = sigma2_over_u2 * u * u
    b = -(2.0 * u * L / 3.0)
    c = -(2.0 * n * sigma2 * L)
    t = (-b + math.sqrt(b * b - 4.0 * c)) / 2.0
    return 2.0 * t * S


if __name__ == "__main__":
    u = UNIT_ROUNDOFF["float32"]

    # regression asserts: lock the derivations
    th = tau_probabilistic(2**20, u, 1.0, 1e-3)
    tb = tau_bernstein(2**20, u, 1.0, 1e-3)
    # corrected per HM Thm 3.1 (union bound over n terms): lambda = sqrt(2 ln(2n/beta))/(1-u)
    assert abs(th - 8.128e-4) < 1e-6, f"hoeffding tau drifted: {th:.6e}"
    assert abs(tb - 3.456e-4) < 1e-6, f"bernstein tau drifted: {tb:.6e}"
    print(f"assert passed.")
    print(f"  tau_hoeffding  (n=2^20) = {th:.6e}")
    print(f"  tau_bernstein  (n=2^20) = {tb:.6e}")
    print(f"  bernstein is {th/tb:.2f}x tighter")
    print()
    print("kappa ceiling at n=2^20, fp32 (worst-case relative error hits 1):")
    for s in ["recursive", "pairwise", "kahan"]:
        print(f"  {s:>10}: kappa_max ~ {kappa_ceiling(s, 2**20, u):.3g}")