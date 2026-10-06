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
  gamma_tilde_n(lambda) = exp(lambda*sqrt(n)*u + n*u^2/(1-u)) - 1.
  (Higham & Mary 2019, eq. (2.1): the 1/(1-u) applies to the second-order term only.
   Verified against the paper 2026-10-06; the earlier form divided both terms by (1-u),
   a relative difference of ~u that does not change any reported value.)

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
    exponent = lam * math.sqrt(n) * u + n * u * u / (1.0 - u)   # HM 2019 eq. (2.1)
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


# ----------------------------------------------------------- variance-aware (Freedman: martingale Bernstein)
SIGMA2_OVER_U2_UNIFORM = 1.0 / 3.0   # theoretical max for round-to-nearest (uniform on [-u,u]); a priori, no measurement


def tau_freedman(n: int, u: float, S: float, alpha: float,
                 sigma2_over_u2: float = SIGMA2_OVER_U2) -> float:
    """
    Variance-aware two-run tolerance via FREEDMAN's inequality -- the martingale
    form of Bernstein -- sound under mean-independence alone (HM Model 2.1),
    which is the assumption the rest of the paper makes. Bernstein requires
    full independence; Freedman does not.

    Freedman (1975): for a martingale difference sequence d_k with |d_k| <= c and
    predictable quadratic variation V = sum_k E[d_k^2 | past],
        P( sum d_k >= t  and  V <= v ) <= exp( -t^2 / (2(v + c t/3)) ).
    With d_k = delta_k, c = u, and v = n*sigma2 as the bound on the accumulated
    conditional variance, the two-sided per-term tail union-bounded over the n
    terms (HM Thm 3.1 structure), per-run failure beta = alpha/2:
        2n exp( -t^2 / (2(n sigma2 + u t/3)) ) = beta
    gives the SAME quadratic as tau_bernstein. Numerically identical; the
    difference is the assumption: mean-independence + a bound on conditional
    variance, instead of independence.

    sigma2_over_u2 choices:
      SIGMA2_OVER_U2 (0.18, MEASURED)  -> calibrated tolerance, ~2.35x tighter than Hoeffding
      SIGMA2_OVER_U2_UNIFORM (1/3)     -> a-priori tolerance using the theoretical maximum
                                          variance of round-to-nearest; no measurement
                                          enters; ~1.7x tighter than Hoeffding, fully sound
    The a-priori version is a true bound; the measured version is a bound under
    the calibrated conditional-variance model. Report both.
    """
    if n < 1 or not (0.0 < alpha < 1.0) or not (0.0 <= u < 1.0) or S < 0.0:
        raise ValueError("bad args")
    beta = alpha / 2.0
    L = math.log(2.0 * n / beta)
    v = sigma2_over_u2 * u * u          # per-term conditional-variance bound
    b = -(2.0 * u * L / 3.0)
    c = -(2.0 * n * v * L)
    t = (-b + math.sqrt(b * b - 4.0 * c)) / 2.0
    return 2.0 * t * S


if __name__ == "__main__":
    u = UNIT_ROUNDOFF["float32"]

    # regression asserts: lock the derivations
    th = tau_probabilistic(2**20, u, 1.0, 1e-3)
    tb = tau_bernstein(2**20, u, 1.0, 1e-3)
    # corrected per HM Thm 3.1 (union bound over n terms): lambda = sqrt(2 ln(2n/beta))/(1-u)
    tf_m = tau_freedman(2**20, u, 1.0, 1e-3)                                   # measured sigma2
    tf_a = tau_freedman(2**20, u, 1.0, 1e-3, sigma2_over_u2=SIGMA2_OVER_U2_UNIFORM)  # a priori
    assert abs(th - 8.128e-4) < 1e-6, f"hoeffding tau drifted: {th:.6e}"
    assert abs(tb - 3.456e-4) < 1e-6, f"bernstein tau drifted: {tb:.6e}"
    assert abs(tf_m - tb) < 1e-12, "freedman(measured) must equal bernstein numerically"
    print(f"assert passed.")
    print(f"  tau_hoeffding            (n=2^20) = {th:.6e}")
    print(f"  tau_bernstein            (n=2^20) = {tb:.6e}   ({th/tb:.2f}x tighter; assumes independence)")
    print(f"  tau_freedman (measured)  (n=2^20) = {tf_m:.6e}   ({th/tf_m:.2f}x tighter; mean-independence + calibrated var)")
    print(f"  tau_freedman (a priori)  (n=2^20) = {tf_a:.6e}   ({th/tf_a:.2f}x tighter; mean-independence, sigma2=u^2/3, no measurement)")
    print()
    print("kappa ceiling at n=2^20, fp32 (worst-case relative error hits 1):")
    for s in ["recursive", "pairwise", "kahan"]:
        print(f"  {s:>10}: kappa_max ~ {kappa_ceiling(s, 2**20, u):.3g}")


# ---------------------------------------------------------------------------
# WP2 (journal version): ORDER-ROBUST PARTIAL-SUM TOLERANCES
#
# Exact error identity for any evaluation tree (chain, pairwise, atomic order):
#     s_hat - s = sum_v delta_v * t_v,       t_v = (exact sum of the two operands at node v)
# t_v is measurable w.r.t. the roundings before node v, so under the
# mean-independence model this is a martingale transform -- no first-order
# truncation, and no union bound over n products (Hallman 2021).
#
# The verifier does not know the other party's order, so we bound
#     sum_v t_v^2  <=  fac^2 * Phi(|x|),   fac = (1+gamma_n)/(1-u),
#     Phi(|x|) = max over ALL evaluation trees of sum_v (sum_{j in v} |x_j|)^2
#              = sum_{k=2}^{n} (sum of the k largest |x_j|)^2        (Lemma; one sort)
# The lemma (descending chain maximizes over every tree) is proved by an
# exchange argument and verified by brute force over all trees for n <= 6.
#
#   tau_ps (Azuma-Hoeffding with predictable bounds, no variance assumption):
#       |s_hat - s| <= lambda u fac sqrt(Phi)   w.p. >= 1 - 2 exp(-lambda^2/2)
#       two-run: tau_PS = 2 lambda(alpha/2) u fac sqrt(Phi),  lambda(beta)=sqrt(2 ln(2/beta))
#   tau_ps_freedman (adds a conditional-variance bound sigma^2 per rounding):
#       c = u fac S,  v = sigma^2 fac^2 Phi,  L = ln(4/alpha)
#       a = cL/3 + sqrt((cL/3)^2 + 2 v L);   tau_PSF = 2a
# ---------------------------------------------------------------------------

def phi_abs(x) -> float:
    """Phi(|x|) = sum_{k=2}^{n} (sum of the k largest |x_j|)^2, in float64."""
    import numpy as _np
    xs = _np.sort(_np.abs(_np.asarray(x, dtype=_np.float64)))[::-1]
    cs = _np.cumsum(xs)
    return float((cs[1:] ** 2).sum())


def _fac(n: int, u: float) -> float:
    return (1.0 + gamma(n, u)) / (1.0 - u)


def tau_ps(n: int, u: float, phi: float, alpha: float) -> float:
    """Order-robust two-run tolerance, boundedness + mean independence only (no union bound)."""
    lam = math.sqrt(2.0 * math.log(4.0 / alpha))          # per-run failure alpha/2: 2exp(-lam^2/2)=alpha/2
    return 2.0 * lam * u * _fac(n, u) * math.sqrt(phi)


def tau_ps_freedman(n: int, u: float, S: float, phi: float, alpha: float,
                    sigma2_over_u2: float = SIGMA2_OVER_U2) -> float:
    """Order-robust two-run tolerance with a conditional-variance bound sigma^2 = sigma2_over_u2 * u^2."""
    fac = _fac(n, u)
    c = u * fac * S
    v = sigma2_over_u2 * u * u * fac * fac * phi
    L = math.log(4.0 / alpha)
    a = c * L / 3.0 + math.sqrt((c * L / 3.0) ** 2 + 2.0 * v * L)
    return 2.0 * a


if __name__ == "__main__":
    # sanity for the WP2 tolerances on uniform-[0.5,1.5] data at n=2^20
    import numpy as _np
    _rng = _np.random.default_rng(0); _n = 2 ** 20; _u = UNIT_ROUNDOFF["float32"]
    _x = (_rng.random(_n) + 0.5).astype(_np.float32)
    _S = float(_np.abs(_x.astype(_np.float64)).sum()); _phi = phi_abs(_x)
    _tH = tau_probabilistic(_n, _u, _S, 1e-3); _tFm = tau_freedman(_n, _u, _S, 1e-3)
    _tPS = tau_ps(_n, _u, _phi, 1e-3); _tPSF = tau_ps_freedman(_n, _u, _S, _phi, 1e-3)
    _tPSFa = tau_ps_freedman(_n, _u, _S, _phi, 1e-3, SIGMA2_OVER_U2_UNIFORM)
    print(f"Phi/(n S^2) = {_phi/(_n*_S*_S):.4f}   (uniform[0.5,1.5]: ~0.425)")
    print(f"tau_H/S   = {_tH/_S:.3e}")
    print(f"tau_Fm/S  = {_tFm/_S:.3e}   ({_tH/_tFm:.2f}x tighter than tau_H)")
    print(f"tau_PS/S  = {_tPS/_S:.3e}   ({_tH/_tPS:.2f}x; no variance assumption)")
    print(f"tau_PSFa/S= {_tPSFa/_S:.3e}   ({_tH/_tPSFa:.2f}x; uniform model)")
    print(f"tau_PSF/S = {_tPSF/_S:.3e}   ({_tH/_tPSF:.2f}x; calibrated 0.18u^2)")


# ---------------------------------------------------------------------------
# WP3: ELEMENTWISE TWO-RUN TOLERANCE FOR MATRIX PRODUCTS  C = A B,  A: MxK, B: KxN
#
# Each entry c_ij is an inner product of length K. HM 2019 Thm 3.1 gives, for
# any evaluation order, |c_hat - c|_ij <= gamma~_K(lambda) (|A||B|)_ij with
# probability Q(lambda, K) per entry. Rounding models for the accumulation:
#   "scalar"  : fp32 FMA chain -> K roundings per entry          (cuBLAS fp32, TF32 off)
#   "block"   : tensor-core block FMA with exact products and exact in-block
#               sums of length b, one fp32 rounding per block (Blanchard et al.
#               2020, model with exact block accumulation) -> q = ceil(K/b)
#               roundings per entry.  b = 4 (Volta/Ampere fp16), treat as parameter.
# Simultaneous guarantee over all M*N entries by a union bound:
#   lambda = lambda_for_alpha(alpha/2, q*M*N)  ->  per-run failure alpha/2 for
#   the whole matrix.  (Per-entry alpha is the alternative; both are exposed.)
#   tau_ij = 2 gamma~_q(lambda) (|A||B|)_ij
# ---------------------------------------------------------------------------

def tau_gemm(A, B, alpha: float, u: float = 2.0 ** -24, model: str = "scalar", b: int = 4,
             per_entry: bool = False):
    """Elementwise two-run tolerance matrix for C = A B (returns numpy MxN float64)."""
    import numpy as _np
    A = _np.asarray(A, dtype=_np.float64); B = _np.asarray(B, dtype=_np.float64)
    M, K = A.shape; K2, N = B.shape
    assert K == K2
    q = K if model == "scalar" else int(math.ceil(K / b))
    events = q if per_entry else q * M * N
    lam = lambda_for_alpha(alpha / 2.0, events, u)
    gt = math.expm1(lam * math.sqrt(q) * u + q * u * u / (1.0 - u))
    return 2.0 * gt * (_np.abs(A) @ _np.abs(B))
