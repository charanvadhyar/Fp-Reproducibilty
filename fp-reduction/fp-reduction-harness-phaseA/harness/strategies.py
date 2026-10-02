"""
strategies.py — fixed-order fp32 summation strategies.

Every strategy here has a FIXED accumulation order, so it is bitwise
deterministic. Phase A measures ERROR (vs the exact sum) as a function of
n and kappa; nondeterminism (spread) comes in Phase B on the GPU.

All arithmetic is genuinely float32: inputs are float32, accumulators are
float32, and every intermediate is float32. On x86 (SSE) numpy float32 ops
are true fp32 with no extended precision, so measured errors are real.

IMPORTANT: np.sum() uses PAIRWISE summation internally. It is NOT the naive
recursive loop. Never use np.sum() as the "recursive" baseline.

Strategies:
  recursive  — left-to-right running accumulator (the classical reference order)
  pairwise   — recursive halving (tree), depth ~ log2(n)
  kahan      — compensated summation (Neumaier variant), error ~ 2u
  numpy_sum  — np.sum(), i.e. numpy's own pairwise, for comparison

Source of bounds these correspond to: Higham, Accuracy and Stability of
Numerical Algorithms, Ch. 4 (recursive: gamma_{n-1}; pairwise: gamma_{log2 n};
compensated: ~2u).
"""

import numpy as np

try:
    from numba import njit
    _HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    _HAVE_NUMBA = False

    def njit(*a, **k):  # no-op decorator fallback (slow but correct)
        if a and callable(a[0]):
            return a[0]
        return lambda f: f


@njit(cache=True)
def _recursive_f32(x):
    s = np.float32(0.0)
    for i in range(x.shape[0]):
        s = np.float32(s + x[i])
    return s


@njit(cache=True)
def _kahan_f32(x):
    # Neumaier's improved Kahan: handles the case where the addend is larger
    # than the running sum. Compensation term c accumulates the lost low bits.
    s = np.float32(0.0)
    c = np.float32(0.0)
    for i in range(x.shape[0]):
        v = x[i]
        t = np.float32(s + v)
        if abs(s) >= abs(v):
            c = np.float32(c + np.float32(np.float32(s - t) + v))
        else:
            c = np.float32(c + np.float32(np.float32(v - t) + s))
        s = t
    return np.float32(s + c)


def _pairwise_f32(x):
    # Recursive halving. Base case small enough that the leaf sum is a short
    # recursive loop; tree depth ~ log2(n). All intermediates float32.
    n = x.shape[0]
    if n <= 8:
        return _recursive_f32(x)
    h = n // 2
    return np.float32(_pairwise_f32(x[:h]) + _pairwise_f32(x[h:]))


def recursive(x: np.ndarray) -> np.float32:
    return _recursive_f32(np.ascontiguousarray(x, dtype=np.float32))


def pairwise(x: np.ndarray) -> np.float32:
    return _pairwise_f32(np.ascontiguousarray(x, dtype=np.float32))


def kahan(x: np.ndarray) -> np.float32:
    return _kahan_f32(np.ascontiguousarray(x, dtype=np.float32))


def numpy_sum(x: np.ndarray) -> np.float32:
    return np.float32(np.sum(np.ascontiguousarray(x, dtype=np.float32), dtype=np.float32))


STRATEGIES = {
    "recursive": recursive,
    "pairwise": pairwise,
    "kahan": kahan,
    "numpy_sum": numpy_sum,
}

# Chain length used by the deterministic bound for each strategy.
# recursive: n-1 roundings in series. pairwise: ~ceil(log2 n). kahan: ~2 (n-independent).
def chain_length(strategy: str, n: int) -> float:
    if strategy == "recursive":
        return max(n - 1, 1)
    if strategy in ("pairwise", "numpy_sum"):
        return max(int(np.ceil(np.log2(n))), 1)
    if strategy == "kahan":
        return 2.0
    raise ValueError(strategy)


if __name__ == "__main__":
    # smoke: the 2^24 non-associativity case must show up in recursive
    a = np.array([2**24, 1, 1], dtype=np.float32)
    print("recursive(2^24,1,1) =", recursive(a), "(expect 16777216: both 1s swamped)")
    print("kahan    (2^24,1,1) =", kahan(a), "(expect 16777218: compensated)")
    print("numba available:", _HAVE_NUMBA)
