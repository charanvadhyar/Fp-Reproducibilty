"""
gpu_smoketest.py — does a GPU atomic reduction actually produce run-to-run spread?

This is the de-risking step for the whole GPU phase. Before building strategy
comparisons, confirm the phenomenon exists on THIS hardware:

  - a naive atomicAdd reduction has NO fixed summation order (threads arrive in
    hardware-scheduler order), so repeated identical launches can give
    DIFFERENT results. If they do, the nondeterminism your tau bounds is real.
  - a fixed-order (tree) reduction should give IDENTICAL results every launch.

If atomicAdd shows zero spread across many launches, either the input is too
small (reduction serialized), atomics are being ordered, or the driver is doing
something deterministic -- in any of those cases we learn it NOW, cheaply,
before building the full experiment.

Prediction from the CPU work: at kappa=1, the spread should land near
  ~0.78 * u * sqrt(n) * S   (the real deviation multiple measured on permuted CPU sums)
if GPU atomic reordering behaves like the CPU permutation model.

Run (on the pod, CuPy + CUDA installed):
  python gpu_smoketest.py
  python gpu_smoketest.py --n 1048576 --launches 200
"""

import argparse
import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None


# --- naive atomicAdd reduction: every thread atomicAdds one element into one
#     accumulator. Arrival order is nondeterministic -> result can vary. ---
_ATOMIC_KERNEL = r'''
extern "C" __global__
void atomic_sum(const float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(out, x[i]);
    }
}
'''


def atomic_reduce(x_gpu, n, threads=256):
    """One naive-atomic reduction. Returns a Python float."""
    out = cp.zeros(1, dtype=cp.float32)
    blocks = (n + threads - 1) // threads
    ker = cp.RawKernel(_ATOMIC_KERNEL, "atomic_sum")
    ker((blocks,), (threads,), (x_gpu, out, np.int32(n)))
    cp.cuda.Stream.null.synchronize()
    return float(out[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1 << 20)   # 2^20
    ap.add_argument("--launches", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if cp is None:
        print("CuPy not installed. On the pod:  pip install cupy-cuda12x")
        print("(match the CUDA version: cupy-cuda11x / cupy-cuda12x)")
        return

    # provenance — matters as much here as in the CPU harness
    dev = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(dev.id)
    print("GPU:", props["name"].decode())
    print("CuPy:", cp.__version__, " CUDA runtime:", cp.cuda.runtime.runtimeGetVersion())
    print(f"n = {args.n} (2^{int(np.log2(args.n))}), launches = {args.launches}\n")

    # fixed input, well-conditioned (kappa ~ 1): all positive
    rng = np.random.default_rng(args.seed)
    x = (rng.random(args.n).astype(np.float32) + np.float32(0.5))
    x_gpu = cp.asarray(x)

    # reference (exact-ish): fp64 sum on host
    import math
    s_ref = math.fsum(x.astype(np.float64).tolist())
    S = math.fsum(np.abs(x).astype(np.float64).tolist())
    u = 2.0 ** -24
    predicted_spread = 0.78 * u * math.sqrt(args.n) * S   # from CPU deviation multiple

    # --- repeated atomic launches ---
    results = [atomic_reduce(x_gpu, args.n) for _ in range(args.launches)]
    results = np.array(results, dtype=np.float64)
    unique = np.unique(results)
    spread = float(results.max() - results.min())

    print("ATOMIC reduction (nondeterministic order expected):")
    print(f"  distinct results over {args.launches} launches : {len(unique)}")
    print(f"  min            = {results.min():.6f}")
    print(f"  max            = {results.max():.6f}")
    print(f"  spread (max-min) = {spread:.3e}")
    print(f"  predicted spread ~ 0.78*u*sqrt(n)*S = {predicted_spread:.3e}")
    print(f"  ratio measured/predicted = {spread/predicted_spread:.2f}" if predicted_spread else "")
    print(f"  mean error vs fp64 ref   = {abs(results.mean() - s_ref):.3e}")
    print()

    if len(unique) == 1:
        print(">> NO run-to-run spread. Atomics gave identical results every launch.")
        print(">> Investigate before proceeding: input may be too small (serialized),")
        print(">> or this GPU/driver orders the atomics. Try larger --n.")
    else:
        print(f">> REAL nondeterminism confirmed: {len(unique)} distinct results.")
        print(">> The phenomenon tau bounds exists on this hardware. GPU phase unlocked.")


if __name__ == "__main__":
    main()
