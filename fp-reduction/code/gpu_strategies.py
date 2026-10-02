"""
gpu_strategies.py — GPU reduction strategies + determinism control.

Three strategies spanning the contention/order spectrum, plus the control that
proves the nondeterminism comes from atomics and nothing else:

  naive_atomic   : every thread atomicAdds into ONE global accumulator.
                   Max contention, nondeterministic order. (smoke-test kernel)
  block_atomic   : each block reduces locally in shared memory (fixed order),
                   then one atomicAdd per block into the global accumulator.
                   This is what real libraries do -> far fewer atomics, so
                   different (usually smaller) nondeterminism.
  tree_fixed     : fully deterministic multi-pass tree reduction, no atomics.
                   CONTROL: repeated launches must be BITWISE IDENTICAL.
  cupy_sum       : cp.sum(), the library baseline. Also expected deterministic
                   for a fixed shape.

The control test (run each 100x, count distinct results) is the A/B that makes
the atomic finding airtight: atomic strategies -> many distinct results;
tree_fixed / cupy_sum -> exactly 1.

Run (on the pod):
  python gpu_strategies.py --n 1048576 --launches 100
"""

import argparse
import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None


# ---- naive atomic: one global accumulator, max contention ----
_NAIVE = r'''
extern "C" __global__
void naive_atomic(const float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(out, x[i]);
}
'''

# ---- block-local reduce in shared memory (fixed order within block),
#      then one atomicAdd per block. This is the hierarchical pattern. ----
_BLOCK = r'''
extern "C" __global__
void block_atomic(const float* x, float* out, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? x[i] : 0.0f;
    __syncthreads();
    // tree reduction within the block: deterministic order
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    // one atomic per block -> few atomics, less nondeterminism
    if (tid == 0) atomicAdd(out, sdata[0]);
}
'''

# ---- fully deterministic tree: pairwise-add passes, no atomics at all ----
_TREE = r'''
extern "C" __global__
void tree_pass(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // each output element = in[2i] + in[2i+1], fixed order
    int j = 2 * i;
    if (j + 1 < n)      out[i] = in[j] + in[j + 1];
    else if (j < n)     out[i] = in[j];
}
'''


def _load():
    return (cp.RawKernel(_NAIVE, "naive_atomic"),
            cp.RawKernel(_BLOCK, "block_atomic"),
            cp.RawKernel(_TREE, "tree_pass"))


def naive_atomic(x_gpu, n, kernels, threads=256):
    knaive, _, _ = kernels
    out = cp.zeros(1, dtype=cp.float32)
    blocks = (n + threads - 1) // threads
    knaive((blocks,), (threads,), (x_gpu, out, np.int32(n)))
    cp.cuda.Stream.null.synchronize()
    return float(out[0])


def block_atomic(x_gpu, n, kernels, threads=256):
    _, kblock, _ = kernels
    out = cp.zeros(1, dtype=cp.float32)
    blocks = (n + threads - 1) // threads
    shmem = threads * 4  # bytes, one float per thread
    kblock((blocks,), (threads,), (x_gpu, out, np.int32(n)), shared_mem=shmem)
    cp.cuda.Stream.null.synchronize()
    return float(out[0])


def tree_fixed(x_gpu, n, kernels, threads=256):
    _, _, ktree = kernels
    cur = x_gpu
    m = n
    while m > 1:
        out_m = (m + 1) // 2
        nxt = cp.empty(out_m, dtype=cp.float32)
        blocks = (out_m + threads - 1) // threads
        ktree((blocks,), (threads,), (cur, nxt, np.int32(m)))
        cur = nxt
        m = out_m
    cp.cuda.Stream.null.synchronize()
    return float(cur[0])


def cupy_sum(x_gpu, n, kernels=None):
    return float(cp.sum(x_gpu, dtype=cp.float32))


STRATEGIES = {
    "naive_atomic": naive_atomic,
    "block_atomic": block_atomic,
    "tree_fixed": tree_fixed,
    "cupy_sum": cupy_sum,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1 << 20)
    ap.add_argument("--launches", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if cp is None:
        print("CuPy not installed. On the pod:  pip install cupy-cuda12x")
        return

    import math
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    print("GPU:", props["name"].decode(), " CuPy:", cp.__version__)
    print(f"n = {args.n} (2^{int(np.log2(args.n))}), launches = {args.launches}\n")

    rng = np.random.default_rng(args.seed)
    x = (rng.random(args.n).astype(np.float32) + np.float32(0.5))
    x_gpu = cp.asarray(x)
    s_ref = math.fsum(x.astype(np.float64).tolist())

    kernels = _load()

    print(f"{'strategy':>14} {'distinct':>9} {'spread':>12} {'mean_err_vs_ref':>16} {'verdict':>14}")
    print("-" * 70)
    for name, fn in STRATEGIES.items():
        res = np.array([fn(x_gpu, args.n, kernels) if name != "cupy_sum"
                        else fn(x_gpu, args.n) for _ in range(args.launches)],
                       dtype=np.float64)
        distinct = len(np.unique(res))
        spread = float(res.max() - res.min())
        mean_err = abs(res.mean() - s_ref)
        verdict = "NONDET" if distinct > 1 else "deterministic"
        print(f"{name:>14} {distinct:>9} {spread:>12.3e} {mean_err:>16.3e} {verdict:>14}")

    print()
    print("Expected: naive_atomic & block_atomic -> NONDET (many distinct);")
    print("          tree_fixed & cupy_sum -> deterministic (1 distinct).")
    print("If tree_fixed shows 1 while naive shows many, the nondeterminism is")
    print("PROVEN to come from the atomics -- the clean A/B control.")


if __name__ == "__main__":
    main()
