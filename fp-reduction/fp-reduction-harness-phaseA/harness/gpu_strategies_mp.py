"""
gpu_strategies_mp.py — multi-precision GPU reduction strategies.

Extends gpu_strategies.py across data types. Key hardware facts:
  - fp32, fp64 atomicAdd: native on all modern GPUs.
  - fp16 atomicAdd: requires compute capability >= 7.0 (RTX 4090 = 8.9, OK).
  - bf16 atomicAdd: supported on recent archs; we try and fall back if unsupported.
  - tf32 / fp8: these are TENSOR-CORE modes, not storage types for scalar
    reductions, so they are OUT OF SCOPE for a summation study. Noted, not run.

So the dtype axis for scalar summation is: float64, float32, float16, bfloat16.

Strategies per dtype:
  naive_atomic : one global accumulator, atomicAdd -> nondeterministic
  block_atomic : block-local tree + one atomic per block
  tree_fixed   : deterministic multi-pass tree, no atomics (works for all dtypes)
  cupy_sum     : library baseline

The CUDA type and the accumulator type are logged separately (storage vs accumulate).
Here storage == accumulate per dtype; mixed-precision accumulation (e.g. bf16
storage, fp32 accumulate) is a documented extension, not run here.

Usage: imported by gpu_exceedance_mp.py; run that, not this.
"""

import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None

# CUDA C type name per numpy dtype
CTYPE = {
    "float64": "double",
    "float32": "float",
    "float16": "__half",
    "bfloat16": "__nv_bfloat16",
}
CP_DTYPE = {
    "float64": "float64",
    "float32": "float32",
    "float16": "float16",
    # bf16 handled specially via cupy (cp.float16 is fp16; bf16 needs ml_dtypes/cupy support)
}

# kernels are templated by substituting the C type. half/bf16 need headers.
_HEADERS = {
    "float64": "",
    "float32": "",
    "float16": "#include <cuda_fp16.h>\n",
    "bfloat16": "#include <cuda_bf16.h>\n",
}

def _naive_src(ct, hdr):
    return hdr + f'''
extern "C" __global__
void naive_atomic(const {ct}* x, {ct}* out, int n) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(out, x[i]);
}}
'''

def _tree_src(ct, hdr):
    # tree works for all dtypes; half/bf16 need operator+ which the headers provide
    return hdr + f'''
extern "C" __global__
void tree_pass(const {ct}* in, {ct}* out, int n) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = 2*i;
    if (j+1 < n)      out[i] = in[j] + in[j+1];
    else if (j < n)   out[i] = in[j];
}}
'''

def build_kernels(dtype):
    """Compile kernels for a dtype. Returns dict; missing keys = unsupported."""
    ct = CTYPE[dtype]; hdr = _HEADERS[dtype]
    k = {}
    try:
        k["naive"] = cp.RawKernel(_naive_src(ct, hdr), "naive_atomic",
                                  options=('--std=c++14',))
        # force a compile to catch atomicAdd-unsupported early
        _probe(k["naive"], dtype)
    except Exception as e:
        k["naive"] = None
        k["naive_err"] = str(e)[:120]
    try:
        k["tree"] = cp.RawKernel(_tree_src(ct, hdr), "tree_pass",
                                 options=('--std=c++14',))
    except Exception as e:
        k["tree"] = None
        k["tree_err"] = str(e)[:120]
    return k


def _cp_array(x_np, dtype):
    if dtype == "bfloat16":
        # cupy supports bf16 via cupy.bfloat16 in recent versions; else via ml_dtypes
        try:
            return cp.asarray(x_np, dtype=cp.bfloat16)
        except AttributeError:
            import ml_dtypes
            return cp.asarray(x_np.astype(ml_dtypes.bfloat16))
    return cp.asarray(x_np.astype(getattr(np, dtype)))


def _probe(kernel, dtype):
    # tiny launch to trigger JIT compile; raises if atomicAdd unsupported for dtype
    x = _cp_array(np.ones(4, dtype=np.float32), dtype)
    out = cp.zeros(1, dtype=x.dtype)
    kernel((1,), (4,), (x, out, np.int32(4)))
    cp.cuda.Stream.null.synchronize()


def naive_atomic(x_gpu, n, k, threads=256):
    out = cp.zeros(1, dtype=x_gpu.dtype)
    blocks = (n + threads - 1)//threads
    k["naive"]((blocks,), (threads,), (x_gpu, out, np.int32(n)))
    cp.cuda.Stream.null.synchronize()
    return float(out[0])


def tree_fixed(x_gpu, n, k, threads=256):
    cur = x_gpu; m = n
    while m > 1:
        om = (m+1)//2
        nxt = cp.empty(om, dtype=x_gpu.dtype)
        blocks = (om + threads - 1)//threads
        k["tree"]((blocks,), (threads,), (cur, nxt, np.int32(m)))
        cur = nxt; m = om
    cp.cuda.Stream.null.synchronize()
    return float(cur[0])


def cupy_sum(x_gpu, n, k=None):
    return float(cp.sum(x_gpu))


def to_gpu(x_np, dtype):
    return _cp_array(x_np, dtype)


DTYPES = ["float64", "float32", "float16", "bfloat16"]
