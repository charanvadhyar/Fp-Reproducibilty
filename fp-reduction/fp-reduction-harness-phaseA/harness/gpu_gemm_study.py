"""
gpu_gemm_study.py — WP3: does the elementwise two-run tolerance hold for real
nondeterministic MATRIX PRODUCTS, and does it detect faults there?

Sources of nondeterminism tested (each gives pairs of "correct" results of C = A B):
  split_k_atomic : our own tiled GEMM in which the K dimension is split into
                   `splits` chunks and each chunk's partial dot products are
                   atomicAdd-ed into C in fp32 -> genuine run-to-run spread
                   per entry (the GEMM analogue of naive_atomic).
  cublas         : cp.matmul (fp32, no TF32) -> deterministic control.
  torch_fp16     : (optional, needs torch) fp16 inputs, fp32 accumulate on
                   tensor cores; the two "runs" are the SAME rows computed at
                   batch size 1 and at batch size B (the He / Thinking Machines
                   batch-invariance effect) -> different evaluation trees.
  torch_fp32     : same with TF32 disabled (scalar fp32 FMA chain) as a control.

For every source: spread_ij = |C1_ij - C2_ij| vs tau_ij from bounds.tau_gemm
(model "scalar" for fp32 paths, "block" with b for the tensor-core path),
counting exceedances over ALL M*N entries and launches, plus the error against
an fp64 reference and its scaling exponent in K.

Fault injection: a single bit-flip in one entry of A (bit position swept) and
an additive perturbation sized in units of the local tau; detection = any
entry of the affected row exceeds tau_ij against a clean run.

Run (pod):
  python gpu_gemm_study.py --M 256 --N 256 --K 4096 --launches 200 --out results/gemm_4090.json
  python gpu_gemm_study.py --cpu-emulate --K 1024 --launches 50          # logic check without a GPU
"""

import argparse, json, math, os, time
import numpy as np
from bounds import tau_gemm, UNIT_ROUNDOFF

try:
    import cupy as cp
except ImportError:
    cp = None

U32 = UNIT_ROUNDOFF["float32"]

_SPLITK = r'''
extern "C" __global__
void splitk_atomic(const float* A, const float* B, float* C, int M, int N, int K, int splits) {
    // grid: (splits, ceil(N/bx), ceil(M/by)); the split index is the FASTEST-varying block
    // dimension so that the K-chunks of one output tile are launched back-to-back and
    // genuinely race on the atomicAdd (with splits in grid.z they land in order).
    int s = blockIdx.x;
    int j = blockIdx.y * blockDim.x + threadIdx.x;
    int i = blockIdx.z * blockDim.y + threadIdx.y;
    if (i >= M || j >= N) return;
    int chunk = (K + splits - 1) / splits;
    int k0 = s * chunk, k1 = min(K, k0 + chunk);
    float acc = 0.0f;
    for (int k = k0; k < k1; ++k) acc = fmaf(A[i * K + k], B[k * N + j], acc);   // fp32 FMA chain
    atomicAdd(&C[i * N + j], acc);                                                // nondeterministic order across splits
}
'''


def gen(M, N, K, rng, kind="uniform"):
    if kind == "uniform":
        A = (rng.random((M, K)) + 0.5).astype(np.float32); B = (rng.random((K, N)) + 0.5).astype(np.float32)
    else:  # mixed sign, cancelling
        A = (rng.random((M, K)) * 2 - 1).astype(np.float32); B = (rng.random((K, N)) * 2 - 1).astype(np.float32)
    return A, B


def ref64(A, B):
    return A.astype(np.float64) @ B.astype(np.float64)


class SplitK:
    def __init__(self, splits, bx=16, by=16):
        self.k = cp.RawKernel(_SPLITK, "splitk_atomic"); self.splits = splits; self.bx = bx; self.by = by
    def __call__(self, Ag, Bg):
        M, K = Ag.shape; N = Bg.shape[1]
        C = cp.zeros((M, N), dtype=cp.float32)
        grid = (self.splits, (N + self.bx - 1) // self.bx, (M + self.by - 1) // self.by)
        self.k(grid, (self.bx, self.by), (Ag, Bg, C, np.int32(M), np.int32(N), np.int32(K), np.int32(self.splits)))
        cp.cuda.Stream.null.synchronize()
        return cp.asnumpy(C)


def splitk_emulate(A, B, splits, rng):
    """CPU emulation: K-chunks accumulated in fp32 in a random order (one random order per call)."""
    M, K = A.shape; N = B.shape[1]
    chunk = (K + splits - 1) // splits
    parts = []
    for s in range(splits):
        k0, k1 = s * chunk, min(K, (s + 1) * chunk)
        acc = np.zeros((M, N), dtype=np.float32)
        for k in range(k0, k1):
            acc = (acc + np.outer(A[:, k], B[k, :]).astype(np.float32)).astype(np.float32)
        parts.append(acc)
    C = np.zeros((M, N), dtype=np.float32)
    for s in rng.permutation(splits):
        C = (C + parts[s]).astype(np.float32)
    return C


def pair_stats(C1, C2, tau, Cref):
    d = np.abs(C1.astype(np.float64) - C2.astype(np.float64))
    err = np.abs(C1.astype(np.float64) - Cref)
    return {"entries": int(d.size), "nondet_entries": int((d > 0).sum()),
            "max_spread_over_tau": float((d / tau).max()), "median_spread_over_tau": float(np.median(d / tau)),
            "exceedances": int((d > tau).sum()), "max_err_over_tau": float((err / tau).max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=256); ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--K", type=int, default=4096)
    ap.add_argument("--splits", type=int, default=64)
    ap.add_argument("--launches", type=int, default=200)
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--data", choices=["uniform", "mixed"], default="uniform")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu-emulate", action="store_true")
    ap.add_argument("--no-torch", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    M, N, K = args.M, args.N, args.K
    if args.cpu_emulate:
        M, N = min(M, 32), min(N, 32)
    A, B = gen(M, N, K, rng, args.data)
    Cref = ref64(A, B)
    tau_s = tau_gemm(A, B, args.alpha, U32, "scalar")                # whole-matrix alpha
    out = {"M": M, "N": N, "K": K, "splits": args.splits, "alpha": args.alpha, "data": args.data,
           "tau_scalar_over_absAB_median": float(np.median(tau_s / (np.abs(A.astype(np.float64)) @ np.abs(B.astype(np.float64))))),
           "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "sources": {}}
    print(f"GEMM {M}x{K} @ {K}x{N}, data={args.data}, alpha={args.alpha}; tau_scalar/(|A||B|) median = {out['tau_scalar_over_absAB_median']:.3e}")

    # ---------------- source 1: split-K atomic (real GPU) or emulation ----------------
    if args.cpu_emulate:
        runs = [splitk_emulate(A, B, args.splits, rng) for _ in range(args.launches)]
        out["provenance"] = {"device": "cpu-emulation"}
    else:
        if cp is None:
            print("CuPy not installed"); return
        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        out["provenance"] = {"gpu": props["name"].decode(), "cupy": cp.__version__}
        Ag, Bg = cp.asarray(A), cp.asarray(B)
        sk = SplitK(args.splits)
        runs = [sk(Ag, Bg) for _ in range(args.launches)]
        # cuBLAS control
        c1 = cp.asnumpy(cp.matmul(Ag, Bg)); c2 = cp.asnumpy(cp.matmul(Ag, Bg))
        out["sources"]["cublas_fp32"] = pair_stats(c1, c2, tau_s, Cref)
        out["sources"]["cublas_fp32"]["bitwise_identical"] = bool(np.array_equal(c1.view(np.int32), c2.view(np.int32)))
        print("cuBLAS fp32 control:", out["sources"]["cublas_fp32"])
    stats = [pair_stats(runs[2 * i], runs[2 * i + 1], tau_s, Cref) for i in range(len(runs) // 2)]
    agg = {"pairs": len(stats),
           "entries_tested": sum(s["entries"] for s in stats),
           "nondet_entry_fraction": float(np.mean([s["nondet_entries"] / s["entries"] for s in stats])),
           "exceedances": int(sum(s["exceedances"] for s in stats)),
           "max_spread_over_tau": float(max(s["max_spread_over_tau"] for s in stats)),
           "median_spread_over_tau": float(np.median([s["median_spread_over_tau"] for s in stats])),
           "max_err_over_tau": float(max(s["max_err_over_tau"] for s in stats))}
    out["sources"]["split_k_atomic"] = agg
    print(f"split-K atomic: {agg['pairs']} pairs, {agg['entries_tested']:,} entries; nondeterministic fraction {agg['nondet_entry_fraction']:.3f}; "
          f"exceedances {agg['exceedances']}; max spread/tau {agg['max_spread_over_tau']:.3e}; max err/tau {agg['max_err_over_tau']:.3e}")

    # ---------------- source 2: tensor cores via torch (optional) ----------------
    if not args.cpu_emulate and not args.no_torch:
        try:
            import torch
            torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
            # force fp32 accumulation inside split-K reductions of half-precision GEMMs
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
            At = torch.tensor(A, device="cuda"); Bt = torch.tensor(B, device="cuda")

            def mm(a, b, want_fp32_out):
                """half-precision inputs: fp32 output if this torch supports out_dtype, else half output."""
                if want_fp32_out:
                    try:
                        return torch.mm(a, b, out_dtype=torch.float32), True
                    except TypeError:
                        pass
                return torch.mm(a, b), False

            U16 = 2.0 ** -11
            cases = (("torch_fp32_scalar", torch.float32, "scalar", 4, False),
                     ("torch_fp16_tc_fp32out", torch.float16, "block", 4, True),
                     ("torch_fp16_tc_fp16out", torch.float16, "block", 4, False))
            for name, dt, model, b, want32 in cases:
                Ad, Bd = At.to(dt), Bt.to(dt)
                Aq, Bq = Ad.float().cpu().numpy(), Bd.float().cpu().numpy()      # the values actually multiplied
                Cq = ref64(Aq, Bq)
                tau_m = tau_gemm(Aq, Bq, args.alpha, U32, model, b)
                full_t, got32 = mm(Ad, Bd, want32)
                if want32 and not got32:
                    print(f"{name}: out_dtype=float32 unsupported by this torch; skipped"); continue
                if dt == torch.float16 and not got32:
                    # half-precision OUTPUT: each run rounds c_ij to fp16 -> add the deterministic output term
                    tau_m = tau_m + 2.0 * U16 * np.abs(Cq)
                full = full_t.float().cpu().numpy()                                # all rows at once
                rows1 = np.vstack([mm(Ad[i:i + 1], Bd, want32)[0].float().cpu().numpy() for i in range(M)])  # batch 1
                st = pair_stats(full, rows1, tau_m, Cq)
                st["bitwise_identical"] = bool(np.array_equal(full.view(np.int32), rows1.view(np.int32)))
                st["output_dtype"] = "float32" if (dt == torch.float32 or got32) else "float16"
                st["tolerance"] = model + ("+fp16_output_rounding" if (dt == torch.float16 and not got32) else "")
                # error scaling in K for this path (exponent of median|err| vs K on sub-products)
                Ks = [k for k in (256, 512, 1024, 2048, 4096, 8192) if k <= K]
                errs = []
                for k in Ks:
                    e = np.abs(mm(Ad[:, :k].contiguous(), Bd[:k, :].contiguous(), want32)[0].float().cpu().numpy() - ref64(Aq[:, :k], Bq[:k, :]))
                    errs.append(float(np.median(e / (np.abs(Aq[:, :k]) @ np.abs(Bq[:k, :])))))
                if len(Ks) >= 3:
                    st["err_scaling_exponent"] = float(np.polyfit(np.log(Ks), np.log(errs), 1)[0])
                    st["err_over_absAB_by_K"] = dict(zip(map(str, Ks), errs))
                out["sources"][name] = st
                print(f"{name}: batch-1 vs full rows -> {st}")
        except Exception as e:
            print("torch path skipped:", repr(e))

    # ---------------- faults: bit-flips in A, detected per entry against a clean run ----------------
    if not args.cpu_emulate:
        Bg_f = cp.asarray(B); sk_f = SplitK(args.splits)
    def clean():
        return splitk_emulate(A, B, args.splits, rng) if args.cpu_emulate else sk_f(cp.asarray(A), Bg_f)
    def faulty(Af):
        return splitk_emulate(Af, B, args.splits, rng) if args.cpu_emulate else sk_f(cp.asarray(Af), Bg_f)
    fault_rows = []
    trials = 20 if args.cpu_emulate else 50
    for bit in (10, 16, 20, 22, 23, 25, 27, 30):
        det = 0; chg = 0; done = 0
        for _ in range(trials):
            Af = A.copy(); i = rng.integers(M); k = rng.integers(K)
            v = np.float32(Af[i, k]); fv = (v.view(np.int32) ^ np.int32(1 << bit)).view(np.float32)
            if not np.isfinite(fv): continue
            Af[i, k] = fv
            d = np.abs(faulty(Af).astype(np.float64) - clean().astype(np.float64))
            det += int((d[i] > tau_s[i]).any()); done += 1
        if done:
            fault_rows.append({"bit": bit, "trials": done, "detected_row_rate": det / done})
            print(f"  bit {bit:>2}: detected (any entry of the affected row > tau) {det}/{done}")
    out["bitflip_A"] = fault_rows

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=1)
        print("saved", args.out)


if __name__ == "__main__":
    main()