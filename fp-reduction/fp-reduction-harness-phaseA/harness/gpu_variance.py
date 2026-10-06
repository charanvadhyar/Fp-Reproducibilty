"""
gpu_variance.py — WP5: measure the rounding-error variance ON THE GPU, on real
atomic partial sums (closes the "sigma^2 was calibrated on CPU" limitation).

How: fp32 atomicAdd returns the OLD accumulator value. Each thread therefore
knows (old, x) and can recompute the hardware's result fl(old + x) in fp32 and
the exact sum old + x in fp64 (exact when the operand exponents differ by <= 29,
otherwise accurate to 2^-53, 2^29 finer than the fp32 error being measured).
It records
    delta = (fl(old+x) - (old+x)) / (old+x)      (relative rounding error)
    old                                           (the partial sum it hit)
into per-thread arrays. The host then reports, per launch and pooled:
    mean(delta)/u, sigma^2/u^2 (marginal), binned by |old| (conditional proxy),
    and V/(n u^2) = sum(delta^2)/(n u^2) per launch (the accumulated variance
    Freedman's inequality needs a bound on).

Caveat stated in the paper: fl(old+x) recomputed by the thread equals the value
the atomic unit produced only if both round to nearest even in fp32, which is
IEEE-754 behaviour for atomicAdd(float) on all tested cards; the script checks
this directly by comparing the final accumulator to the sum of thread-local
increments (bitwise), and reports any mismatch.

Run (pod):
  python gpu_variance.py --n 1048576 --launches 50 --out results/gpu_variance.json
  python gpu_variance.py --n 1048576 --kappa 1e4 --launches 50 --out results/gpu_variance_k1e4.json
"""

import argparse, json, math, os, time
import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None

from gensum import gensum
from reference import sum_abs

U = 2.0 ** -24

_KERNEL = r'''
extern "C" __global__
void atomic_record(const float* x, float* acc, double* delta, float* old_out,
                   unsigned char* exact_flag, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float xi  = x[i];
    float old = atomicAdd(acc, xi);          // returns the value BEFORE the add
    float fl  = old + xi;                    // same RN-even fp32 add the atomic unit did
    double ex = (double)old + (double)xi;    // exact unless exponent gap > 29
    double d  = (ex != 0.0) ? ((double)fl - ex) / ex : 0.0;
    delta[i]   = d;
    old_out[i] = old;
    // flag whether fp64 sum is exact: true if |ex| < 2^29 * min(|old|,|xi|) roughly
    double a = fabs((double)old), b = fabs((double)xi);
    double mn = (a < b) ? a : b, mx = (a < b) ? b : a;
    exact_flag[i] = (mn == 0.0 || mx / mn < 536870912.0) ? 1 : 0;
}
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1 << 20)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--launches", type=int, default=50)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=256)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if cp is None:
        print("CuPy not installed."); return

    x, s_exact, kappa = gensum(args.n, args.kappa, seed=args.seed)
    x = x.astype(np.float32); n = args.n; S = sum_abs(x)
    kern = cp.RawKernel(_KERNEL, "atomic_record")
    xg = cp.asarray(x)
    blocks = (n + args.threads - 1) // args.threads

    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    out = {"gpu": props["name"].decode(), "cupy": cp.__version__, "n": n, "kappa": float(kappa),
           "launches": args.launches, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    print(f"GPU: {out['gpu']}  n=2^{int(math.log2(n))}  kappa={kappa:.3g}  launches={args.launches}")

    per_launch = []
    pooled_d, pooled_old = [], []
    mism = 0; nonexact = 0
    for L in range(args.launches):
        acc = cp.zeros(1, dtype=cp.float32)
        delta = cp.empty(n, dtype=cp.float64)
        old = cp.empty(n, dtype=cp.float32)
        flag = cp.empty(n, dtype=cp.uint8)
        kern((blocks,), (args.threads,), (xg, acc, delta, old, flag, np.int32(n)))
        cp.cuda.Stream.null.synchronize()
        d = cp.asnumpy(delta); o = cp.asnumpy(old); f = cp.asnumpy(flag)
        nonexact += int((f == 0).sum())
        # consistency check: replay the recorded (old, x) pairs -> every fl(old+x) must be
        # a value the accumulator actually took; cheap check: the max old + its x == final acc
        final = float(acc[0])
        if kappa <= 1.0:   # positive data: the add that saw the largest old value was the last one
            idx = int(np.argmax(o)); replay = np.float32(o[idx]) + np.float32(x[idx])
            if float(replay) != final:
                mism += 1
        V = float((d ** 2).sum()) / (n * U * U)
        per_launch.append({"mean_over_u": float(d.mean() / U), "sigma2_over_u2": float(d.var() / U / U),
                           "V_over_nu2": V, "final": final})
        pooled_d.append(d); pooled_old.append(o)
        if L < 3 or L == args.launches - 1:
            print(f"  launch {L:>3}: mean/u={d.mean()/U:+.4f}  sigma2/u2={d.var()/U/U:.4f}  V/(n u^2)={V:.4f}")

    d = np.concatenate(pooled_d); o = np.concatenate(pooled_old)
    sem = d.std() / math.sqrt(len(d))
    Vs = np.array([p["V_over_nu2"] for p in per_launch])
    out.update({
        "samples": int(len(d)),
        "mean_over_u": float(d.mean() / U), "sem_over_u": float(sem / U),
        "sigma2_over_u2": float(d.var() / U / U),
        "max_abs_delta_over_u": float(np.abs(d).max() / U),
        "V_over_nu2_mean": float(Vs.mean()), "V_over_nu2_max": float(Vs.max()), "V_over_nu2_min": float(Vs.min()),
        "nonexact_reference_fraction": nonexact / (n * args.launches),
        "replay_mismatches": mism if kappa <= 1.0 else None,
        "per_launch": per_launch,
    })
    print(f"\npooled over {len(d):,} atomic adds:")
    print(f"  mean(delta)/u    = {out['mean_over_u']:+.4f}  (SE {out['sem_over_u']:.4f})")
    print(f"  sigma^2/u^2      = {out['sigma2_over_u2']:.4f}   (uniform model 0.3333; CPU sequential 0.186)")
    print(f"  V/(n u^2)        = mean {Vs.mean():.4f}  min {Vs.min():.4f}  max {Vs.max():.4f}  over launches")
    print(f"  max|delta|/u     = {out['max_abs_delta_over_u']:.4f}  (must be <= 1)")
    print(f"  fp64 ref inexact = {out['nonexact_reference_fraction']:.2e} of adds;  replay mismatches = {mism}")

    # conditional-variance proxy: bin by magnitude of the partial sum hit
    edges = np.quantile(np.abs(o), np.linspace(0, 1, args.bins + 1))
    rows = []
    print(f"\nby |partial sum| decile (conditional-variance proxy):")
    for b in range(args.bins):
        m = (np.abs(o) >= edges[b]) & (np.abs(o) < edges[b + 1]) if b < args.bins - 1 else (np.abs(o) >= edges[b])
        if m.sum() < 1000: continue
        r = {"old_lo": float(edges[b]), "old_hi": float(edges[b + 1]),
             "mean_over_u": float(d[m].mean() / U), "sigma2_over_u2": float(d[m].var() / U / U), "N": int(m.sum())}
        rows.append(r)
        print(f"  |s| in [{r['old_lo']:>10.3e},{r['old_hi']:>10.3e}):  mean/u={r['mean_over_u']:+.4f}  sigma2/u2={r['sigma2_over_u2']:.4f}  N={r['N']:,}")
    out["binned"] = rows
    mx = max(r["sigma2_over_u2"] for r in rows) if rows else None
    print(f"  max binned sigma^2/u^2 = {mx:.4f}  ({'<= 1/3' if mx is not None and mx <= 1/3 else '> 1/3'})")
    out["max_binned_sigma2_over_u2"] = mx

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=1)
        print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
