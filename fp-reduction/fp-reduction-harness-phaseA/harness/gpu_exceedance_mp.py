"""
gpu_exceedance_mp.py — multi-precision exceedance sweep on one GPU.

Runs the reduction strategies across data types {float64, float32, float16,
bfloat16}, measuring real run-to-run spread and testing it against BOTH the
Hoeffding tolerance (tau_probabilistic) and the Bernstein tolerance
(tau_bernstein). One pass tests both tolerances on all supported dtypes.

Per dtype the unit roundoff u changes (bounds.UNIT_ROUNDOFF), so tau scales
accordingly. Strategies whose atomicAdd is unsupported for a dtype are skipped
and noted, not crashed.

tf32 and fp8 are tensor-core modes, not scalar storage types, so they are out
of scope for a summation study and are not run (noted in the paper).

Run (on the 4090 pod, before tearing it down):
  python gpu_exceedance_mp.py --out results/gpu_mp.jsonl
  python gpu_exceedance_mp.py --out results/gpu_mp_quick.jsonl --quick
"""

import argparse, json, math, os, time
import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None

from gensum import gensum
from reference import sum_abs, exact_sum
from bounds import tau_probabilistic, tau_bernstein, UNIT_ROUNDOFF
import gpu_strategies_mp as mp


def run_cell(name, fn, x_np, n, kappa, dtype, u, alpha, launches, kernels):
    x_gpu = mp.to_gpu(x_np, dtype)
    S = sum_abs(x_np)
    s_ref = exact_sum(x_np)
    tauH = tau_probabilistic(n, u, S, alpha)
    tauB = tau_bernstein(n, u, S, alpha)
    res = np.array([fn(x_gpu, n, kernels) if name != "cupy_sum" else fn(x_gpu, n)
                    for _ in range(launches)], dtype=np.float64)
    distinct = int(len(np.unique(res)))
    spread = float(res.max() - res.min())
    return {
        "strategy": name, "n": n, "kappa": kappa, "dtype": dtype, "u": u,
        "alpha": alpha, "launches": launches, "sum_abs": S,
        "tau_hoeffding": tauH, "tau_bernstein": tauB,
        "distinct_results": distinct, "spread_maxmin": spread,
        "mean_err_vs_ref": float(abs(res.mean() - s_ref)),
        "exceeds_hoeffding": bool(spread > tauH),
        "exceeds_bernstein": bool(spread > tauB),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--launches", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if cp is None:
        print("CuPy not installed."); return

    if args.quick:
        ns = [2**k for k in (14, 18, 20)]; kappas = [1, 1e4]
    else:
        ns = [2**k for k in (14, 16, 18, 20, 22)]; kappas = [1, 1e2, 1e4, 1e6]

    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    prov = {"gpu": props["name"].decode(), "cupy": cp.__version__,
            "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "launches": args.launches}
    print(f"GPU: {prov['gpu']}  multi-precision sweep\n")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    print(f"{'dtype':>10} {'strategy':>13} {'n':>6} {'kappa':>8} "
          f"{'spread':>10} {'tauH':>10} {'tauB':>10} {'H?':>4} {'B?':>4}")
    print("-" * 86)

    with open(args.out, "a") as fh:
        for dtype in mp.DTYPES:
            u = UNIT_ROUNDOFF[dtype]
            kernels = mp.build_kernels(dtype)
            # which strategies are available for this dtype
            strat = {}
            if kernels.get("naive") is not None:
                strat["naive_atomic"] = mp.naive_atomic
            else:
                print(f"  [{dtype}] naive_atomic unavailable: {kernels.get('naive_err','?')}")
            if kernels.get("tree") is not None:
                strat["tree_fixed"] = mp.tree_fixed
            strat["cupy_sum"] = mp.cupy_sum

            for name, fn in strat.items():
                for n in ns:
                    for kt in kappas:
                        x, s_exact, kappa = gensum(n, kt, seed=args.seed)
                        try:
                            rec = run_cell(name, fn, x, n, kappa, dtype, u,
                                           args.alpha, args.launches, kernels)
                        except Exception as e:
                            print(f"  [{dtype}/{name}/n=2^{int(math.log2(n))}] skipped: {str(e)[:80]}")
                            continue
                        rec["provenance"] = prov
                        fh.write(json.dumps(rec) + "\n"); fh.flush()
                        print(f"{dtype:>10} {name:>13} 2^{int(math.log2(n)):<4} "
                              f"{kappa:>8.2g} {rec['spread_maxmin']:>10.2e} "
                              f"{rec['tau_hoeffding']:>10.2e} {rec['tau_bernstein']:>10.2e} "
                              f"{'no' if not rec['exceeds_hoeffding'] else 'YES':>4} "
                              f"{'no' if not rec['exceeds_bernstein'] else 'YES':>4}")
    print(f"\nwrote {args.out}")
    print("H? / B? = does spread exceed Hoeffding / Bernstein tau. 'no' everywhere = both hold.")


if __name__ == "__main__":
    main()
