"""
gpu_exceedance.py — Phase B core: does the derived tau hold on REAL GPU spread?

The CPU exceedance test created spread by permuting inputs (a model of
reordering). This uses the genuine run-to-run nondeterminism of GPU atomic
reductions -- no permutation, the hardware scheduler does it.

For each (strategy, n, kappa):
  1. generate x with prescribed kappa (gensum) and put it on the GPU
  2. run the strategy `launches` times -> real run-to-run results
  3. spread = pairwise |s_i - s_j| over launches (or max-min as a summary)
  4. tau = bounds.tau_probabilistic(n, u, S, alpha)
  5. report: does spread exceed tau? how far below (tightening factor)?

Strategies:
  naive_atomic, block_atomic  -> nondeterministic (real spread)
  tree_fixed,  cupy_sum       -> deterministic (spread = 0, sanity controls)

Deterministic strategies should show spread 0 and tightening_factor 0: a built-in
sanity check that the harness isn't inventing spread.

Needs: gensum.py, reference.py, bounds.py (same dir), cupy, gpu_strategies.py.

Run (on the pod):
  python gpu_exceedance.py --out results/gpu_exceedance.jsonl
  python gpu_exceedance.py --out results/quick_gpu.jsonl --quick
"""

import argparse, json, math, os, time
import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None

from gensum import gensum
from reference import sum_abs, exact_sum
from bounds import tau_probabilistic, UNIT_ROUNDOFF
import gpu_strategies as gs


def run_cell(name, x_cpu, n, kappa, u, alpha, launches, kernels):
    """Run one strategy `launches` times on the GPU, compare spread to tau."""
    x_gpu = cp.asarray(x_cpu)
    S = sum_abs(x_cpu)
    s_ref = exact_sum(x_cpu)
    tau = tau_probabilistic(n, u, S, alpha)
    fn = gs.STRATEGIES[name]

    res = np.array(
        [fn(x_gpu, n, kernels) if name != "cupy_sum" else fn(x_gpu, n)
         for _ in range(launches)],
        dtype=np.float64,
    )

    distinct = int(len(np.unique(res)))
    spread_maxmin = float(res.max() - res.min())
    # a robust high quantile of pairwise spread: compare each run to the first
    pair_spreads = np.abs(res - res[0])
    spread_q = float(np.quantile(pair_spreads, 1.0 - alpha)) if distinct > 1 else 0.0
    mean_err = float(abs(res.mean() - s_ref))

    exceed = spread_maxmin > tau
    tightening = (spread_q / tau) if tau > 0 else float("nan")

    return {
        "strategy": name, "n": n, "kappa_target_note": "see kappa", "kappa": kappa,
        "dtype": "float32", "u": u, "alpha": alpha, "launches": launches,
        "sum_abs": S, "tau": tau,
        "distinct_results": distinct,
        "spread_maxmin": spread_maxmin,
        "spread_q_at_1_minus_alpha": spread_q,
        "mean_err_vs_ref": mean_err,
        "exceeds_tau": bool(exceed),
        "tightening_factor": tightening,   # spread_q / tau; 0 for deterministic
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
        print("CuPy not installed. On the pod:  pip install cupy-cuda12x")
        return

    if args.quick:
        ns = [2**k for k in (14, 18, 20)]
        kappas = [1, 1e4]
        strategies = ["naive_atomic", "block_atomic", "tree_fixed"]
    else:
        ns = [2**k for k in (14, 16, 18, 20, 22)]
        kappas = [1, 1e2, 1e4, 1e6]
        strategies = ["naive_atomic", "block_atomic", "tree_fixed", "cupy_sum"]

    u = UNIT_ROUNDOFF["float32"]
    kernels = gs._load()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    prov = {
        "gpu": props["name"].decode(),
        "cupy": cp.__version__,
        "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "launches": args.launches,
    }
    print(f"GPU: {prov['gpu']}  alpha={args.alpha:g}  launches={args.launches}\n")
    print(f"{'strategy':>13} {'n':>6} {'kappa':>9} {'distinct':>8} "
          f"{'spread':>11} {'tau':>11} {'t=sp/tau':>10} {'exceed':>7}")
    print("-" * 80)

    with open(args.out, "a") as fh:
        for strategy in strategies:
            for n in ns:
                for kt in kappas:
                    x, s_exact, kappa = gensum(n, kt, seed=args.seed)
                    rec = run_cell(strategy, x, n, kappa, u, args.alpha,
                                   args.launches, kernels)
                    rec["provenance"] = prov
                    fh.write(json.dumps(rec) + "\n"); fh.flush()
                    print(f"{strategy:>13} 2^{int(math.log2(n)):<4} {kappa:>9.2g} "
                          f"{rec['distinct_results']:>8} {rec['spread_maxmin']:>11.2e} "
                          f"{rec['tau']:>11.2e} {rec['tightening_factor']:>10.4f} "
                          f"{'YES' if rec['exceeds_tau'] else 'no':>7}")
    print(f"\nwrote {args.out}")
    print("\ntightening_factor t = (GPU spread quantile) / tau:")
    print("  deterministic strategies -> t = 0 (spread 0): sanity check")
    print("  atomic strategies -> t = how close real GPU spread gets to tau")
    print("  exceed=YES anywhere -> tau is violated by real hardware (would be a finding)")


if __name__ == "__main__":
    main()
