"""
sweep.py — Phase A measurement sweep (CPU, fp32, fixed-order strategies).

For every (strategy, n, kappa_target, seed):
  1. generate x with prescribed kappa (gensum)         -> x, s_exact, kappa_achieved
  2. compute s_hat with the strategy
  3. record error metrics + the deterministic bound + provenance

One JSON record per config, appended to a .jsonl file. Never aggregate at
write time; analysis happens in analyze.py. Raw records are the artifact.

Usage:
  python sweep.py --out results/phaseA.jsonl
  python sweep.py --out results/quick.jsonl --quick     # small grid, ~seconds
"""

import argparse, json, math, os, platform, socket, subprocess, sys, time
import numpy as np

from strategies import STRATEGIES
from gensum import gensum
from reference import sum_abs
from bounds import UNIT_ROUNDOFF, deterministic_bound


def ulp_distance(a: float, b: float) -> int:
    """Distance in representable fp32 steps between a and b."""
    a32 = np.float32(a); b32 = np.float32(b)
    ia = a32.view(np.int32).astype(np.int64); ib = b32.view(np.int32).astype(np.int64)
    # remap negatives to a continuous ordering
    if ia < 0: ia = -(2**31) - ia
    if ib < 0: ib = -(2**31) - ib
    return int(abs(ia - ib))


def provenance() -> dict:
    """Log this with EVERY record. It is half the paper."""
    rec = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }
    try:
        import numba; rec["numba"] = numba.__version__
    except ImportError:
        rec["numba"] = None
    # GPU fields populated in Phase B; harmless here
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5)
        rec["gpu"] = out.stdout.strip() or None
    except Exception:
        rec["gpu"] = None
    try:
        rec["git_commit"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
        ).stdout.strip() or None
    except Exception:
        rec["git_commit"] = None
    return rec


def run_one(strategy: str, n: int, kappa_target: float, seed: int, u: float) -> dict:
    x, s_exact, kappa = gensum(n, kappa_target, seed=seed)
    S = sum_abs(x)
    f = STRATEGIES[strategy]

    t0 = time.perf_counter(); s_hat = float(f(x)); t1 = time.perf_counter()

    err = abs(s_hat - s_exact)
    bound = deterministic_bound(strategy, n, u, S)
    # dimensionless normalized error: should be O(n) worst-case or O(sqrt n) probabilistic
    e_hat = err / (u * S) if S > 0 else float("nan")

    return {
        "strategy": strategy, "n": n, "kappa_target": kappa_target, "kappa": kappa,
        "seed": seed, "dtype": "float32", "u": u,
        "s_exact": s_exact, "s_hat": s_hat, "sum_abs": S,
        "abs_error": err,
        "rel_error": err / abs(s_exact) if s_exact != 0 else float("nan"),
        "ulp_error": ulp_distance(s_hat, s_exact),
        "e_hat": e_hat,                       # err / (u * S)
        "bound_det": bound,
        "tightness": err / bound if bound not in (0, math.inf) else float("nan"),
        "time_s": t1 - t0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    if args.quick:
        ns = [2**k for k in range(10, 17, 2)]       # 2^10 .. 2^16
        kappas = [1, 1e2, 1e4]
        strategies = ["recursive", "pairwise", "kahan"]
    else:
        ns = [2**k for k in range(10, 25, 2)]       # 2^10 .. 2^24
        kappas = [1, 1e2, 1e4, 1e6]
        strategies = ["recursive", "pairwise", "kahan", "numpy_sum"]

    u = UNIT_ROUNDOFF["float32"]
    prov = provenance()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    total = len(strategies) * len(ns) * len(kappas) * args.seeds
    done = 0
    with open(args.out, "a") as fh:
        for strategy in strategies:
            for n in ns:
                for kt in kappas:
                    for seed in range(args.seeds):
                        rec = run_one(strategy, n, kt, seed, u)
                        rec["provenance"] = prov
                        fh.write(json.dumps(rec) + "\n"); fh.flush()
                        done += 1
                        if done % 10 == 0 or done == total:
                            print(f"[{done}/{total}] {strategy:>10} n=2^{int(math.log2(n)):<2} "
                                  f"kappa={rec['kappa']:>9.3g} e_hat={rec['e_hat']:>10.3g} "
                                  f"tight={rec['tightness']:>8.2e}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
