"""
tail_test.py — WP1: does the bound DOMINATE the empirical tail at every level?

The theorem is a statement for every confidence level, not one alpha:
    P( |s1 - s2| > tau(alpha) ) <= alpha      for all alpha in (0,1).
The exceedance sweeps only checked alpha = 1e-3 with 200 launches, where zero
exceedances is uninformative. This script measures the whole tail:

  1. Run the naive-atomic reduction L times (GPU) or L permutation pairs (CPU).
  2. Form L/2 INDEPENDENT two-run spreads d_i = |r_{2i} - r_{2i+1}|.
  3. For a grid of alpha in {0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 1e-3, 1e-4, 1e-5}:
        tau(alpha)         -> from bounds (two-run, alpha/2 per run)
        observed rate      -> fraction of d_i > tau(alpha)
        95% upper bound    -> Clopper-Pearson on that fraction
     Dominance holds at alpha iff the upper bound <= alpha.
  4. Report the empirical quantile in natural units:
        lambda_emp(q) = Q_d(1-q) / (sqrt(2) * u * sqrt(n) * S)
     against lambda(alpha/2, n) from the bound, for each level -> the gap in
     lambda-units as a function of level (is it flat? does the tail decay as
     exp(-lambda^2/2)?).
  5. Sub-Gaussian check: regress log P(d > x) on x^2 over the tail; report slope
     and R^2.

Outputs: <out>.json (summary per n) and <out>_<n>.npy (raw spreads) for plots.

Run (GPU pod):
  python tail_test.py --device gpu --launches 20000 --out results/tail_gpu.json
  python tail_test.py --device gpu --launches 100000 --ns 20 --out results/tail_gpu_2p20.json
Run (CPU, laptop):
  python tail_test.py --device cpu --launches 200000 --ns 16 18 --out results/tail_cpu.json

Needs: bounds.py, gensum.py, reference.py (and gpu_strategies.py + cupy for --device gpu).
"""

import argparse, json, math, os, time
import numpy as np
from scipy.stats import beta as _beta

from gensum import gensum
from reference import sum_abs
from bounds import tau_probabilistic, tau_freedman, lambda_for_alpha, UNIT_ROUNDOFF, SIGMA2_OVER_U2_UNIFORM

ALPHAS = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 1e-3, 1e-4, 1e-5]


def cp_upper(k, m, conf=0.95):
    """Clopper-Pearson one-sided upper bound on a binomial rate with k of m."""
    if k >= m:
        return 1.0
    return float(_beta.ppf(conf, k + 1, m - k))


def spreads_gpu(x, n, launches):
    import cupy as cp
    import gpu_strategies as gs
    kern = gs._load()
    xg = cp.asarray(x)
    r = np.empty(launches, dtype=np.float64)
    for i in range(launches):
        r[i] = gs.naive_atomic(xg, n, kern)
    return r


def spreads_cpu(x, n, launches, rng):
    """Permutation-induced spread: sequential fp32 sum of permuted input.
    np.cumsum is a true left-to-right accumulation (np.sum is pairwise)."""
    r = np.empty(launches, dtype=np.float64)
    for i in range(launches):
        r[i] = float(np.cumsum(x[rng.permutation(n)], dtype=np.float32)[-1])
    return r


def analyse(d, n, u, S, alpha_design):
    m = len(d)
    out = {"pairs": int(m), "spread_median": float(np.median(d)), "spread_max": float(d.max())}
    rows = []
    nat = math.sqrt(2.0) * u * math.sqrt(n) * S          # two-run natural unit
    for a in ALPHAS:
        tauH = tau_probabilistic(n, u, S, a)
        tauFa = tau_freedman(n, u, S, a, SIGMA2_OVER_U2_UNIFORM)
        tauFm = tau_freedman(n, u, S, a)
        kH = int((d > tauH).sum()); kFa = int((d > tauFa).sum()); kFm = int((d > tauFm).sum())
        q = float(np.quantile(d, 1 - a)) if m * a >= 1 else float("nan")
        rows.append({
            "alpha": a,
            "tauH": tauH, "tauFa": tauFa, "tauFm": tauFm,
            "rate_H": kH / m, "ub95_H": cp_upper(kH, m),
            "rate_Fa": kFa / m, "ub95_Fa": cp_upper(kFa, m),
            "rate_Fm": kFm / m, "ub95_Fm": cp_upper(kFm, m),
            "dominates_H": cp_upper(kH, m) <= a,
            "dominates_Fm": cp_upper(kFm, m) <= a,
            "resolvable": m * a >= 10,                     # >=10 expected exceedances
            "lambda_bound": lambda_for_alpha(a / 2, n, u),
            "lambda_emp": q / nat if q == q else float("nan"),   # measured quantile in natural units
            "utilization_H": q / tauH if q == q else float("nan"),
        })
    out["levels"] = rows
    # sub-Gaussian tail fit: log P(d > x) vs x^2 over the upper tail (P in [1e-3, 0.3])
    xs = np.quantile(d, 1 - np.array([0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]))
    ps = np.array([(d > x).mean() for x in xs])
    ok = ps > 0
    if ok.sum() >= 4:
        A = np.vstack([xs[ok] ** 2 / nat ** 2, np.ones(ok.sum())]).T
        coef, res, *_ = np.linalg.lstsq(A, np.log(ps[ok]), rcond=None)
        pred = A @ coef
        ss = ((np.log(ps[ok]) - np.log(ps[ok]).mean()) ** 2).sum()
        r2 = 1 - ((np.log(ps[ok]) - pred) ** 2).sum() / ss if ss > 0 else float("nan")
        # Gaussian in natural units would give slope -1/2 (P ~ exp(-lambda^2/2))
        out["subgaussian_fit"] = {"slope_per_lambda2": float(coef[0]), "gaussian_slope": -0.5,
                                  "r2": float(r2), "effective_sigma_nat": float(math.sqrt(-1 / (2 * coef[0]))) if coef[0] < 0 else None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--ns", type=int, nargs="+", default=[16, 18, 20, 22], help="log2 sizes")
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--launches", type=int, default=20000, help="runs (pairs = launches/2)")
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    u = UNIT_ROUNDOFF["float32"] if "float32" in UNIT_ROUNDOFF else 2.0 ** -24
    rng = np.random.default_rng(args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    prov = {"device": args.device, "launches": args.launches, "kappa": args.kappa,
            "seed": args.seed, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if args.device == "gpu":
        import cupy as cp
        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        prov["gpu"] = props["name"].decode(); prov["cupy"] = cp.__version__
    print(json.dumps(prov))

    summary = {"provenance": prov, "cells": []}
    for k in args.ns:
        n = 2 ** k
        x, s_exact, kappa = gensum(n, args.kappa, seed=args.seed)
        x = x.astype(np.float32)
        S = sum_abs(x)
        t0 = time.time()
        r = spreads_gpu(x, n, args.launches) if args.device == "gpu" else spreads_cpu(x, n, args.launches, rng)
        dt = time.time() - t0
        d = np.abs(r[0::2] - r[1::2])            # independent pairs
        np.save(args.out.replace(".json", f"_{k}.npy"), d)
        cell = {"n": n, "log2n": k, "kappa_achieved": float(kappa), "S": float(S),
                "distinct": int(len(np.unique(r))), "seconds": dt}
        cell.update(analyse(d, n, u, S, args.alpha))
        summary["cells"].append(cell)
        print(f"\nn=2^{k}  kappa={kappa:.3g}  pairs={len(d)}  distinct results={cell['distinct']}  ({dt:.0f}s)")
        print(f"{'alpha':>7} {'tauH':>10} {'rate_H':>8} {'ub95':>8} {'dom?':>5} | {'tauFm':>10} {'rate_Fm':>8} {'ub95':>8} {'dom?':>5} | {'lam_bnd':>7} {'lam_emp':>7} {'util_H':>7}")
        for row in cell["levels"]:
            flag = "" if row["resolvable"] else " (<10 expected; not resolvable)"
            print(f"{row['alpha']:>7.0e} {row['tauH']:>10.3e} {row['rate_H']:>8.2e} {row['ub95_H']:>8.2e} {str(row['dominates_H']):>5} | "
                  f"{row['tauFm']:>10.3e} {row['rate_Fm']:>8.2e} {row['ub95_Fm']:>8.2e} {str(row['dominates_Fm']):>5} | "
                  f"{row['lambda_bound']:>7.2f} {row['lambda_emp']:>7.2f} {row['utilization_H']:>7.3f}{flag}")
        if "subgaussian_fit" in cell:
            f = cell["subgaussian_fit"]
            print(f"tail fit: log P ~ {f['slope_per_lambda2']:+.3f}*lambda^2 (Gaussian: -0.5), R^2={f['r2']:.3f}, "
                  f"effective sigma = {f['effective_sigma_nat']} natural units")
        with open(args.out, "w") as fh:
            json.dump(summary, fh, indent=1)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
