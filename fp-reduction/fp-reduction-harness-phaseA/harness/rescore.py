"""
rescore.py — WP2: re-test every stored spread against the ORDER-ROBUST tolerances
(tau_PS, tau_PSFa, tau_PSF from bounds.py) without re-running anything.

Inputs are regenerated from the recorded (n, kappa_target, seed) with gensum, so
Phi(|x|) is computed for exactly the vectors the sweeps used.

Works on:
  CPU records (exceedance.py):   fields n, kappa_target, sum_abs, empirical_q_at_1_minus_alpha, spread_max
  GPU records (gpu_exceedance.py / multi-arch): fields n, kappa (achieved), sum_abs, spread_maxmin, strategy
    (kappa_target inferred as the nearest decade of the achieved kappa; seed assumed 0 unless --seed)

Usage:
  python rescore.py results/exceedance.jsonl --kind cpu
  python rescore.py results/gpu_exceedance.jsonl results/gpu_a100.jsonl results/gpu_h100.jsonl --kind gpu
  python rescore.py ... --sigma2 0.20          # calibrated variance to use for tau_PSF (default 0.18)
"""

import argparse, json, math, sys
import numpy as np
from gensum import gensum
from bounds import (tau_probabilistic, tau_freedman, tau_ps, tau_ps_freedman, phi_abs,
                    UNIT_ROUNDOFF, SIGMA2_OVER_U2_UNIFORM)


def load(paths):
    recs = []
    for p in paths:
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if "n" in r and ("sum_abs" in r):
                    r["_file"] = p
                    recs.append(r)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--kind", choices=["cpu", "gpu"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--sigma2", type=float, default=0.18)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    u = UNIT_ROUNDOFF["float32"]
    recs = load(args.files)
    if args.kind == "gpu":
        recs = [r for r in recs if r.get("strategy") in ("naive_atomic", "block_atomic")]
    print(f"{len(recs)} nondeterministic cells loaded")

    phi_cache = {}
    rows = []
    hdr = f"{'file':>18} {'strat':>12} {'n':>7} {'kappa':>8} {'spread':>10} | {'tau_H':>9} {'tau_Fm':>9} {'tau_PS':>9} {'tau_PSFa':>9} {'tau_PSF':>9} | {'u_H':>6} {'u_PSF':>6} {'exc':>12}"
    print(hdr); print("-" * len(hdr))
    counts = {k: 0 for k in ("H", "Fm", "PS", "PSFa", "PSF")}
    for r in recs:
        n = int(r["n"]); S = float(r["sum_abs"])
        if args.kind == "cpu":
            kt = float(r["kappa_target"]); spread = float(r["empirical_q_at_1_minus_alpha"]); spread_max = float(r["spread_max"])
        else:
            kt = 10.0 ** round(math.log10(float(r["kappa"]))); spread = float(r["spread_maxmin"]); spread_max = spread
        key = (n, kt, args.seed)
        if key not in phi_cache:
            x, _, _ = gensum(n, kt, seed=args.seed)
            phi_cache[key] = (phi_abs(x), float(np.abs(np.asarray(x, dtype=np.float64)).sum()))
        phi, S_regen = phi_cache[key]
        if abs(S_regen - S) / S > 1e-6:
            print(f"  WARNING: regenerated S differs from stored S for n={n} kappa={kt} ({S_regen:.6e} vs {S:.6e}); seed mismatch?")
        taus = {
            "H": tau_probabilistic(n, u, S, args.alpha),
            "Fm": tau_freedman(n, u, S, args.alpha),
            "PS": tau_ps(n, u, phi, args.alpha),
            "PSFa": tau_ps_freedman(n, u, S, phi, args.alpha, SIGMA2_OVER_U2_UNIFORM),
            "PSF": tau_ps_freedman(n, u, S, phi, args.alpha, args.sigma2),
        }
        exc = [k for k in taus if spread_max > taus[k]]
        for k in exc: counts[k] += 1
        row = {"file": r["_file"], "strategy": r.get("strategy", "cpu"), "n": n, "kappa_target": kt, "spread": spread,
               "spread_max": spread_max, "phi_over_nS2": phi / (n * S * S), **{f"tau_{k}": v for k, v in taus.items()},
               "exceeded": exc}
        rows.append(row)
        print(f"{r['_file'][-18:]:>18} {row['strategy']:>12} 2^{int(math.log2(n)):<5} {kt:>8.0e} {spread:>10.3e} | "
              f"{taus['H']:>9.2e} {taus['Fm']:>9.2e} {taus['PS']:>9.2e} {taus['PSFa']:>9.2e} {taus['PSF']:>9.2e} | "
              f"{spread/taus['H']:>6.3f} {spread/taus['PSF']:>6.3f} {','.join(exc) if exc else '-':>12}")
    print("\nexceedances (spread_max > tau) out of", len(rows), "cells:")
    for k, v in counts.items():
        print(f"  tau_{k:<5}: {v}")
    util = {k: max(rw["spread"] / rw[f"tau_{k}"] for rw in rows) for k in counts}
    print("max utilization (spread/tau) over cells:", {k: round(v, 3) for k, v in util.items()})
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"cells": rows, "exceedance_counts": counts, "max_utilization": util,
                       "sigma2_over_u2": args.sigma2, "alpha": args.alpha}, fh, indent=1)
        print("saved", args.out)


if __name__ == "__main__":
    main()
