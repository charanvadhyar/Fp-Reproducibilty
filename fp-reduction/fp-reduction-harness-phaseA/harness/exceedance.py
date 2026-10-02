"""
exceedance.py — Phase 3 core experiment: does the derived tolerance hold?

Phase A measured ERROR (distance from exact sum) and confirmed ~sqrt(n) scaling.
This measures SPREAD (distance between two computed results) and asks the
question the whole project turns on:

    does |s_hat_1 - s_hat_2| exceed tau more often than the designed rate alpha?

tau is the two-run tolerance from bounds.tau_probabilistic. If the probabilistic
model holds, the measured exceedance rate should be <= alpha. The ratio

    c = measured_exceedance / alpha

is the CORRECTION FACTOR:
    c ~ 1   -> the derived tolerance holds as-is (model valid)
    c > 1   -> real rounding departs from the mean-zero model; tau must be
               inflated by c. c(kappa) is the 'cost of correlation'.

How spread is generated (CPU, no GPU needed):
    fixed-order strategies are deterministic, so two identical runs give zero
    spread. We create legitimate same-computation spread by PERMUTING the input:
    same numbers, different order -> different rounding -> different result.
    This is exactly the reordering that a parallel reduction does.

Usage:
    python exceedance.py --out results/exceedance.jsonl
    python exceedance.py --out results/quick_exc.jsonl --quick
"""

import argparse, json, math, os, time
import numpy as np

from strategies import recursive, pairwise, kahan
from gensum import gensum
from reference import sum_abs
from bounds import tau_probabilistic, UNIT_ROUNDOFF

STRAT_FN = {"recursive": recursive, "pairwise": pairwise, "kahan": kahan}


def run_cell(strategy, n, kappa_target, u, alpha, trials, base_seed, rng):
    """
    For one (strategy, n, kappa) cell: generate x once, then over `trials`
    permutation-pairs measure how often the two summations differ by > tau.
    """
    x, s_exact, kappa = gensum(n, kappa_target, seed=base_seed)
    S = sum_abs(x)
    tau = tau_probabilistic(n, u, S, alpha)
    f = STRAT_FN[strategy]

    exceed = 0
    max_spread = 0.0
    spreads = []
    for _ in range(trials):
        p1 = rng.permutation(n)
        p2 = rng.permutation(n)
        s1 = float(f(x[p1]))
        s2 = float(f(x[p2]))
        spread = abs(s1 - s2)
        spreads.append(spread)
        if spread > tau:
            exceed += 1
        if spread > max_spread:
            max_spread = spread

    rate = exceed / trials
    spreads = np.array(spreads)

    # The derived tau is often very conservative -> exceedance rate is 0 and
    # uninformative. The more useful quantity: what scaling t of tau is
    # exceeded at exactly rate alpha? t is the empirical tightening factor.
    #   t ~ 1    : derived tau is well-calibrated
    #   t << 1   : derived tau is loose by 1/t (could tighten and still hold)
    #   t > 1    : derived tau is too tight (would need to loosen) -> underestimate
    # Estimate as the (1-alpha) quantile of spread, divided by tau.
    q = float(np.quantile(spreads, 1.0 - alpha))
    tightening_factor = q / tau if tau > 0 else float("nan")

    return {
        "strategy": strategy, "n": n, "kappa_target": kappa_target, "kappa": kappa,
        "dtype": "float32", "u": u, "alpha": alpha, "trials": trials,
        "sum_abs": S, "tau": tau,
        "exceedances": exceed,
        "exceedance_rate": rate,
        "correction_factor": (rate / alpha) if alpha > 0 else float("nan"),
        # the informative metric when tau is conservative:
        "empirical_q_at_1_minus_alpha": q,     # the spread quantile tau should match
        "tightening_factor": tightening_factor, # q/tau: how loose tau is (0.1 = 10x loose)
        # spread distribution summary
        "spread_median": float(np.median(spreads)),
        "spread_p99": float(np.quantile(spreads, 0.99)),
        "spread_max": max_spread,
        "spread_over_tau_median": float(np.median(spreads)) / tau if tau > 0 else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.quick:
        ns = [2**k for k in range(10, 17, 2)]
        kappas = [1, 1e2, 1e4]
        strategies = ["recursive"]
        trials = min(args.trials, 500)
    else:
        ns = [2**k for k in range(10, 23, 2)]     # 2^10..2^22 (2^24 slow w/ permute)
        kappas = [1, 1e2, 1e4, 1e6]
        strategies = ["recursive", "pairwise", "kahan"]
        trials = args.trials

    u = UNIT_ROUNDOFF["float32"]
    rng = np.random.default_rng(args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    total = len(strategies) * len(ns) * len(kappas)
    done = 0
    print(f"alpha = {args.alpha:g}, trials/cell = {trials}\n")
    print(f"{'strategy':>10} {'n':>8} {'kappa':>10} {'exc_rate':>10} {'tighten_t':>13}")
    print("-" * 56)
    with open(args.out, "a") as fh:
        for strategy in strategies:
            for n in ns:
                for kt in kappas:
                    rec = run_cell(strategy, n, kt, u, args.alpha, trials, args.seed, rng)
                    fh.write(json.dumps(rec) + "\n"); fh.flush()
                    done += 1
                    print(f"{strategy:>10} 2^{int(math.log2(n)):<6} {rec['kappa']:>10.2g} "
                          f"{rec['exceedance_rate']:>10.4f} {rec['tightening_factor']:>13.4f}")
    print(f"\nwrote {args.out}")
    print("\ntightening_factor t = (spread quantile at 1-alpha) / tau:")
    print("  t ~ 1   -> derived tau well-calibrated")
    print("  t << 1  -> derived tau is loose by 1/t (headroom; low false-positive rate)")
    print("  t > 1   -> derived tau too tight (would underestimate; real spread bigger)")


if __name__ == "__main__":
    main()