"""
wp7_exponent_generator.py — WP7: is the exponent -> 0 at high kappa (Fig. 2) a property
of the arithmetic or of the cancelling input generator?

Measures the recursive (sequential fp32) summation error |s_hat - s| / (u * sum|x|)
against n for two generators at the same target kappa:
  (a) gensum           : the paper's generator (exact-cancellation construction)
  (b) gaussian_offset  : x ~ N(mu, 1) with mu set so that sum|x| / |sum x| ~ kappa
                         (no constructed cancellation; kappa arises statistically)
and fits the scaling exponent beta (error ~ n^beta) for each. Exact reference: math.fsum.

Run:  python wp7_exponent_generator.py            (~2-5 min on a laptop)
"""
import math, json, sys, os
import numpy as np
sys.path.insert(0, os.getcwd())
try:
    from gensum import gensum
except Exception:
    gensum = None

U = 2.0 ** -24
NS = [2 ** k for k in range(10, 23, 2)]
KAPPAS = [1.0, 1e2, 1e4, 1e6]
SEEDS = 10

def gaussian_offset(n, kappa, rng):
    """N(0,1) draws shifted by one constant so that sum|x|/|sum x| = kappa exactly in fp64
    (cancellation arises statistically, not by construction). kappa=1 -> all positive."""
    if kappa <= 1:
        return (rng.random(n) + 0.5).astype(np.float32)
    x = rng.normal(0.0, 1.0, n)
    S = np.abs(x).sum(); c = (S / kappa - x.sum()) / n
    return (x + c).astype(np.float32)

def err_norm(x):
    s_hat = float(np.cumsum(x, dtype=np.float32)[-1])
    s = math.fsum(map(float, x))
    return abs(s_hat - s) / (U * float(np.abs(x.astype(np.float64)).sum())), abs(float(np.abs(x.astype(np.float64)).sum()) / s) if s else float('inf')

def fit(ns, errs):
    return float(np.polyfit(np.log(ns), np.log(errs), 1)[0])

out = {}
for gen_name in (["gensum"] if gensum else []) + ["gaussian_offset"]:
    out[gen_name] = {}
    for kap in KAPPAS:
        med = []; kach = []
        for n in NS:
            e = []; k_ = []
            for sd in range(SEEDS):
                rng = np.random.default_rng(1000 * sd + int(math.log2(n)))
                if gen_name == "gensum":
                    x, _, ka = gensum(n, kap, seed=sd); x = x.astype(np.float32)
                else:
                    x = gaussian_offset(n, kap, rng)
                en, ka = err_norm(x); e.append(en); k_.append(ka)
            med.append(float(np.median(e))); kach.append(float(np.median(k_)))
        beta = fit(NS, med)
        out[gen_name][str(kap)] = {"beta": beta, "median_err_norm": med, "kappa_achieved_median": kach}
        print(f"{gen_name:>16} kappa={kap:>7.0e} (achieved ~{np.median(kach):.2g}): beta = {beta:+.3f}   err_norm by n: "
              + " ".join(f"{m:.2g}" for m in med))
json.dump(out, open("results/wp7_exponents.json", "w"), indent=1)
print("saved results/wp7_exponents.json")


# ---------------------------------------------------------------------------
# Part 2: MECHANISM.  The exact identity  s_hat - s = sum_k delta_k t_k  gives
#   Var(err) = sigma^2 * sum_k t_k^2   (sigma^2 = 0.186 u^2 measured per rounding)
# so  RMS(err) ~ u sqrt(0.186 sum_k s_k^2).  Partial sums grow like k for
# positive data (sum ~ n^3 -> err ~ n^1.5 -> normalized exponent 0.5) and like
# sqrt(k) for cancelling data (sum ~ n^2 -> err ~ n -> normalized exponent 0).
# This part checks the prediction against the measured RMS error.
# ---------------------------------------------------------------------------
print("\nmechanism check: RMS(err)/(uS) measured vs predicted u*sqrt(0.186*sum s_k^2)/(uS)")
mech = {}
for n in (2 ** 14, 2 ** 18, 2 ** 20):
    for kap in KAPPAS:
        errs, preds = [], []
        for sd in range(12):
            rng = np.random.default_rng(sd); x = gaussian_offset(n, kap, rng)
            cs = np.cumsum(x, dtype=np.float32); s_hat = float(cs[-1]); s = math.fsum(map(float, x))
            errs.append((s_hat - s) ** 2); preds.append(0.186 * U * U * float((cs.astype(np.float64)[1:] ** 2).sum()))
        rms = math.sqrt(np.mean(errs)); pred = math.sqrt(np.mean(preds)); S = float(np.abs(x.astype(np.float64)).sum())
        mech[f"n=2^{int(math.log2(n))},kappa={kap:.0e}"] = {"measured": rms / (U * S), "predicted": pred / (U * S), "ratio": rms / pred}
        print(f"  n=2^{int(math.log2(n))} kappa={kap:>5.0e}: measured {rms/(U*S):8.3f}  predicted {pred/(U*S):8.3f}  ratio {rms/pred:5.2f}")
out["mechanism_check"] = mech
json.dump(out, open("results/wp7_exponents.json", "w"), indent=1)
