"""
analyze.py — turn raw records into Figure 1 and the numbers the panel asked for.

Reads the .jsonl written by sweep.py and reports, per strategy:
  * the SCALING EXPONENT beta1 from   log e_hat = beta0 + beta1*log n
        beta1 ~ 1.0  -> error grows like n      (worst-case regime)
        beta1 ~ 0.5  -> error grows like sqrt n (probabilistic regime)
  * the kappa coefficient beta2 from  log e_hat = ... + beta2*log kappa
        should be ~0 if the normalization by S was right; if not, the
        normalization is wrong — this is a built-in sanity check
  * tightness ratio = measured error / deterministic bound, per n
        near 1   -> bound is tight
        << 1     -> bound is uselessly loose (expected at large n)

Usage:
  python analyze.py results/phaseA.jsonl
  python analyze.py results/phaseA.jsonl --plot fig1.png
"""

import argparse, json, math
from collections import defaultdict
import numpy as np


def load(path):
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def fit_exponent(recs):
    """OLS: log e_hat ~ b0 + b1 log n + b2 log kappa. Returns (b1, b2, n_points)."""
    rows = [(r["n"], r["kappa"], r["e_hat"]) for r in recs
            if r["e_hat"] and r["e_hat"] > 0 and math.isfinite(r["e_hat"])]
    if len(rows) < 4:
        return float("nan"), float("nan"), len(rows)
    n = np.array([math.log(r[0]) for r in rows])
    k = np.array([math.log(max(r[1], 1.0)) for r in rows])
    y = np.array([math.log(r[2]) for r in rows])
    X = np.column_stack([np.ones_like(n), n, k])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[1]), float(beta[2]), len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--plot", default=None)
    args = ap.parse_args()
    recs = load(args.path)

    by_strat = defaultdict(list)
    for r in recs:
        by_strat[r["strategy"]].append(r)

    print(f"{'strategy':>10} | {'beta1 (n exp)':>13} | {'beta2 (kappa)':>13} | pts")
    print("-" * 52)
    for s, rs in by_strat.items():
        b1, b2, m = fit_exponent(rs)
        print(f"{s:>10} | {b1:>13.3f} | {b2:>13.3f} | {m}")
    print()
    print("beta1 ~1 = worst-case regime, ~0.5 = probabilistic (sqrt n) regime.")
    print("beta2 should be ~0 (normalization check).")
    print()
    # per-kappa exponent: pooling across kappa mixes very different data structures
    # (all-positive vs gensum cancellation). Fit the n-exponent within each kappa level.
    print("Scaling exponent beta1 fitted PER kappa level (n only):")
    kts = sorted({r["kappa_target"] for r in recs})
    print(f"{'strategy':>10} " + " ".join(f"{'k='+format(k,'g'):>10}" for k in kts))
    for s, rs in by_strat.items():
        cells = []
        for kt in kts:
            sub = [(r["n"], r["e_hat"]) for r in rs
                   if r["kappa_target"] == kt and r["e_hat"] and r["e_hat"] > 0]
            if len(sub) >= 3:
                ln = np.array([math.log(a) for a, _ in sub]); ly = np.array([math.log(b) for _, b in sub])
                b1 = float(np.polyfit(ln, ly, 1)[0]); cells.append(f"{b1:>10.3f}")
            else:
                cells.append(f"{'-':>10}")
        print(f"{s:>10} " + " ".join(cells))
    print()

    # tightness ratio vs n, per strategy, median over kappa & seeds
    print("Tightness ratio (measured error / deterministic bound), median over kappa & seeds:")
    print(f"{'strategy':>10} " + " ".join(f"{'2^'+str(int(math.log2(n))):>8}"
          for n in sorted({r['n'] for r in recs})))
    for s, rs in by_strat.items():
        by_n = defaultdict(list)
        for r in rs:
            if math.isfinite(r["tightness"]):
                by_n[r["n"]].append(r["tightness"])
        line = f"{s:>10} " + " ".join(f"{np.median(by_n[n]):>8.1e}" if by_n[n] else f"{'-':>8}"
                                      for n in sorted({r['n'] for r in recs}))
        print(line)

    if args.plot:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # ---- Figure 1: clean regime, kappa = 1, slopes anchored to data ----
        fig, ax = plt.subplots(figsize=(7, 5))
        anchor = None  # (n0, y0) from recursive's first point, to place slope refs
        for s, rs in by_strat.items():
            by_n = defaultdict(list)
            for r in rs:
                if r["kappa_target"] == 1 and r["e_hat"] and r["e_hat"] > 0:
                    by_n[r["n"]].append(r["e_hat"])
            if not by_n:
                continue
            ns = sorted(by_n); med = [np.median(by_n[n]) for n in ns]
            ax.loglog(ns, med, "o-", label=s)
            if s == "recursive":
                anchor = (ns[0], med[0])

        # slope reference lines, anchored to recursive's first point so that
        # "is recursive parallel to sqrt(n)?" is a direct visual check
        ns_all = sorted({r["n"] for r in recs})
        if anchor:
            n0, y0 = anchor
            ax.loglog(ns_all, [y0 * (n / n0) for n in ns_all],
                      "k--", alpha=.4, label="slope ~ n (worst-case)")
            ax.loglog(ns_all, [y0 * (n / n0) ** 0.5 for n in ns_all],
                      "k:", alpha=.6, label="slope ~ sqrt(n) (probabilistic)")
        ax.set_xlabel("n (terms summed)")
        ax.set_ylabel("normalized error  e_hat = |err| / (u * sum|x|)")
        ax.set_title("Figure 1: error scaling at kappa = 1 (fp32, CPU, fixed order)")
        ax.legend(); ax.grid(True, which="both", alpha=.3)
        fig.tight_layout(); fig.savefig(args.plot, dpi=150)
        print("saved", args.plot)

        # ---- Figure 2: recursive only, one line per kappa (regime breakdown) ----
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        rec = by_strat.get("recursive", [])
        kts = sorted({r["kappa_target"] for r in rec})
        for kt in kts:
            by_n = defaultdict(list)
            for r in rec:
                if r["kappa_target"] == kt and r["e_hat"] and r["e_hat"] > 0:
                    by_n[r["n"]].append(r["e_hat"])
            if not by_n:
                continue
            ns = sorted(by_n); med = [np.median(by_n[n]) for n in ns]
            ax2.loglog(ns, med, "o-", label=f"kappa={kt:g}")
        if anchor:
            n0, y0 = anchor
            ax2.loglog(ns_all, [y0 * (n / n0) ** 0.5 for n in ns_all],
                       "k:", alpha=.6, label="slope ~ sqrt(n)")
        ax2.set_xlabel("n (terms summed)")
        ax2.set_ylabel("normalized error  e_hat")
        ax2.set_title("Figure 2: recursive error vs n, by conditioning (kappa)")
        ax2.legend(); ax2.grid(True, which="both", alpha=.3)
        fig2.tight_layout()
        fig2_path = args.plot.replace(".png", "_kappa.png")
        fig2.savefig(fig2_path, dpi=150)
        print("saved", fig2_path)


if __name__ == "__main__":
    main()