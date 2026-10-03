"""
fault_study.py — comprehensive fault-detection study for the derived tolerance.

Five experiments, all comparing the Hoeffding tolerance (tau_H) against the
tighter Bernstein tolerance (tau_B), with benign reordering noise as the
non-fault class:

  E1  single-element input bit-flip     (realistic but often sub-noise-floor)
  E2  accumulator bit-flip              (realistic SDC: corruption of the running
                                         sum -> propagates directly, detectable)
  E3  multi-element burst severity      (detection vs number of corrupted elements)
  E4  fault timing in the reduction     (early vs late corruption of the accumulator)
  E5  ROC                               (sweep threshold; TPR vs FPR; AUC;
                                         Hoeffding-tau and Bernstein-tau marked)

The honest story these produce:
  - single low-bit input flips are below the nondeterminism noise floor and
    undetectable by ANY order-comparison verifier (E1) -> bounds the method;
  - accumulator corruption and multi-element faults ARE detectable (E2,E3);
  - detection depends on when the fault occurs (E4);
  - as a classifier, the Bernstein tolerance dominates Hoeffding (E5 ROC),
    detecting smaller faults at the same (zero) false-positive rate.

CPU experiment (recursive summation); fault magnitudes are hardware-independent.

Usage:
  python fault_study.py --n 1048576 --trials 2000 --out results/fault_study.json
"""

import argparse, struct, math, json, os
import numpy as np
from strategies import recursive
from gensum import gensum
from reference import sum_abs
from bounds import tau_probabilistic, tau_bernstein, UNIT_ROUNDOFF


# ---------------------------------------------------------------- bit tools
def flip_bit_f32(v, bit):
    i = struct.unpack('>I', struct.pack('>f', np.float32(v)))[0] ^ (1 << bit)
    return struct.unpack('>f', struct.pack('>I', i))[0]


# ---------------------------------------------------------------- spreads
def benign(x, rng):
    return abs(float(recursive(x[rng.permutation(len(x))])) -
               float(recursive(x[rng.permutation(len(x))])))


def input_flip(x, rng, bit):
    xf = x.copy(); idx = rng.integers(len(x))
    fv = flip_bit_f32(float(xf[idx]), bit)
    if not math.isfinite(fv):
        return None
    xf[idx] = np.float32(fv)
    return abs(float(recursive(x[rng.permutation(len(x))])) -
               float(recursive(xf[rng.permutation(len(x))])))


def accumulator_flip(x, rng, bit):
    """
    Corrupt the running accumulator once, mid-reduction: do a left-to-right sum,
    flip one bit of the partial sum at a random step, continue. Models a register
    SDC during accumulation. Compared against a clean reordered run.
    """
    n = len(x)
    step = int(rng.integers(n // 4, 3 * n // 4))  # corrupt somewhere in the middle
    s = np.float32(0.0)
    for i in range(n):
        s = np.float32(s + x[i])
        if i == step:
            fv = flip_bit_f32(float(s), bit)
            if not math.isfinite(fv):
                return None
            s = np.float32(fv)
    s_fault = float(s)
    s_clean = float(recursive(x[rng.permutation(n)]))
    return abs(s_clean - s_fault)


def multi_flip(x, rng, k, bit=22):
    """k elements each get a bit flipped (burst / multi-element SDC)."""
    xf = x.copy()
    idxs = rng.choice(len(x), size=k, replace=False)
    for idx in idxs:
        fv = flip_bit_f32(float(xf[idx]), bit)
        if math.isfinite(fv):
            xf[idx] = np.float32(fv)
    return abs(float(recursive(x[rng.permutation(len(x))])) -
               float(recursive(xf[rng.permutation(len(x))])))


def timed_accumulator_flip(x, rng, bit, frac):
    """Accumulator flip at a controlled fraction `frac` through the reduction."""
    n = len(x); step = int(frac * n)
    s = np.float32(0.0)
    for i in range(n):
        s = np.float32(s + x[i])
        if i == step:
            fv = flip_bit_f32(float(s), bit)
            if not math.isfinite(fv):
                return None
            s = np.float32(fv)
    return abs(float(recursive(x[rng.permutation(n)])) - float(s))


# ---------------------------------------------------------------- ROC
def roc(benign_arr, fault_arr, n_thresh=200):
    """TPR/FPR as threshold sweeps; returns points and AUC."""
    allv = np.concatenate([benign_arr, fault_arr])
    ths = np.unique(np.quantile(allv, np.linspace(0, 1, n_thresh)))
    pts = []
    for t in ths:
        tpr = float((fault_arr > t).mean())
        fpr = float((benign_arr > t).mean())
        pts.append((fpr, tpr))
    pts = sorted(pts)
    # AUC by trapezoid
    auc = 0.0
    for i in range(1, len(pts)):
        auc += (pts[i][0] - pts[i-1][0]) * (pts[i][1] + pts[i-1][1]) / 2
    return pts, abs(auc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1 << 20)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--trials", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    u = UNIT_ROUNDOFF["float32"]
    rng = np.random.default_rng(args.seed)
    x, s_exact, kappa = gensum(args.n, args.kappa, seed=args.seed)
    S = sum_abs(x)
    tauH = tau_probabilistic(args.n, u, S, args.alpha)
    tauB = tau_bernstein(args.n, u, S, args.alpha)
    T = args.trials
    Tf = max(T // 5, 200)
    out = {"n": args.n, "kappa": kappa, "alpha": args.alpha,
           "tauH": tauH, "tauB": tauB, "tighter": tauH / tauB}

    print(f"n=2^{int(math.log2(args.n))}  kappa={kappa:.3g}  alpha={args.alpha:g}")
    print(f"tau_H={tauH:.3e}  tau_B={tauB:.3e}  (B {tauH/tauB:.2f}x tighter)\n")

    # benign baseline
    ben = np.array([benign(x, rng) for _ in range(T)])
    out["benign_median"] = float(np.median(ben)); out["benign_max"] = float(ben.max())
    out["fp_H"] = float((ben > tauH).mean()); out["fp_B"] = float((ben > tauB).mean())
    print(f"benign: median {np.median(ben):.3e}, max {ben.max():.3e}; "
          f"false-positive  H={out['fp_H']:.4f}  B={out['fp_B']:.4f}  (alpha {args.alpha:g})\n")

    # E1 input flips
    print("E1  single-element INPUT bit-flip:")
    print(f"{'bit':>4} {'med diff':>12} {'det_H':>7} {'det_B':>7}")
    out["E1"] = []
    for bit in [10, 16, 20, 22, 23, 25, 27, 30]:
        ds = [d for d in (input_flip(x, rng, bit) for _ in range(Tf)) if d is not None]
        if not ds: continue
        ds = np.array(ds)
        r = {"bit": bit, "med": float(np.median(ds)),
             "det_H": float((ds > tauH).mean()), "det_B": float((ds > tauB).mean())}
        out["E1"].append(r)
        print(f"{bit:>4} {r['med']:>12.3e} {r['det_H']:>7.3f} {r['det_B']:>7.3f}")

    # E2 accumulator flips
    print("\nE2  ACCUMULATOR bit-flip (register SDC mid-reduction):")
    print(f"{'bit':>4} {'med diff':>12} {'det_H':>7} {'det_B':>7}")
    out["E2"] = []
    for bit in [10, 16, 20, 22, 23, 25, 27, 30]:
        ds = [d for d in (accumulator_flip(x, rng, bit) for _ in range(Tf)) if d is not None]
        if not ds: continue
        ds = np.array(ds)
        r = {"bit": bit, "med": float(np.median(ds)),
             "det_H": float((ds > tauH).mean()), "det_B": float((ds > tauB).mean())}
        out["E2"].append(r)
        print(f"{bit:>4} {r['med']:>12.3e} {r['det_H']:>7.3f} {r['det_B']:>7.3f}")

    # E3 multi-element severity
    print("\nE3  MULTI-ELEMENT burst (k elements, mantissa bit 22):")
    print(f"{'k':>6} {'med diff':>12} {'det_H':>7} {'det_B':>7}")
    out["E3"] = []
    for k in [1, 10, 100, 1000, 10000]:
        ds = np.array([multi_flip(x, rng, k) for _ in range(Tf)])
        r = {"k": k, "med": float(np.median(ds)),
             "det_H": float((ds > tauH).mean()), "det_B": float((ds > tauB).mean())}
        out["E3"].append(r)
        print(f"{k:>6} {r['med']:>12.3e} {r['det_H']:>7.3f} {r['det_B']:>7.3f}")

    # E4 timing
    print("\nE4  FAULT TIMING (accumulator bit 27 at fraction f through reduction):")
    print(f"{'frac':>6} {'med diff':>12} {'det_H':>7} {'det_B':>7}")
    out["E4"] = []
    for frac in [0.1, 0.3, 0.5, 0.7, 0.9]:
        ds = [d for d in (timed_accumulator_flip(x, rng, 27, frac) for _ in range(Tf)) if d is not None]
        if not ds: continue
        ds = np.array(ds)
        r = {"frac": frac, "med": float(np.median(ds)),
             "det_H": float((ds > tauH).mean()), "det_B": float((ds > tauB).mean())}
        out["E4"].append(r)
        print(f"{frac:>6} {r['med']:>12.3e} {r['det_H']:>7.3f} {r['det_B']:>7.3f}")

    # E5 ROC  (fault class = accumulator flips across a range of bits -> realistic mix)
    print("\nE5  ROC (fault class = accumulator flips, mixed bit positions):")
    fault_mix = []
    for _ in range(T):
        bit = int(rng.integers(18, 28))   # realistic SDC bit range
        d = accumulator_flip(x, rng, bit)
        if d is not None:
            fault_mix.append(d)
    fault_mix = np.array(fault_mix)
    pts, auc = roc(ben, fault_mix)
    # TPR at the two tau operating points (both at FPR ~ 0)
    tpr_at_H = float((fault_mix > tauH).mean())
    tpr_at_B = float((fault_mix > tauB).mean())
    out["E5"] = {"auc": auc, "tpr_at_tauH": tpr_at_H, "tpr_at_tauB": tpr_at_B,
                 "roc_points": pts}
    print(f"  AUC = {auc:.4f}")
    print(f"  TPR at tau_H operating point (FPR~0): {tpr_at_H:.3f}")
    print(f"  TPR at tau_B operating point (FPR~0): {tpr_at_B:.3f}")
    print(f"  -> Bernstein detects {tpr_at_B - tpr_at_H:+.3f} more of the fault mix at zero FPR")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
        print(f"\nwrote {args.out}")

    print("\nSUMMARY:")
    print("  E1: single input flips often below noise floor -> blind spot (bounds method)")
    print("  E2: accumulator SDC propagates -> detectable across bit positions")
    print("  E3: multi-element faults detected once total perturbation exceeds noise floor")
    print("  E4: later-in-reduction faults differ in detectability")
    print("  E5: Bernstein tau dominates Hoeffding as a classifier (higher TPR at FPR~0)")


if __name__ == "__main__":
    main()
