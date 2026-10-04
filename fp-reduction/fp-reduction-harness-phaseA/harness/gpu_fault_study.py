"""
gpu_fault_study.py — fault detection against REAL GPU nondeterminism.

Unlike fault_study.py (CPU, benign class = input permutation), here BOTH classes
live on the GPU:
  - benign class : genuine run-to-run spread of the naive atomic reduction
                   (no fault; the hardware scheduler reorders the sum)
  - fault class  : the same reduction with an injected fault, compared to a clean run

So the detector is tested against the actual nondeterminism the paper is about,
on real hardware, with no permutation model and no cross-hardware hand-wave.

Fault models:
  - input bit-flip      : flip one bit of one input element (often sub-noise-floor)
  - input additive f    : add a controlled magnitude f to one element (sweeps the
                          tau_B..tau_H band to show the Bernstein advantage)
  - multi-element burst : corrupt k elements

Both tolerances (Hoeffding, Bernstein) are evaluated, plus an ROC over the
controlled-magnitude fault class where the Bernstein advantage is visible.

Run (on the 4090 pod):
  python gpu_fault_study.py --n 1048576 --trials 500 --out results/gpu_fault.json
"""

import argparse, struct, math, json, os
import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None

from gensum import gensum
from reference import sum_abs
from bounds import tau_probabilistic, tau_bernstein, tau_freedman, SIGMA2_OVER_U2_UNIFORM, UNIT_ROUNDOFF
import gpu_strategies as gs


def flip_bit_f32(v, bit):
    i = struct.unpack('>I', struct.pack('>f', np.float32(v)))[0] ^ (1 << bit)
    return struct.unpack('>f', struct.pack('>I', i))[0]


def roc(ben, flt, n=200):
    allv = np.concatenate([ben, flt])
    ths = np.unique(np.quantile(allv, np.linspace(0, 1, n)))
    pts = sorted((float((ben > t).mean()), float((flt > t).mean())) for t in ths)
    auc = sum((pts[i][0]-pts[i-1][0])*(pts[i][1]+pts[i-1][1])/2 for i in range(1, len(pts)))
    return pts, abs(auc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1 << 20)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--trials", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if cp is None:
        print("CuPy not installed."); return

    u = UNIT_ROUNDOFF["float32"]
    rng = np.random.default_rng(args.seed)
    x, s_exact, kappa = gensum(args.n, args.kappa, seed=args.seed)
    S = sum_abs(x)
    tauH = tau_probabilistic(args.n, u, S, args.alpha)
    tauB = tau_bernstein(args.n, u, S, args.alpha)
    tauFa = tau_freedman(args.n, u, S, args.alpha, sigma2_over_u2=SIGMA2_OVER_U2_UNIFORM)  # a priori, no measurement
    kern = gs._load()
    x_gpu = cp.asarray(x)
    T = args.trials

    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    print(f"GPU: {props['name'].decode()}   n=2^{int(math.log2(args.n))} kappa={kappa:.3g} alpha={args.alpha:g}")
    print(f"tau_H={tauH:.3e}  tau_B={tauB:.3e}  tau_Fa={tauFa:.3e}  (B {tauH/tauB:.2f}x, Fa {tauH/tauFa:.2f}x tighter than H)\n")

    def atomic(xg):
        return gs.naive_atomic(xg, args.n, kern)

    out = {"gpu": props['name'].decode(), "n": args.n, "kappa": kappa,
           "alpha": args.alpha, "tauH": tauH, "tauB": tauB, "tauFa": tauFa}

    # --- benign class: real atomic run-to-run spread (two clean launches) ---
    ben = np.array([abs(atomic(x_gpu) - atomic(x_gpu)) for _ in range(T)])
    out["fp_H"] = float((ben > tauH).mean()); out["fp_B"] = float((ben > tauB).mean()); out["fp_Fa"] = float((ben > tauFa).mean())
    out["benign_median"] = float(np.median(ben)); out["benign_max"] = float(ben.max())
    print(f"benign (real atomic spread): median {np.median(ben):.3e}, max {ben.max():.3e}")
    print(f"false-positive: H={out['fp_H']:.4f}  B={out['fp_B']:.4f}  Fa={out['fp_Fa']:.4f}  (alpha {args.alpha:g})\n")

    # --- fault class 1: input bit-flip, vs a clean atomic run ---
    print("Input bit-flip faults (vs clean atomic run):")
    print(f"{'bit':>4} {'med diff':>12} {'det_H':>7} {'det_B':>7} {'det_Fa':>7}")
    out["bitflip"] = []
    Tf = max(T//2, 150)
    for bit in [10, 16, 20, 22, 23, 25, 27, 30]:
        ds = []
        for _ in range(Tf):
            xf = x.copy(); idx = rng.integers(args.n)
            fv = flip_bit_f32(float(xf[idx]), bit)
            if not math.isfinite(fv): continue
            xf[idx] = np.float32(fv)
            d = abs(atomic(x_gpu) - atomic(cp.asarray(xf)))
            ds.append(d)
        if not ds: continue
        ds = np.array(ds)
        r = {"bit": bit, "med": float(np.median(ds)),
             "det_H": float((ds > tauH).mean()), "det_B": float((ds > tauB).mean()),
             "det_Fa": float((ds > tauFa).mean())}
        out["bitflip"].append(r)
        print(f"{bit:>4} {r['med']:>12.3e} {r['det_H']:>7.3f} {r['det_B']:>7.3f} {r['det_Fa']:>7.3f}")

    # --- fault class 2: controlled additive magnitude, sweeping tau_B..tau_H ---
    print("\nControlled additive fault (sweeps the tau_B..tau_H band):")
    print(f"{'f/tauB':>7} {'f/tauH':>7} {'det_H':>7} {'det_B':>7} {'det_Fa':>7}")
    out["additive"] = []
    for mult in [0.3, 0.6, 1.0, 1.5, 2.0, tauH/tauB, 1.2*tauH/tauB, 3.0]:
        f = mult * tauB
        ds = []
        for _ in range(Tf):
            xf = x.copy(); idx = rng.integers(args.n)
            xf[idx] = np.float32(float(xf[idx]) + f)
            ds.append(abs(atomic(x_gpu) - atomic(cp.asarray(xf))))
        ds = np.array(ds)
        r = {"f": float(f), "f_over_tauB": float(mult), "f_over_tauH": float(f/tauH),
             "det_H": float((ds > tauH).mean()), "det_B": float((ds > tauB).mean()),
             "det_Fa": float((ds > tauFa).mean())}
        out["additive"].append(r)
        print(f"{mult:>7.2f} {f/tauH:>7.2f} {r['det_H']:>7.3f} {r['det_B']:>7.3f} {r['det_Fa']:>7.3f}")

    # --- ROC on the controlled-additive fault class (where Bernstein's edge shows) ---
    flt = []
    for _ in range(T):
        f = rng.uniform(0.5, 2.5) * tauB   # faults spanning the discriminating band
        xf = x.copy(); idx = rng.integers(args.n)
        xf[idx] = np.float32(float(xf[idx]) + f)
        flt.append(abs(atomic(x_gpu) - atomic(cp.asarray(xf))))
    flt = np.array(flt)
    pts, auc = roc(ben, flt)
    out["roc"] = {"auc": auc, "tpr_at_tauH": float((flt > tauH).mean()),
                  "tpr_at_tauB": float((flt > tauB).mean()),
                  "tpr_at_tauFa": float((flt > tauFa).mean()), "points": pts}
    print(f"\nROC (fault class spans 0.5..2.5 x tau_B):")
    print(f"  AUC = {auc:.4f}")
    print(f"  TPR at tau_H (FPR~{out['fp_H']:.3f}): {out['roc']['tpr_at_tauH']:.3f}")
    print(f"  TPR at tau_B (FPR~{out['fp_B']:.3f}): {out['roc']['tpr_at_tauB']:.3f}")
    print(f"  TPR at tau_Fa a-priori (FPR~{out['fp_Fa']:.3f}): {out['roc']['tpr_at_tauFa']:.3f}")
    print(f"  -> Bernstein detects {out['roc']['tpr_at_tauB']-out['roc']['tpr_at_tauH']:+.3f} "
          f"more at ~zero FPR, on REAL GPU nondeterminism")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()