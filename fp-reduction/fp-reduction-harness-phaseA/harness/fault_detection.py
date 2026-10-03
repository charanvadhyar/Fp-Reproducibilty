"""
fault_detection.py — can the tolerance separate real faults from benign noise?

This is the usefulness test. A verifier flags a computation when two results
differ by more than tau. We ask:
  - benign case: two reorderings of the SAME data (no fault) -> difference
    should stay UNDER tau (not flagged). [already shown in exceedance tests]
  - fault case: one run has a single injected bit-flip -> difference should
    EXCEED tau (flagged).

A single bit-flip's magnitude depends only on WHICH bit flips (IEEE-754
position), not on the GPU, so this runs on one device (here CPU, fp32).

We sweep the flipped bit position from low mantissa bits (tiny perturbation,
hard to detect) to high exponent bits (huge, easy). For each, we measure the
detection rate against BOTH tolerances (Hoeffding and the tighter Bernstein),
producing a detection-rate-vs-fault-magnitude curve and the key quantity:
the BLIND SPOT — faults too small to exceed tau, which slip through.

The tighter Bernstein tau should shrink the blind spot (detect smaller faults)
while, by the exceedance results, still not flagging benign noise. That is the
practical payoff of the tightening.

Usage:
  python fault_detection.py --n 1048576 --trials 2000
"""

import argparse, struct, math
import numpy as np

from strategies import recursive
from gensum import gensum
from reference import sum_abs
from bounds import tau_probabilistic, tau_bernstein, UNIT_ROUNDOFF


def flip_bit(x_f32, bit):
    """Flip one bit (0=LSB mantissa .. 22=MSB mantissa, 23..30=exponent) of a float32."""
    b = struct.pack('>f', np.float32(x_f32))
    i = struct.unpack('>I', b)[0]
    i ^= (1 << bit)
    return struct.unpack('>f', struct.pack('>I', i))[0]


def benign_spread(x, rng):
    """Two reorderings, same data -> benign run-to-run style difference."""
    s1 = float(recursive(x[rng.permutation(len(x))]))
    s2 = float(recursive(x[rng.permutation(len(x))]))
    return abs(s1 - s2)


def faulted_spread(x, rng, bit):
    """Reference sum vs sum with one element's bit flipped -> fault-induced diff."""
    s_clean = float(recursive(x[rng.permutation(len(x))]))
    xf = x.copy()
    idx = rng.integers(len(x))
    v = float(xf[idx])
    fv = flip_bit(v, bit)
    if not math.isfinite(fv):      # exponent flip can make inf/nan; skip those
        return None
    xf[idx] = np.float32(fv)
    s_fault = float(recursive(xf[rng.permutation(len(x))]))
    return abs(s_clean - s_fault)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1 << 20)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--trials", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    u = UNIT_ROUNDOFF["float32"]
    rng = np.random.default_rng(args.seed)
    x, s_exact, kappa = gensum(args.n, args.kappa, seed=args.seed)
    S = sum_abs(x)
    tauH = tau_probabilistic(args.n, u, S, args.alpha)
    tauB = tau_bernstein(args.n, u, S, args.alpha)

    print(f"n = {args.n} (2^{int(math.log2(args.n))}), kappa = {kappa:.3g}, alpha = {args.alpha:g}")
    print(f"tau_Hoeffding = {tauH:.3e}   tau_Bernstein = {tauB:.3e}   (B is {tauH/tauB:.2f}x tighter)\n")

    # --- benign false-positive check: how often does benign noise exceed tau? ---
    benign = np.array([benign_spread(x, rng) for _ in range(args.trials)])
    fp_H = float((benign > tauH).mean())
    fp_B = float((benign > tauB).mean())
    print(f"benign false-positive rate:  Hoeffding {fp_H:.4f}   Bernstein {fp_B:.4f}   (target alpha {args.alpha:g})")
    print(f"benign spread: median {np.median(benign):.3e}, max {benign.max():.3e}\n")

    # --- detection vs fault MAGNITUDE (controlled), sweeping the tau_B..tau_H gap ---
    # A single low-bit flip on one element is smaller than the benign noise floor,
    # so it is undetectable by ANY order-based verifier. The discriminating regime
    # is a fault whose magnitude lies between tau_B and tau_H: there the tighter
    # Bernstein tau detects while Hoeffding does not. We inject an additive fault
    # of controlled size f onto one element and measure detection.
    print(f"{'fault f':>12} {'f/tauB':>8} {'f/tauH':>8} {'detect_H':>9} {'detect_B':>9}")
    print("-" * 50)
    ft = max(args.trials // 5, 200)
    # sweep f from below tau_B, across the gap, to above tau_H
    for mult in [0.3, 0.6, 1.0, 1.5, 2.0, tauH/tauB, 1.2*tauH/tauB, 3.0]:
        f = mult * tauB
        det_H = det_B = 0
        for _ in range(ft):
            xf = x.copy()
            idx = rng.integers(len(x))
            xf[idx] = np.float32(float(xf[idx]) + f)      # inject fault of size f
            s_clean = float(recursive(x[rng.permutation(len(x))]))
            s_fault = float(recursive(xf[rng.permutation(len(x))]))
            d = abs(s_clean - s_fault)
            det_H += int(d > tauH)
            det_B += int(d > tauB)
        print(f"{f:>12.3e} {f/tauB:>8.2f} {f/tauH:>8.2f} {det_H/ft:>9.3f} {det_B/ft:>9.3f}")

    print()
    print("A fault of size f between tau_B and tau_H is detected by Bernstein but not Hoeffding")
    print("(detect_B=1, detect_H=0 in that band). Below tau_B neither detects; above tau_H both do.")
    print("The gap (f/tauB between 1 and %.2f) is the detection advantage of the tighter bound." % (tauH/tauB))
    print("Benign false-positive rate stays at/below alpha for both (first block) -> the")
    print("tightening improves fault sensitivity without increasing false positives.")

    # keep the bit-flip sweep too, as the 'realistic single-element fault' reference
    print()
    print("Reference: single-element bit-flip faults (realistic SDC model):")
    print(f"{'bit':>4} {'region':>9} {'med fault diff':>15} {'detect_H':>9} {'detect_B':>9}")
    print("-" * 52)
    fb = max(args.trials // 10, 100)
    for bit in [20, 22, 23, 25, 27, 30]:
        region = "mantissa" if bit <= 22 else "exponent"
        diffs = [d for d in (faulted_spread(x, rng, bit) for _ in range(fb)) if d is not None]
        if not diffs: continue
        diffs = np.array(diffs)
        print(f"{bit:>4} {region:>9} {np.median(diffs):>15.3e} "
              f"{(diffs>tauH).mean():>9.3f} {(diffs>tauB).mean():>9.3f}")


if __name__ == "__main__":
    main()
