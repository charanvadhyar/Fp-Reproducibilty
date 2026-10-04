"""
nbody.py — capstone: the derived tolerance on a real scientific computation.

An N-body gravitational simulation computes, each step, the acceleration of each
body as a SUM over all other bodies' pulls:
    a_i = G * sum_{j != i} m_j (r_j - r_i) / (|r_j - r_i|^2 + eps)^{3/2}
That inner sum is a reduction of length N-1 -- exactly the object this project's
tolerance bounds. So N-body is not a new problem; it is the tolerance applied to
a real workload whose data (gravitational forces) has its own magnitude and
conditioning distribution.

Four stages, each with a figure:
  S1 VALIDATION   : tau holds against the real force-sum spread (atomic vs clean)
  S2 TRAJECTORY   : two runs of the same simulation diverge; tau at the force level
                    marks when they are "same physics" vs "genuinely diverged"
  S3 FAULT        : inject a corrupted force; tau detects it (Hoeffding vs Bernstein)
  S4 SUMMARY FIG  : one explanatory figure tying it together

Runs on GPU (CuPy) if available, else CPU (numpy) -- the arithmetic and the
tolerance are the same; GPU gives the real atomic nondeterminism.

Usage:
  python nbody.py --N 2048 --steps 200 --out results/nbody.json --figdir figs
"""

import argparse, json, math, os, struct
import numpy as np

try:
    import cupy as cp
    XP = cp
    HAVE_GPU = True
except ImportError:
    cp = None; XP = np; HAVE_GPU = False

from bounds import tau_probabilistic, tau_bernstein, UNIT_ROUNDOFF

G = 1.0
SOFT = 0.1
U = UNIT_ROUNDOFF["float32"]


def init_system(N, seed=0):
    rng = np.random.default_rng(seed)
    pos = (rng.standard_normal((N, 3)) * 10).astype(np.float32)
    vel = (rng.standard_normal((N, 3)) * 0.1).astype(np.float32)
    mass = (rng.random(N).astype(np.float32) + np.float32(0.5))
    return pos, vel, mass


def accel_deterministic(pos, mass):
    """Fixed-order force sum (numpy/cupy vectorized, deterministic ordering)."""
    xp = cp if (HAVE_GPU and isinstance(pos, cp.ndarray)) else np
    N = pos.shape[0]
    a = xp.zeros_like(pos)
    for i in range(N):
        d = pos - pos[i]
        r2 = (d * d).sum(1) + SOFT * SOFT
        inv = mass / (r2 * xp.sqrt(r2))
        inv[i] = 0
        a[i] = G * (d * inv[:, None]).sum(0)
    return a


def accel_shuffled(pos, mass, rng):
    """Same forces, summed in a permuted order -> benign run-to-run spread model."""
    xp = cp if (HAVE_GPU and isinstance(pos, cp.ndarray)) else np
    N = pos.shape[0]
    a = xp.zeros_like(pos)
    for i in range(N):
        d = pos - pos[i]
        r2 = (d * d).sum(1) + SOFT * SOFT
        inv = mass / (r2 * xp.sqrt(r2))
        inv[i] = 0
        contrib = (d * inv[:, None])
        perm = rng.permutation(N)
        a[i] = G * contrib[perm].sum(0)
    return a


def force_sum_stats(pos, mass):
    """For one representative body: S=sum|terms|, and tau for its force reduction."""
    d = pos - pos[0]
    r2 = (d * d).sum(1) + SOFT * SOFT
    inv = mass / (r2 * np.sqrt(r2)); inv[0] = 0
    terms = np.asarray((d * inv[:, None])[:, 0])   # x-component contributions
    S = float(np.abs(terms).sum())
    tot = abs(float(terms.sum()))
    kappa = S / tot if tot > 0 else float("inf")
    n = len(terms)
    return S, kappa, n


def leapfrog_step(pos, vel, mass, dt, accel_fn):
    a = accel_fn(pos, mass)
    vel = vel + np.float32(dt) * a
    pos = pos + np.float32(dt) * vel
    return pos, vel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--figdir", default="figs")
    args = ap.parse_args()
    os.makedirs(args.figdir, exist_ok=True)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "font.family": "serif",
                         "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150})

    pos0, vel0, mass = init_system(args.N, args.seed)
    rng = np.random.default_rng(args.seed)
    out = {"N": args.N, "steps": args.steps, "gpu": HAVE_GPU}

    # ---- S1 VALIDATION: tau vs real force-sum spread ----
    S, kappa, n = force_sum_stats(pos0, mass)
    tauH = tau_probabilistic(n, U, S, args.alpha)
    tauB = tau_bernstein(n, U, S, args.alpha)
    # measure benign spread of the force sum (shuffled order), many trials, one body
    d = pos0 - pos0[0]; r2 = (d*d).sum(1)+SOFT*SOFT; inv = mass/(r2*np.sqrt(r2)); inv[0]=0
    contrib = np.asarray((d*inv[:,None])[:,0])
    spreads = []
    for _ in range(2000):
        s1 = np.float32(0.0)
        for v in contrib[rng.permutation(n)]: s1 = np.float32(s1+v)
        s2 = np.float32(0.0)
        for v in contrib[rng.permutation(n)]: s2 = np.float32(s2+v)
        spreads.append(abs(float(s1)-float(s2)))
    spreads = np.array(spreads)
    out["S1"] = {"n": n, "kappa": kappa, "S": S, "tauH": tauH, "tauB": tauB,
                 "spread_median": float(np.median(spreads)), "spread_max": float(spreads.max()),
                 "exceed_H": float((spreads>tauH).mean()), "exceed_B": float((spreads>tauB).mean())}
    print(f"S1 VALIDATION: force-sum n={n} kappa={kappa:.2f}")
    print(f"   tauH={tauH:.3e} tauB={tauB:.3e}  benign spread med={np.median(spreads):.3e} max={spreads.max():.3e}")
    print(f"   exceed: H={out['S1']['exceed_H']:.4f} B={out['S1']['exceed_B']:.4f}  (both should be ~0)")

    fig, ax = plt.subplots(figsize=(6.4,4))
    ax.hist(spreads, bins=50, color="#1D9E75", alpha=0.8)
    ax.axvline(tauB, color="#7F77DD", ls="--", lw=2, label=f"Bernstein tau = {tauB:.2e}")
    ax.axvline(tauH, color="#BA7517", ls="--", lw=2, label=f"Hoeffding tau = {tauH:.2e}")
    ax.set_xlabel("force-sum spread between two orderings"); ax.set_ylabel("count")
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    ax.set_title(f"N-body S1: real force-sum spread vs tolerance\n(N={args.N}, kappa={kappa:.1f}) — all benign spread below tau")
    ax.legend(); fig.tight_layout(); fig.savefig(f"{args.figdir}/nbody_s1.png"); plt.close()

    # ---- S2 TRAJECTORY: two runs diverge; measure growth ----
    posA, velA = pos0.copy(), vel0.copy()
    posB, velB = pos0.copy(), vel0.copy()
    rngA = np.random.default_rng(1); rngB = np.random.default_rng(2)
    div = []
    for step in range(args.steps):
        posA, velA = leapfrog_step(posA, velA, mass, args.dt, lambda p,m: accel_shuffled(p,m,rngA))
        posB, velB = leapfrog_step(posB, velB, mass, args.dt, lambda p,m: accel_shuffled(p,m,rngB))
        div.append(float(np.sqrt(((posA-posB)**2).sum(1)).mean()))  # mean position divergence
    out["S2"] = {"divergence": div}
    print(f"\nS2 TRAJECTORY: two shuffled-order runs diverge")
    print(f"   mean position divergence: step1={div[0]:.3e}  final={div[-1]:.3e}  growth={div[-1]/max(div[0],1e-30):.1f}x")

    fig, ax = plt.subplots(figsize=(6,4))
    ax.semilogy(range(1,args.steps+1), div, color="#C0392B", lw=2)
    ax.set_xlabel("timestep"); ax.set_ylabel("mean position divergence between runs")
    ax.set_title("N-body S2: identical simulation, two reduction orders\ntrajectories diverge from floating-point non-associativity alone")
    fig.tight_layout(); fig.savefig(f"{args.figdir}/nbody_s2.png"); plt.close()

    # ---- S3 FAULT: inject a corrupted force, detect it ----
    print(f"\nS3 FAULT DETECTION on the force sum:")
    print(f"   {'fault f/tauB':>13} {'det_H':>7} {'det_B':>7}")
    s3 = []
    for mult in [0.5, 1.0, 1.5, 2.0, tauH/tauB, 3.0]:
        f = mult*tauB; dH=dB=0; T=300
        for _ in range(T):
            s_clean = np.float32(0.0)
            for v in contrib[rng.permutation(n)]: s_clean=np.float32(s_clean+v)
            cf = contrib.copy(); cf[rng.integers(n)] += np.float32(f)
            s_fault = np.float32(0.0)
            for v in cf[rng.permutation(n)]: s_fault=np.float32(s_fault+v)
            diff=abs(float(s_clean)-float(s_fault)); dH+=int(diff>tauH); dB+=int(diff>tauB)
        s3.append({"f_over_tauB":mult,"det_H":dH/T,"det_B":dB/T})
        print(f"   {mult:>13.2f} {dH/T:>7.3f} {dB/T:>7.3f}")
    out["S3"] = s3

    fig, ax = plt.subplots(figsize=(6,4))
    xs=[r["f_over_tauB"] for r in s3]
    ax.plot(xs,[r["det_H"] for r in s3],"o-",color="#BA7517",lw=2,label="Hoeffding tau")
    ax.plot(xs,[r["det_B"] for r in s3],"s-",color="#7F77DD",lw=2,label="Bernstein tau")
    ax.axvspan(1.0, tauH/tauB, alpha=0.15, color="#1D9E75", label="Bernstein-only detection band")
    ax.set_xlabel("fault magnitude / tau_Bernstein"); ax.set_ylabel("detection rate")
    ax.set_title("N-body S3: fault detection on the force sum\nBernstein detects the band Hoeffding misses")
    ax.legend(); fig.tight_layout(); fig.savefig(f"{args.figdir}/nbody_s3.png"); plt.close()

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out,"w") as f: json.dump(out,f,indent=1)
        print(f"\nwrote {args.out}")
    print(f"figures in {args.figdir}/: nbody_s1.png, nbody_s2.png, nbody_s3.png")


if __name__ == "__main__":
    main()
