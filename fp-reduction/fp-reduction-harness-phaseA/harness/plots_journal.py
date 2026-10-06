"""
plots_journal.py — figures for the journal version from the WP result files.
Uses the paper's notation (τ_H, τ_Fa, τ_Fm, τ_PS, τ_PSF; utilization t; margin 1/t) and NO in-image titles.

  python plots_journal.py --results results --figs figs_journal
Produces whatever inputs exist:
  fig_tail.png        tail_gpu*.json / tail_cpu*.json : empirical exceedance curve vs the bound's level
  fig_lambda_gap.png  λ_emp(α) vs λ(α/2,n) per n
  fig_rescore.png     rescore_*.json : utilization per tolerance across cells
  fig_gpu_variance.png gpu_variance*.json : binned σ²/u² on real atomic partial sums
  fig_gemm.png        gemm_*.json : spread/τ and bit-flip detection
  fig_llm.png         llm_*.json : margin/τ_pair distribution + near-tie fault detection
  fig_wp7.png         wp7_exponents.json : exponent per generator + mechanism check
"""
import argparse, glob, json, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = {"H": "#b8860b", "Fa": "#1f77b4", "Fm": "#7b68ee", "PS": "#2ca02c", "PSFa": "#17becf", "PSF": "#d62728"}
LBL = {"H": "τ_H", "Fa": "τ_Fa", "Fm": "τ_Fm", "PS": "τ_PS", "PSFa": "τ_PSFa", "PSF": "τ_PSF"}


def save(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig); print("wrote", path)


def fig_tail(files, out):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for f in files:
        d = json.load(open(f)); dev = d["provenance"].get("gpu", d["provenance"]["device"])
        for c in d["cells"]:
            k = c["log2n"]; lv = [r for r in c["levels"] if r["resolvable"]]
            a = [r["alpha"] for r in lv]
            ax[0].plot(a, [max(r["rate_H"], 1e-9) for r in lv], "o-", label=f"{dev[:12]} n=2^{k} τ_H")
            ax[1].plot(a, [r["lambda_emp"] / r["lambda_bound"] for r in lv], "s-", label=f"{dev[:12]} n=2^{k}")
    x = np.logspace(-5, 0, 50); ax[0].plot(x, x, "k--", label="bound: rate = α")
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("α"); ax[0].set_ylabel("observed exceedance rate of τ_H(α)")
    ax[0].set_ylim(1e-6, 1); ax[0].legend(fontsize=7)
    ax[1].set_xscale("log"); ax[1].set_xlabel("α"); ax[1].set_ylabel("λ_emp(α) / λ(α/2, n)"); ax[1].axhline(1, color="k", ls="--"); ax[1].legend(fontsize=7)
    save(fig, out)


def fig_rescore(files, out):
    fig, ax = plt.subplots(figsize=(7, 4))
    for f in files:
        d = json.load(open(f)); cells = d["cells"]
        keys = ["H", "Fm", "PS", "PSFa", "PSF"]
        for i, k in enumerate(keys):
            u = [c["spread"] / c[f"tau_{k}"] for c in cells]
            ax.scatter(np.full(len(u), i) + (0.15 if "gpu" in f else -0.15) + np.random.uniform(-0.05, 0.05, len(u)), u,
                       s=10, color=C[k], alpha=0.6, label=(os.path.basename(f) if i == 0 else None))
    ax.axhline(1, color="k", ls="--", lw=1); ax.set_yscale("log"); ax.set_xticks(range(5)); ax.set_xticklabels([LBL[k] for k in ["H", "Fm", "PS", "PSFa", "PSF"]])
    ax.set_ylabel("utilization t = spread / τ   (t = 1: exceedance)"); ax.legend(fontsize=7)
    save(fig, out)


def fig_gpu_variance(files, out):
    fig, ax = plt.subplots(figsize=(6, 4))
    for f in files:
        d = json.load(open(f)); b = d["binned"]
        ax.plot([(r["old_lo"] + r["old_hi"]) / 2 for r in b], [r["sigma2_over_u2"] for r in b], "o-", label=f"{d['gpu'][:14]} κ={d['kappa']:.0e} (pooled {d['sigma2_over_u2']:.3f})")
    ax.axhline(1 / 3, color="k", ls="--", label="uniform model u²/3"); ax.axhline(0.18, color="gray", ls=":", label="CPU calibration 0.18u²")
    ax.set_xscale("log"); ax.set_xlabel("|partial sum| at the atomic add"); ax.set_ylabel("σ² / u² (binned)"); ax.legend(fontsize=7)
    save(fig, out)


def fig_gemm(files, out):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for f in files:
        d = json.load(open(f)); tag = f"{d['M']}x{d['K']}x{d['N']} {d['data']}"
        src = d["sources"]
        names = list(src.keys()); vals = [src[s].get("max_spread_over_tau", 0) for s in names]
        ax[0].bar([f"{n}\n{tag}" for n in names], [max(v, 1e-6) for v in vals], alpha=0.7)
        if d.get("bitflip_A"):
            ax[1].plot([r["bit"] for r in d["bitflip_A"]], [r["detected_row_rate"] for r in d["bitflip_A"]], "o-", label=tag)
    ax[0].set_yscale("log"); ax[0].axhline(1, color="k", ls="--"); ax[0].set_ylabel("max spread / τ_ij"); ax[0].tick_params(axis="x", labelsize=6)
    ax[1].set_xlabel("flipped bit of one entry of A"); ax[1].set_ylabel("detected (row-wise, any entry > τ_ij)"); ax[1].legend(fontsize=7)
    save(fig, out)


def fig_llm(files, out):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for f in files:
        d = json.load(open(f)); pc = d["certified_argmax"]["margin_over_taupair_percentiles"]
        qs = sorted(pc, key=float); ax[0].plot([float(q) for q in qs], [pc[q] for q in qs], "o-", label=os.path.basename(f))
        for key, xl in (("near_tie_bitflips", "bit"), ("near_tie_additive_margin_units", "fault_over_margin")):
            rows = d.get(key, [])
            if rows and key == "near_tie_additive_margin_units":
                ax[1].plot([r[xl] for r in rows], [r["detected"] for r in rows], "o-", label="detected")
                ax[1].plot([r[xl] for r in rows], [r["top1_changed"] for r in rows], "s--", label="top-1 changed")
    ax[0].set_yscale("log"); ax[0].axhline(1, color="k", ls="--", label="certification threshold"); ax[0].set_xlabel("percentile of prompts"); ax[0].set_ylabel("margin / (τ_top1 + τ_top2)"); ax[0].legend(fontsize=7)
    ax[1].set_xlabel("injected fault / margin (near-tie prompts)"); ax[1].set_ylabel("rate"); ax[1].legend(fontsize=7)
    save(fig, out)


def fig_wp7(f, out):
    d = json.load(open(f)); fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for g in d:
        if g == "mechanism_check": continue
        ks = [float(k) for k in d[g]]; ax[0].plot(ks, [d[g][k]["beta"] for k in d[g]], "o-", label=g)
    ax[0].set_xscale("log"); ax[0].axhline(0.5, color="k", ls=":"); ax[0].axhline(0, color="k", ls=":"); ax[0].set_xlabel("κ"); ax[0].set_ylabel("fitted exponent β"); ax[0].legend()
    m = d.get("mechanism_check", {})
    if m:
        xs = [v["predicted"] for v in m.values()]; ys = [v["measured"] for v in m.values()]
        ax[1].loglog(xs, ys, "o"); lim = [min(xs) / 2, max(xs) * 2]; ax[1].plot(lim, lim, "k--")
        ax[1].set_xlabel("predicted  u·√(0.186 Σ s_k²) / (u S)"); ax[1].set_ylabel("measured RMS error / (u S)")
    save(fig, out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results", default="results"); ap.add_argument("--figs", default="figs_journal")
    a = ap.parse_args(); os.makedirs(a.figs, exist_ok=True); R = a.results
    g = lambda p: sorted(glob.glob(os.path.join(R, p)))
    if g("tail_*.json"): fig_tail(g("tail_*.json"), f"{a.figs}/fig_tail.png")
    if g("rescore_*.json"): fig_rescore(g("rescore_*.json"), f"{a.figs}/fig_rescore.png")
    if g("gpu_variance*.json"): fig_gpu_variance(g("gpu_variance*.json"), f"{a.figs}/fig_gpu_variance.png")
    if g("gemm_*.json"): fig_gemm(g("gemm_*.json"), f"{a.figs}/fig_gemm.png")
    if g("llm_*.json"): fig_llm(g("llm_*.json"), f"{a.figs}/fig_llm.png")
    if g("wp7_exponents.json"): fig_wp7(g("wp7_exponents.json")[0], f"{a.figs}/fig_wp7.png")


if __name__ == "__main__":
    main()
