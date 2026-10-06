"""
ml_capstone.py — the derived tolerance on real neural-network reductions.

The ML counterpart of nbody.py. Three stages, three tolerance tiers
(tau_H Hoeffding, tau_Fa Freedman a priori, tau_Fm Freedman calibrated).

SCOPE (state this in the paper): the bounds are for SCALAR summation. We test
reductions that are scalar sums inside a network. Tensor-core matrix multiply
(block-FMA accumulation) is OUT of scope; TF32 is disabled so matmuls run in
plain fp32, and any matmul effect we observe (batch-size dependence) is
reported as context, never as something tau bounds.

Model: a bag-of-tokens classifier on a synthetic task.
    emb[V,d] --gather--> segment-SUM pooling --> /L --> Linear(d,H) --> LayerNorm(H)
             --> ReLU --> Linear(H,K) --> logits
Segment-sum pooling is implemented two ways:
    'scatter' : pooled.index_add_(0, seg_id, gathered)   -> CUDA atomics, NONDETERMINISTIC
    'reshape' : gathered.view(B,L,d).sum(1)              -> fixed order, deterministic (control)
index_add_ is the scatter-reduce pattern that Shanmugavelu et al. (2024) identify
as a dominant source of nondeterminism in deep-learning pipelines; it is the same
operation as embedding-bag pooling and GNN message aggregation.

M1  REDUCTION-LEVEL VALIDATION on real network data
    (a) scatter pooling     : real trained-embedding contributions, real PyTorch atomics
    (b) vocab softmax denom : sum of exp(logits) over a 32k vocabulary (tied readout
                              over the trained embedding table), via our atomic kernel
    (c) LayerNorm mean      : sum over H real hidden activations, via our atomic kernel
    (d) gradient accumulation: per-example contributions to one weight-gradient
                              entry summed over a batch, via our atomic kernel
    For each: measured kappa, tau (3 tiers), spread over R launches, exceedances.
M2  PROPAGATION (observed, NOT bounded)
    Train twice from identical init with nondeterministic pooling; track parameter
    and logit divergence per step. Deterministic control: two runs with 'reshape'
    pooling + deterministic algorithms -> should be bitwise identical.
    Batch-invariance probe: same inputs at different batch sizes (matmul effect).
M3  FAULT DETECTION at the reduction level (verification use case)
    benign: two nondeterministic inference runs. faults: (i) controlled additive
    fault of size m*tau_Fm into one contribution, (ii) bit-flip SDC in one
    trained embedding weight. Detection per tier; plus whether each fault changes
    the model's top-1 prediction (are the faults we miss the ones that matter?).

Run (on the 4090):
    python ml_capstone.py --out results/ml_capstone.json --figdir figs_ml
    python ml_capstone.py --quick --out results/ml_quick.json --figdir figs_ml   # smoke test
"""

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # must precede CUDA init

import argparse, json, math, struct, time
import numpy as np
import torch

from bounds import (tau_probabilistic, tau_freedman, UNIT_ROUNDOFF,
                    SIGMA2_OVER_U2, SIGMA2_OVER_U2_UNIFORM)

try:
    import cupy as cp
    import gpu_strategies as gs
    HAVE_CUPY = True
except Exception:
    cp = None; gs = None; HAVE_CUPY = False

U = UNIT_ROUNDOFF["float32"]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# plain fp32 everywhere: no TF32 tensor-core shortcuts in matmul/conv
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


# ------------------------------------------------------------------ tolerances (vectorized)
def _lam(alpha, n):
    return np.sqrt(2.0 * np.log(2.0 * n / alpha)) / (1.0 - U)

def tauH_vec(n, S, alpha):
    """Hoeffding two-run tolerance; matches bounds.tau_probabilistic."""
    n = np.asarray(n, float); S = np.asarray(S, float)
    lam = _lam(alpha / 2.0, n)
    return 2.0 * np.expm1((lam * np.sqrt(n) * U + n * U * U) / (1.0 - U)) * S

def tauF_vec(n, S, alpha, s2):
    """Freedman two-run tolerance; matches bounds.tau_freedman."""
    n = np.asarray(n, float); S = np.asarray(S, float)
    L = np.log(2.0 * n / (alpha / 2.0)); v = s2 * U * U
    b = -(2.0 * U * L / 3.0); c = -(2.0 * n * v * L)
    t = (-b + np.sqrt(b * b - 4.0 * c)) / 2.0
    return 2.0 * t * S

def tiers(n, S, alpha):
    return (tauH_vec(n, S, alpha),
            tauF_vec(n, S, alpha, SIGMA2_OVER_U2_UNIFORM),   # Fa: a priori
            tauF_vec(n, S, alpha, SIGMA2_OVER_U2))           # Fm: calibrated

def _check_against_bounds(alpha):
    n, S = 2**20, 1.0
    a = (float(tauH_vec(n, S, alpha)), tau_probabilistic(n, U, S, alpha))
    b = (float(tauF_vec(n, S, alpha, SIGMA2_OVER_U2_UNIFORM)),
         tau_freedman(n, U, S, alpha, sigma2_over_u2=SIGMA2_OVER_U2_UNIFORM))
    c = (float(tauF_vec(n, S, alpha, SIGMA2_OVER_U2)), tau_freedman(n, U, S, alpha))
    for x, y in (a, b, c):
        assert abs(x - y) <= 1e-12 * max(1.0, abs(y)), (x, y)


# ------------------------------------------------------------------ synthetic task + model
def make_task(V, K, seed):
    g = torch.Generator().manual_seed(seed)
    # each class prefers its own random subset of the vocabulary
    pref = torch.rand(K, V, generator=g) ** 8
    pref = pref / pref.sum(1, keepdim=True)
    return pref

def sample_batch(pref, B, L, gen):
    K, V = pref.shape
    y = torch.randint(0, K, (B,), generator=gen)
    tok = torch.multinomial(pref[y], L, replacement=True, generator=gen)   # (B, L)
    return tok, y

class BagNet(torch.nn.Module):
    def __init__(self, V, d, H, K):
        super().__init__()
        self.emb = torch.nn.Embedding(V, d)
        self.fc1 = torch.nn.Linear(d, H)
        self.ln = torch.nn.LayerNorm(H)
        self.fc2 = torch.nn.Linear(H, K)

    def pool_sum(self, tok, mode, gathered=None):
        B, L = tok.shape
        if gathered is None:
            gathered = self.emb(tok.reshape(-1))                 # (B*L, d)
        if mode == "scatter":
            seg = torch.arange(B, device=tok.device).repeat_interleave(L)
            out = torch.zeros(B, gathered.shape[1], device=tok.device, dtype=gathered.dtype)
            return out.index_add_(0, seg, gathered)              # atomic on CUDA
        return gathered.view(B, L, -1).sum(1)                     # fixed order

    def forward(self, tok, mode="scatter", return_hidden=False):
        L = tok.shape[1]
        p = self.pool_sum(tok, mode) / L
        h_pre = self.fc1(p)
        h = torch.relu(self.ln(h_pre))
        logits = self.fc2(h)
        return (logits, h_pre, h) if return_hidden else logits


def train(model, pref, steps, B, L, lr, mode, seed, eval_tok=None, track=None):
    gen = torch.Generator().manual_seed(seed)                     # identical data order
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(steps):
        tok, y = sample_batch(pref, B, L, gen)
        tok, y = tok.to(DEV), y.to(DEV)
        loss = torch.nn.functional.cross_entropy(model(tok, mode=mode), y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if track is not None:
            track(step, model)
    return model


# ------------------------------------------------------------------ hardware atomic reduction of a real vector
_KERN = None
def atomic_reduce_runs(vec_f32_np, R, rng):
    """R nondeterministic reductions of one real vector. GPU: naive atomicAdd kernel.
    CPU fallback (no CuPy): sequential sums of random permutations (labelled)."""
    n = vec_f32_np.size
    if HAVE_CUPY:
        global _KERN
        if _KERN is None:
            _KERN = gs._load()
        xg = cp.asarray(vec_f32_np.astype(np.float32))
        k = _KERN
        return np.array([gs.naive_atomic(xg, n, k) for _ in range(R)], dtype=np.float64)
    out = []
    for _ in range(R):
        s = np.float32(0.0)
        for v in vec_f32_np[rng.permutation(n)].astype(np.float32):
            s = np.float32(s + v)
        out.append(float(s))
    return np.array(out)


def reduction_record(name, vecs, R, alpha, rng):
    """vecs: list of 1-D float32 numpy arrays, each one real reduction."""
    rows = []
    for v in vecs:
        v = v.astype(np.float32); n = v.size
        S = math.fsum(np.abs(v).astype(np.float64).tolist())
        s = math.fsum(v.astype(np.float64).tolist())
        kappa = S / abs(s) if s != 0 else float("inf")
        runs = atomic_reduce_runs(v, R, rng)
        spread = float(runs.max() - runs.min())
        tH, tFa, tFm = (float(x) for x in tiers(n, S, alpha))
        rows.append(dict(n=n, S=S, kappa=kappa, spread=spread, distinct=int(len(np.unique(runs))),
                         tauH=tH, tauFa=tFa, tauFm=tFm,
                         t_H=spread / tH, exc_H=spread > tH, exc_Fa=spread > tFa, exc_Fm=spread > tFm))
    k = np.array([r["kappa"] for r in rows])
    summ = dict(name=name, count=len(rows), n=int(np.median([r["n"] for r in rows])),
                kappa_median=float(np.median(k)), kappa_max=float(np.max(k)),
                nondeterministic=int(sum(r["distinct"] > 1 for r in rows)),
                exceed_H=int(sum(r["exc_H"] for r in rows)),
                exceed_Fa=int(sum(r["exc_Fa"] for r in rows)),
                exceed_Fm=int(sum(r["exc_Fm"] for r in rows)),
                max_t_H=float(max(r["t_H"] for r in rows)),
                median_t_H=float(np.median([r["t_H"] for r in rows])))
    return summ, rows


# ------------------------------------------------------------------ bit flip
def flip_bit_f32(v, bit):
    i = struct.unpack('>I', struct.pack('>f', np.float32(v)))[0] ^ (1 << bit)
    return struct.unpack('>f', struct.pack('>I', i))[0]


# ================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--figdir", default="figs_ml")
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    Q = a.quick
    V, d, H, K = (4096, 32, 256, 10) if Q else (32768, 64, 1024, 10)
    L_train, B_train = (64, 32) if Q else (512, 64)
    steps = 40 if Q else 400
    L_m1, B_m1 = (512, 8) if Q else (4096, 64)        # long segments: reductions of n = L_m1
    R = 10 if Q else 100                               # launches per reduction
    n_vec = 4 if Q else 32                             # vectors per M1 reduction type
    B_grad = 512 if Q else 8192
    M3_pairs = 6 if Q else 50

    _check_against_bounds(a.alpha)
    os.makedirs(a.figdir, exist_ok=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    rng = np.random.default_rng(a.seed)
    gpu_name = torch.cuda.get_device_name(0) if DEV.type == "cuda" else "cpu"
    prov = dict(device=gpu_name, torch=torch.__version__, cupy=(cp.__version__ if HAVE_CUPY else None),
                tf32=False, timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                V=V, d=d, H=H, K=K, L_train=L_train, B_train=B_train, steps=steps,
                L_m1=L_m1, B_m1=B_m1, R=R, alpha=a.alpha)
    print(f"device={gpu_name}  torch={torch.__version__}  cupy={'yes' if HAVE_CUPY else 'NO (CPU permutation fallback)'}")
    if not HAVE_CUPY or DEV.type != "cuda":
        print("WARNING: not on a CUDA GPU with CuPy -- spreads are a CPU fallback, not hardware nondeterminism.")
    out = {"provenance": prov}
    pref = make_task(V, K, a.seed).to(DEV)
    pref_cpu = pref.cpu()

    # ============================================================ M2 first (it produces the trained model)
    print("\n=== M2  PROPAGATION through training (observed, not bounded) ===")
    evgen = torch.Generator().manual_seed(a.seed + 99)
    ev_tok, _ = sample_batch(pref_cpu, 256, L_train, evgen)
    ev_tok = ev_tok.to(DEV)

    def train_pair(mode, deterministic):
        """Train two models in lockstep from identical init on identical batches;
        record only divergence scalars per step (no snapshots -> bounded memory)."""
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
        torch.manual_seed(a.seed); mA_ = BagNet(V, d, H, K).to(DEV)
        torch.manual_seed(a.seed); mB_ = BagNet(V, d, H, K).to(DEV)
        oA = torch.optim.Adam(mA_.parameters(), lr=1e-3); oB = torch.optim.Adam(mB_.parameters(), lr=1e-3)
        gen = torch.Generator().manual_seed(a.seed)
        pdiv, ldiv = [], []
        for step in range(steps):
            tok, y = sample_batch(pref_cpu, B_train, L_train, gen)
            tok, y = tok.to(DEV), y.to(DEV)
            for m_, o_ in ((mA_, oA), (mB_, oB)):
                loss = torch.nn.functional.cross_entropy(m_(tok, mode=mode), y)
                o_.zero_grad(set_to_none=True); loss.backward(); o_.step()
            with torch.no_grad():
                pa = torch.cat([p.flatten() for p in mA_.parameters()])
                pb = torch.cat([p.flatten() for p in mB_.parameters()])
                pdiv.append(float((pa - pb).norm() / (pa.norm() + 1e-30)))
                ldiv.append(float((mA_(ev_tok, mode="reshape") - mB_(ev_tok, mode="reshape")).abs().max()))
        torch.use_deterministic_algorithms(False)
        return mA_, mB_, pdiv, ldiv

    mA, mB, p_nd, l_nd = train_pair("scatter", False)
    cA, cB, p_dc, l_dc = train_pair("reshape", True)
    first_nz = next((i + 1 for i, v in enumerate(p_nd) if v > 0), None)
    with torch.no_grad():
        predA = mA(ev_tok, mode="reshape").argmax(1); predB = mB(ev_tok, mode="reshape").argmax(1)
        disagree = float((predA != predB).float().mean())
    print(f"nondeterministic pair: param rel-div step1={p_nd[0]:.2e} final={p_nd[-1]:.2e}; "
          f"first divergent step={first_nz}; max logit diff final={l_nd[-1]:.2e}; "
          f"top-1 disagreement on eval set={disagree:.3%}")
    print(f"deterministic control: param rel-div final={p_dc[-1]:.2e}, logit diff final={l_dc[-1]:.2e}  (expect exactly 0)")
    out["M2"] = dict(param_div_nondet=p_nd, logit_div_nondet=l_nd,
                     param_div_control=p_dc, logit_div_control=l_dc,
                     first_divergent_step=first_nz, top1_disagreement_final=disagree,
                     control_bitwise_identical=bool(max(p_dc) == 0.0 and max(l_dc) == 0.0))

    # batch-invariance probe (matmul; out of scalar scope -> context only)
    torch.use_deterministic_algorithms(True, warn_only=True)
    with torch.no_grad():
        ref = cA(ev_tok, mode="reshape")
        bi = {}
        for bs in (1, 8, 64, 256):
            parts = [cA(ev_tok[i:i + bs], mode="reshape") for i in range(0, ev_tok.shape[0], bs)]
            bi[bs] = float((torch.cat(parts) - ref).abs().max())
    torch.use_deterministic_algorithms(False)
    print(f"batch-invariance probe (deterministic mode, max |logit diff| vs batch 256): {bi}")
    out["M2"]["batch_invariance_maxdiff"] = bi

    model = mA.eval()

    # ============================================================ M1 reduction-level validation
    print("\n=== M1  REDUCTION-LEVEL VALIDATION on real network data ===")
    gen1 = torch.Generator().manual_seed(a.seed + 1)
    tok1, y1 = sample_batch(pref_cpu, B_m1, L_m1, gen1)
    tok1 = tok1.to(DEV)
    M1 = {}

    # (a) scatter pooling with real PyTorch atomics: every (segment, dim) element is one reduction of n=L_m1
    with torch.no_grad():
        gathered = model.emb(tok1.reshape(-1))
        runs = torch.stack([model.pool_sum(tok1, "scatter", gathered) for _ in range(R)]).double()
        S_el = gathered.abs().double().view(B_m1, L_m1, d).sum(1)
        exact = gathered.double().view(B_m1, L_m1, d).sum(1)
    spread = (runs.max(0).values - runs.min(0).values).cpu().numpy()
    S_np = S_el.cpu().numpy(); ex = exact.abs().cpu().numpy()
    kap = np.where(ex > 0, S_np / np.maximum(ex, 1e-300), np.inf)
    tH, tFa, tFm = tiers(np.full_like(S_np, L_m1), S_np, a.alpha)
    distinct = (runs.max(0).values != runs.min(0).values).cpu().numpy()
    M1["scatter_pooling"] = dict(name="scatter pooling (index_add_, real PyTorch atomics)",
        count=int(spread.size), n=L_m1, kappa_median=float(np.median(kap)), kappa_max=float(np.max(kap[np.isfinite(kap)])),
        nondeterministic=int(distinct.sum()),
        exceed_H=int((spread > tH).sum()), exceed_Fa=int((spread > tFa).sum()), exceed_Fm=int((spread > tFm).sum()),
        max_t_H=float((spread / tH).max()), median_t_H=float(np.median(spread / tH)))

    with torch.no_grad():
        logits, h_pre, h = model(tok1, mode="reshape", return_hidden=True)
        # (b) vocabulary softmax denominator: tied readout over the trained embedding table
        q = model.pool_sum(tok1, "reshape") / L_m1
        vlog = q @ model.emb.weight.T                                # (B, V)
        vexp = torch.exp(vlog - vlog.max(1, keepdim=True).values)    # the terms the softmax sums
        # (c) LayerNorm mean: sum of H real pre-norm activations
        # (d) gradient accumulation over a batch for one fc2 weight entry
        gen_g = torch.Generator().manual_seed(a.seed + 2)
        tg, yg = sample_batch(pref_cpu, B_grad, L_train, gen_g)
        tg, yg = tg.to(DEV), yg.to(DEV)
        lg, _, hg = model(tg, mode="reshape", return_hidden=True)
        delta = torch.softmax(lg, 1); delta[torch.arange(B_grad), yg] -= 1.0   # dL/dlogits per example
    picks_b = rng.choice(B_m1, size=min(n_vec, B_m1), replace=False)
    vecs_b = [vexp[i].cpu().numpy() for i in picks_b]
    vecs_c = [h_pre[i % B_m1].cpu().numpy() for i in range(n_vec)]
    ent = [(int(rng.integers(K)), int(rng.integers(H))) for _ in range(n_vec)]
    vecs_d = [(delta[:, k] * hg[:, j]).cpu().numpy() for k, j in ent]   # per-example dW2[k,j] contributions

    for key, label, vecs in (("softmax_denominator", "vocab softmax denominator (exp-logits, tied readout)", vecs_b),
                             ("layernorm_mean", "LayerNorm mean (pre-norm hidden activations)", vecs_c),
                             ("grad_accumulation", "gradient accumulation (per-example dW contributions)", vecs_d)):
        summ, _ = reduction_record(label, vecs, R, a.alpha, rng)
        M1[key] = summ
    for k_, s_ in M1.items():
        print(f"  {s_['name']:<55} n={s_['n']:>6}  kappa med={s_['kappa_median']:>9.3g} max={s_['kappa_max']:>9.3g}  "
              f"nondet {s_['nondeterministic']}/{s_['count']}  exceed H/Fa/Fm = "
              f"{s_['exceed_H']}/{s_['exceed_Fa']}/{s_['exceed_Fm']}  max t_H={s_['max_t_H']:.3g}")
    out["M1"] = M1

    # ============================================================ M3 fault detection at the reduction level
    print("\n=== M3  FAULT DETECTION at the reduction level (scatter pooling) ===")
    with torch.no_grad():
        S3 = S_el.cpu().numpy()
        tH3, tFa3, tFm3 = tiers(np.full_like(S3, L_m1), S3, a.alpha)
        # benign: pairs of nondeterministic runs
        fp = np.zeros(3); fp_any = np.zeros(3); n_el = 0
        for _ in range(M3_pairs):
            dlt = (model.pool_sum(tok1, "scatter", gathered) - model.pool_sum(tok1, "scatter", gathered)).abs().double().cpu().numpy()
            ex3 = [dlt > tH3, dlt > tFa3, dlt > tFm3]
            fp += [e.sum() for e in ex3]; fp_any += [e.any() for e in ex3]; n_el += dlt.size
        fp_rate = (fp / n_el).tolist(); fp_any_rate = (fp_any / M3_pairs).tolist()
        print(f"  benign false-positive rate H/Fa/Fm over {n_el:,} element comparisons: "
              f"{fp_rate[0]:.2e} / {fp_rate[1]:.2e} / {fp_rate[2]:.2e};  "
              f"batch-level (any element) over {M3_pairs} pairs: {fp_any_rate[0]:.2f} / {fp_any_rate[1]:.2f} / {fp_any_rate[2]:.2f}")

        clean = model.pool_sum(tok1, "scatter", gathered)
        pred_clean = model.fc2(torch.relu(model.ln(model.fc1(clean / L_m1)))).argmax(1)

        # (i) controlled additive fault of magnitude m * tau_Fm(element) into ONE contribution
        add_rows = []
        for m in [0.3, 0.6, 1.0, 1.5, 2.0, 2.35, 3.0]:
            det = np.zeros(3); T = 0
            for _ in range(M3_pairs):
                s_i = int(rng.integers(B_m1)); j = int(rng.integers(d)); t_i = int(rng.integers(L_m1))
                f = m * tFm3[s_i, j]
                g2 = gathered.clone(); g2[s_i * L_m1 + t_i, j] += float(f)
                diff = float((model.pool_sum(tok1, "scatter", g2)[s_i, j] - model.pool_sum(tok1, "scatter", gathered)[s_i, j]).abs())
                det += [diff > tH3[s_i, j], diff > tFa3[s_i, j], diff > tFm3[s_i, j]]; T += 1
            add_rows.append(dict(f_over_tauFm=m, det_H=det[0] / T, det_Fa=det[1] / T, det_Fm=det[2] / T, trials=T))
            print(f"  additive f={m:>4.2f} x tau_Fm   detect H/Fa/Fm = {det[0]/T:.2f} / {det[1]/T:.2f} / {det[2]/T:.2f}")

        # (ii) realistic SDC: bit flip in one trained embedding weight used by the batch
        used = torch.unique(tok1).cpu().numpy()
        sdc_rows = []
        for bit in [16, 20, 22, 23, 25, 27, 30]:
            det = np.zeros(3); changed = 0; missed_changed = 0; T = 0
            for _ in range(M3_pairs):
                row = int(rng.choice(used)); col = int(rng.integers(d))
                w = model.emb.weight; old = float(w[row, col])
                nv = flip_bit_f32(old, bit)
                if not math.isfinite(nv):
                    continue
                w[row, col] = nv
                g2 = model.emb(tok1.reshape(-1))
                faulty = model.pool_sum(tok1, "scatter", g2)
                w[row, col] = old
                dlt = (faulty - clean).abs().double().cpu().numpy()
                flagged = [(dlt > tH3).any(), (dlt > tFa3).any(), (dlt > tFm3).any()]
                pred_f = model.fc2(torch.relu(model.ln(model.fc1(faulty / L_m1)))).argmax(1)
                ch = bool((pred_f != pred_clean).any())
                det += flagged; changed += ch; missed_changed += (ch and not flagged[2]); T += 1
            if T:
                sdc_rows.append(dict(bit=bit, trials=T, det_H=det[0] / T, det_Fa=det[1] / T, det_Fm=det[2] / T,
                                     changes_prediction=changed / T, missed_and_changed=missed_changed / T))
                print(f"  SDC bit {bit:>2}: detect H/Fa/Fm = {det[0]/T:.2f}/{det[1]/T:.2f}/{det[2]/T:.2f}   "
                      f"changes top-1 = {changed/T:.2f}   missed-by-Fm AND changes top-1 = {missed_changed/T:.2f}")
    out["M3"] = dict(benign_fp_rate_H_Fa_Fm=fp_rate, benign_comparisons=int(n_el),
                     benign_batch_fp_rate_H_Fa_Fm=fp_any_rate, benign_pairs=M3_pairs,
                     additive=add_rows, sdc_bitflip=sdc_rows)

    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {a.out}")
    make_figs(out, a.figdir)
    print(f"figures in {a.figdir}/: ml_m1.png ml_m2.png ml_m3.png")


def make_figs(out, figdir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "font.family": "serif", "axes.grid": True,
                         "grid.alpha": 0.3, "figure.dpi": 150})
    # M1: max tightening factor per reduction type
    M1 = out["M1"]; keys = list(M1)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    vals = [max(M1[k]["max_t_H"], 1e-12) for k in keys]
    ax.bar(range(len(keys)), vals, color=["#1D9E75", "#065A82", "#BA7517", "#7F77DD"][:len(keys)])
    ax.axhline(1.0, color="#C0392B", ls="--", label="tau_H exceeded (t = 1)")
    ax.set_yscale("log"); ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([f"{k.replace('_', chr(10))}\nkappa~{M1[k]['kappa_median']:.2g}" for k in keys], fontsize=8.5)
    ax.set_ylabel("max spread / tau_H over all reductions")
    ax.set_title("ML M1: tolerance on real network reductions")
    ax.legend(fontsize=9); fig.tight_layout(); fig.savefig(f"{figdir}/ml_m1.png"); plt.close()
    # M2: divergence through training
    M2 = out["M2"]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    st = np.arange(1, len(M2["param_div_nondet"]) + 1)
    ax.semilogy(st, np.maximum(M2["param_div_nondet"], 1e-20), color="#C0392B", lw=2,
                label="nondeterministic pooling (two runs)")
    ax.semilogy(st, np.maximum(M2["param_div_control"], 1e-20), color="#1D9E75", lw=2, ls="--",
                label="deterministic control (floor = exactly 0)")
    ax.set_xlabel("training step"); ax.set_ylabel("relative parameter divergence")
    ax.set_title("ML M2: identical training, reduction order alone")
    ax.legend(fontsize=9); fig.tight_layout(); fig.savefig(f"{figdir}/ml_m2.png"); plt.close()
    # M3: three-tier detection
    rows = out["M3"]["additive"]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    xs = [r["f_over_tauFm"] for r in rows]
    ax.plot(xs, [r["det_H"] for r in rows], "o-", color="#BA7517", lw=2, label="tau_H Hoeffding")
    ax.plot(xs, [r["det_Fa"] for r in rows], "v-", color="#1C7293", lw=2, label="tau_Fa Freedman a priori")
    ax.plot(xs, [r["det_Fm"] for r in rows], "s-", color="#7F77DD", lw=2, label="tau_Fm Freedman calibrated")
    ax.set_xlabel("injected fault / tau_Fm"); ax.set_ylabel("detection rate")
    ax.set_title("ML M3: fault detection in scatter pooling")
    ax.legend(fontsize=9); fig.tight_layout(); fig.savefig(f"{figdir}/ml_m3.png"); plt.close()


if __name__ == "__main__":
    main()
