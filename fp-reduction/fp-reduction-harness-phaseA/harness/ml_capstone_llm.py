"""
ml_capstone_llm.py — WP4: the tolerance on a real language model (Qwen2.5-0.5B),
including the near-tie case the v5 paper admits it never tested.

What it measures (all on one GPU, fp32 weights, TF32 off):
  1. LOGITS AS A GEMM.  logits = h · W_vocabᵀ is a (P × V) matrix product with
     K = hidden size (896). We compute it with our split-K atomic GEMM (real
     nondeterministic reduction on the real model weights, `launches` times) and
     test every entry against the elementwise tolerance tau_gemm (scalar fp32
     model, whole-matrix alpha).  -> exceedance count over P*V*launches entries.
  2. CERTIFIED ARGMAX.  For each prompt, margin = logit[top1] - logit[top2].
     If margin > tau[top1] + tau[top2], then (under the model) NO correct
     evaluation order can change the top-1 token: the prediction is certified
     order-invariant. We report the fraction of prompts that are certifiable
     and check that every observed top-1 disagreement (across our launches and
     across the batch-size probe) occurs at an UNCERTIFIED prompt.
  3. BATCH-SIZE PROBE (He / Thinking Machines effect).  The model's own logits
     at batch size 1 vs batch size P for the same prompts: spread vs tau,
     top-1 flips vs certification.
  4. SCALAR REDUCTIONS INSIDE THE MODEL.  The final RMSNorm sum of squares
     (n = 896, positive terms) and the softmax denominator over the vocabulary
     (n = V) reduced by the naive-atomic kernel: spread vs tau_H / tau_Fm / tau_PSF.
  5. NEAR-TIE FAULTS.  Take the prompts with the smallest margins. Inject single
     bit-flips (bit sweep) into the top-1 row of W_vocab and additive faults
     sized in units of the local tau; record (detected at logit level vs clean
     run, top-1 changed). This produces prediction-changing faults of
     INTERMEDIATE magnitude, which the bag-of-tokens study could not.

Needs: torch, transformers, cupy, bounds.py, gpu_gemm_study.py, gpu_strategies.py.
  pip install transformers accelerate
Run:
  python ml_capstone_llm.py --model Qwen/Qwen2.5-0.5B --prompts 128 --launches 100 --out results/llm_qwen05.json
  python ml_capstone_llm.py --quick     # 16 prompts, 10 launches
"""

import argparse, json, math, os, time
import numpy as np

from bounds import (tau_gemm, tau_probabilistic, tau_freedman, tau_ps_freedman, phi_abs, UNIT_ROUNDOFF)

U = UNIT_ROUNDOFF["float32"]

PROMPTS = [
    "The capital of France is", "Water boils at a temperature of", "The first president of the United States was",
    "In mathematics, a prime number is", "The largest planet in the solar system is", "To be or not to be, that is the",
    "The chemical symbol for gold is", "Photosynthesis converts sunlight into", "The speed of light is approximately",
    "A triangle has three", "The Great Wall is located in", "Shakespeare wrote the play", "The square root of 144 is",
    "The opposite of hot is", "DNA stands for", "The Pacific is the largest", "Mount Everest is the tallest",
    "The author of 1984 is", "An octopus has eight", "The currency of Japan is", "Ice is water in its",
    "The sun rises in the", "A year has twelve", "The human heart has four", "Pi is approximately equal to",
    "The Amazon is the largest", "Electrons carry a negative", "The freezing point of water is", "Bees produce",
    "The longest river in Africa is", "In the beginning God created the", "My favourite colour is",
    "The meeting will start at", "She opened the door and saw", "He decided to", "The weather today is",
    "I think the best approach would be to", "After the storm, the", "The results of the experiment were",
    "One reason this might fail is", "The function returns", "If x is greater than y then", "The patient was given",
    "Yesterday I went to the", "The committee decided to", "This is probably because", "The next step is to",
    "It was a dark and stormy", "Once upon a time there was a", "In conclusion, we", "The main difference between",
    "The recipe calls for two cups of", "The train leaves at", "Please remember to", "The error occurs when",
    "He looked at the sky and", "The study found that", "The price of oil", "The algorithm runs in",
    "The teacher asked the students to", "Breaking news:", "According to the report,", "The song begins with",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--prompts", type=int, default=128)
    ap.add_argument("--prompt-file", default=None, help="text file, one prompt per line (to hunt for near-ties use thousands)")
    ap.add_argument("--launches", type=int, default=100)
    ap.add_argument("--splits", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--near-tie", type=int, default=16, help="number of smallest-margin prompts for fault injection")
    ap.add_argument("--fault-trials", type=int, default=30)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/llm.json")
    args = ap.parse_args()
    if args.quick:
        args.prompts, args.launches, args.near_tie, args.fault_trials = 16, 10, 4, 8

    import torch, cupy as cp
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from gpu_gemm_study import SplitK
    import gpu_strategies as gs
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    dev = args.device
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(dev).eval()
    W = model.get_output_embeddings().weight.detach()                # (V, d) fp32
    V, d = W.shape
    prov = {"model": args.model, "V": int(V), "d": int(d), "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else dev,
            "torch": torch.__version__, "cupy": cp.__version__, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    print(json.dumps(prov))

    # prompts: cycle the built-in list with light variation to reach --prompts
    if args.prompt_file:
        with open(args.prompt_file) as fh:
            pool = [l.strip() for l in fh if l.strip()]
    else:
        pool = PROMPTS
    prompts = [pool[i % len(pool)] + ("" if i < len(pool) else f" ({i // len(pool)})") for i in range(args.prompts)]
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"
    enc = tok(prompts, return_tensors="pt", padding=True).to(dev)

    # ---- hidden states at the last position (the input of lm_head) and the model's own logits ----
    # mask-aware position ids so that batch-P and batch-1 see IDENTICAL inputs (left padding)
    pos = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
    with torch.no_grad():
        out_full = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], position_ids=pos,
                         output_hidden_states=True)
        h = out_full.hidden_states[-1][:, -1, :].contiguous()        # (P, d) after final norm
        logits_full = out_full.logits[:, -1, :].float()               # batch of P
        logits_b1 = torch.cat([model(input_ids=enc["input_ids"][i:i + 1], attention_mask=enc["attention_mask"][i:i + 1],
                                     position_ids=pos[i:i + 1]).logits[:, -1, :] for i in range(len(prompts))]).float()
        # pre-norm hidden for the RMSNorm sum-of-squares reduction
        pre = out_full.hidden_states[-2][:, -1, :].contiguous() if len(out_full.hidden_states) > 1 else h
    P = h.shape[0]
    # sanity: the hidden state we took must be the lm_head input (post final norm)
    chk = float((h @ W.T - logits_full).abs().max())
    if chk > 1e-2 * float(logits_full.abs().max()):
        h = model.model.norm(out_full.hidden_states[-1][:, -1, :]).contiguous()
        chk = float((h @ W.T - logits_full).abs().max())
    print(f"lm_head input check: max |h W^T - logits| = {chk:.3e}")
    hn = h.cpu().numpy().astype(np.float32); Wn = W.cpu().numpy().astype(np.float32)
    ref = hn.astype(np.float64) @ Wn.T.astype(np.float64)              # (P, V) exact-ish reference
    tau = tau_gemm(hn, Wn.T, args.alpha, U, "scalar")                 # (P, V), whole-matrix alpha
    print(f"P={P} prompts, V={V}, d={d}; median tau/|logit| = {np.median(tau / np.maximum(np.abs(ref), 1e-30)):.3e}")

    # ---- 1. split-K atomic logits: real nondeterministic reduction on real weights ----
    sk = SplitK(args.splits)
    A_g = cp.asarray(hn); B_g = cp.asarray(np.ascontiguousarray(Wn.T))
    exc = 0; nondet = 0; maxu = 0.0; top1_flips = 0; flip_prompts = set()
    base_top1 = ref.argmax(1)
    npairs = args.launches // 2
    for i in range(npairs):                                   # pairs processed on the fly (memory)
        r1 = sk(A_g, B_g); r2 = sk(A_g, B_g)
        dsp = np.abs(r1.astype(np.float64) - r2.astype(np.float64))
        exc += int((dsp > tau).sum()); nondet += int((dsp > 0).sum()); maxu = max(maxu, float((dsp / tau).max()))
        for r in (r1, r2):
            f = np.nonzero(r.argmax(1) != base_top1)[0]
            top1_flips += len(f); flip_prompts |= set(f.tolist())
    res_gemm = {"launches": 2 * npairs, "pairs": npairs, "entries_per_pair": int(P * V),
                "nondet_entry_fraction": nondet / (npairs * P * V), "exceedances": exc,
                "max_spread_over_tau": maxu, "top1_disagreements_vs_reference": top1_flips, "top1_checks": 2 * npairs * P}
    print("split-K atomic logits:", res_gemm)

    # ---- 2. certified argmax ----
    srt = np.sort(ref, axis=1); top2 = np.argsort(ref, axis=1)[:, -2:]
    margin = srt[:, -1] - srt[:, -2]
    tau_pair = tau[np.arange(P), top2[:, 1]] + tau[np.arange(P), top2[:, 0]]
    certified = margin > tau_pair
    mr = margin / tau_pair
    res_cert = {"certified_fraction": float(certified.mean()), "n_uncertified": int((~certified).sum()),
                "margin_over_taupair_percentiles": {str(q): float(np.percentile(mr, q)) for q in (0, 1, 5, 25, 50, 75, 100)},
                "median_margin_over_taupair": float(np.median(margin / tau_pair)),
                "min_margin_over_taupair": float((margin / tau_pair).min()),
                "flips_only_at_uncertified": all((not certified[p]) for p in flip_prompts),
                "flip_prompt_indices": sorted(flip_prompts)}
    print("certified argmax:", res_cert)

    # ---- 3. batch-size probe ----
    lf = logits_full.cpu().numpy().astype(np.float64); lb = logits_b1.cpu().numpy().astype(np.float64)
    dsp = np.abs(lf - lb)
    res_batch = {"bitwise_identical": bool(np.array_equal(lf, lb)), "nondet_entry_fraction": float((dsp > 0).mean()),
                 "exceedances": int((dsp > tau).sum()), "max_spread_over_tau": float((dsp / tau).max()),
                 "top1_flips": int((lf.argmax(1) != lb.argmax(1)).sum()),
                 "flips_only_at_uncertified": all((not certified[p]) for p in np.nonzero(lf.argmax(1) != lb.argmax(1))[0])}
    print("batch-size probe (batch P vs batch 1):", res_batch)

    # ---- 4. scalar reductions inside the model ----
    kern = gs._load()
    scal = {}
    for name, vec_fn, n in (("rmsnorm_sumsq", lambda i: (pre[i].float() ** 2).cpu().numpy().astype(np.float32), d),
                            ("softmax_denominator", lambda i: np.exp(ref[i] - ref[i].max()).astype(np.float32), V)):
        cells = []
        for i in range(min(P, 32)):
            x = vec_fn(i); S = float(np.abs(x.astype(np.float64)).sum()); phi = phi_abs(x)
            xg = cp.asarray(x)
            r = np.array([gs.naive_atomic(xg, n, kern) for _ in range(args.launches)])
            spread = float(r.max() - r.min())
            # swamping diagnostic: mass of terms below half an ulp of the final sum -- such terms are
            # dropped deterministically when added to a full accumulator (biased, order-dependent error)
            half_ulp = 0.5 * np.spacing(np.float32(S))
            swamped = float(np.abs(x[np.abs(x) < half_ulp]).astype(np.float64).sum()) / S
            cells.append({"spread": spread, "tauH": tau_probabilistic(n, U, S, args.alpha),
                          "tauFm": tau_freedman(n, U, S, args.alpha), "tauPSF": tau_ps_freedman(n, U, S, phi, args.alpha),
                          "distinct": int(len(np.unique(r))), "swamped_mass_fraction": swamped,
                          "swamped_count_fraction": float((np.abs(x) < half_ulp).mean()),
                          "phi_over_nS2": phi / (n * S * S)})
        scal[name] = {"n": n, "instances": len(cells),
                      "cells": cells,
                      "swamped_mass_fraction_median": float(np.median([c["swamped_mass_fraction"] for c in cells])),
                      "swamped_count_fraction_median": float(np.median([c["swamped_count_fraction"] for c in cells])),
                      "util_H_per_instance": [c["spread"] / c["tauH"] for c in cells],
                      "util_PSF_per_instance": [c["spread"] / c["tauPSF"] for c in cells],
                      "nondet_instances": sum(c["distinct"] > 1 for c in cells),
                      "exceed_H": sum(c["spread"] > c["tauH"] for c in cells),
                      "exceed_Fm": sum(c["spread"] > c["tauFm"] for c in cells),
                      "exceed_PSF": sum(c["spread"] > c["tauPSF"] for c in cells),
                      "max_util_H": max(c["spread"] / c["tauH"] for c in cells),
                      "max_util_PSF": max(c["spread"] / c["tauPSF"] for c in cells)}
        print(name, {k: v for k, v in scal[name].items() if k not in ('cells', 'util_H_per_instance', 'util_PSF_per_instance')},
              ' max util_PSF per instance:', round(max(scal[name]['util_PSF_per_instance']), 3))

    # ---- 5. near-tie faults ----
    order = np.argsort(margin / tau_pair)
    tie_idx = order[:args.near_tie]
    faults = []
    def logits_with_fault(row, k, value):
        """set W[row,k] = value on the GPU copy (B_g is W^T), compute, restore."""
        saved = float(B_g[k, row]); B_g[k, row] = np.float32(value)
        L = sk(A_g, B_g); B_g[k, row] = np.float32(saved)
        return L
    for bit in (12, 16, 18, 20, 22, 23, 25, 27, 30):
        det = chg = chg_missed = done = 0
        for t in range(args.fault_trials):
            p = int(tie_idx[t % len(tie_idx)]); row = int(base_top1[p]); k = int(rng.integers(d))
            v = np.float32(Wn[row, k]); fv = (v.view(np.int32) ^ np.int32(1 << bit)).view(np.float32)
            if not np.isfinite(fv): continue
            Lf = logits_with_fault(row, k, fv).astype(np.float64); Lc = sk(A_g, B_g).astype(np.float64)
            dd = np.abs(Lf[p] - Lc[p])
            is_det = bool((dd > tau[p]).any()); is_chg = bool(Lf[p].argmax() != Lc[p].argmax())
            det += is_det; chg += is_chg; chg_missed += (is_chg and not is_det); done += 1
        if done:
            faults.append({"bit": bit, "trials": done, "detected": det / done, "top1_changed": chg / done, "changed_and_missed": chg_missed})
            print(f"  bit {bit:>2}: detected {det}/{done}  top1 changed {chg}/{done}  changed-and-missed {chg_missed}")
    # additive faults in units of the local tau on the top-1 logit path (perturb one weight so the logit moves by m*tau)
    add = []
    for m in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0):
        det = chg = chg_missed = done = 0
        for t in range(args.fault_trials):
            p = int(tie_idx[t % len(tie_idx)]); row = int(base_top1[p]); k = int(rng.integers(d))
            if hn[p, k] == 0: continue
            delta = -m * tau[p, row] / hn[p, k]            # push the top-1 logit DOWN by m*tau
            Lf = logits_with_fault(row, k, np.float32(Wn[row, k] + delta)).astype(np.float64); Lc = sk(A_g, B_g).astype(np.float64)
            dd = np.abs(Lf[p] - Lc[p])
            is_det = bool((dd > tau[p]).any()); is_chg = bool(Lf[p].argmax() != Lc[p].argmax())
            det += is_det; chg += is_chg; chg_missed += (is_chg and not is_det); done += 1
        if done:
            add.append({"fault_over_tau": m, "trials": done, "detected": det / done, "top1_changed": chg / done, "changed_and_missed": chg_missed})
            print(f"  additive {m:>3.1f} tau: detected {det}/{done}  top1 changed {chg}/{done}  changed-and-missed {chg_missed}")

    # additive faults in units of the MARGIN: a certified prompt (margin > tau_pair) implies any
    # prediction-changing fault moves a logit by more than tau, so it must be detected.
    addm = []
    for m in (0.5, 0.9, 1.1, 1.5, 3.0):
        det = chg = chg_missed = done = 0
        for t in range(args.fault_trials):
            p = int(tie_idx[t % len(tie_idx)]); row = int(base_top1[p]); k = int(rng.integers(d))
            if hn[p, k] == 0: continue
            delta = -m * margin[p] / hn[p, k]
            Lf = logits_with_fault(row, k, np.float32(Wn[row, k] + delta)).astype(np.float64); Lc = sk(A_g, B_g).astype(np.float64)
            dd = np.abs(Lf[p] - Lc[p])
            is_det = bool((dd > tau[p]).any()); is_chg = bool(Lf[p].argmax() != Lc[p].argmax())
            det += is_det; chg += is_chg; chg_missed += (is_chg and not is_det); done += 1
        if done:
            addm.append({"fault_over_margin": m, "trials": done, "detected": det / done, "top1_changed": chg / done, "changed_and_missed": chg_missed})
            print(f"  additive {m:>3.1f} margin: detected {det}/{done}  top1 changed {chg}/{done}  changed-and-missed {chg_missed}")

    out = {"provenance": prov, "P": int(P), "alpha": args.alpha, "splits": args.splits,
           "gemm_logits": res_gemm, "certified_argmax": res_cert, "batch_probe": res_batch,
           "scalar_reductions": scal, "near_tie_bitflips": faults, "near_tie_additive_tau_units": add, "near_tie_additive_margin_units": addm,
           "near_tie_prompts": [prompts[i] for i in tie_idx], "near_tie_margin_over_taupair": [float(margin[i] / tau_pair[i]) for i in tie_idx]}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()