"""
det_baseline.py — the referee question "why not just use deterministic mode?"

Measures, for the operations studied in the paper, the COST and the COVERAGE of
the deterministic alternatives:

  cost     : wall time of the deterministic path vs the nondeterministic one
  coverage : what nondeterminism REMAINS after switching deterministic mode on
             (run-to-run spread should go to zero; batch-size / shape dependence
             does not, because deterministic mode fixes the order for a given
             shape, not across shapes -- so a tolerance is still needed whenever
             two parties may run different shapes or different libraries)

Operations:
  A. our kernels: naive_atomic vs block_atomic vs tree_fixed (CuPy) at n = 2^20, 2^22
  B. PyTorch index_add_ (scatter pooling): default vs torch.use_deterministic_algorithms(True)
  C. PyTorch fp32 GEMM: default vs CUBLAS_WORKSPACE_CONFIG=:4096:8 + deterministic mode,
     plus batch-1 vs batched outputs under BOTH settings
  D. (optional) Qwen2.5-0.5B forward: default vs deterministic mode, time and batch invariance

CUBLAS_WORKSPACE_CONFIG must be in the environment before CUDA initialises, so this
script sets it in-process when --det is given and otherwise leaves it unset; run it
TWICE (once per mode) and compare the two JSONs:
  python det_baseline.py --out results/det_off.json
  python det_baseline.py --det --out results/det_on.json
  python det_baseline.py --det --llm --out results/det_on_llm.json     # adds D
"""

import argparse, json, os, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--det", action="store_true", help="deterministic mode (sets CUBLAS_WORKSPACE_CONFIG before CUDA init)")
ap.add_argument("--llm", action="store_true")
ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
ap.add_argument("--reps", type=int, default=50)
ap.add_argument("--device", default="cuda")
ap.add_argument("--out", required=True)
args = ap.parse_args()
if args.det:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch
if args.det:
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
dev = args.device


def timeit(fn, reps):
    fn()
    if dev == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps): fn()
    if dev == "cuda": torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps


def spread(fn, reps):
    r = [fn() for _ in range(reps)]
    return float(max(float((a - r[0]).abs().max()) for a in r)), len({a.reshape(-1)[0].item() for a in r})


out = {"deterministic_mode": args.det, "device": dev, "torch": torch.__version__,
       "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else dev, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
g = torch.Generator(device="cpu").manual_seed(0)

# ---------------- A. our kernels (CuPy) ----------------
try:
    import cupy as cp
    import gpu_strategies as gs
    kern = gs._load(); rows = []
    for k in (20, 22):
        n = 2 ** k
        x = (torch.rand(n, generator=g) + 0.5).numpy().astype(np.float32); xg = cp.asarray(x)
        for name in ("naive_atomic", "block_atomic", "tree_fixed", "cupy_sum"):
            fn = (lambda: gs.STRATEGIES[name](xg, n, kern)) if name != "cupy_sum" else (lambda: gs.cupy_sum(xg, n))
            fn(); cp.cuda.Stream.null.synchronize()
            t0 = time.perf_counter()
            for _ in range(args.reps): fn()
            cp.cuda.Stream.null.synchronize(); t = (time.perf_counter() - t0) / args.reps
            vals = [fn() for _ in range(20)]
            rows.append({"n": n, "strategy": name, "ms": 1000 * t, "distinct_of_20": len(set(vals))})
            print(f"A {name:>13} n=2^{k}: {1000*t:8.3f} ms   distinct results of 20: {len(set(vals))}")
    out["kernels"] = rows
except Exception as e:
    print("A skipped:", repr(e))

# ---------------- B. index_add_ scatter pooling ----------------
try:
    V, d, T, B = 32768, 64, 4096, 8192
    emb = torch.randn(V, d, generator=g).to(dev)
    tok = torch.randint(0, V, (B * T // 64,), generator=g).to(dev)          # 524,288 tokens
    seg = torch.randint(0, B, (tok.numel(),), generator=g).to(dev)
    def pool():
        outp = torch.zeros(B, d, device=dev)
        return outp.index_add_(0, seg, emb[tok])
    t = timeit(pool, args.reps); sp, dist = spread(pool, 20)
    out["index_add"] = {"ms": 1000 * t, "max_run_to_run_spread": sp, "distinct_of_20": dist}
    print(f"B index_add_ ({tok.numel():,} tokens): {1000*t:8.3f} ms   run-to-run spread {sp:.3e}  distinct {dist}")
except Exception as e:
    print("B skipped:", repr(e))

# ---------------- C. fp32 GEMM ----------------
try:
    M, K, N = 256, 4096, 256
    A = (torch.rand(M, K, generator=g) + 0.5).to(dev); Bm = (torch.rand(K, N, generator=g) + 0.5).to(dev)
    mm = lambda: A @ Bm
    t = timeit(mm, args.reps); sp, dist = spread(mm, 20)
    full = (A @ Bm); rows1 = torch.cat([A[i:i + 1] @ Bm for i in range(M)])
    bi = float((full - rows1).abs().max())
    out["gemm_fp32"] = {"shape": [M, K, N], "ms": 1000 * t, "max_run_to_run_spread": sp, "distinct_of_20": dist,
                        "batch1_vs_batched_max_diff": bi, "batch_invariant": bi == 0.0}
    print(f"C GEMM {M}x{K}x{N}: {1000*t:8.3f} ms   run-to-run spread {sp:.3e}  batch-1 vs batched max diff {bi:.3e}  batch-invariant: {bi == 0.0}")
    # fp16 tensor-core path too
    Ah, Bh = A.half(), Bm.half()
    mmh = lambda: Ah @ Bh
    t = timeit(mmh, args.reps); sp, dist = spread(mmh, 20)
    fullh = (Ah @ Bh); rows1h = torch.cat([Ah[i:i + 1] @ Bh for i in range(M)])
    bih = float((fullh.float() - rows1h.float()).abs().max())
    out["gemm_fp16"] = {"shape": [M, K, N], "ms": 1000 * t, "max_run_to_run_spread": sp, "distinct_of_20": dist,
                        "batch1_vs_batched_max_diff": bih, "batch_invariant": bih == 0.0}
    print(f"C GEMM fp16 {M}x{K}x{N}: {1000*t:8.3f} ms   run-to-run spread {sp:.3e}  batch-1 vs batched max diff {bih:.3e}  batch-invariant: {bih == 0.0}")
except Exception as e:
    print("C skipped:", repr(e))

# ---------------- D. LLM forward ----------------
if args.llm:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokz = AutoTokenizer.from_pretrained(args.model); tokz.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(dev).eval()
        prompts = ["The capital of France is", "Water boils at a temperature of", "The first president of the United States was",
                   "In mathematics, a prime number is", "The largest planet in the solar system is", "To be or not to be, that is the",
                   "The chemical symbol for gold is", "Photosynthesis converts sunlight into"] * 2
        enc = tokz(prompts, return_tensors="pt", padding=True).to(dev)
        pos = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
        with torch.no_grad():
            fwd = lambda: model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], position_ids=pos).logits[:, -1, :]
            t = timeit(fwd, max(5, args.reps // 10)); sp, dist = spread(fwd, 10)
            full = fwd()
            b1 = torch.cat([model(input_ids=enc["input_ids"][i:i + 1], attention_mask=enc["attention_mask"][i:i + 1], position_ids=pos[i:i + 1]).logits[:, -1, :] for i in range(len(prompts))])
            bi = float((full - b1).abs().max()); flips = int((full.argmax(1) != b1.argmax(1)).sum())
        out["llm_forward"] = {"model": args.model, "prompts": len(prompts), "ms": 1000 * t, "max_run_to_run_spread": sp, "distinct_of_10": dist,
                              "batch1_vs_batched_max_logit_diff": bi, "batch_invariant": bi == 0.0, "top1_flips": flips}
        print(f"D LLM forward ({len(prompts)} prompts): {1000*t:8.1f} ms   run-to-run spread {sp:.3e}  batch-1 vs batched {bi:.3e}  top-1 flips {flips}")
    except Exception as e:
        print("D skipped:", repr(e))

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
with open(args.out, "w") as fh:
    json.dump(out, fh, indent=1)
print("saved", args.out)