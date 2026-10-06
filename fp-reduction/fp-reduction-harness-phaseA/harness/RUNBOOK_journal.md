# Runbook — journal-version experiments (JPDC target)

Everything below drops into `harness/` next to bounds.py, gensum.py, reference.py, gpu_strategies.py.
`bounds.py` is updated (γ̃ exponent form; `phi_abs`, `tau_ps`, `tau_ps_freedman`, `tau_gemm`) — replace it first and run `python bounds.py` (asserts + the WP2 table).

## Verification done before hand-off
- Every CUDA kernel (`gpu_strategies`, `gpu_variance`, `gpu_gemm_study`) compiled with NVRTC for compute_80.
- Every GPU script executed end-to-end under a NumPy mock of CuPy that emulates each kernel (random-order atomics, split-K), and `ml_capstone_llm.py` against a tiny mock Qwen-shaped model; JSON outputs fed to `plots_journal.py` successfully.
- `rescore.py`, `tail_test.py --device cpu`, `wp7_exponent_generator.py`, `derivation_wp2_check.py` ran for real on CPU.
- Not verifiable here: real cuBLAS/tensor-core behaviour, Qwen download, timings.

## Status

| WP | What | Code | Needs | Status |
|---|---|---|---|---|
| 1 | Tail-dominance test ("α-test as a curve") | `tail_test.py` | 4090 (+ A100/H100 if available) | ready; CPU mode validated |
| 2 | Order-robust partial-sum tolerances τ_PS, τ_PSFa, τ_PSF | `bounds.py`, `rescore.py`, `derivation_wp2_order_robust.md`, `derivation_wp2_check.py` | nothing (re-scores stored results) | **done** (derivation + lemma proved + brute-force verified; rescore tested) |
| 3 | GEMM tolerance (scalar + block-FMA) and split-K atomic GEMM study | `bounds.tau_gemm`, `gpu_gemm_study.py` | 4090; torch for tensor-core path | ready; CPU emulation validated |
| 4 | Qwen2.5-0.5B: logits-as-GEMM, certified argmax, batch probe, near-tie faults | `ml_capstone_llm.py` | 4090, `pip install transformers` | ready; untested on GPU |
| 5 | Rounding variance measured ON the GPU (instrumented atomicAdd) | `gpu_variance.py` | 4090 | ready; untested on GPU |
| 6 | L40S same-architecture control | existing `gpu_exceedance.py` | L40S pod | nothing to write |
| 7 | Why the exponent → 0 at high κ | `wp7_exponent_generator.py` | CPU | **done**: mechanism found (below) |
| 8 | LaTeX rewrite | — | — | after results |
| — | figures | `plots_journal.py` | result JSONs | ready; tested on mock outputs |

## Pod session (≈ 2–3 h on a 4090)

```
cd /workspace/Fp-Reproducibilty/fp-reduction/fp-reduction-harness-phaseA/harness
pip install transformers accelerate scipy matplotlib
python bounds.py                                                     # asserts + WP2 table
mkdir -p results figs_journal

# WP5  (~5 min)
python gpu_variance.py --n 1048576 --launches 50 --out results/gpu_variance_k1.json
python gpu_variance.py --n 1048576 --kappa 1e4 --launches 50 --out results/gpu_variance_k1e4.json

# WP1  (~30-60 min; 2^22 dominates)
python tail_test.py --device gpu --launches 20000 --ns 16 18 20 --out results/tail_gpu.json
python tail_test.py --device gpu --launches 100000 --ns 20 --out results/tail_gpu_2p20_100k.json

# WP2  (seconds)  -- paths to your stored jsonl files
python rescore.py results/exceedance.jsonl --kind cpu --out results/rescore_cpu.json
python rescore.py results/gpu_exceedance.jsonl results/gpu_a100.jsonl results/gpu_h100.jsonl --kind gpu --out results/rescore_gpu.json

# WP3  (~20 min)
python gpu_gemm_study.py --M 256 --N 256 --K 4096 --launches 200 --out results/gemm_4090_uniform.json
python gpu_gemm_study.py --M 256 --N 256 --K 4096 --launches 200 --data mixed --out results/gemm_4090_mixed.json
python gpu_gemm_study.py --M 256 --N 256 --K 16384 --launches 100 --out results/gemm_4090_k16k.json

# WP4  (~30-60 min; 0.5B model fp32 ~2 GB)
python ml_capstone_llm.py --quick --out results/llm_quick.json        # smoke test first
python ml_capstone_llm.py --prompts 128 --launches 100 --out results/llm_qwen05.json

# fp16/bf16 inside-regime cells (from review 2)
python gpu_exceedance_mp.py --inside-regime --out results/gpu_mp_inside.jsonl

# WP7 (CPU, ~7 min, can run on the pod while GPU jobs run in another shell)
python wp7_exponent_generator.py

# figures from everything above
python plots_journal.py --results results --figs figs_journal
```

If a GPU script crashes partway, the per-step JSON is written at the end only for gpu_gemm_study/ml_capstone_llm; tail_test writes after every n. Re-run with smaller `--launches` first if in doubt.
On the A100/H100/L40S pod: `tail_test.py --ns 20 --launches 20000`, `gpu_variance.py`, and the standard `gpu_exceedance.py` sweep (L40S only).

Send back: every `results/*.json` / `*.jsonl` / `*.npy` written above plus the console logs.

## Results already in hand (sandbox, CPU)

**WP2 (n = 2²⁰, fp32, α = 10⁻³, uniform data):** τ_H/S 8.13e-4 · τ_Fm 3.46e-4 (2.35×) · **τ_PS 3.46e-4 (2.35×, no variance assumption, no union bound)** · τ_PSFa 2.00e-4 (4.06×) · **τ_PSF 1.47e-4 (5.53×)**. On CPU permutation spread the max over 20,000 pairs uses 43% of τ_PSF (was 7.8% of τ_H). Cost: factor (1+γ_{n−1})/(1−u) = 1.07 at 2²⁰, 1.33 at 2²².

**WP1 (CPU, 20,000 pairs, n = 2¹⁴, 2¹⁶):** the two-run spread tail is Gaussian (R² = 0.999) with effective σ ≈ 0.155 natural units; utilization of τ_H is 1.5–7% at every level from α = 0.5 to 10⁻³ — the bound-to-empirical gap in λ is ~10× and flat across levels.

**WP7 (mechanism of Fig. 2):** a second generator with *statistical* cancellation (Gaussian + constant shift to exact κ) gives the same exponents (0.51, 0.41, 0.05, −0.11 at κ = 1, 10², 10⁴, 10⁶) as the constructed generator, so it is arithmetic, not the generator. Mechanism: from the exact identity e = Σ δ_k t_k, Var(e) = σ²Σ s_k²; partial sums grow like k for positive data (Σs_k² ~ n³ → error ~ n^{1.5} → normalized exponent 0.5) and like √k for cancelling data (Σs_k² ~ n² → error ~ n → exponent 0). Prediction u√(0.186 Σ s_k²) matches the measured RMS error within a factor 0.6–1.3 across 12 (n, κ) cells spanning three orders of magnitude. This replaces the "we do not have a mechanism" sentence in v5 §V-A and is the same identity the WP2 tolerance is built on.

**WP3 (CPU emulation, K = 1024):** logic validated; in a GEMM the per-entry tolerance is small enough that a single-input bit-flip at bit ≥ 20 is detected (20/20), unlike the scalar n = 2²⁰ case — expected to be a finding on the GPU as well.
