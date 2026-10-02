# fp-reduction harness — Phase A (CPU, fp32, fixed-order)

Measures summation ERROR vs the exact sum as a function of n, kappa and
strategy, and checks it against the classical bounds. No GPU needed.
Phase B (GPU nondeterminism / spread) builds on this.

## Files
| file | what | status |
|---|---|---|
| `strategies.py` | recursive / pairwise / kahan / numpy_sum, all fp32 fixed-order (numba) | tested |
| `gensum.py` | inputs with PRESCRIBED kappa + known exact sum (Ogita–Rump–Oishi style) | tested, hits target within ~1% |
| `reference.py` | exact sum via `fsum`, validated against `Fraction` | tested, PASS |
| `bounds.py` | deterministic bounds (Higham Ch.4) + **probabilistic tau — YOURS, stubbed** | partial |
| `sweep.py` | n × kappa × strategy × seed → one JSON record each, with provenance | tested |
| `analyze.py` | scaling exponent (pooled + per-kappa), tightness ratios, Figure 1 | tested |

## Run
```bash
pip install numpy numba matplotlib
python reference.py                                   # must print PASS
python sweep.py --out results/quick.jsonl --quick     # ~30 s pipeline test
python sweep.py --out results/phaseA.jsonl --seeds 30 # the real thing (minutes)
python analyze.py results/phaseA.jsonl --plot fig1.png
```

## What the quick run already showed (do not treat as results)
* Seed-to-seed error variance is large (2→49 ULP at the same n). **3 seeds
  cannot fit a slope. Use ≥30.**
* beta2 (kappa coefficient) is strongly negative, not ~0. The design-doc
  "beta2≈0" sanity check assumed error scales with sum|x| regardless of data
  structure — that is the WORST-CASE assumption. For cancelling data the
  partial sums are much smaller than sum|x|, so the bound is loose in a
  kappa-dependent way. This is a finding to interpret, not a bug to fix.
  Consequence: fit the n-exponent PER kappa level (analyze.py does both).

## Rules baked in (why they matter)
* `np.sum` is pairwise, NOT the naive loop. Recursive is an explicit numba loop.
* Every intermediate is float32. On x86 that is true fp32 arithmetic.
* Reference is `fsum` (fp64 correctly-rounded), never GPU fp64 — that is
  still a parallel reduction with the same ordering problem.
* kappa is REPORTED (achieved), never assumed. Analyse with `kappa`, not
  `kappa_target`.
* One JSON record per config; aggregation only in analyze.py. Raw records
  are the artifact.
* Provenance (host, versions, git commit, GPU if present) in every record.

## Yours to do
1. `bounds.py`: derive `lambda_for_alpha` and `tau_probabilistic`, verify
   the constants against Higham & Mary 2019, then remove the
   NotImplementedError. This is the closed-form tolerance the panel asked
   about — and what Phase B's spread measurements get checked against.
2. Interpret the per-kappa exponents from the 30-seed run. Is recursive at
   kappa=1 near 0.5 (probabilistic regime)? What happens at high kappa?
3. Decide the kappa levels for the full sweep using `bounds.kappa_ceiling`
   so no cell is past the point of "no correct digits".
