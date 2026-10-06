# WP2 — Order-robust partial-sum tolerances (derivation for the journal version)

Notation: inputs x₁,…,xₙ in floating point, unit roundoff u, S = Σ|xⱼ|. An *evaluation tree* T is any full binary tree with the xⱼ at its leaves; each internal node v computes ŝ_v = fl(ŝ_L + ŝ_R) = (ŝ_L + ŝ_R)(1 + δ_v), |δ_v| ≤ u. Chains (sequential sums in any order), pairwise trees, block-then-atomic hybrids, and the random order realised by atomic accumulation are all evaluation trees. Fix any order of the internal nodes consistent with evaluation (children before parents) and let 𝓕_v be the σ-algebra of the roundings at nodes evaluated before v.

**Model (mean independence).** E[δ_v | 𝓕_v] = 0 and |δ_v| ≤ u for every internal node v. (Model 2.1 of Higham–Mary with independence weakened to mean independence, as in HM 2020 / Connolly–Higham–Mary 2021.)

## 1. Exact error identity

Let t_v = ŝ_L + ŝ_R be the exact sum of the two computed operands at v, and let s_v be the exact sum of the leaves below v. Write e_v = ŝ_v − s_v.

ŝ_v = t_v(1 + δ_v) = ŝ_L + ŝ_R + δ_v t_v, hence e_v = e_L + e_R + δ_v t_v. Unrolling to the root,

  **ŝ − s = Σ_v δ_v t_v.**  (1)

This is exact — no first-order truncation — and t_v is 𝓕_v-measurable, so under the model (δ_v t_v) is a martingale difference sequence: (1) is a martingale transform. (This is the identity underlying Hallman 2021 and Hallman–Ipsen 2022; we use it for an arbitrary tree.)

## 2. Deterministic control of the weights

Let A_v = Σ_{j∈v}|xⱼ| (sum of magnitudes under v) and m_v the number of leaves under v. The classical bound gives |ŝ_v| ≤ (1 + γ_{m_v−1}) A_v, so

  |t_v| = |ŝ_v|/|1 + δ_v| ≤ (1 + γ_{n−1})/(1 − u) · A_v =: f · A_v,  (2)

with f = 1 + O(nu). Therefore Σ_v t_v² ≤ f² Σ_v A_v², and |δ_v t_v| ≤ u f S for every v.

## 3. Lemma (the worst tree is the descending chain)

**Lemma.** For nonnegative a₁ ≥ a₂ ≥ … ≥ aₙ, over all evaluation trees T on these leaves,

  max_T Σ_{v internal} (Σ_{j∈v} aⱼ)² = Σ_{k=2}^{n} (a₁ + … + a_k)² =: Φ(a).

*Proof.* Induction on n (trivial for n ≤ 2). Let T have root children A, B with sums S_A, S_B, and let A be the subtree containing the largest leaf a₁; then S_A ≥ a₁ ≥ any leaf of B. Σ_T = (S_A + S_B)² + Σ_A + Σ_B ≤ S² + Φ(A) + Φ(B) by induction. Consider the chain that evaluates A's leaves in descending order first and then B's leaves in descending order: its internal-node values are the partial sums P₂,…,P_{|A|} of A (contributing exactly Φ(A)) followed by S_A + B₁, …, S_A + B_{|B|} where B_j are B's descending partial sums, the last equal to S. Since S_A ≥ every leaf of B, S_A + B_j ≥ B_{j+1}, so Σ_{j=1}^{|B|−1}(S_A + B_j)² ≥ Σ_{j=2}^{|B|} B_j² = Φ(B), and the final term is S². Hence this chain has Σ ≥ Σ_T. Among chains, an adjacent exchange that moves a larger element earlier does not decrease any partial sum, so the descending chain maximises. ∎

Verified by exhaustive enumeration of all trees and leaf assignments for n ≤ 6 (random positive data, 80 instances) in `derivation_wp2_check.py`.

**Corollary (any sign).** For arbitrary xⱼ, Σ_v A_v² ≤ Φ(|x|) because the A_v are exactly the node sums of the tree applied to |x|. Hence for every evaluation tree,

  Σ_v t_v² ≤ f² Φ(|x|),  Φ(|x|) = Σ_{k=2}^{n} (sum of the k largest |xⱼ|)².  (3)

Φ(|x|) is order-independent, data-dependent, and computable in one sort. Φ(|x|) ≤ (n − 1)S² always; for well-conditioned data it is markedly smaller (Φ/(nS²) ≈ 0.425 for values uniform on [0.5, 1.5]; = 1/3 for equal values).

## 4. Azuma–Hoeffding with predictable bounds (no union bound)

**Lemma (standard).** If (d_k) are martingale differences with |d_k| ≤ c_k, c_k 𝓕_{k−1}-measurable, and Σ c_k² ≤ C almost surely, then P(|Σ d_k| ≥ a) ≤ 2 exp(−a²/(2C)).
*Proof sketch.* The conditional Hoeffding lemma gives E[e^{θd_k} | 𝓕_{k−1}] ≤ e^{θ²c_k²/2}, so M_k = exp(θΣ_{j≤k} d_j − θ²Σ_{j≤k} c_j²/2) is a supermartingale with E M_n ≤ 1; since Σc_j² ≤ C, P(Σd ≥ a) ≤ e^{−θa + θ²C/2} and θ = a/C gives the bound. (Freedman 1975, §1; Hallman 2021 uses the maximal form.)

Apply with d_v = δ_v t_v, c_v = u|t_v|, C = u² f² Φ(|x|):

**Theorem 1 (order-robust single-run bound).** Under the model, for every evaluation tree and every λ > 0,
  |ŝ − s| ≤ λ u f √Φ(|x|)  with probability ≥ 1 − 2 exp(−λ²/2).

No factor n appears in the probability: the whole error is one martingale, not n separate products.

**Two-run tolerance.** With α/2 per run and the triangle inequality,
  **τ_PS(α) = 2 λ₁(α) u f √Φ(|x|),  λ₁(α) = √(2 ln(4/α)).**  (4)
For α = 10⁻³, λ₁ = 4.07 (vs λ(α/2, 2²⁰) = 6.66 in the union-bounded form).

## 5. Freedman form (adds a conditional-variance bound)

If additionally E[δ_v² | 𝓕_v] ≤ σ² for every v, then V = Σ_v t_v² E[δ_v²|𝓕_v] ≤ σ² f² Φ(|x|) =: v̄ and |d_v| ≤ u f S =: c, so Freedman's inequality gives P(|ŝ − s| ≥ a) ≤ 2 exp(−a²/(2(v̄ + ca/3))). Solving 2 exp(·) = α/2:

  **τ_PSF(α) = 2a,  a = cL/3 + √((cL/3)² + 2 v̄ L),  L = ln(4/α).**  (5)

σ² = u²/3 gives the uniform-model tolerance τ_PSFa; σ² = 0.18u² (or the covering value 0.20u²) gives the calibrated τ_PSF.

## 6. What it buys (n = 2²⁰, fp32, α = 10⁻³, values uniform on [0.5, 1.5]; `python bounds.py`)

| tolerance | assumption | τ/S | tighter than τ_H |
|---|---|---|---|
| τ_H | bounded + mean-indep., union bound, S-weighted | 8.13 × 10⁻⁴ | 1 |
| τ_Fm | + calibrated variance | 3.46 × 10⁻⁴ | 2.35× |
| **τ_PS** | bounded + mean-indep. only (no variance), order-robust | 3.46 × 10⁻⁴ | **2.35×** |
| τ_PSFa | + uniform-rounding model | 2.00 × 10⁻⁴ | 4.06× |
| **τ_PSF** | + calibrated variance | 1.47 × 10⁻⁴ | **5.53×** |

τ_PS matches the calibrated Freedman gain with *no* distributional assumption: the two refinements (no union bound, partial-sum weighting) are worth as much as the variance information, and they compose. The price is the factor f = (1 + γ_{n−1})/(1 − u) from the deterministic control of the weights: 1.07 at n = 2²⁰ and 1.33 at n = 2²², so the gain shrinks as n·u approaches 1 — which is the regime boundary again.

## 7. Remarks for the paper

- (1) is exact, so τ_PS and τ_PSF carry no second-order remainder; the only approximation is the deterministic control (2), which is a rigorous inequality.
- The verifier computes Φ(|x|) from the data in O(n log n) and needs nothing about the other party's order — the quantity is a maximum over all evaluation trees.
- For positive data the lemma is tight (the descending chain attains it); for cancelling data the bound passes through |x| and is not attained, which is a source of conservatism to report.
- All stored spreads can be re-tested against (4)–(5) without new runs: `rescore.py` regenerates each cell's input from its seed.
- Open: a sharper control of Σ t_v² than (2) for n·u not small (e.g. a probabilistic (1 + γ̃) with its own union bound, or a self-normalised inequality that uses the verifier's own partial sums).
