# Specialized relaxations for Bayesian-optimization acquisition functions

Status: investigation (October 2026). Nothing here is in the solver yet. The numbers come from
`benchmarks/acq_gap.py`, run on the 15-point Matérn 5/2 GP of `tests/test_gp.py`
(target `fmin − 0.05`, 25 random square boxes per width in [−1, 1]²).

## 1. What exists

| function | where | relaxation |
|---|---|---|
| Matérn 1/2, 3/2, 5/2, SE kernels | `kernels.py` | one convex, nonincreasing function of the squared distance [schweidtmann2021deterministic] |
| GP mean `k(x)ᵀα` | `gp.gp_mean` | plain sum of kernel primitives, relaxed term by term |
| GP std | `gp.gp_std` | `√(σ_f² − ‖Wᵀk‖²)` intersected with `σ(c) ± √(2σ_f²(1 − k(d_max)))` |
| EI | `gp.expected_improvement` | jointly convex and monotone in (μ, σ); concave envelope over the (μ, σ) box |
| PI | `gp.probability_of_improvement` | monotonicity bound plus exact Φ envelopes on z = (t − μ)/σ |
| LCB/UCB | none needed | `μ − κσ`, composed |

## 2. Where the gap is now

**The EI and PI rules are no longer the bottleneck.** For each box we compare the sampled
minimum with the node bound (`make_bounder`, univariate + αBB). We also compute the best bound
that *any* rule on (μ, σ) alone could give: the acquisition at the worst corner of the exact
(sampled) μ and σ ranges. The share of the gap that remains with those exact ranges is:

| width | LCB | −EI | −PI | −log EI |
|---:|---:|---:|---:|---:|
| 0.5 | 0.16 | 0.02 | 0.04 | 0.17 |
| 0.05 | 0.33 | 0.00 | 0.01 | 0.10 |
| 0.002 | 0.53 | 0.02 | 0.08 | 0.51 |

For EI and PI, nearly all of the gap comes from how loosely **μ and σ themselves** are
enclosed. A tighter (μ, σ) → EI rule cannot help much. LCB at small widths is the exception:
there, half of the gap comes from treating μ and σ as independent, so a joint bound is needed.

**The GP mean is the loosest piece.** Width of each enclosure divided by the true range:

| width | μ, current | μ, Taylor + RKHS (§3) | σ, `gp_std` | σ, Taylor + RKHS | remainder `e_max` |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 8.8 | 4.4 | 1.8 | 3.3 | 6.8e−1 |
| 0.2 | 7.6 | 3.0 | 2.9 | 2.2 | 1.4e−1 |
| 0.05 | 6.8 | 1.65 | 3.4 | 1.4 | 1.0e−2 |
| 0.01 | 7.3 | 1.14 | 2.5 | 1.06 | 4.3e−4 |
| 0.002 | 6.6 | 1.05 | 2.5 | 1.01 | 1.7e−5 |

The current μ enclosure does not converge to the true range: it stays about 7× too wide
however small the box. `k(x)ᵀα` is a sum of N kernel terms with signed weights
(‖α‖₁ = 21 here, against ‖μ‖_H = 2.7). Each term is relaxed on its own, so the cancellation
between terms is lost.

## 3. Proposal: Taylor models with RKHS remainders

Write `δ = x − c`, with `c` the box center, and `f ~ GP`. For a kernel that is differentiable
at 0 (Matérn 5/2, SE):

* **Mean.** μ is in the prior RKHS H with `‖μ‖_H² = αᵀK_fα` (K_f is the noise-free Gram
  matrix, a constant of the trained GP). The remainder of its first-order Taylor expansion is
  `⟨μ, k_x − k_c − Dk_c δ⟩`, so

  `μ(x) ∈ μ(c) + ∇μ(c)·δ ± ‖μ‖_H · e(δ)`.
* **Std.** By the triangle inequality for standard deviations (as in `gp_std`'s existing
  Lipschitz bound),

  `|σ(x) − s̃(δ)| ≤ sd_post(f(x) − f(c) − ∇f(c)·δ) ≤ e(δ)`,

  with `s̃(δ) = sd_post(f(c) + ∇f(c)·δ)`. This is the square root of a PSD quadratic in δ
  (the norm of an affine map), so it is **convex**. It needs `k_post(c, c)`, `∂₂k_post(c, c)`
  and `∂₁∂₂k_post(c, c)` once per node.
* **Remainder, in closed form.** With `s = ‖δ/ℓ‖²` and `k = σ_f² φ(s)`:

  `e(δ)² = ‖k_x − k_c − Dk_c δ‖²_H = 2σ_f² [1 − φ(s) + 2sφ'(s) − sφ'(0)]`.

  This matches autodiff to 1e−15. It is nondecreasing in s, so its box maximum is at the
  farthest corner, and `e = O(s) = O(w²)`. Neither bound goes through `K⁻¹`'s conditioning
  except via ‖μ‖_H, which is ≤ `√(yᵀK⁻¹y)`.

Both models are second order. Intersected with the current rules (which are better on large
boxes for σ), they give tighter μ and σ for every acquisition. The structure also gives a
**joint** bound directly at the acquisition level:

* **LCB.** `μ(c) + ∇μ(c)·δ − κ s̃(δ)` is concave, so its box minimum is at a vertex. The bound
  is that minimum minus `(‖μ‖_H + κ) e_max`.
* **EI** (maximized). EI is jointly convex, nonincreasing in μ and nondecreasing in σ, so
  `EI(μ(c) + ∇μ(c)·δ − ‖μ‖_H e, s̃(δ) + e)` is convex in δ and lies above EI. Its box maximum
  is therefore at a vertex. The same holds for log EI: log is increasing, so the vertex
  maximizer is the same.

Node bounds (median of sampled min − bound; lower is tighter):

| width | LCB current | LCB vertex | −EI current | −EI vertex |
|---:|---:|---:|---:|---:|
| 0.5 | 2.3 | 3.9 | 2.2 | 1.7 |
| 0.2 | 0.65 | 0.79 | 0.28 | 0.019 |
| 0.05 | 0.13 | 0.058 | 7.5e−4 | 2.2e−4 |
| 0.01 | 0.018 | 0.0027 | 1.1e−5 | 2.9e−7 |

EI's bound improves 1.3–40× at every width, and LCB's 2–7× on small boxes. Keep the max of
both bounds (LCB is worse with the Taylor model on large boxes).

### Open points before implementing

1. **Vertices are 2ⁿ.** That is fine for typical BO (n ≤ 6–8). Above that, bound the
   convex/concave model's maximum/minimum with cheaper rules (e.g.
   `s̃(0) + ‖∇s̃‖-type` bounds, or an LP over a few linearizations), or keep the cv/cc form
   below.
2. **Pointwise relaxations, not just a node bound.** B&B cuts need cv/cc functions.
   * cc of σ: `s̃(δ) + e` is convex, so a concave overestimator needs its vertex
     interpolant or a tangent-plus-curvature bound.
   * cv of σ: `s̃(δ) − e` is already convex.
   * μ: affine ± a constant, trivial.
   These belong inside `gp_std`'s rule and a new `gp_mean` primitive, so every acquisition
   benefits through the existing EI/PI rules.
3. **Matérn 1/2 and 3/2** are not differentiable at 0 (no `∇f(c)`). For them only the
   zeroth-order version applies: `μ(c) ± ‖μ‖_H √(2σ_f²(1 − φ(s)))`, a ~2× tightening on large
   boxes (measured 3.7–5.1× against 6–9× for Matérn 5/2), and first order.
4. **Rounding.** ‖μ‖_H, `k_post(c, c)` and `s̃` involve cancellation where σ is small; `safe=True`
   needs outward widening like the other GP rules.
5. Measure with notebook 06's LCB/EI/PI solves (333 / 765 / 451 nodes now) and the
   chance-constrained case (2,807 nodes).

## 4. Other acquisition functions: what each would need

* **log EI** (Ament et al., NeurIPS 2023). The relaxation is already right: log is increasing and
  concave, so cc = log(EI.cc). The problem is numerical: EI underflows to 0 far from the data,
  giving −∞. It needs a primitive with the stable `log ψ(z)` evaluation (asymptotic expansion
  for z ≪ 0). `log ψ` is concave (ψ is the integral of the log-concave Φ), which gives exact
  envelopes in z if needed.
* **log PI** = log Φ(z) is concave and increasing, so cc = log Φ(z.cc) exactly. It needs a
  stable `log_ndtr` rule.
* **Constrained EI**, `EI · Π PI_j` (independent constraint GPs). Products of nonnegative
  monotone factors: bound the product from the factors' bounds. Tightness comes from the
  factors, so it benefits from §3 directly.
* **Thompson sampling with random Fourier features**: a sum of cosines. Generic rules apply,
  with the same cancellation problem as the mean. An RKHS-style bound needs the feature
  weights' norm.
* **Batch / Monte Carlo q-EI, knowledge gradient, entropy search**: these need the joint
  posterior over q points (a q×q Cholesky inside the relaxation) or inner optimization. They
  are out of scope for now.

Refs: [schweidtmann2021deterministic] (*Math. Prog. Comp.* 13(3), 2021); Ament et al.,
"Unexpected improvements to expected improvement for Bayesian optimization", NeurIPS 2023
(not in `references.bib`); on Taylor models with remainders in global optimization,
[bompadre2013convergence].
