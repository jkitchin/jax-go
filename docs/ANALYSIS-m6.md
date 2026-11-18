# m6_del_asl_hvar in 60 s cold (2026-11-17)

**Question.** Why does m6 (151 variables, continuous, nonconvex; SCIP solves it in 1.1 s) end at the time limit cold, with the bound short by about 1e-4 relative?

**Short answer.**
- SCIP's lower optimum is a tolerance artifact, so the incumbent is already optimal.
- Our bound stalled because pseudocost branching split near-fixed complementarity variables. The McCormick-gap weights that prevent this were applied only on quadratic forms, and m6 has a polynomial form.
- The cold setup spent ~11 s waiting for a bound-kernel compile that only started after `bound_domain`.

Fixing both brings m6 from 0/4 to 3/4 solved cold at load 6–7.

## 1. Diagnosis

### The incumbent is optimal
- SCIP reports 1.5358754, and we report 1.5364152. Both points have the same h, u and y.
- SCIP's complementarity variables sit about 2e-8 below their bound of 0, inside its feasibility tolerance. Times the 1000 penalty weight, the penalty becomes −5.4e-4.
- Our box is exact, so 1.5364152 is the true optimum. The bound must reach 1.536262 (rel_tol 1e-4).

### The relaxation is not the gap
- Root bound: 1.432 for us (1.512 with root OBBT), against 1.339 for SCIP after cuts.
- Nodes to close: we close the stricter gap in 1300–1800 nodes; SCIP closes its looser one in 1507.
- Throughput is the gap: about 30 nodes/s against SCIP's ~1400 (see section 4).

### Branching stall
- Traced on the best box (warm cache, 120 s limit, root OBBT on), the global bound sat at 1.5277566 from iteration 10 to iteration 30.
- In each of those iterations pseudocost split a complementarity variable (x20–x59 or a u variable) of absolute width about 1e-5.
  - `bound_domain` had nearly fixed these variables using the incumbent cutoff.
  - Measured against that tiny root width, their relative width is 0.3–0.6.
  - The LP point sits at their bound, so one child keeps the parent's bound. The other child is pruned, which looks like a large gain in the pseudocosts.
- The McCormick-gap weights (`_gap_weights`) score such a variable near zero, because the envelope is exact at a bound. But they were applied only when `K.quadratic` exists, and m6's lifted LP is a polynomial form (51 products, h·y²).

### Cold setup (load 6–7)
- `bound_domain` took 11 s: the FBBT kernel compile, the pounce build plus the first solve's JIT (about 4 s), LP bounds (1 s) and split bounds (0.7 s).
- The bound kernel's trace and compile then started, and the root bound arrived at about 25 s.

## 2. Changes (`jaxglobal/bnb.py`)

1. **Gap weights on polynomial forms (`_gap_scores`, `_poly_prep`, `_GAP_POLY`).**
   - A product whose factor is a product column takes the column's exact value at the relaxation point and its interval bounds.
   - The product's gap counts for every variable of the monomial.
   - It is computed level by level, and spread over the variables with a sparse support matrix.
   - The quadratic case is unchanged. Only branching scores are affected.
2. **Speculative kernels (`_SPECULATE`, `_speculate`, `_kernel_opts`).**
   - On a constrained, continuous problem with unbounded variables, the kernels are built on a background thread before `bound_domain`, with no cut pool.
   - If the lifted LP applies, the bound kernel's trace and compile start there.
   - `minimize` uses those kernels if its own options agree up to `cut_pool`/`exact`, which the lifted LP ignores. Otherwise the cost is CPU only.
   - It is restricted to continuous problems. One integer model under heavy load (m391, load 13–15) lost its root-dive incumbent with it.
   - `_cached` now locks its dictionary.

## 3. Measurements

| | base | new |
|---|---|---|
| m6 cold, `run.py --jobs 1`, 4 back-to-back pairs (load 6.3–9.2) | 0/4; bound 1.5262–1.5305 | 3/4 at 55.7–58.2 s; 1 time limit at 1.535957 |
| m6 warm, nodes / iterations to close | 2813 / 60 | 1777 / 36 (1311 cold) |
| m6 cold, search start | about 25 s | about 16–19 s |

**Tree OBBT:**
- Warm: 1287 nodes against 1777, with the same iterations and wall time.
- Cold: worse (time limit at bound 1.53548–1.53579).
- Not made automatic.

**MINLPLib A/B on the instances the change can reach (5 int50, 14 start50):**
- Same status and incumbents.
- Node counts within noise; ex1252a went 6353 → 5943, alkyl 105 → 87.

**stage2b:**
- Only m473 is reachable: identical bound and nodes.
- The other 17 run identical code paths.

check_wrong seeds 0 and 1: clean.

## 4. What is left

- **Node throughput.**
  - The node LP is a dense 202×202 Cholesky IPM; the normal matrix has 1050 nonzeros out of 40,804.
  - The vmapped IPM loop runs the batch maximum: 70.8 iterations executed per node against 22.4 of its own.
  - A 2-box iteration costs a full 16-box call.
  - Measured warm (same 1777 nodes, but not at equal load): `batch_sizes=[4]` 40.7 and 48.8 s, against 63.5 s for 16. Left alone: this is global and belongs to the node-LP work.
- **Multi-variable splits** in a call's padded slots when few boxes are open, to cut iteration depth (about 36 sequential iterations on m6). Not tried.
- **`bound_domain` is still 11–15 s cold,** mostly the FBBT compile and the pounce first-solve JIT.
- **The margin on m6 is thin at load ≥ 7.**
