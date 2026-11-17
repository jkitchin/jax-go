# Node LP of the lifted quadratic path: block normal equations (m441, m400, m410), 2026-11-17

**Question.** Why do m410_CLay0203HRS, m441_kClus_3_10_2_HRc_1 and m400_CLay0304HRcone fail in 60 s cold, when SCIP solves them in 0.6–4 s (92 / 351 / 586 nodes), all with a root bound of 0?

## Diagnosis

**m441 (k-means, hull with rotated cones `i·z ≥ x² + y²`).**
- *SCIP ablations* (pyscipopt, nodes to optimal):

  | SCIP setting | nodes |
  |---|---:|
  | default | 351 |
  | SOC and perspective handlers off | 278–313 |
  | symmetry off | 351 |
  | propagation off | 510 |
  | separation off | 4476 |
  | no cuts at nodes | 6415 |

  SCIP wins by converging its gradient cuts at every node, not through the cone.
- *Our node bound.* On boxes that fix the optimal assignment for a share of the points:

  | Kelley rounds | 50 % fixed | 75 % fixed | all fixed (leaf) |
  |---|---:|---:|---:|
  | 2 (the assignment-model default) | 0.14 | 0.51 | 0.89 |
  | 4 | 0.17 | 0.58 | 0.994 |
  | 10 | 0.17 | 0.59 | 1.0 |

  Values are lb / optimum. A leaf must reach 1 − 1e-4 to be pruned, so with 2 rounds no leaf ever prunes.
- *Cost.* Each round was an IPM whose iterations each ran a dense 316×316 batched Cholesky: 4–6 ms per box per iteration, 1.0 s per 16-box call, with occasional 30 s outliers. The Cholesky was ~2/3 of the iteration.

**m400.** The search is not bound-limited. Node-box local solves take 23.5 of 25 s of search on the main thread and never find an incumbent; only 3 bound calls run in 60 s.

**m410 (reduced space).**
- Cold profile at load 3.3:

  | stage | time |
  |---|---:|
  | trace | 10.4 s |
  | compile | 13 s |
  | root bound arrives | 28.8 s |
  | root stages | 29–45 s |
  | – dives (of which ~5 s waiting for the dive compile, 5.5 s local solves) | 11.2 s |
  | search, 55 nodes (of which ~10 s local solves) | 15 s |

- Background OBBT and OA traces and compiles (12.7 + 18 s) compete with the main thread.
- With a primed cache it needs 255 nodes (88 s). Its node LP is not the bottleneck.

## Fix (`jaxglobal/quadratic.py`)

- **`_blocks`.**
  - The coupling graph of the normal matrix has an edge between two columns that share a row. Pool rows that may enter the separation slots are included.
  - Greedy separator: the column of most couplings in the largest component moves to the separator, later several at a time, up to N/2. The step with the fewest estimated flops is kept: `nb·bs³/3 + nb·bs²·ns + nb·bs·ns² + ns³/3`, with components packed first-fit-decreasing into blocks of the largest component's size.
  - Used when N ≥ 64 and the estimate is ≤ 1/3 of the dense Cholesky's (`BLOCK_SOLVE`, `BLOCK_MIN_N`, `BLOCK_GAIN`).
  - Results:

    | model | columns | separator | blocks |
    |---|---:|---:|---|
    | m441 | 316 | 34 | 10 × 29 |
    | m400 | 248 | 8 | 10 × 27 |
    | m399 | 153 | 6 | 6 × 27 |
    | m472 | 139 | 28 | 14 × 8 |

  - The k-means big-M models and m473/m474 keep the dense solve.
- **`_assemble` / `_arrow`.** With a static pattern, the scaled normal matrix is scattered straight into the packed `[D | B | C]` storage. Jacobi scaling is the same as before. Then per block `L_k = chol(D_k)`, `X_k = L_k⁻¹B_k`, `S = C − Σ X_kᵀX_k`, and two triangular sweeps. With run-time pool columns the matrix is built dense and gathered. A breakdown gives NaN, as with the dense factor.
- **`KELLEY_SPREAD`, default `()`.** Kelley round k also adds square tangents at `x + d·(u−l)/2^k`.

## Validity
Every bound is still `_certificate(y)` with the same `safe` margins; the linear solve only steers the IPM. One-LP bounds agree with the dense solve to ~1e-5. With Kelley rounds the tangent points can differ, because the IPM stops at a different point inside the optimal face; on one small-model box this gave 0.0122 vs 0.0075, both valid. Tests check that no bound is above a feasible point, with safe on and off. check_wrong seeds 0 and 1: no wrong answers.

## Measurements

**IPM iteration per box, batch 16, dense → block:**

| model | dense | block | non-solve part |
|---|---:|---:|---:|
| m441 | 4.1–5.9 ms | 0.30 ms | 0.19 ms |
| m400 | 1.1–4.2 ms | 0.20–0.28 ms | |

**16-box call:**

| model | dense | block |
|---|---:|---:|
| m441 | 1.03 s | 0.31–0.36 s |
| m400 | 2.0–4.5 s | 0.39 s |
| m399 | 0.46–0.58 s | 0.19–0.21 s |
| m394 | 0.12–0.15 s | 0.07–0.09 s |
| m391 | 0.075 s | 0.075 s |

**Screens, 40 iterations without a time limit (200 for start50), block off → on.** Nodes are identical on m396, m403, m406, m391, m472, kall_circlespolygons_c1p12 and kall_diffcircles_8. m399 goes 607 → 497, m394 6725 → 7501 (Kelley point variation; identical m441 screens differ by 1383 vs 1827).

int50 is unreachable by the block solve (gate script); in start50 only the two kall_* instances are reached, both screened above.

**m441:**

| configuration | load | result |
|---|---|---|
| base, 120 s | 6 | bound 0.0053, 701 nodes |
| block, 300 s | 6.5 | optimal in 198 s, 4443 nodes |
| block + spread, 300 s | 6 | optimal in 175 s, 1807 nodes |
| base, 60 s cold | 2.3 | 597 nodes, bound 0 |
| block, 60 s cold | 3 | 1021 nodes, bound 0.0052 |

**Spread screens, off → on:**
- Better: m426 lb 0.0420 → 0.0429; m391 6693 → 6537 nodes; ex1252a 8423 → 6303 nodes; m420 lb 0.63704 → 0.63721.
- Worse: m406 625 → 2441 nodes; m394 lb 8055 → 7920.
- Left off: the evidence is mixed.

## What is left
- **m441 in 60 s** needs about 3x more:
  1. **Incumbent.** The same configuration ends at 0.329, 0.392 or 0.909; with a bad incumbent nothing prunes.
  2. **Search loop.** Kernels are idle 39 of 105 s while the main thread runs local solves and dives.
  3. **Leaf convergence.** Extra Kelley rounds or spread only on boxes whose integers are all fixed, or tangents inherited from the parent's LP point.
  4. **Batch size.** The vmapped IPM runs the slowest box's iterations for all 16; the m6 analysis found smaller batches faster.
  5. **Root stages.** OBBT 7 s and dives 8 s.
- **m400.** Cap the time share of search local solves; the incumbent heuristic for CLay hull models.
- **m410.**
  1. Trace: vectorize the linear parts in nl.py, and run the alphaBB Hessian only on the nonlinear rows (~6 s).
  2. Root dives: skip the wait for the dive compile when root local solves already gave an incumbent.
  3. Cap the time share of search local solves.
  4. Gate the OA trace and compile once an incumbent exists.
- **Remaining cost per IPM iteration** (0.19 ms per box) is the segment sums and the certificate; the Cholesky is no longer dominant.
