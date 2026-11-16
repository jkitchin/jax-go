# Stage-2b nearly closed gaps (m399, m474, m6, m473), 2026-11-16

**Question.** Why do m399, m474, m6 and m473 end at the time limit when their gaps are already small?

**Short answer.**

| instance | what blocks it |
|---|---|
| m399 | the incumbent |
| m474 | the incumbent |
| m6 | the bound |
| m473 | the bound; it is not actually nearly closed, and SCIP times out on it too |

**Change.** Local searches for integer problems now start from the LP points of the most promising boxes, and are solved inside those boxes. Slightly infeasible local solves are retried with warm barrier options. OBBT only tightens variables that enter nonlinearly, plus integers.

## 1. Diagnosis

### Termination rules

- `run.py` gives `minimize` `rel_tol` 1e-4 and `abs_tol` 1e-6. `minimize` stops when `ub - glb <= max(abs_tol, rel_tol*|ub|)`.
- `report.classify` counts a run as solved when its status is optimal and its primal value is within 1e-3 of the best known.
- So the gap at which we report optimal is the same 1e-4 SCIP is given. Nothing was changed here.

### m474 (Bental5 FixedCharge1, maximize, 10 binaries): an incumbent problem

The bound reaches 6271.63 after about 50 nodes, but the base's incumbent stays at 6146.61 for the whole run. At the end of a run, the LP points of the 4 best open boxes all carry the optimal assignment `[0 1 0 0 1 0 1 1 0 0]`. Starting from those 4 points:

| method | default pounce options | `mu_init` 1e-6 |
|---|---|---|
| fix-and-solve from the LP point | 6271.10 (violation 0.14), 6271.61, 5164 (violation 1.1), 6267.9 (violation 25) | 6271.6146 in 3 of 4 |
| NLP relaxation in the box, round, then fix-and-solve | 6271.556 (violation 4.9e-6), 6268.66, 6271.54 (violation 1.5), 6268.65 (violation 3e-3) | 6271.6146 in 4 of 4 |

The base never starts a local solve at an LP point. With integers, `update_incumbent` solves the NLP relaxation of the *root* box from the kernel's candidate point. That solve slides back to the root relaxation's optimum, whose roundings have already been tried. In the base, 18 s of the 60 s go to those local solves, plus 6.6 s to polish, with no gain.

### m399 (CLay0303HRcone, convex): an incumbent problem

- With a warm cache, the bound closes to 26668.9 at about 467 nodes, whatever the incumbent.
- The incumbent 28351 comes from the root dives.
- The OA heuristic finds 26669.11 at 38–57 s, depending on load. In the cold merge run it never did.
- Cold at load 8, the root alone took 43 s.

### m6_del_asl_hvar (pure NLP): a bound problem

The terms are `h·x²`, `u·h` and `1000·(uun·yyn + uup·yyp)`. The root bound is 1.432, against the optimum 1.5364.

- **Root OBBT** is skipped by the cost rule (`_ROOT_OBBT_SHARE`). Forced on, it lifts the root bound to 1.512, but the call takes 18–20 s (2 LPs × 151 variables × 2 rounds).
- **The search stalls after OBBT.** The best box's bound sits at 1.528264 for 6 iterations, with the h variables barely split. Bound reached in 60 s, with OBBT forced (1.53627 is needed):

  | branching | bound |
  |---|---|
  | default (pseudocost) | 1.528264 (stalls) |
  | width | 1.5327, 675 nodes |
  | reliability | 1.5296, 391 nodes |
  | strong | 1.5343, 741 nodes |

  Without OBBT, the relaxation, multivariate and batch-64 variants were worse.
- **Node and root costs.** Node throughput is about 30 boxes/s, and about 85% of the search time is in the bound kernel. `bound_domain` takes 7–13 s, mostly JAX tracing in `nl` and the feasible-point local solves.
- **A separate bug:** `obbt="tree"` on m6 ran for 298 s and used 5.6 GB, ignoring the time limit.

### m473 (FixedCharge0, pure NLP, maximize)

The bound has been stuck at 7256.67 since about iteration 10 (1231 nodes), against the incumbent 7221.61. SCIP also times out (7227.9). This is a weak relaxation, not a nearly closed gap.

### Load and compile time dominate cold runs

On m474, cold at load 6, the root bound arrives at about 45 s; warm, at about 9 s. Each code edit also changes the kernel cache's source hash, so every arm of every A/B runs cold.

## 2. What changed

In `jaxglobal/bnb.py`, `lp.py` and `quadratic.py`:

1. **Node-box local searches with integers** (`_NODE_LOCAL = True`; `update_incumbent` gains an `xr` argument; `local()`).
   - With integer variables, the starts from the most promising boxes are now their LP points.
   - Each start's continuous relaxation is solved in its own box instead of the root box, then rounded at the `ROUNDINGS` thresholds and fix-and-solved as before.
   - The rounded LP point is also fix-and-solved, starting from the LP point, with `_WARM[0]` (`mu_init` 1e-6).
   - This replaces the root-box solve for those starts, so the number of starts is unchanged. Problems without integers keep the existing behaviour.
2. **`retry_warm`.** Local solves that end slightly infeasible are solved again from where they stopped, with `_WARM[0]` and then `_WARM[1]`.
   - It applies when the violation is in (feas_tol, 1e-2], the point is integral, and its value would beat the incumbent.
   - It is limited to 2 points per batch, and to 1 s plus 5% of the elapsed time.
   - It applies in `local()` and `fixed_solve`. Only screened points come out of it.
3. **OBBT on nonlinear variables plus integers** (`_Kernels.obbt_variables`; the `variables` argument of `make_quadratic_obbt` and `lp.make_obbt`).
   - A variable that enters only linearly gets from OBBT the range the LP already implies, and that range changes no relaxation.
   - On m6 this is 71 of 151 variables, about half the root OBBT call.
   - The root OBBT cost estimate now counts only the tightened variables.
   - The bounds stay weak-duality certificates.
4. **Tests.**
   - `test_quadratic.py::test_obbt_variable_subset`
   - `test_incumbent.py::test_integer_node_local_solves[True/False]`
   - `test_incumbent.py::test_slightly_infeasible_local_solves_retried`

Changes 1 and 2 only propose incumbents, and every point passes `screen`, so no lower bound is affected.

**Tried and dropped:**

- Warm options on all node-box solves: slower (57.8 s vs 50.2 s on m474).
- Keeping the root-box solve and adding the box solve: 15 nodes and 6268.75 on m474.
- One rounded-LP-point solve per iteration, outside the backoff: no gain.

## 3. Measurements

**Cold, `run.py --jobs 1`, both arms back to back, fresh caches.** Two runs each, as primal / dual; TL is the time limit. CSVs: `benchmarks/results/stage2bclose_ab_{base,new}_targets_run{1,2}.csv`.

| instance | base, run 1 | base, run 2 | new, run 1 | new, run 2 |
|---|---|---|---|---|
| m474 (max) | TL, 6146.61 / 6319.01 | TL, 6146.61 / 6272.43 | TL, **6271.6146** / 6272.67 | TL, 6146.61 / 6272.67 |
| m399 | TL, 28351.42 / 26669.04 | TL, 26669.11 / 18623.6 | **optimal 58.2 s** | **optimal 60.0 s** |
| m6 | TL, 1.53642 / 1.53077 | TL, 1.53642 / 1.53249 | TL, 1.53642 / 1.53051 | TL, 1.53642 / 1.53249 |
| m473 (max) | TL, 7221.61 / 7256.67 | TL, 7221.61 / 7256.67 | TL, 7221.61 / 7256.70 | TL, 7221.61 / 7256.70 |

Load was 0.5–5.5 during these runs.

**Warm kernel cache, final code:**

- m474: optimal in 59.0 s and 55.8 s (load 12.6–13.3); one TL with the optimal incumbent at load 16. Base: TL with 6146.61 in both runs.
- m399: optimal in 49.7 s (load 14.9) and 60.5 s (load 12.8). Base: optimal in 52–56 s at load 5–6, TL at load 13.

**A/B, 60 s cold, `--jobs 2 --threads-per-job 2`:**

| set | load | base solved | new solved | wrong | geometric mean time, both solved |
|---|---|---|---|---|---|
| int50 | 9–16 | 35 | 36 | 0 / 0 | 27.92 → 27.29 s; m3 gained, no losses |
| start50 | 5–11 | 36 | 36 | 0 / 0 | 16.49 → 14.31 s; st_e16 gained |

The start50 run also showed ramsey as lost. That was load: the new arm was still compiling at 60.1 s. Rerun alone, both arms solve it (48.4 s vs 41.1 s).

check_wrong seeds 0 and 1: no wrong answers.

## 4. Risks and what is left

- The node-box searches change which local solves run on every integer problem. The int50 A/B shows no loss, but it ran at load 9–16.
- **m474 and m399, cold:** under load, the root stages and compile take 45–55 s of the 60 s. Cutting cold root cost is now the limit: the compile, the root local solves, and the OA heuristic's 4–9 s on m474, where it is useless.
- **m6:** two ideas.
  - Root OBBT that runs in chunks and can stop at a deadline, so the cost rule need not skip it.
  - Stall-aware branching: switch a box to width or strong branching when both children keep the parent's bound.
- **m473:** needs a stronger relaxation.
