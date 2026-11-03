# Status: what is done and what is next

This is the single place for the project's current state. Update it whenever
a PR changes what is done, a measurement is re-run, or the priorities move.
The other documents keep the detail and history:

- [`ROADMAP.md`](ROADMAP.md): the original phased plan, with the measurements
  behind each phase.
- [`PLAN-scip-gap-v2.md`](PLAN-scip-gap-v2.md): the SCIP-gap effort on
  `start50`/`int50` (§10 has its final status).
- [`PLAN-acquisition-relaxations.md`](PLAN-acquisition-relaxations.md): the
  GP and Bayesian-optimisation relaxations.
- [`../benchmarks/results/README.md`](../benchmarks/results/README.md): what
  each results file is.

Last updated: 2026-11-03, at `main` ce16aa8 (after PR #86). Development is
paused (October 2026).

## Summary

- **Correctness is solid.** jaxglobal has given 0 wrong answers on every
  MINLPLib set run so far (global50, `start50`, `int50`, held out: 357
  instances), all checked against reference bounds. On the same runs SCIP 10.0
  gave 2 wrong answers and BARON 1.
- **Speed is not competitive with mature solvers.** Shifted geometric mean
  time is 8–45× SCIP's cold and 3–13× warm, depending on the set, and most of its "passes" rely
  on the 10 s floor of the pass rule. Cold runs, which include JAX compilation,
  are the fair comparison, and they are where jaxglobal loses most.
- **It stops scaling at about 50 variables.** It solved 1 of the 108 MINLPLib
  integer instances with 51–100 variables, because setup (tracing, compiling,
  the first dense LP) uses up the time limit.
- **The niche (models defined by code) is promising but not yet
  demonstrated against other solvers.** Big gains in node counts on GP and NN
  models, but the flagship comparison with SCIP, Couenne and BARON has only
  a smoke run, in which jaxglobal solved 1 of 5.

## Current numbers

### MINLPLib against SCIP

Re-run on 2026-10-20 at `main` 9812f39 (after PRs #76, #77, #81). Machine: a
4-core 2.1 GHz Xeon cloud VM, two solves at a time, 60 s limit, SCIP 10.0
through `pyscipopt` 6.3. **Pass** = jaxglobal proves the optimum within
max(2 × SCIP's time, 10 s). **Cold** includes compilation; **warm** reuses
compiled kernels. Reproduce with `benchmarks/rerun_2026_10.sh`; tables from
`benchmarks/rerun_report.py`; rows in `benchmarks/results/rerun_*.csv`.

| set | instances | solved cold / warm | pass cold | pass warm | faster than SCIP (warm) | wrong: jaxglobal / SCIP | SCIP solved |
|---|---:|---:|---:|---:|---:|---:|---:|
| global50 | 48 | 40 / 42 | 6 | 33 | 0 | 0 / 0 | 44 |
| `start50` | 50 | 37 / 40 | 15 | 36 | 1 | 0 / 1 | 38 |
| `int50` | 50 | 37 / 43 | 9 | 27 | 3 | 0 / 0 | 47 |
| held out | 209 | 169 / 180 | 66 | 157 | 2 | 0 / 1 | 204 |

The instance lists are in [`../benchmarks/sets/`](../benchmarks/sets/).
The same code on a faster laptop (58bb551, before PR #76) passed more: on
global50 it solved 45 of 48 cold, as did SCIP, against BARON's 47. Generalization
is weak: on held-out instances where SCIP needs at least 0.5 s, 10 of 26 passed
warm (laptop run), against 20 of 23 on the instances the SCIP-gap work was tuned on.

### Speed

Shifted geometric mean time (shift 1 s; unsolved instances count at the 60 s
limit), from the 2026-10-20 re-run above (`benchmarks/rerun_report.py`):

| set | jaxglobal cold | jaxglobal warm | SCIP | ratio, cold / warm |
|---|---:|---:|---:|---:|
| global50 | 25.2 s | 7.3 s | 0.56 s | 45× / 13× |
| `start50` | 22.0 s | 8.6 s | 2.70 s | 8× / 3.2× |
| `int50` | 28.4 s | 12.7 s | 1.22 s | 23× / 10× |
| held out | 17.9 s | 6.4 s | 0.55 s | 33× / 12× |

Median jaxglobal time on solved instances, warm: 3.3–9.1 s depending on the
set. A repeat solve of a built-in problem (kernel cache hit) takes 0.3–1 s.
The 2026-10-17 cold numbers quoted in the README (30.7 s against 1.93 s on
`start50`, 31.8 s against 1.35 s on `int50`) predate PR #76 and a different
machine; use the table above.

### Scale

On 328 MINLPLib integer instances with at most 100 variables (60 s, cold,
2026-10-17, heavily loaded machine; **stale**): 72 of 220 solved with ≤ 50
variables, 1 of 108 with 51–100. In the 51–100 range a profile found 0–1
nodes in 60 s: tracing and compiling the kernels took 9–37 s each, first
bound calls 7–50 s (dense IPM, rows × n² per iteration).

### Code-defined models (the niche)

| model | before | after | what did it |
|---|---:|---:|---|
| `nn_tanh` 5×8×2 | 883,471 nodes | 13,411 | CROWN bounds (`crown=True`, opt-in) |
| GP mean, N = 400 | 5,519 nodes | 725 | Taylor/RKHS model of `gp_mean` |
| GP chance constraint | not closed in 120 s | 153 nodes | `gp_std` primitive, then Taylor model |
| EI / PI acquisition | 1,401 / 727 nodes | 527 / 375 | EI and PI primitives, then Taylor models |

Flagship smoke run against SCIP, Couenne and BARON
(`benchmarks/results/flagship_smoke.csv`, 5 models, 60 s): jaxglobal solved
only `tariff_blend` (19 s; SCIP 0.03 s, BARON 0.02 s). On `pipes_parallel`,
`hx_water`, `gasoil_h0.1` and `methanol_h0.2` it timed out or was killed;
BARON solved `pipes_parallel` in 0.34 s, and nobody solved the other three.
**No full flagship run exists.**

## What is done

By [`ROADMAP.md`](ROADMAP.md) phase. "Opt-in" means implemented and tested
but not the default.

| phase | done | not done |
|---|---|---|
| 0. Benchmark harness | `.nl` reader, MINLPLib sets, `run.py`, SCIP/Couenne/BARON comparison, export to PySCIPOpt and `.nl`, unbounded-domain derivation | — |
| 1. Bound tightening | FBBT on the jaxpr every node, duality-based range reduction, batched OBBT at the root (`obbt="tree"` opt-in); 3.0× fewer nodes (geometric mean, 21 problems) | FBBT ranges fed back into the relaxations; choosing tree-OBBT variables by impact; probing on continuous variables |
| 2. Relaxations | univariate envelopes; scaled-Gershgorin αBB; convexity detection and outer approximation; sparse lifted LP for quadratic and polynomial (degree ≤ 6) problems; general lifted LP (`lifted=True`, opt-in) | multivariate McCormick; trilinear, fractional and edge-concave envelopes; SDP/eigenvalue cuts; stronger bilinear (pooling) bounds |
| 3. Search | `branch_var="auto"` (pseudocost, or width with integers); `strong` and `reliability` opt-in; `branch_point="relax"` opt-in; symmetry pruning; dives and incumbent search | exit criterion (within 2× of Couenne on ≤ 20 variables) never measured; gap-attribution branching scores; the decision on making `strong` the default (issue #11) |
| 4. Device-resident B&B | — | node pool on device, GPU LP (PDHG), multi-GPU; **GPU throughput never measured** |
| 5. Niche | CROWN bounds (opt-in); GP kernel primitives, `gp_mean`, `gp_std`, EI and PI primitives with Taylor/RKHS models; RK4 kinetics via `lax.scan`; `jaxglobal.applications` with Pyomo twins | alpha-CROWN; full flagship benchmark; parametric batches of many instances; differentiable optimal value |
| 6. Integers and rigor | MINLP (integer branching, root probing, reduced-cost fixing, pseudo-Boolean elimination, binary cuts); float64 check of every returned point; `safe=True` outward rounding (opt-in, about 1.7× slower cold) | verified transcendental enclosures; exact LP-dual check (a fully rigorous mode); MILP cuts (Gomory, cover, MIR) |

Compile and setup work (the largest cold-time cost): kernel cache in memory
and on disk, two kernels per problem instead of four, Kelley rounds as
`lax.scan`, lean `.nl` tracing, root OBBT skipped when the root closes the
gap, structural Jacobian/Hessian sparsity from the tape (`structure.py`),
interpreted relaxers for large tapes, deadline enforcement during compile.

SCIP-gap plan v2: outer approximation for convex rows (item B, PR #31) and
cuts on the lifted LP (item E, PR #34) merged, followed by per-instance fixes
in PRs #35–#67; §10 of that plan gives the final status (74 of 79 targets pass
warm, 68 cold). A3 (interpreter thresholds) has tooling but no valid timing, so
`relax_interp.RULE` stays `"rows"`. R0 (re-baseline) was superseded by the
2026-10-20 re-run. F (circle packing) was dropped.

## Known failures

| instance(s) | why | what would fix it |
|---|---|---|
| tln4, tln5 | weak RLT root bound on integer-scaled products; the bound stalls | global branching or a primal heuristic for those products |
| wastewater15m1, pooling instances in `start50`, kall_circlespolygons_c1p12/c1p13/c1r13 | bilinear McCormick bounds too weak | stronger bilinear relaxations |
| smallinvDAX*, flay04m, `ex4`, cvxnonsep_* | big-M, cardinality and convex nonlinear constraints relax weakly | MILP cuts; better outer approximation |
| autocorr_bern*, hadamard_5, nvs09 (cold) | compile time on dense polynomials | lower compile cost (A3, leaner tracing) |
| clay0303hfsg, tanksize, tspn05, flay03m (global50) | timeouts at 60 s on the cloud VM | not diagnosed individually |
| any model with 51–100 variables | setup and dense LP exhaust the time limit | sparse or first-order LP; lower setup cost |

## What is next

In priority order. The order is a proposal based on the measurements above:
measure first where the numbers are stale or missing, then attack the
largest documented cost.

1. **Refresh the stale baselines** (no library change). Re-run shifted geometric
   mean times for `start50`/`int50` and the 328-instance scale run on current
   `main`, on one machine with the load recorded (the rules in
   `PLAN-scip-gap-v2.md` §3). Every later item is judged against these.
2. **minlp.org smoke tests** (issue #87). Stage 1: 10–20 problems with ≤ 50
   variables and constraints against SCIP, to find which algorithmic gaps
   matter on applied models. Stage 2: larger problems.
3. **Full flagship benchmark** (`benchmarks/flagship.py`). The niche is the
   project's main claim, and it has only a 5-model smoke run in which jaxglobal
   loses or times out on 4. Find out why `pipes_parallel` and the kinetics
   models do not solve.
4. **Node LP beyond polynomials.** Replace the dense IPM with a sparse or
   first-order (PDHG) LP, which the weak-duality certificate keeps safe. This
   is the main blocker at 50–100 variables and also the core of a GPU
   version.
5. **Setup and compile cost.** Finish A3 (time the interpreter thresholds and
   decide `RULE`), make relaxation traces leaner, and fold remaining setup steps
   into the tape analysis. This is what separates cold from warm results.
6. **Stronger relaxations.** FBBT intermediate ranges fed into McCormick;
   bilinear and pooling bounds (issue #12 history; tln, wastewater, circle
   packing); alpha-CROWN for deeper NNs.
7. **Measure GPU.** The bounding kernel is already `jit(vmap)`; measure node
   throughput on a GPU before committing to phase 4 (device-resident loop).
8. **Rigorous mode.** Verified transcendental enclosures and an exact LP-dual
   check, to make `safe=True` a proof.

Decisions still open:

- Make `branch_var="strong"` the default? (issue #11; it needs fewer nodes but
  was slower in total warm time, 127 s against 54 s on 25 problems)
- Make `crown="auto"` the default for NN models? (it adds compile time; the
  evidence is a handful of models)
- Make `relax_interp.RULE = "ops"` the default? (waits on A3 timings)

## Updating this file

- When a PR finishes an item, move it from *What is next* to *What is done*
  and cite the PR.
- When a benchmark is re-run, replace the row, give the commit, date and
  machine, and remove its **stale** mark.
- Keep detail and history in `ROADMAP.md` and the plan documents; this file
  holds only the current state.
