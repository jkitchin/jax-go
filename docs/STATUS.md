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

Last updated: 2026-11-03, on branch `claude/epic-thompson-l8fpwd` (after
`main` ce16aa8, PR #86). Development is
paused (October 2026).

## Summary

- **Correctness is solid, with one hole found and fixed.** jaxglobal gave 0
  wrong answers on the 357 instances of the 2026-10-20 re-run (global50,
  `start50`, `int50`, held out), all checked against reference bounds; SCIP
  10.0 gave 2 and BARON 1. The 2026-11-03 scale smoke run found one: `jit1`,
  where a point that used the whole `feas_tol` on badly scaled rows was
  reported optimal 209 below the true optimum. Fixed (see *What is done*).
- **Speed is not competitive with mature solvers.** Shifted geometric mean
  time is 8–45× SCIP's cold and 3–13× warm, depending on the set, and most of its "passes" rely
  on the 10 s floor of the pass rule. Cold runs, which include JAX compilation,
  are the fair comparison, and they are where jaxglobal loses most.
- **Every solve pays a large fixed cost.** Instances solved at the root node
  take a median of 13.6 s cold (SCIP: well under 1 s), and some never finish
  setup in 60 s. At 51–100 variables this is why almost nothing solves.
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

### Scale and fixed cost

Smoke run on 2026-11-03 (this branch, 4-core cloud VM, load about 2–5, 60 s,
cold, two at a time): the first 150 of the 328 integer instances with at most
100 variables, in order of size (`sets/int100.txt`;
`results/scale_int100_cold.csv`; `benchmarks/scale_report.py`). It was stopped
there on purpose: the pattern was clear (see `CLAUDE.md`, *Measurement
strategy*).

| variables | tried | solved now | solved 2026-10-17 (same instances) |
|---|---:|---:|---:|
| 1–10 | 66 | 56 | 53 |
| 11–20 | 31 | 23 | 13 |
| 21–50 | 53 of 123 | 24 | 6 |
| 51–100 | 0 of 108 | — | 1 of 108 |

31 instances solve now that did not on 2026-10-17 and none were lost (that run
was on a heavily loaded machine, so part of the gain is load).

**Fixed cost per solve.** The 55 instances solved at the root node (1 node)
took a median of 13.6 s (3.3–59.9 s). 16 instances reached 0 nodes in 60 s
(the `eg_*` family, `autocorr_bern*` with 15+ terms, `cvxnonsep_psig*`,
`ghg_1veh`, `hmittelman`, `nvs09`) and 18 more stopped at 1 node (all
`smallinvDAX*`). On 2026-10-17 a profile of the 51–100 range found the same:
tracing and compiling the kernels took 9–37 s each, first bound calls 7–50 s
(dense IPM, rows × n² per iteration).

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

Strict feasibility of the reported point (2026-11-03): when the incumbent uses
more than 1e-3 × `feas_tol` of a constraint's tolerance, the strictly feasible
point found by the repair solve is now always reported, and the status follows
from its gap to the proven bound (`optimal` or `unproven`). Before, a repair
that raised `f` by more than 100 × `feas_tol` × (1 + |f|) was rejected and the
tolerance-exploiting point kept with status `optimal`, which reported
MINLPLib `jit1` 209 below its optimum (small variables tied to integers, with
objective coefficients up to 1e7). Test:
`tests/test_incumbent.py::test_badly_scaled_repair_never_reports_below_optimum`
(a synthetic model of the same structure). Checked on 26 other instances from
other families (`results/smoke_repair_after*.csv`): same status and node
counts as before.

Time limit during setup (2026-11-03): the limit was only checked by the
FBBT compile of `bound_domain` and at the root bound, so the uninterruptible
setup steps in between (tracing, structure analysis, building the local
solver) all ran: `eg_disc_s` (8 variables, 110k tape rows) returned after
275 s with a 60 s limit. The limit is now checked between those steps
(`out_of_time` in `minimize`, `check_time` in `bound_domain`), so a solve
overruns by at most one step (`eg_disc_s` with 75 s: 96 s). Smoke check on
six other instances (`results/smoke_deadline.csv`): unchanged.

Local solver ready-first (2026-11-03): with `local_solver="auto"` pounce is
built in a background (daemon) thread; the first local solve waits up to
`local.POUNCE_WAIT` (10 s) for it, and until it is ready the gradient-only
SciPy solver (L-BFGS-B / SLSQP) is used. pounce's build compiles the exact
Lagrangian Hessian, which on loop-unrolled code (RK4 by `lax.scan`) takes many
minutes: `gasoil_h0.1` with a 60 s limit had stayed in setup for more than
4 min; it now returns at 60 s. On 8 MINLPLib instances from other families
(`results/smoke_local_ready.csv`) status and node counts are unchanged
(ex1252a, at the 60 s edge, 8,383 nodes both ways at 120 s).

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
| `eg_*`, `autocorr_bern*` (15+ terms), `cvxnonsep_psig*`, `ghg_1veh`, `hmittelman`, `nvs09` | 0 nodes in 60 s cold: setup or compile never finishes | profile one of each; lower setup cost |
| `smallinvDAX*` (cold) | stop at 1 node in 60 s | profile the root |
| badly scaled models in general | `feas_tol` is absolute, so on rows over small variables it is worth far more than intended; the repair now reports the strict point, but the bound is still that of the relaxed problem, so such solves can end `unproven` | row- or variable-scaled feasibility tolerance |

## Failure census (2026-11-04)

`benchmarks/failure_census.py` classifies every instance of the 2026-10-20
re-run that jaxglobal did not solve **warm** (kernels cached, so not a
compile problem) by comparing its incumbent and bound with the MINLPLib
optimum (`results/failure_census.csv`). 51 instances (an instance in two
sets counted once):

| cause | count | SCIP solves | what it means |
|---|---:|---:|---|
| bound-limited | 16 | 11 | incumbent within tolerance of the optimum, the bound is not: cannot prove it |
| unsupported | 16 | 12 | no finite bounds derived for some variables (mostly tiny models SCIP solves in < 0.1 s) |
| setup-bound | 9 | 7 | at most one node in 60 s: the root never finishes even warm |
| both | 7 | 4 | neither incumbent nor bound within tolerance |
| error | 3 | 3 | crashed: `_DeadlockError` in scipy's import lock (fixed, see below) |

No failure is incumbent-limited only: where the search runs, it finds the
optimum; the bound is what is missing (23 instances bound-limited or both).
The bound-limited ones fall into families: circle packing and point packing
(kall_circles*, pointpack08, himmel16, inscribedsquare02), pooling-like
bilinear models with large dual gaps (wastewater15m1: 392, chenery: 114,
ex5_4_4: 1,885), convex MINLPs (cvxnonsep_normcon40, pcon40), integer
bilinear (tln4, tln5), and single models (tspn05, ex6_1_3, ex7_3_5, ex8_4_4,
ex14_1_7, ex1252a).

The three errors were a real bug: two threads importing `scipy.linalg` for
the first time at once (`_DeadlockError: deadlock detected by
_ModuleLock('scipy.linalg.cython_blas')`). `jaxglobal/__init__.py` now imports
the lazily used scipy modules up front; re-run now, flay03m and ex14_2_1
solve, wastewater05m1 times out at 17,053 nodes (an algorithmic failure).

## What is next

In priority order, based on the measurements above. Work from smoke tests
and profiles, not long runs, and validate each fix on instances it was not
developed on (`CLAUDE.md`).

1. **Fixed cost per solve.** The largest and most general cost: a median of
   13.6 s cold for problems SCIP solves at once, and 0 nodes in 60 s on some.
   *Profiled (2026-11-03, `profile_root.py`, nvs04, st_e15, nvs08, ex1221,
   eg_disc_s, ghg_1veh):* on 1-node instances with 2–8 variables about 23 s of
   wall time is the node kernel: trace 3.5–4 s, XLA compile 13–14 s (HLO about
   17k lines: FBBT about 7k, the bounder about 10k), run 0.05–0.1 s. Ruled out:
   XLA flags (`xla_llvm_disable_expensive_passes`,
   `xla_backend_optimization_level=1`: about 10%, and slower solves);
   compile contention (1 compile worker instead of 4: no change); splitting
   FBBT and the bounder into two executables (no gain: compile time is about
   linear in size); no single relaxation dominates (αBB, univariate envelopes
   or Kelley off: each about −30%). The levers left are structural:
   (a) problem-independent kernels (the tape as data, compiled once per shape
   bucket, so the disk cache serves new problems); (b) a cheap first root bound
   (interval / McCormick only) while the full kernel compiles, enough to
   close many root-solved problems — *measured and ruled out*: without αBB and
   univariate envelopes, 10 root-solved instances still solve at the root but
   the mean wall time is unchanged (19.4 against 19.6 s), because the dive and
   OBBT kernels compile in the same window (`results/lean_*.csv`); (c) smaller relaxation code (fewer
   primitive-rule variants per row). Large tapes (`eg_*`: 110k rows on 8
   variables) spend minutes in tracing alone, in `bound_domain` and the
   structure analysis. So (a) is the remaining lever: a redesign that passes
   the tape arrays (`relax_interp`, `fbbt_interp`, αBB, LP rows) as kernel
   arguments with padded shape buckets instead of baking them in as constants.
   A multi-day project; not started. *Spike (2026-11-03, st_e15 / nvs08,
   pieces compiled alone, batch 16):* the FBBT interpreter compiles in 0.7 s
   (straight-line: 2.8–3.1 s) and runs in 1.7 ms (0.5–0.8 ms); the interpreted
   bounder compiles *slower* than the straight-line one (full: 5.4–5.7 s against
   2.9–3.2 s) and runs 2× slower. So interpretation alone does not cut
   compile time; the gain must come from compiling an interpreted kernel once
   per shape bucket and reusing it from the disk cache across problems (also
   across the instances of one benchmark set, which share a cache directory).
   Work involved: a fixed opcode set with parameters as data (today the
   per-opcode branches are built from the opcodes a problem uses, e.g.
   `integer_pow`'s exponent), tape arrays and schedules as kernel arguments
   padded to buckets, and the same for αBB, the univariate hulls and the LP
   rows. First increment: FBBT.
   *Increment 1 done (2026-11-03): `jaxglobal/fbbt_generic.py`.* FBBT with
   the tape as kernel arguments: a universal opcode table (59 opcodes: every
   elementwise primitive of the interpreter in each operand variant,
   `integer_pow` for exponents 2–8 and −1 to −3) and a 21-opcode core table
   that covers most models; programs padded to power-of-two buckets of rows,
   slots, variables and outputs; the row interpreter's sweeps with a
   `lax.switch` per row; stops at an exact fixpoint. Exported kernels go to the
   disk cache (`kcache`), so a new process skips tracing. Bounds are bitwise
   equal to `fbbt_interp`'s row interpreter on 14 MINLPLib models and the
   tests (`tests/test_fbbt_generic.py`). 342 of 343 benchmark models
   encode; the 209 held-out models fall into 33 buckets (52 in the largest).
   Costs: a cold compile per bucket of about 2 s (core) or 5 s (full); a warm
   process gets it in 0.3 s; per call it runs 3–6× slower than the
   straight-line FBBT kernel (16 boxes: 2.7–9 ms against 0.4–1.7 ms), so it
   does not belong in the node kernel by itself. Wired into `bound_domain`
   behind `domain.GENERIC_FBBT`, **off**: `bound_domain` got faster once kernels
   were shared (17.0 against 20.7 s over six processes), but whole solves did
   not (318 against 303 s on 12 instances, two at a time,
   `results/smoke_genfbbt_*.csv`). FBBT is the cheap part of the compile;
   the gain depends on the bounder being shared too.
   *Increment 2 done (2026-11-03): `jaxglobal/bound_generic.py`.* The node
   bound with the tape as data: `relax_interp`'s McCormick rules (with
   forward-mode subgradients) row by row over the opcode table, the parts as
   signed output slots, `_cut_rows` as a mask (dead rows), and
   `make_constrained_bounder`'s LP, certificate, Kelley rounds and range
   reduction on padded variables. McCormick only (no univariate hulls, αBB,
   convexity certificates or cut pool). Relaxation values and subgradients
   equal the interpreter's to 1e-12; bounds agree to LP accuracy (the padded
   LP takes other interior-point iterates) and are valid
   (`tests/test_bound_generic.py`). Cold compile 9–12 s per bucket (the
   per-problem bounder: 5.5–9 s), reused across problems.
   *Measured end to end, and the conclusion for the redesign:* with
   `bnb.GENERIC_EARLY` (shared FBBT and bound used while the problem's node
   kernel compiles), 8 root-solved instances in sequence with one cache
   directory took 141 s against 106 s without, same results. The shared
   kernels bounded only the root (1–2 calls); root OBBT, the dives and the
   next nodes still waited for the per-problem kernels, and each new bucket
   added a shared compile. So `GENERIC_EARLY` is off. Sharing kernels pays
   only when a problem can be solved with no per-problem kernel at all: the
   whole root pipeline shared (FBBT, bound, OBBT, dives, the incumbent
   candidates) and its bounds as tight as the default ones (univariate hulls
   and αBB are what close many root gaps). That is a larger project than
   these two increments suggested; see *What is next*.
   *Evidence for the scope (2026-11-03/04):*
   - Why instances fail (the 2026-10-20 re-run, cold against warm): 26 fail
     only cold (compile-bound), 36 fail even warm (algorithmic; SCIP solves
     25 of them), 16 are unsupported. Against the pass rule, removing compile
     turns 157 failures into passes (6 → 33, 15 → 36, 9 → 27, 66 → 157 by set),
     the algorithmic items can touch at most 52: the redesign's ceiling is the
     larger, if it is finished.
   - Tightness (20 instances, warm, `results/tight_*.csv`): McCormick alone
     (no αBB, no univariate hulls) solves the same 16 of 20 as the defaults
     (319 s against 337 s). With the lifted LP, cut pool and convexity
     certificates off as well, 14 of 20 (388 s): `nous1` needs the lifted
     quadratic LP, `cvxnonsep_normcon30` the convexity certificates.
   - Which bounder models use (`benchmarks/structure_census.py`, all 357,
     `results/structure_census.csv`): the lifted quadratic/polynomial LP for
     230 (64%), convexity certificates for 97 (27%), either for 278; plain
     reduced-space McCormick only for 79.
   So a shared pipeline does not need αBB or univariate hulls, but must have
   the lifted LP (the main bounder of two thirds of the models) and the
   convexity outer approximation, plus OBBT, the dives and the local
   solver's functions.
   *Increment 3 done (2026-11-04): `jaxglobal/quadratic_generic.py`,* the
   lifted LP with the `QuadraticForm` as data (padded block layout; no
   separation pool or lattice). Bounds agree with `make_quadratic_bounder` to
   LP accuracy (`tests/test_quadratic_generic.py`).
   *Whole-pipeline measurement (`bnb.GENERIC_ONLY`, experiment):* every node
   bounded by shared kernels only (shared FBBT, then the shared lifted LP or
   McCormick bound, with `f` at the LP point as the incumbent candidate);
   root OBBT, cut pool and dives off. On the 20 instances
   (`results/tight_genonly_*.csv`): solved 11 cold / 12 warm, against the
   defaults' 14 cold / 16 warm; on the 11 solved cold by both, 162 s against
   148 s. Three causes, each measured:
   1. The shared kernels are 3–6× slower per node (row-by-row interpretation):
      inscribedsquare01 5,835 nodes in 43 s against 5,775 in 13 s with the
      problem's own McCormick kernel; kriging_peaks-red030 747 nodes in 60 s
      against 973 in 11.6 s.
   2. A cold compile of the shared kernels for a new bucket costs 20–50 s
      (mathopt5_8: 55.7 s cold, 5 s warm), which on this sample cancels the
      per-problem compiles saved.
   3. Missing pieces: cvxnonsep_normcon30 (convexity certificates), nous1
      (root OBBT or the cut pool), fac2.
   So the redesign as built does not reach the warm ceiling: interpretation
   costs per-node speed and the shared kernels are large to compile. Reaching
   it would need shared kernels that are both fast per node and cheap to
   compile, plus the missing pieces. `GENERIC_ONLY` stays off.
   *Spike on fast shared kernels (2026-11-04, time-boxed;
   `benchmarks/spikes/`): no viable design found.* McCormick bound, 16
   boxes, per call:

   | model (rows) | per-problem straight-line | shared, row by row | shared, scheduled (all opcodes per level) |
   |---|---:|---:|---:|
   | st_e15 (32) | 3.1 ms (compile 6.6 s) | 12 ms (10 s) | 66 ms (12 s) |
   | pooling_haverly1pq (80) | 9.0 ms (5.0 s) | 160 ms (12 s) | 1,022 ms (9 s) |
   | ex2_1_7 (409) | 9.8 ms (9.7 s) | 813 ms (9.4 s) | — |
   | clay0203m (384) | 52 ms (10.4 s) | 896 ms (12 s) | — |
   | kriging_peaks-red030 (1,162) | 8.5 ms (123 s) | 232 ms (8.7 s) | — |

   Cause, measured (`shared_sweep_scaling.py`): for one box and one point a
   row costs a constant 7–24 µs (loop overhead), but batched over 16 boxes ×
   17 cut points it grows with slots × variables (104, 204, 884 µs per row at
   64×8, 256×16, 1,024×32): each row update moves memory in proportion to
   the carried state (forward-mode subgradients, an S × 2 × N array per point
   and box), not to its arithmetic. Applying every opcode per level trades
   loop steps for 22× the work and is slower still. The best interpreter in
   the code (`relax_interp`'s, problem-specific widths per opcode) runs about
   2× slower than straight-line code; a shared kernel cannot have
   problem-specific widths without padding to every opcode. A faster shared
   design would need another layout (boxes innermost, block writes per level,
   subgradients by reverse sweeps): untested, and nothing measured so far
   suggests it would reach straight-line speed. **Conclusion: the kernel
   redesign is stopped.** The shared kernels stay in the code, off; what they
   showed is recorded above. Also seen: `ex1221`
   (same structure as `st_e15`) solves in 4 s cold because the existing
   per-problem disk cache, keyed by the model's fingerprint, already serves
   structurally identical models.
   Smaller waste seen on the way: `pool_judge` starts a second compile of the
   node kernel (without pool rows) even when the time left (smallinvDAX: 0.5 s)
   cannot amortize it.
   *Tried and reverted:* compiling pounce's evaluation functions in its
   background build (a one-iteration warm-up solve) instead of on the first
   local solve. Alone it helped (smallinvDAXr1b020-022: 9 → 333 nodes in 60 s),
   but in the benchmark setting (two solves at a time on 4 cores) the extra
   concurrent compile cost as much as it saved: 396 s against 389 s on 10
   instances, same node counts (`results/smoke_pounce_*.csv`). On this
   machine the cores are saturated by compilation; moving compile work between
   threads does not help, only removing it does.
2. **minlp.org smoke tests** (issue #87). *Blocked on a permission:* the
   instances are GAMS models (480 at minlp.org; 15 small nonlinear ones are
   picked, with sizes, in the session notes). `gamspy` installs GAMS with CONVERT
   (which writes `.nl`), but running downloaded GAMS code was refused by the
   session's safety check (GAMS can run shell commands). Needs the user's
   go-ahead, or `.nl` files converted elsewhere. Stage 1: 10–20 problems with ≤ 50
   variables and constraints against SCIP, to find which algorithmic gaps
   matter on applied models. Stage 2: larger problems.
3. **Flagship smoke tests** (`benchmarks/flagship.py`). The niche is the
   project's main claim, and in the 5-model smoke run jaxglobal loses or times
   out on 4. *Diagnosed (2026-11-03):* `gasoil_h0.1` (3 variables, RK4 by
   `lax.scan`) never reached a bound: the relaxation kernel of the unrolled
   integration still compiled at 60 s, its tracing starved the main thread
   (the GIL is held for long stretches while lowering: a 2 s analysis took
   34 s), and building pounce compiled the exact Hessian through every RK4
   step (> 400 s) and ran far past the time limit. L-BFGS-B with the gradient
   (5 s compile) finds the reference optimum in 0.02 s. `hx_water` (3 variables,
   `where` logic): root bound 39 s, root OBBT 13 s, no lower bound at 60 s.
   Fixed: the local solver no longer blocks setup (see *What is done*). Open:
   the relaxation kernels of loop-unrolled models (item 1), and an incumbent
   before the root bound (tried: a local solve while the kernel compiles; under
   the GIL contention its own gradient compile overran the limit, so it was
   reverted). The kernel compile threads also keep the Python process alive
   after `minimize` returns — fixed (2026-11-03): background builds run on
   daemon threads (`kcache.DaemonExecutor`); a script exits within about a
   second of `minimize` returning. Builds still running at exit are dropped;
   call `kcache.wait_for_builds` first to keep them for the disk cache.
4. **Node LP beyond polynomials.** Replace the dense IPM with a sparse or
   first-order (PDHG) LP, which the weak-duality certificate keeps safe. This
   is the main blocker at 50–100 variables and also the core of a GPU
   version.
5. **Scaled feasibility tolerance.** Make `feas_tol` relative to row and
   variable scale, so badly scaled models are neither exploited by the search
   nor left `unproven` after the repair.
6. **Stronger relaxations.** FBBT intermediate ranges fed into McCormick;
   bilinear and pooling bounds (issue #12 history; tln, wastewater, circle
   packing); alpha-CROWN for deeper NNs.
7. **Measure GPU.** The bounding kernel is already `jit(vmap)`; measure node
   throughput on a GPU before committing to phase 4 (device-resident loop).
8. **Rigorous mode.** Verified transcendental enclosures and an exact LP-dual
   check, to make `safe=True` a proof.

Done from the earlier list: the shifted geometric means were refreshed from
the 2026-10-20 data (above), and the scale run was replaced by the smoke run
above. Full-set numbers are re-run only to report a finished change.

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
