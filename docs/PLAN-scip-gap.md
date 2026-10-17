# Plan: closing the jaxglobal vs SCIP gap on MINLPLib (start50 / int50)

Written 2026-10-17. Each workstream below is meant to be carried out by one
agent on its own branch, with an independent reviewer agent per PR (section 9).

Baseline (cold, 60 s, 4 jobs): start50 31/50 vs SCIP 44; int50 29/50 vs SCIP
45; SGM 32.6 s vs 1.9 s and 35.9 s vs 1.35 s; 0 wrong answers
(`benchmarks/results/issue14_start50_cold.csv`, `issue14_int50_cold.csv`,
`int50_scip.csv`).

## 0. What the data says

Every unsolved instance, with size, free variables, scalarized tape rows,
whether `quadratic_form` accepts it, and the warm int50 run
(`int50_probe_warm.csv`):

| instance | n | int | free | rows | quadratic path on? | cold result | warm result / SCIP |
|---|---|---|---|---|---|---|---|
| ex6_2_6 | 3 | 0 | 0 | 184 | no | 1 node, 60 s | SCIP < 1 s |
| ex8_4_5 / ex8_4_4 | 15/17 | 0 | 0 | 142/127 | no | 1 / 0 nodes | SCIP fast |
| ramsey | 33 | 0 | 21 | 176 | no | 1 node, gap 3e-4 abs | SCIP 0.01 s |
| portfol_buyin | 17 | 8 | 9 | 943 | no | 1 node, 74 s | warm: optimal 40 s, 95 nodes |
| batch | 46 | 24 | 0 | 388 | no | 1 node | warm: optimal 34.5 s, 19 nodes; SCIP 0.21 s |
| hadamard_4 | 17 | 16 | 1 | 101 | no (degree 4) | 1 node | warm: optimal 11 s, 429 nodes; SCIP 0.2 s |
| csched1a | 28 | 15 | 10 | 179 | no (div, exp) | 1355 nodes, weak | warm: optimal 47 s; SCIP 4 s |
| flay04m | 42 | 24 | 0 | 308 | no (div) | 1 node, 79 s | warm: 15 nodes in 61 s (4 s/node) |
| smallinvDAXr3b/r4b/r5b | 31 | 30 | 31 | 1583 | no (pow) | 0 nodes, incumbent found | warm: still 0 nodes; SCIP 1.3-1.5 s |
| cvxnonsep_nsig30 / pcon40 | 30/40 | 15/20 | 0 | 122/199 | no (pow) | 0 nodes, nan | SCIP fails nsig30, pcon40 too |
| cvxnonsep_normcon30/40 | 30/40 | 15/20 | 0 | 123 | no (sqrt) | 2303 / 763 nodes, bound -54 vs -34 | warm: 3839 nodes still -51.6; SCIP 3 and 5 nodes, 0.1 s |
| hadamard_5, autocorr_bern20-10, _30-04 | - | all | - | 541-3003 .nl ops, 480-2534 mul | no (degree 4) | 0 nodes | SCIP 5-31 s |
| tln4 / tln5 | 24 / 35 | all | 0 | 184 | yes | 24k / 11k nodes, 8.4 vs 3.05 | warm same; SCIP 7 nodes / 1 node |
| wastewater15m1 | 46 | 0 | 0 | 359 | yes | 7679 nodes, no incumbent | SCIP 2446 |
| ex5_4_3 | 16 | 0 | 0 | 86 | no | 3705 nodes, no incumbent | SCIP 4845.46 |
| st_e16 | 12 | 0 | 0 | 77 | no | 3749 nodes, incumbent 12305 worse than SCIP's 12292 | |
| kall_circles_c6a/b/c, c7a, diffcircles_8 | 14-28 | 0 | 0 | 150-260 | yes | 1.3-2.8k nodes, dual stays at 0 | SCIP fails c6a, c6c; c6b 38 s, c7a 59.6 s |
| kall_circlespolygons_c1p12 | 43 | 0 | 0 | 283 | yes | 5375 nodes 0.34 / 0.19 | |
| ex8_4_1 | 22 | 0 | 0 | 120 | yes | 8187 nodes, 0.6186 / 0.6080 | |
| ex6_1_1, ex6_1_3 | - | 0 | 0 | - | no (log) | 9359 / 3281 nodes, weak | |
| m3, clay0203m, launch, ex8_5_3, least | 26/30/38/5/3 | 6/18/0/0/0 | 12/6/24/5/2 | | clay: quad | unsupported (free variables) | SCIP 0.24 / 0.58 s; least also fails in SCIP |

(hadamard_4 "dual 8.68 > primal 3.0" is a maximization problem, `O0 1` in the
.nl; `nl.py:76` flips sign. Not a soundness problem.)

Conclusions:

1. **Setup + compile at the root is the single largest bucket** (14-16
   instances across both sets): ex6_2_6 has 3 variables and 184 tape rows and
   still spends 60 s on one node; portfol_buyin, batch, hadamard_4, csched1a
   all *solve* warm within the limit and fail cold; ramsey's root gap is
   already 3e-4 absolute. The stage-5 profile in `docs/ROADMAP.md`
   (trace/lower 9-37 s, compile 9-25 s, first bound 7-50 s, local solver build
   up to 24 s) says the same. This bucket also dominates the SGM gap, which is
   mostly fixed per-instance overhead, not node throughput. No LP improvement
   helps here.
2. **Convexity is not exploited**: cvxnonsep_normcon30/40 are convex MINLPs
   that SCIP solves in 3-5 nodes; jaxglobal's bound after 3839 nodes is still
   50 % off because McCormick of `sqrt(sum of squares)` composes
   concave-increasing after convex and loses convexity, and the whole-function
   αBB (`relax.alphabb_rule`, `relax.py:674-705`) disables itself when the
   interval Hessian is unbounded (sqrt near 0 gives `ok_a = False` → `cv =
   flo`). portfol_buyin, batch (exp), flay04m, cvxnonsep_* and clay0203m are
   convex-constrained. OA cuts on detected-convex rows are a cheap,
   high-confidence lever (6-8 instances), independent of the lifted LP.
3. **The sparse lifted LP is already on for 10 of the 17 "searched but did not
   close" instances** (all kall_*, wastewater15m1, tln4/5, ex8_4_1, clay0203m)
   and they still fail. So a general lifted LP alone is not the biggest lever;
   it is a *prerequisite* for cuts (OA on lifted columns, RLT, degree-4
   products in hadamard/autocorr) and for div/exp/log/pow problems (csched1a,
   flay04m, ex8_4_x, st_e16, ex5_4_3, ex6_1_x, m3). Build it as a
   generalization of `quadratic.py` (static sparse pattern, `_ipm`,
   `_certificate`), not of `lifted.py` (dense, per-call tracing of relax
   rules).
4. **Two instances have no incumbent after thousands of nodes** (ex5_4_3,
   wastewater15m1: equality-constrained NLPs). st_e16's incumbent is worse
   than SCIP's primal. Incumbent search is a separate, cheap lever (2-3
   instances).
5. **Unsupported due to free variables**: m3 and clay0203m are layout problems
   whose free variables are bounded by big-M linear rows; `bound_domain`'s
   FBBT does not derive them; an LP-based bound derivation on the linear rows
   would (2 instances; launch/ex8_5_3 maybe; least is unsolvable for SCIP
   too).
6. Circle packing (kall_*) is hard for SCIP as well (fails 2 of 5, 38-60 s on
   the rest). Lowest return; drop first.

Ordering of levers by (expected instances gained) / effort: **A
setup/compile (de-risk first) > B convex OA cuts > C incumbent + domain
derivation > D general sparse lifted LP > E cuts on the lifted LP (RLT,
degree-4, integer) > F tree OBBT / branching for circle packing.**

## 1. Workstream 0 (first, 1 agent, 1 day): de-risking profile

Goal: split wall time at the root for stuck instances into trace/lower, XLA
compile, `bound_domain`, local-solver build, `nonlinear_variables`/structure,
first bound call, dives/enumeration. Without this, Workstream A is guesswork.

- Instances: ex6_2_6 (3 vars; pure compile pathology?), ramsey (tiny gap at
  root, 21 free vars → `bound_domain` cost?), portfol_buyin (943 rows, warm
  solves in 40 s), smallinvDAXr3b010-011 (1583 rows, 31 free, 0 nodes warm and
  cold), batch, hadamard_5 (480 mul, 0 nodes), cvxnonsep_nsig30 (0 nodes, 122
  rows: why?).
- Method: run `benchmarks/run.py --worker <name> --time-limit 120` with
  `jax.log_compiles(True)` and the existing `_Precompiled` timing
  (`bnb.py:225-300`); time `bound_domain` (`bnb.py:752`), `make_local`
  (`bnb.py:826`), `K.nonlinear`/`nonlinear_variables` (`bnb.py:817-819`), the
  root `evaluate` (`bnb.py:988`), root dives + `_ENUM` (`bnb.py:997-1010`),
  root OBBT (`bnb.py:1011`), probing (`bnb.py:1024`). Also record
  `use_interpreter` / `use_relax_interpreter` decisions (`fbbt_interp.py`
  `SCHEDULE_MIN_ROWS=300`, `relax_interp.py` `MIN_ROWS=600`) and whether
  `alphabb` prepare falls back to `jacfwd(jacfwd)` (`alphabb.py:60-140`,
  `TAYLOR_BUDGET=30_000`). Cold (`JAXGLOBAL_CACHE=0`) and warm.
- Deliverable: a table in `docs/ROADMAP.md` (new subsection "SCIP gap: root
  profile"), CSV `benchmarks/results/scipgap_profile.csv`. No solver change.
- Exit: every one of the 7 instances has a time split summing to within 10 %
  of wall time; the plan for A is adjusted to the two largest buckets.
- **Done**: `docs/ROADMAP.md` "SCIP gap: root profile". Compile is ≥ 30 % on
  5/7 and the first LP ≤ 1.6 %, so A stays first. Section 2 is revised.
- Risk being measured: if compile is < 30 % everywhere and the first LP (dense
  `lp.ipm`, `lp.py:147`) dominates, Workstream D moves ahead of A.

## 2. Workstream A: setup and compile cost (1 agent, 1-2 weeks; critical for SGM)

**Workstream 0 result** (`docs/ROADMAP.md`, "SCIP gap: root profile";
`benchmarks/results/scipgap_profile.csv`).

A stays ahead of D:

- Kernel compile is 31-97 % of the cold wall time on 5 of 7 instances.
- The first LP takes 0.1-2.1 s, at most 1.6 % of the wall time.

The two largest buckets:

1. XLA compile of the bound and OBBT kernels. The cost is heavy-tailed:
   hadamard_5 takes 273 s and 301 s, cvxnonsep_nsig30 32 s and 42 s. It is
   driven by alphaBB's straight-line `jacfwd(jacfwd(f))` interval Hessian;
   with `alphabb=False`, compile is 7× and 2× faster.
2. Root dives on smallinvDAX: 36-50 s for one dive kernel call.

Re-baselining:

- On a quiet machine, ramsey, portfol_buyin and batch already solve cold,
  both at 1 job / 120 s and at 4 jobs / 60 s.
- The `issue14` cold baseline was taken under heavy load. Re-run it, with load
  recorded, before measuring A.
- ex6_2_6 is not a setup problem: first bound at 6 s, then 110k nodes stuck
  at a gap of 1e-5 near a zero optimum. It leaves A's targets.

Goal:

- Cold start to first bound under 10 s for every n ≤ 50 instance. Today it is
  286 s on hadamard_5 and 38 s on nsig30.
- No run overshoots `time_limit` by more than one kernel call.
- hadamard_4/5, csched1a and flay04m reach the search cold.

Owned files:

- `jaxglobal/bnb.py`: `_Precompiled`, `_Kernels`, the `minimize` setup
  section, `evaluate`/`run`, `dive`.
- `jaxglobal/kcache.py`
- `jaxglobal/relax_interp.py`
- `jaxglobal/fbbt_interp.py` (thresholds only)
- `jaxglobal/alphabb.py`
- `jaxglobal/local.py` (build cost)

Design, ordered by measured payoff:

1. **alphaBB interval Hessian (new, first).** `alphabb.make_alphabb` traces
   `jacfwd(jacfwd(f))` and evaluates it in interval arithmetic straight-line.
   It does this in two cases:
   - on every straight-line path;
   - on the interpreted path when `make_taylor2` exceeds `TAYLOR_BUDGET`
     (portfol_buyin 0.5M, smallinvDAX 5.0M, hadamard_5 0.7M, nsig30 0.2M).

   This graph dominates XLA compile. Options, to be measured on
   hadamard_4/5, nsig30, pcon40 and autocorr:
   - (a) skip alphaBB when the Hessian jaxpr size × n² is over a budget;
   - (b) a taylor2 that keeps only the entries of `structure.py`'s Hessian
     pattern, so it fits the budget;
   - (c) per-row alphaBB on small rows only;
   - (d) alphaBB in a second kernel that replaces the first once compiled
     (item 2).

   Soundness: dropping the alphaBB term only weakens the relaxation, because
   the McCormick and univariate terms are still valid.
2. **Cheap root bound before the full kernel compiles (revised).**
   - A batch-1 kernel is *not* cheaper to compile: OBBT at batch 1 takes 301 s
     against 273 s for the bound kernel at batch 16 on hadamard_5.
   - The cheap kernel is the same `node` function without the alphaBB term and
     with the relaxation interpreter. It compiles first, and the full kernel
     keeps compiling in the background and replaces it when ready.
   - Measured on first bound: hadamard_5 286 → 48 s, nsig30 38 → 15-22 s.
   - Soundness: both kernels give valid bounds, and boxes bounded by the cheap
     kernel stay valid.
3. **Deadline enforcement (new).**
   - `minimize` checks `time_limit` once per iteration, but one iteration is
     up to 16 kernel calls of 16 boxes (21-28 s on nsig30 and smallinvDAX).
     The dive kernel has no deadline at all.
   - Measured overruns: 14-20 s on nsig30, 15 s on smallinvDAX cold, and
     143 s on smallinvDAX warm.
   - Fix:
     - check the deadline between the calls in `run` (an unbounded box stays
       open with its parent's bound);
     - skip a dive when the last dive's duration exceeds the time left.
4. **Dive cost (new).**
   - On smallinvDAX one dive call (16 boxes × 30 integers × 2 roundings ×
     interpreted FBBT over 1583 rows) takes 36-50 s, about 20 bound calls.
   - Budget dives by measured cost, for example fewer boxes or the
     integers-per-call cap.
5. **Relaxation-interpreter threshold: on graph size, not rows.**
   - On nsig30 (122 rows) the interpreter cuts compile 31.6 → 12.0 s and
     raises the node rate 7×.
   - On ex6_2_6 (184 rows) it halves the node rate.
   - Choose by straight-line jaxpr size (or op count × Kelley rounds), and
     measure the crossover on about 10 instances.
6. **Cache: an unfinished compile is never cached.**
   - hadamard_5 warm pays the full compile again because the XLA persistent
     cache writes only on completion. Items 1 and 2 fix this.
   - nsig30 warm still waits 5.6 s on compile: check why.
   - The cache key itself works: the warm first bound comes in 0.7-2.6 s on
     the others.
7. **Local solver build: low priority.** `make_local` takes at most 2.2 s and
   the root local solves at most 5 s (pounce jit on first use).

Dropped after Workstream 0:

- Kelley as `lax.scan` in the lifted path: none of the 7 uses the lifted or
  quadratic path. Revisit with D.
- Fingerprinting the Program instead of the Hessian jaxpr: structure,
  nonlinear, quadratic and fingerprint take under 1 s on every instance.

Stays pure jit/vmap: the cheap kernel is the same `node` function with one
relaxation term switched off; no host logic inside.

Tests:

- A test that `minimize` on a small degree-4 integer problem returns a finite
  bound within N seconds cold with `JAXGLOBAL_CACHE=0`. This is a regression
  guard with a generous bound.
- A test that `minimize` returns within `time_limit` plus one kernel call when
  the search and dives are still running (deadline enforcement).
- Bounds with the cheap kernel are ≤ bounds with the full kernel on the fuzz
  problems.
- The `tests/test_fuzz.py` variants must pass.

Measurement:

- `python benchmarks/profile_root.py --names hadamard_5,cvxnonsep_nsig30,smallinvDAXr3b010-011,portfol_buyin,batch,ramsey,ex6_2_6`
  before and after: first bound time, overrun past `time_limit`, and the
  compile buckets.
- `python benchmarks/run.py --solvers jaxglobal --tag scipgap_A
  --time-limit 60 --names ex8_4_5,ex8_4_4,portfol_buyin,batch,hadamard_4,hadamard_5,csched1a,flay04m,cvxnonsep_nsig30,cvxnonsep_pcon40,smallinvDAXr3b010-011,autocorr_bern20-10`,
  cold and warm.
- Then the full start50 and int50 cold runs, against a re-baseline taken on
  the same machine with load recorded.

Exit:

- First bound under 10 s cold for every n ≤ 50 instance, against 286 s on
  hadamard_5 today.
- No overrun of `time_limit` beyond one kernel call.
- Solved, counted against the re-baseline rather than `issue14`:
  - hadamard_4 and ex8_4_5, with high confidence;
  - csched1a and flay04m, with medium confidence.

  ramsey, portfol_buyin and batch already solve on a quiet machine.
- SGM time reduced by ≥ 1.5× on both sets.
- Hard gate: 0 wrong.

## 3. Workstream B: outer-approximation cuts for convex rows (1 agent, 1-2 weeks; independent of A and D)

Goal: convex constraints (and convex objectives) get exact tangent-plane cuts
at every LP point so convex MINLPs close at the root as they do in SCIP.

Owned files: new `jaxglobal/convexity.py`; `jaxglobal/lp.py` (`_parts_fn`,
`_cuts`, `make_constrained_bounder` cut assembly, `lp.py:207-400`);
`jaxglobal/relax.py` only to export helpers.

Design:
- `convexity.py`: a static walk of the scalarized `Program`
  (`fbbt_interp.scalarize`) with DCP-style rules: affine rows → affine; `exp`,
  `square`, even `integer_pow`, `pow` with exponent ≥ 1 on a nonnegative
  domain, `-log`, `-sqrt`, `div` by a positive constant, `max` → convex (given
  input curvature and monotonicity); sums and positive scaling preserve
  curvature; `sqrt(convex)` is not convex but `sqrt(sum of squares)` is:
  detect `sqrt(sum(square(affine)))` explicitly (cvxnonsep_normcon). The
  verdict is per output row, using the FBBT box for sign conditions
  (domain-dependent flags are traced `jnp.where(lo >= 0, ...)` gates, so the
  function stays jit-pure; the candidate set of rows is static).
- Cut: for a row g_i flagged convex on the box, add `g_i(x_k) + ∇g_i(x_k)·(x −
  x_k) ≤ 0` at the LP point of each Kelley round (and the objective as `t ≥
  f(x_k) + ∇f(x_k)(x − x_k)` when f is convex). Reuse the `_cuts` jacfwd
  machinery (`lp.py:207`) by replacing the row's McCormick `cv` with the
  function value when `convex_i` is true (`cv = jnp.where(convex_i, g_i(x),
  mc.cv)`), which also makes αBB's `ok_a` failure irrelevant for those rows.
  Safe mode: evaluate `g_i(x_k)` in interval arithmetic at the point (as
  `alphabb.apply` already does) and round the constant outward.
- Also apply inside Workstream D's lifted LP: a lifted column `w = φ(u)` with φ
  convex gets tangent rows `w ≥ φ(u_k) + φ'(u_k)(u − u_k)` (as `quadratic.py`
  already does for squares).

Soundness: a tangent of a convex function is a valid underestimator; the
certificate (`lp.dual`, `quadratic._certificate`) is unchanged. The risk is a
wrong convexity verdict, so the detector is tested adversarially: random DCP
expression trees with known curvature, a numerical Hessian PSD check at 1000
random points per box, and a fuzz test that `g_i(x) ≥ tangent(x)` for all
flagged rows. Build the fuzz before writing any OA cut.

Tests: `tests/test_convexity.py` (rules, the norm pattern; non-convex
compositions must return False: `sqrt(x)`, `x*y`, `exp(-x^2)`; `log(sum exp)`
must come out convex), `tests/test_constrained.py` cut-validity test extended
to the OA rows; `tests/test_fuzz.py` gets a convex-constrained generator.

Measurement: `--names cvxnonsep_normcon30,cvxnonsep_normcon40,cvxnonsep_nsig30,cvxnonsep_pcon40,portfol_buyin,batch,flay04m,clay0203m,syn05m02m,meanvarx`
with `--tag scipgap_B`, then the full sets.

Exit: cvxnonsep_normcon30/40 solve in < 50 nodes (currently 2300-3800); +3 to
+5 solved (normcon30/40 high confidence; nsig30/pcon40 medium, and they need A
first since they are 0-node instances). 0 wrong.

## 4. Workstream C: incumbents and domain derivation (1 agent, 1 week; independent)

Goal: a feasible point for every instance SCIP finds one for; free variables
bounded from linear rows so m3, clay0203m (and maybe launch) are no longer
"unsupported".

Owned files: `jaxglobal/domain.py`, `jaxglobal/bnb.py` only the `local()` /
`_feasible_points` call sites (`bnb.py:829-860, 898-910, 978`),
`jaxglobal/local.py` (fix-and-solve options).

Design:
- C1 **LP-based bound derivation for free variables** in `bound_domain`:
  extract the affine rows of the joint tape (`lp._affine_parts`) plus variable
  bounds, solve min/max x_j with HiGHS (`scipy.optimize.linprog`, host side,
  once at setup) for each free j; then rerun FBBT. Nonlinear rows are ignored
  (a relaxation, so sound). If some variables remain free, fall back to the
  current `"unbounded_domain"` status.
- C2 **Equality-constrained feasibility search**: ex5_4_3 (13 equalities, 16
  vars) and wastewater15m1 (28 equalities, 46 vars) never get an incumbent.
  Use pounce with a feasibility objective (min ‖h‖²) from `n_starts` Latin
  hypercube points and from the LP point of each node; add a periodic
  feasibility pump every `_LOCAL_BACKOFF` nodes while `ub = inf`.
- C3 **Polish the incumbent**: st_e16's incumbent 12305 is above SCIP's 12292;
  run the local solver from the best few LP points on the tightened box each
  time the bound improves.

Tests: `tests/test_domain.py` (LP-derived bounds are valid: for random LPs
with implied bounds the derived box contains all feasible points; an
infeasible linear part gives infeasible); existing integer domain tests still
pass; a new test that ex5_4_3 gets a feasible point within 10 s.

Measurement: `--names m3,clay0203m,launch,ex8_5_3,ex5_4_3,wastewater15m1,st_e16,ramsey,portfol_buyin,smallinvDAXr3b010-011`.

Exit: m3, clay0203m become supported (high confidence) and solve (medium
without B); ex5_4_3 gets an incumbent and closes (+1, high confidence: its
dual already equals SCIP's optimum); wastewater15m1 gets an incumbent (closing
it also needs D/E). Expected +2 to +4. 0 wrong, and `bound_domain` must still
reject genuinely unbounded problems (least).

## 5. Workstream D: general sparse lifted LP (1 agent, 3-4 weeks)

Goal: one sparse lifted LP for any scalarized tape, generalizing
`quadratic.py` and retiring the dense `lifted.py`; rows are the per-node
relaxation of every non-affine Program row; solver is the sparse-row IPM with
the N×N Cholesky normal matrix (`quadratic._ipm`); certificate is
`quadratic._certificate`.

Owned files: new `jaxglobal/lifted_sparse.py`; `jaxglobal/quadratic.py`
(factor `_Pattern`, `_layout`, `_unit`, `_scale`, `_certificate`, `_ipm` into a
shared `jaxglobal/sparse_lp.py`); `jaxglobal/bnb.py` dispatch in
`_Kernels.__init__` (`bnb.py:314-332`) and `lp_tighten` (`bnb.py:340-345`).
`jaxglobal/relax.py` unchanged (rule functions reused as point evaluators).

Design:
- **Columns**: x (n) plus one column per non-affine Program row. Column bounds
  from the tape FBBT. Affine rows are folded into a static sparse matrix `E`
  mapping columns to outputs (f, g, h) as `QuadraticForm.E` does, but sparse.
- **Rows per lifted column** (static count, traced values):
  - `mul(u, v)`: 4 McCormick facets. Products of lifted columns (degree 4 in
    hadamard/autocorr) are `mul` rows whose inputs are other columns.
  - `div(u, v)`: lift as `w·v = u`, McCormick on the bilinear `w·v` with `w`
    bounded by FBBT (sound when v's interval excludes 0, which FBBT
    guarantees, or the row is dropped).
  - univariate convex/concave (`exp`, `log`, `sqrt`, `pow`, `integer_pow`,
    `square`): a secant row on the concave side plus tangent rows (box ends,
    midpoint, and the LP point of each Kelley round) on the convex side;
    nonconvex univariates (odd powers, `sin`) use `cv/cc` and subgradients
    from `relax.apply_rule` with point MC inputs (as `lifted.node_cuts`
    does).
  - Dead rows (`0 ≤ 1`) when a column is fixed, as `quadratic._unit` does.
- **Kelley rounds as `lax.scan`** over a fixed buffer of tangent rows; the LP
  point of round k gives the tangent points of round k+1; Workstream B's OA
  rows at the same points.
- **Warm start**: the parent node's `(x, y, s)` as the child's IPM starting
  point, pulled back to the interior. Measured separately; the certificate
  does not depend on it.
- **Range reduction** (`reduce_ranges`) on x columns from the same duals;
  lifted OBBT for `obbt="root"`.
- **Size guard**: count non-affine rows only, and set the column ceiling from
  a measurement of the N×N Cholesky cost (risk 2 below). Problems above it
  fall back to the reduced-space bounder.
- Keep `lifted` tri-state (`False`, `"quadratic"`, `"sparse"`) and benchmark
  before changing the default.

Soundness: every row must hold at the lifted point `(x, w(x))` for every x in
the box, as `tests/test_quadratic.py` checks for RLT rows; replicate per
opcode with random boxes and 1000 points, and compare the LP optimum with
HiGHS on the same rows (`highs_bound` helper in `tests/test_quadratic.py`).
The bound reported is always the certificate value, never the IPM primal.

Tests: `tests/test_lifted_sparse.py` (row validity per opcode including div
and degree-4 products; HiGHS agreement; lifted bound ≥ reduced-space bound on
haverly and nous1; infeasible box pruned; safe-mode rounding);
`tests/test_fuzz.py` gains `lifted="sparse"` with and without `safe=True`.

Measurement: `--names nous1,hadamard_4,hadamard_5,autocorr_bern20-10,csched1a,flay04m,ex8_4_4,ex8_4_5,st_e16,ex5_4_3,ex6_1_1,ex6_1_3,ex8_4_1,wastewater15m1,m3,batch`
cold and warm; full start50/int50 cold with and without the new path, since
regressions on solved instances (nous1, spectra2) are possible.

Exit: no regression on the solved instances; +3 to +6 solved (csched1a,
flay04m, ex8_4_4/5, hadamard_5 medium; st_e16/ex5_4_3/ex6_1_x low-medium).
Compile time of the lifted kernel ≤ that of the current quadratic kernel on
nous1.

## 6. Workstream E: cuts on the lifted LP (1 agent, 2 weeks; depends on D)

Owned files: `jaxglobal/lifted_sparse.py` (row blocks), `jaxglobal/quadratic.py`
(RLT builder, to generalize), `jaxglobal/bnb.py` only for branching scores.

- E1 **RLT rows** for linear equalities × variables, ported to the general
  sparse form (the equality's products must all be columns).
- E2 **Shared product columns**: one column per distinct `(opcode, a, b)` on
  the Program, so products shared across constraints are merged.
- E3 **Integer rounding of lifted bounds**: for integer x, round FBBT bounds of
  `x²`, `x·y` columns inward after branching.
- E4 **Simple MILP cuts** on affine rows with integer columns: Chvátal-Gomory
  rounding of single rows with integer coefficients and knapsack covers for
  binaries, with outward rounding of the right-hand side. Validity test:
  random pure-integer knapsack rows, enumerate all integer points, every cut
  must hold. Expected to matter for tln4/tln5 (SCIP 7 and 1 nodes) and
  csched1a.
- E5 **Branching score from the lifted gap**: extend `_gap_scores`
  (`bnb.py:50`) to all lifted rows.

Measurement: `--names tln4,tln5,csched1a,hadamard_4,hadamard_5,autocorr_bern20-10,wastewater15m1,ex8_4_1,kall_circlespolygons_c1p12,nous1`.

Exit: tln4 or tln5 solved; nodes on hadamard_4 cut by ≥ 3× vs D alone; +2 to
+3 solved, medium-low confidence.

## 7. Workstream F (optional, drop first): tree OBBT and circle packing

kall_circles_c6a/c6c are unsolved by SCIP; c6b/c7a take SCIP 38-60 s. Only if
A-E land early: OBBT every k levels on the lifted LP, symmetry-breaking rows
for identical circles (only if the model is detected symmetric), strong
branching on the lifted gap. Expected 0 to +2 at high effort.

## 8. Dependencies, parallelism, critical path

```
W0 profile (1 d) ──> A setup/compile ───────────────┐
                     B convex OA (independent) ─────┼─> full start50/int50 cold+warm runs, docs
                     C incumbent/domain (indep.) ───┤
                     D sparse lifted LP ──> E cuts ─┘
                                            F (optional)
```

- A, B, C, D can run in parallel after W0 (W0 may re-rank A vs D). Benchmark
  runs from parallel agents load the same machine: record load, and confirm
  borderline results on a quiet machine before merging.
- Merge order: C (smallest surface), then B (D rebases onto B's convexity
  flags), then A, then D (`bnb.py` is the only file A and D both touch: D
  keeps to lines 314-345, A to `_Precompiled` / `minimize` setup), then E.
- Critical path for solve count: D → E (5-6 weeks). For SGM time: W0 → A (2
  weeks). B and C are the fastest wins (1-2 weeks each).
- If short on time, drop in this order: F, E4, E1-E3, D's warm start, then D
  entirely (A+B+C alone are expected to give +9 to +13).

## 9. Verification protocol per PR (an independent agent, not the author)

1. `pytest -q tests` and `JAXGLOBAL_FUZZ=200 pytest -q tests/test_fuzz.py`
   with the new variant in `VARIANTS` (both `safe=False` and `safe=True`).
2. `python benchmarks/check_wrong.py` on a sample (e.g. `--sample 80
   --time-limit 30 --seed 1 --jobs 4`), plain and safe; any wrong row blocks
   the merge.
3. Cut validity, as a one-off script in a scratch directory: for each
   instance in the workstream's list, evaluate every root LP row at SCIP's
   solution point (lifted where needed), assert `A z ≤ b + 1e-7·scale`, and
   assert jaxglobal's root dual ≤ SCIP's primal + tol in the model's sense.
4. Adversarial: perturb the cut generator in a local copy (flip a facet sign)
   and confirm the tests catch it; try fixed variables, zero-width boxes,
   variables at 0 for `div`/`log`/`sqrt`/`pow`, infinite FBBT bounds, and
   integer boxes `[k, k]`.
5. Re-run the author's named-instance benchmark cold (`JAXGLOBAL_CACHE=0`)
   and warm, record `uptime` load, and compare with the baseline CSV through
   `benchmarks/report.py`.

## 10. Repo conventions

- One branch per workstream (`scipgap-w0-profile`, `scipgap-a-setup`,
  `scipgap-b-convex`, ...), a PR, merged with a merge commit. Commit messages
  end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; PR
  bodies end with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`
  and are passed with `--body-file`.
- Results CSVs in `benchmarks/results/` named
  `scipgap_<ws>_<set>_{cold,warm}.csv` via `--tag`; a one-line summary in the
  results table of `benchmarks/sets/README.md` and a dated subsection in
  `docs/ROADMAP.md` (what changed, instances gained and lost, SGM cold/warm,
  machine load, CSV link). Update `README.md` limitations when one is removed.
- Full-set runs use the command in `benchmarks/sets/README.md` with
  `--jobs 4`, cold and warm. Load matters near the 60 s limit (nous1 takes 36
  s alone, 57.8 s under 4 jobs).
- 0 wrong answers on `check_wrong.py` and on both sets is a hard gate for
  every merge.

## 11. Expected gains and confidence

| WS | start50 | int50 | confidence | effort |
|---|---|---|---|---|
| A setup/compile | ex8_4_5, (ex8_4_4); ramsey already solves on a quiet machine (W0) | hadamard_4, (hadamard_5, csched1a, flay04m); portfol_buyin, batch already solve on a quiet machine (W0) | high 2, medium 3 | 1-2 wk; also the SGM lever |
| B convex OA | - | cvxnonsep_normcon30/40, (nsig30, pcon40), flay04m with A | high 2, medium 3 | 1-2 wk |
| C incumbent/domain | ex5_4_3, (st_e16, launch) | m3, clay0203m | high 1-2, medium 2 | 1 wk |
| D sparse lifted LP | (ex8_4_4, st_e16, ex6_1_x) | csched1a, flay04m, hadamard_5, (autocorr) | medium 3, low 3 | 3-4 wk |
| E cuts | wastewater15m1, ex8_4_1 | tln4/5, smallinvDAX | medium-low | 2 wk |
| F circles | 0-2 | - | low | drop first |

Realistic target: start50 31 → 37-40, int50 29 → 38-41, SGM time halved or
better on both sets. The remaining gap to SCIP is circle packing,
autocorr/smallinvDAX scale, and least (SCIP fails too).

## 12. Risks and de-risking measurements (first)

1. Compile vs first-LP split (W0), which decides A vs D priority. Done: A first (compile ≥ 30 % on 5/7, first LP ≤ 1.6 %).
2. Sparse IPM at N ≈ 1000 columns per node (smallinvDAX, autocorr): time one
   `quadratic._ipm` iteration with a synthetic pattern of 1000 columns × 4000
   rows before raising the column ceiling; if > 50 ms per iteration per node,
   the lifted LP stays root-only at that size.
3. Convexity detector false positives: the random-DCP fuzz comes before any
   OA cut.
4. Regressions on solved instances when the lifted path becomes default for
   non-quadratic problems (nous1, spectra2, ex4, gear2).
5. `bnb.py` merge conflicts between A and D: split by line range, land A
   first.
6. Machine load near the 60 s limit: always record cold, warm and load; use
   `--jobs 2` to confirm borderline instances.
