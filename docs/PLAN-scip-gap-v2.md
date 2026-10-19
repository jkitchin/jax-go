# SCIP gap plan, v2 (after Workstream 0)

Revision of `docs/PLAN-scip-gap.md` in the light of W0's root-time profile
(PR #16, branch `scipgap-w0-profile`; results in `benchmarks/results/scipgap_profile.csv`,
`benchmarks/results/scipgap_w0_jobs4_cold.csv`, raw JSON under
`.../scratchpad/prof/json*`). Everything below was checked against the code in the
W0 worktree (`jaxglobal/bnb.py`, `alphabb.py`, `univariate.py`, `relax.py`,
`relax_interp.py`, `fbbt_interp.py`, `lp.py`, `structure.py`, `kcache.py`,
`domain.py`, `benchmarks/profile_root.py`, `benchmarks/run.py`, `report.py`,
`check_wrong.py`). `jaxglobal/convexity.py` does not exist yet.

Sections: 1 corrections to W0 · 2 soundness analysis · 3 measurement rules ·
4 work items · 5 parallelism and merge order · 6 per-PR verification protocol ·
7 conventions · 8 dropped / kept / reordered · 9 top risks.

---

## 1. Corrections and additions to W0's analysis

W0's wall-time buckets are sound and useful. Several of its conclusions are not.

1. **ex6_2_6 is not a tolerance-semantics problem.** Both solvers run with
   abs 1e-6 / rel 1e-4 (`benchmarks/run.py`: `REL_GAP 1e-4, ABS_GAP 1e-6`). SCIP
   needs 3.16 s and 12,364 nodes (`instances.csv`), not "<1 s". The optimum is
   -2.6e-6, so the relative tolerance is meaningless and the absolute one demands
   a bound within 1e-6 of zero; jaxglobal's bound at 60 s is -8.78e-4 (300x the
   tolerance) after 27.9k nodes, and -0.00x after 109k nodes at 120 s. Node
   throughput (~900 nodes/s) is fine; the bound quality on the x·log(x/Σx) terms
   is the problem (McCormick of the composition log(x/Σx)·x is weak near the
   boundary), and the primal is also missing: jaxglobal's incumbent is +7.1e-7,
   worse than the reference -2.6e-6. Note the MINLPLib reference gap itself is
   open (0.344). Levers: C (find the incumbent; closes the abs gap only together
   with a bound) and a perspective/relative-entropy rule (B stretch). Not A.
   Any "interpreter halves the node rate" effect is irrelevant here.

2. **hadamard_5 needs D, not just a compile fix.** With a 900 s limit the bound
   kernel compiles in 273 s (the OBBT kernel at batch 1 another 301 s), then 88k
   nodes reach -7.68 against the optimum -5. Without whole-function alphaBB the
   kernel compiles in 38.5 s but the bound is -14.4 after 9.2k nodes. Neither
   relaxation closes the degree-4 polynomial (480 `mul`, 60 `neg`, one sum in
   the .nl). SCIP: 4.97 s, 1,559 nodes. A2 gets a first bound inside the limit;
   D (lifted LP for products of lifted variables) is the lever. hadamard_5 is a
   maximisation; the harness sign convention has to be kept in mind when
   quoting bounds.

3. **cvxnonsep_nsig30: SCIP fails it too, and the root gap is a relaxation
   problem.** SCIP: time_limit, 131.78 / 108.53 after 26,837 nodes. jaxglobal's
   root bound 23.46 vs optimum 130.63 after one node. The .nl is 30 `pow`, 30
   `mul`, one sum: a 30-fold product of powers (a convex signomial). McCormick
   on a 30-deep product chain is hopeless whatever the kernel speed. Lever: B
   (outer approximation of a structurally convex function), not kernel
   throughput. The 1.3 s per 16-box kernel call (122 LP rows, n=30) is real but
   W0 never split it (univariate hull prepare, alphaBB prepare, IPM x Kelley
   rounds, FBBT) — an ablation profile (A3) must precede any change to
   `relax_interp.MIN_ROWS`.

4. **The W0 cold table mixes loads from 7 to 124.** `batch 24.5 s` was taken at
   1-min load 124 (jobs-4 run: 15.1 s). hadamard_5 cold at load 110. Any
   per-instance seconds claim needs load < 10, recorded. Hence R0 and the
   measurement rules in §3.

5. **Three of A's six targets already solve cold on a quiet machine**
   (ramsey 8.9 s, portfol_buyin 14.9 s, batch ~15 s). A's upside on them is
   bounded by the warm/cold delta (ramsey 8.9 → 2.2 s, portfol 14.9 → 7.6 s).
   The issue14 baselines (ramsey time_limit, portfol 73.9 s, batch 64.6 s, all
   at 1 node) were taken under heavy load and are not usable as a baseline.

6. **Op count, not batch size, drives compile time** (OBBT at batch 1 compiled as
   slowly as the bounder at batch 16). W0 correctly drops the batch-1 root
   kernel. But its "cheap kernel first, then swap" doubles compile work in the
   shared 4-thread pool; at `--jobs 4` that is 4 workers x 2 kernels x 4
   threads. If used at all the cheap kernel must be submitted first and the
   full kernel only after the cheap one is ready (sequential), and the OBBT
   kernel must come from the same (cheap) relaxer or be skipped.

7. **The reason hadamard_5's alphaBB compile explodes is a threshold mismatch,
   not an inherent cost.** `alphabb.prepare` uses the interpreted Hessian
   (`make_taylor2`) only when `use_relax_interpreter(tape, n)` (rows >= 600) and
   `S·n² <= TAYLOR_BUDGET=30000`. hadamard_5's tape has ~540 rows (< 600) so the
   straight-line `jacfwd(jacfwd(flat))` jaxpr is used: ~540 ops x 26² ≈ 3.6e5
   ops to compile. `TAYLOR_BUDGET` is a memory guard, not a speed knob: the
   taylor2 carry is `(S, 2+2d+2d²)` floats per box (`relax_interp.make_taylor2`,
   `cuts = cumsum([0,1,1,d,d,d*d,d*d])`). For hadamard_5 (S≈540, d=26) that is
   ~6 MB per box, ~100 MB for 16 boxes: affordable. Replace the S·n² rule by a
   bytes budget (S·(2+2d+2d²)·8·batch) and decouple the taylor2 decision from
   `MIN_ROWS` (A2). W0's "~1.3 GB for smallinvDAX at 16 boxes" should be
   re-derived from `const_val.size` before anything is tuned.

8. **The smallinvDAX overruns are not compile waits.** 135.5 s at tl=120 cold,
   **263 s warm**, 80.6 s at tl=60 (jobs 4). Warm being slower than cold with
   the same 291 nodes says the time is in un-deadlined run paths (dives: 36–50 s
   per call; `dive()` only checks the deadline while waiting for the executable),
   not in compilation. A1 must explain the warm 263 s from the JSON in
   `prof/json` before choosing between a dive time budget and a smaller dive
   kernel.

9. **Soundness of the two proposed mechanisms is confirmed** (details in §2):
   deferring/dropping whole-function alphaBB cannot invalidate a bound;
   mid-iteration deadline aborts cannot corrupt the certificate. W0 did not
   state either argument; both are needed in the PR bodies.

10. **An unbounded path W0 missed:** `_Precompiled.__call__` falls back to
    `self.jitted(*args)` (a synchronous compile with no deadline) when the
    background compile raised anything other than a timeout. A1 should bound it
    (raise `_CompileTimeout` if the deadline has passed instead of compiling
    synchronously).

11. **Per-primitive alphaBB is independent of the `alphabb=` option.**
    `relax._auto_rule` calls `alphabb_rule` for `AUTO_UNIVARIATE` primitives
    (sin, cos, tanh, logistic, atan, asinh, sinh, erf, erfc) and `pow`; this is
    unaffected by `make_relaxer(alphabb=False)` and must not be touched by A2.

---

## 2. Soundness analysis (to be cited in PR bodies)

**Dropping or deferring whole-function alphaBB.** `univariate.make_relaxer`
(lines 444–460) combines alphaBB with the McCormick/univariate relaxation by
`cv = max(m.cv, acv)`, `cc = min(m.cc, acc)`; `alphabb.apply` returns -inf/+inf
where alpha is infinite. Both components are valid relaxations on their own, so
the max/min is valid, and so is either component alone. A kernel built with
`alphabb=False` therefore produces valid (weaker) bounds; swapping kernels
between B&B iterations is sound because the loop only needs each node's bound
to be a valid lower bound (`glb = min(open_lb.min(), pruned_lb, ub)`). The
`kcache` fingerprint must include the alphabb option in `extra` so the two
kernels never share a cache entry.

**Deadline aborts.** In `minimize` the open-node arrays are reassigned only at
the end of an iteration (`open_lo = np.concatenate([open_lo[rest], new_lo])`,
etc.). An exception raised from `run()` between kernel calls propagates to the
`evaluate` call site in the search loop, which already catches `_CompileTimeout`
and `break`s with the previous iteration's open set intact; the children of the
aborted iteration are never removed from `open_*`, so their parents' bounds
still cover them. Incumbents found by completed calls of the aborted iteration
are discarded, which is sound (only an upper bound is lost). The other call
sites (root bound → `Result(status="time_limit", lower_bound=-inf)`; root OBBT
→ root bound stands; probe → unchanged bounds) are likewise sound. The final
`_verify` is a float64 re-evaluation of the incumbent and does not depend on
the kernel. Dives (`dive()`, `_DIVES=16`, `dive_fbbt`) only produce incumbents,
so skipping, shortening or aborting them is sound.

**OBBT and probe.** `make_obbt` returns box tightenings certified by `cert(y)`;
aborting between the 2n vmapped LPs is not possible mid-kernel, and aborting
before the call leaves the box untightened (sound).

**What is not automatically sound:** changing the IPM iteration count or Kelley
rounds (the certificate handles any `y >= 0`, so it stays valid, but
`reduce_ranges` uses the certificate's reduced costs — also valid for any
`y >= 0`); changing `_row_margin` / `safe` handling; any new convex-function
cut (B) — these need the cut-validity gates in §6.

---

## 3. Measurement rules (apply to every number in every PR)

* **Cold** = `JAXGLOBAL_CACHE=0`, fresh process per instance. **Warm** = second
  run with `JAXGLOBAL_CACHE_DIR` pointing at a directory primed by one cold run
  in the same source state.
* **One instance at a time** for any per-instance timing: `benchmarks/profile_root.py`
  (modes cold/prime/warm, `--opts`, `--set`, `--label`, `--json-dir`) or
  `benchmarks/run.py --jobs 1 --names <one>`.
* **Load recorded**: 1-min load from `uptime` immediately before each run, in the
  CSV `note` column. A number is admissible only if load < 10 (target < 5). If
  another agent's benchmark is running, wait. Coordination: create
  `/private/tmp/claude-501/jaxglobal-bench.lock` with `mkdir` (atomic) before a
  timing run, remove it afterwards; if it exists, wait.
* **Repeats**: timing thresholds are judged on the median of 3 cold runs.
* **Set-level runs** (start50/int50) use the harness as is: `python benchmarks/run.py
  --set start50 --time-limit 60 --jobs 4 --tag <tag>`; the 4 workers themselves
  raise the load to ~4–6, which is why the gate is load < 10 and not < 4.
* **Reference columns**: all comparisons are against R0's CSVs, same mode, same
  load regime, never against `issue14_*_cold.csv` or W0's mixed-load table.

---

## 4. Work items

Every item: branch `scipgap-<item>`, own worktree from `main` after PR #16 is
merged, one PR, one independent verification agent, gates of §6, PR body via
`--body-file`, merged with `gh pr merge --merge`.

### R0 — Re-baseline (no library code)

**Goal.** A trustworthy cold and warm baseline of start50 and int50 at low load,
plus per-instance jobs-1 confirmations of the seven W0 instances, so that every
later acceptance test has a number to beat.

**Files.** New `benchmarks/results/scipgap_r0_start50_cold.csv`,
`scipgap_r0_int50_cold.csv`, `scipgap_r0_start50_warm.csv`,
`scipgap_r0_int50_warm.csv`, `scipgap_r0_seven_cold.csv` (profile_root, one at a
time, 60 s and 120 s limits); `docs/ROADMAP.md` (one table + load note);
`benchmarks/sets/README.md` results-history entry. If `run.py` does not record
load, add a `load` column to its CSV writer (additive change only; the
verification agent checks `report.classify` output is unchanged on an old CSV).

**Design.** `run.py --jobs 4 --time-limit 60`, start50 and int50, cold then warm,
machine otherwise idle, load logged before each set. Then
`profile_root.py --modes cold --time-limit 60` and `--time-limit 120` for
ex6_2_6, ramsey, portfol_buyin, hadamard_5, cvxnonsep_nsig30, smallinvDAX,
batch. Report wall time, status, nodes, primal, dual, first-bound time, and
overrun (wall minus limit). Rejected: re-using W0's table (mixed loads) or the
issue14 CSVs (heavy load, see §1.4–1.5).

**Acceptance.** All four set CSVs committed with load < 10 recorded for every
set; 0 `wrong` in `report.classify` on each; the seven-instance CSV has 3 cold
repeats per instance. Nothing else.

**Must not touch.** Any `jaxglobal/*` file; `report.classify`; tolerances; the
instance sets.

**Dependencies.** None. Everything else compares against it, but other agents
may start their code work before R0 lands and only need R0 for their final
acceptance measurement.

### A1 — Deadline enforcement in the search loop, dives and compile fallback

**Goal.** A `time_limit` solve returns within `time_limit + 3 s` on every
instance; status and certificate remain sound.

**Files / functions.** `jaxglobal/bnb.py` only: `_Precompiled.__call__`
(fallback path), `minimize.run`, `minimize.dive`, the periodic-dive and
root-dive/enum call sites, the per-iteration time check; `tests/test_time_limit.py`
(add tests). Optionally `benchmarks/profile_root.py` to split "dive compile wait"
from "dive run" if the JSON does not already do so.

**Design.**
1. `run()` checks `time.perf_counter() > t0 + time_limit` before every kernel
   dispatch and raises `_CompileTimeout("deadline")` (or a subclass
   `_Deadline(_CompileTimeout)` so existing handlers catch it). Call sites
   unchanged (soundness §2).
2. `_Precompiled.__call__` fallback: if the deadline has passed, raise
   `_CompileTimeout` instead of `self.jitted(*args)`.
3. Dives get a time budget: measure the wall time of each `dive` call; skip a
   dive when `remaining < 1.5 x last_dive_time` (first dive: skip if
   `remaining < 0.25 x time_limit`). Root dives and `_ENUM` likewise. Dives are
   incumbent-only (§2) so this is sound.
4. First explain smallinvDAX's warm 263 s from `prof/json` (which bucket). If the
   dive *run* is 36–50 s per call at n=30 (fori_loop over 30 integers x 2
   roundings x `dive_fbbt` with 5 rounds, vmapped over 16 boxes), also reduce
   per-dive cost: `dive_fbbt` rounds 5 → 2 inside the dive kernel, and/or fewer
   dive boxes. Only if the ablation shows no incumbent-quality loss on
   smallinvDAX, portfol_buyin, batch, ex6_2_6 (primal at 60 s unchanged within
   1e-6 rel).
5. Rejected: a wall-clock check *inside* kernels (`lax` cannot read the clock;
   would need host callbacks); a separate watchdog thread raising into the loop
   (unsafe with JAX dispatch); making `time_limit` cover only the search (the
   tests in `test_time_limit.py` require compilation to be covered).

**Acceptance (cold, one at a time, load < 10, 3 repeats each).** With
`--time-limit 60`: smallinvDAXr3b010-011, cvxnonsep_nsig30, hadamard_5, ex6_2_6
each finish with wall <= 63 s (R0 shows 80.6 s for smallinvDAX at jobs 4; W0
135/263 s at tl 120). The same with `--time-limit 120`: wall <= 123 s. Dual
bounds at the limit are not worse than R0's by more than 1e-6 abs for the
same limit (fewer dives must not change bounds; incumbents may differ). No
change on instances that finish before the limit: ramsey, portfol_buyin, batch
wall within 10 % of R0. New tests: deadline mid-search with a synthetic slow
kernel (monkeypatch `K.bound` to sleep) returns `time_limit` with the
pre-iteration lower bound; dive budget skips when time is short.

**Must not touch.** `_Kernels.__init__`, `alphabb.py`, `univariate.py`, `lp.py`,
`relax*.py`, `fbbt*.py`, default option values other than inside the dive kernel
(and only with the ablation above).

**Dependencies.** None. Merges first among code items.

### A2 — alphaBB interval-Hessian cost (hadamard_5 / nsig30 / smallinvDAX compile)

**Goal.** First bound inside the time limit on hadamard_5 and faster first bound
on nsig30/smallinvDAX, without weakening bounds once the kernel is ready.

**Files / functions.** `jaxglobal/alphabb.py` (`TAYLOR_BUDGET`, `taylor()`,
`prepare`); `jaxglobal/univariate.py` only if a new `interpret`/budget argument
must be threaded through `make_relaxer`; `jaxglobal/lp.py::_parts_fn` only to
thread the same argument; `jaxglobal/bnb.py::_Kernels.__init__` only for the
optional cheap-first kernel (second `_Precompiled`, fingerprint `extra`); new
`tests/test_alphabb_budget.py`; `benchmarks/profile_root.py` already reports
`make_taylor2` size vs budget and the jacfwd fallback.

**Design (in order; stop when the acceptance test passes).**
1. Decouple the taylor2 decision from `MIN_ROWS`: use the interpreted Hessian
   when the straight-line path would be large (`ops x n² > ~5e4`, measured) and
   the taylor2 carry fits a **bytes** budget `S·(2+2d+2d²)·8·batch <= 256 MB`
   (constant `TAYLOR_BYTES`; keep `TAYLOR_BUDGET` as the fallback name for the
   `interpret=True` override). Verify the memory estimate against
   `run.program.const_val.size` on smallinvDAX, hadamard_5, nsig30 and record
   the numbers in the PR.
2. If the interpreted Hessian is too slow per node (W0: interpreter halves the
   node rate on ex6_2_6; measure on hadamard_5 and nsig30), add the cheap-first
   kernel: `_Kernels` builds `bound_fast` (`alphabb=False`) and `bound` (full);
   `bound_fast.start` first, `bound.start` submitted only when the fast one is
   ready; `minimize.evaluate` uses `bound` when ready else `bound_fast`; OBBT
   uses whichever relaxer the root used. Fingerprint `extra` includes the
   alphabb flag. Soundness §2.
3. Stretch (separate PR `scipgap-a2-colored`): sparse interval Hessian from
   `structure.hess` pairs with a greedy column colouring, `n_colors` forward
   seeds instead of `n`. Helps smallinvDAX-type (sparse) Hessians, not
   hadamard_5 (dense).
4. Rejected: raising `TAYLOR_BUDGET` to a bigger S·n² (memory blow-up, §1.7);
   per-node Hessian recomputation elimination via a root-only alpha (alpha from
   the root box is valid on sub-boxes but much weaker; destroys alphaBB's
   quadratic shrinkage); dropping whole-function alphaBB by default
   (hadamard_5 bound -14.4 vs -7.68).

**Acceptance (cold, one at a time, load < 10, median of 3).** hadamard_5 first
bound (profile bucket `root_bound`) < 60 s (R0/W0: > 120 s; 273 s at tl 900);
cvxnonsep_nsig30 first bound < 20 s (W0 38.5 s); smallinvDAX first bound <= W0's.
No regression: ramsey, portfol_buyin, batch, ex6_2_6 cold wall within 10 % of
R0; ex6_2_6 node rate (nodes/s after first bound at tl 120) within 10 % of R0
when the full kernel is in use; final dual at tl 120 on hadamard_5 and ex6_2_6
not worse than R0. Peak RSS of the worker on smallinvDAX and hadamard_5 (via
`/usr/bin/time -l`) < 4 GB.

**Must not touch.** `relax.py` (`_auto_rule`, `alphabb_rule`), the `run`/`dive`/
search-loop code A1 owns, `lp.py` beyond threading one argument, `relax_interp.py`
thresholds (A3 owns them).

**Dependencies.** None for the code. Rebases on A1 before merge (both touch
`bnb.py`, disjoint functions).

### A3 — Per-node kernel ablation and interpreter thresholds

**Goal.** Know where the 1.3 s per 16-box call on nsig30 goes, and replace the
fixed row thresholds (`relax_interp.MIN_ROWS=600`, `fbbt_interp.SCHEDULE_MIN_ROWS=300`,
`use_interpreter(min_rows=50, max_expansion=1.5)`) with a rule that chooses the
interpreter when the straight-line kernel's op count (ops x cut points x
Kelley rounds) would be large, measured on both compile time and node rate.

**Files / functions.** `benchmarks/profile_root.py` (an `--ablate` convenience
that runs a fixed list of `--opts` variants and prints kernel-run seconds per
call and nodes/s; optional), `jaxglobal/relax_interp.py::use_relax_interpreter`,
`jaxglobal/fbbt_interp.py::use_interpreter` (rule only), `docs/ROADMAP.md`.

**Design.** Step 1 (measurement, no code change): on nsig30, smallinvDAX,
ex6_2_6, hadamard_5 run `--opts` variants `{"alphabb":false}`,
`{"univariate":false}`, `{"n_kelley":0}`, `{"fbbt_rounds":1}`, `{"obbt":"none"}`
and `--set relax_interp.MIN_ROWS=0` / `=10000`, `--set fbbt_interp.SCHEDULE_MIN_ROWS=...`;
record compile seconds, kernel-run seconds per call, nodes/s, bound at the
limit. Step 2: a threshold rule on an op-count estimate from the tape (ops,
rows, n) chosen so that no instance in the 7 + 20 start50/int50 instances
loses > 10 % node rate while hadamard_5-class tapes (≈540 rows, just under 600)
get the interpreter. Rejected: changing `MIN_ROWS` to a smaller constant
without the ablation (W0's own ex6_2_6 data shows a 2x node-rate loss); making
the interpreter the default.

**Acceptance.** PR contains the ablation table (cold, load < 10). Threshold
change: compile time on hadamard_5's McCormick-only kernel < 15 s (W0 38.5 s)
and no instance among the 7 plus 20 sampled start50/int50 instances with
> 10 % node-rate loss or > 10 % cold wall increase vs R0; bounds at the limit
unchanged within 1e-6 abs on instances that do not finish.

**Must not touch.** `alphabb.py` (A2), `bnb.py`, `lp.py`, relaxation rules.

**Dependencies.** Code is independent; its threshold interacts with A2's
taylor2 decision, so A3 merges after A2 and re-measures on A2's main.

### B — Outer approximation for structurally convex functions

**Goal.** Tight root relaxations for convex MINLPs whose convexity McCormick
cannot see (cvxnonsep_nsig30, normcon20r/30/40, pcon40, smallinvDAX
constraints, batch, portfol_buyin): where a function (or output component) is
certified convex on the box, use `cv = f` (and `cc = f` where concave) in the
relaxer, so the LP cuts at centre + 2n points + Kelley become outer
approximation.

**Files / functions.** New `jaxglobal/convexity.py`: a DCP-style classifier over
the tape (`fbbt.make_tape`) returning per-output `(convex, concave)` flags that
may depend on box sign conditions (evaluated in-kernel from `lo`, `hi`): affine;
`exp`, `log`, `pow` (p >= 1 or p <= 0 convex on x > 0; 0 < p < 1 concave),
`sqrt`/`log` concave, `square`/`integer_pow` even convex, `dot_general` with PSD
constant (quadratic forms via the existing eigen test), sums with non-negative
weights, monotone compositions, `max` of convex, norm (`sqrt(sum(square))`),
log-sum-exp, and a **monomial/signomial tracker** (slot = c·Π x_i^{a_i} on a
positive box: concave if c > 0, a_i >= 0, Σa_i <= 1; convex if c < 0 under the
same, or c > 0 with one a_i < 0 and others <= 0 — the posynomial/signomial
rules). `jaxglobal/univariate.py::make_relaxer`: combine with the McCormick
result by `max`/`min` as alphaBB is combined today (same `tighten(MC(...))`
path), guarded by the box condition. `jaxglobal/lp.py::_parts_fn` to pass the
flag; `tests/test_convexity.py`. Stretch (separate PR): a relative-entropy /
perspective pattern `x·log(x/y)` (ex6_2_6).

**Design notes.** The certificate is structural, not numeric: an interval
Hessian PSD test (eigen of midpoint minus radius) would be loose on product
chains and is already partly covered by alphaBB's eigen relaxation. Convexity
flags are computed at trace time from the tape, with sign conditions as
`jnp` booleans; where the condition fails the plain relaxation is used.
Rejected: convexity via sampled Hessians (not a certificate); replacing McCormick
entirely by OA for convex problems (McCormick/univariate hulls still tighten
non-convex parts and give FBBT its structure).

**Acceptance (cold, one at a time, load < 10, median of 3).** cvxnonsep_nsig30:
root bound after the first LP >= 117.5 (90 % of 130.63; R0: 23.46) and status
optimal within 120 s (SCIP fails this). smallinvDAXr3b010-011: optimal within
120 s or root bound >= 0.35 (R0: -0.131). int50 convex instances with status
time_limit in R0 (cvxnonsep_normcon20r/30/40, pcon40, smallinvDAXr4/r5 as
present): at least two more solve within 60 s at `--jobs 4`, none regress
(status, wall within 10 %). batch and portfol_buyin not slower than R0 by more
than 10 %.

**Correctness gates (in addition to §6).** `tests/test_convexity.py`: for every
rule, random boxes satisfying/violating the sign condition, 1000 random points,
check `f(x) >= tangent at 50 random points` where convex is claimed and the
flag is False where the condition is violated; a MINLPLib-driven test that for
each instance where any output is flagged convex, 200 random points in the
original box satisfy `f(x) >= cut(x) - 1e-9(1+|f|)` for the root LP cuts;
`check_wrong.py` full sets (not just the sample) because this changes bounds on
many instances.

**Must not touch.** `bnb.py` (except nothing: the flag flows through
`make_relaxer`), `alphabb.py`, `relax.py` rules, `relax_interp.py`.

**Dependencies.** None for code; it edits `make_relaxer` near the lines A2 may
touch for an argument — coordinate: A2 appends arguments, B adds one `convex`
branch; whichever merges second rebases.

### C — Incumbent and domain (kept from v1 with two additions)

**Goal.** Find the incumbent earlier and better: multi-start local solves with
integers fixed from dive roundings, `bound_domain` improvements per v1 §C, plus
(new) a local polish of every improved incumbent and (new) a near-zero
objective special case so that ex6_2_6's -2.6e-6 is found.

**Files / functions.** `jaxglobal/domain.py`, `jaxglobal/local.py` (or wherever
`make_local` lives), `bnb.py` only at the `make_local` / `update_incumbent` /
`fixed_solve` call sites.

**Acceptance (cold, load < 10).** ex6_2_6 primal <= -2.5e-6 within 10 s (R0:
+7.1e-7 at 60 s); smallinvDAXr3b010-011 primal within 1e-4 rel of 0.398797
within 60 s (R0: 0.4004); hadamard_5 primal = -5 (max 5.0) within the limit
once a bound exists (A2); on start50 `solved` group no instance slower than R0
by > 10 %; incumbents on all sets never violate `FEAS_TOL` (classify reports 0
wrong / 0 infeasible incumbents).

**Must not touch.** Kernel construction, `run`/`dive` (A1), relaxers, LP.

**Dependencies.** None; merges after A1.

### D — Sparse lifted LP for degree-3/4 polynomials (kept, now first-class)

**Goal.** Close hadamard_5 (and autocorr_bern20-10/30-04, sporttournament06,
hadamard_4): lift products of lifted variables (`w_ij·w_kl`, `w_ij·x_k`) into
the LP with McCormick/RLT rows, following `quadratic.py`'s row blocks, applied
recursively on the `structure`/`scalarize` program.

**Files / functions.** New `jaxglobal/lifted_poly.py` (or `quadratic.py`
generalised with a `degree` parameter: `quadratic_form` currently accepts
`mul`, `square`, `integer_pow` y∈{1,2}, `div` by constant, `add*`, `sub`, `neg`);
`bnb.py::_Kernels.__init__` only in the `raw_bound` selection branch (lines
~314–345); `tests/test_lifted_poly.py`.

**Acceptance (cold, load < 10).** hadamard_5: root bound after the first LP
strictly better than R0's (report both) and either optimal within 300 s or the
bound at 300 s >= -6.0 (W0: -7.68 at 900 s). autocorr_bern20-10 and hadamard_4
solve within 60 s at `--jobs 4` or improve their R0 dual by >= 50 % of the gap.
No regression on quadratic instances currently using the lifted LP (issue #15
set) by > 10 % wall.

**Must not touch.** `alphabb.py`, `univariate.py`, `run`/`dive`.

**Dependencies.** Code independent; merges after A2 (both touch
`_Kernels.__init__`; D must confine itself to the selection branch).

### E — Cuts on the lifted LP (kept, after D)

As v1 §E, on D's representation. Acceptance set: hadamard_5 optimal within
120 s cold; autocorr_bern30-04 bound improvement >= 50 % of R0's gap; no
regression on the #15 set. Depends on D.

### F — Circle packing: **dropped** from this effort (none of W0's instances;
separate plan).

---

## 5. Parallelism and merge order

Start in parallel (worktrees from `main` after PR #16): R0, A1, A2, A3 (step 1
measurement; step 2 code after A2's design is known), B, C, D.

| item | bnb.py regions | other files |
|---|---|---|
| A1 | `_Precompiled.__call__` fallback, `run`, `dive`, search-loop checks, dive call sites | `tests/test_time_limit.py` |
| A2 | `_Kernels.__init__` (second `_Precompiled`, fingerprint extra), setup lines ~756–842 for `start` ordering | `alphabb.py`, (`univariate.py`, `lp.py` arg threading) |
| C | `make_local`/`update_incumbent`/`fixed_solve` call sites | `domain.py`, `local.py` |
| D | `raw_bound` selection branch (~314–345) | new `lifted_poly.py`/`quadratic.py` |
| A3 | none | `relax_interp.py`, `fbbt_interp.py`, `profile_root.py` |
| B | none | new `convexity.py`, `univariate.py::make_relaxer`, `lp.py::_parts_fn` |
| R0 | none | `benchmarks/results/*`, `docs/ROADMAP.md`, (`run.py` load column) |

No two items edit the same function. Known adjacency: A2 and B both touch
`make_relaxer`'s tail; A2 and D both touch `_Kernels.__init__`. The later
merger rebases.

**Merge order:** R0 → A1 → C → B → A2 → A3 → D → E. Rationale: R0 makes the
numbers comparable; A1 makes every later timing honest (no overruns) and is
small; C is small and independent; B has the largest expected payoff on the
convex half of int50; A2 is a larger `_Kernels` change; A3's thresholds depend
on A2's taylor2 decision; D/E are the biggest and only matter for the
polynomial instances. Each PR re-runs its acceptance test after rebasing on
the then-current `main`.

---

## 6. Per-PR verification protocol (unchanged from v1 §9, made explicit)

Run by the implementing agent, then independently by a verification agent that
reads the diff for soundness (§2 arguments must appear in the PR body) and
re-runs 1–4 and the acceptance test.

1. `python -m pytest -q` (all tests).
2. `JAXGLOBAL_FUZZ=200 python -m pytest -q tests/test_fuzz.py`.
3. `python benchmarks/check_wrong.py --sample 80 --time-limit 30 --seed 1 --jobs 4`
   and the same with `--safe`: 0 wrong. For B and D (bound-changing on many
   instances): full start50 and int50 at `--time-limit 60 --jobs 4`, 0 wrong.
4. Cut validity at the SCIP/reference point for every instance whose bound
   changed (B, D, E): reference primal >= our dual.
5. The item's acceptance test exactly as written (cold, one at a time, load
   recorded, 3 repeats), with R0's number beside it in the PR body.
6. Hard gate: zero wrong answers anywhere; a single `wrong` blocks the merge.

---

## 7. Conventions

Branches `scipgap-<item>`; one PR per item; verification agent per PR; PR body
written to a file and passed with `gh pr create --body-file <path>`; merge with
`gh pr merge --merge`; never `--no-verify`; never bare `git stash`/`git stash pop`
in worktrees. Commit messages end with
`Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; PR descriptions end
with `🤖 Generated with [Claude Code](https://claude.com/claude-code)` (the work
is done by Opus agents). Nobody edits
`benchmarks/run.py`/`report.py`/`check_wrong.py` semantics, `report.classify`,
tolerances, instance sets, or the `issue14_*` CSVs; R0 may add a load column.
Nobody changes `minimize` defaults without an ablation table in the PR.

---

## 8. Dropped / kept / reordered relative to v1

* **Dropped:** W0's batch-1 root kernel; Kelley scan in the lifted LP;
  Hessian-fingerprint caching; local-solver build item; F (circle packing);
  "tolerance semantics" investigation for ex6_2_6.
* **Kept:** B (now the top-payoff item, with a structural-convexity design and
  a signomial tracker), C (with two additions), D, E, the verification protocol,
  zero-wrong gate.
* **Added:** R0 re-baseline; A1 deadline enforcement + dive budget + compile
  fallback; A2 taylor2 bytes budget decoupled from `MIN_ROWS` (cheap-first
  kernel demoted to step 2); A3 ablation before any interpreter threshold change.
* **Reordered:** A split into A1/A2/A3; merge order R0 → A1 → C → B → A2 → A3 →
  D → E (v1: C, B, A, D, E).
* **Expectations lowered:** nsig30 is a SCIP failure too (success = solving it,
  not matching SCIP); hadamard_5 needs D, not A; ramsey/batch/portfol_buyin are
  already solved cold and A's upside on them is the warm/cold delta only.

---

## 9. Top risks

1. **Load-contaminated measurements** recur (several agents benchmarking at
   once). Mitigation: the lock directory and load < 10 rule in §3; the
   verification agent rejects any number without a recorded load.
2. **B's convexity classifier claims convexity wrongly** (sign conditions on
   intermediate slots, signomial rules at box boundaries with lo = 0). This is
   the only item that can produce wrong answers by construction. Mitigation:
   structural rules only, box conditions evaluated in-kernel, tangent tests
   per rule, full-set `check_wrong` plain and `--safe`.
3. **A2's memory estimate is off** and the taylor2 Hessian thrashes at 16 boxes
   x 2n OBBT LPs. Mitigation: estimate verified against `const_val.size` and
   RSS gate < 4 GB; fall back to the straight-line path above the bytes budget.
4. **A1's dive budget removes incumbents** that the search relied on to prune
   (no bound change but more nodes). Mitigation: incumbent-quality ablation
   before shrinking `dive_fbbt`; A1 only skips dives when the deadline is near.
5. **D's lifting blows up the LP** (hadamard_5: 480 products of degree up to 4
   → thousands of lifted columns, IPM at 40 iterations too slow per node).
   Mitigation: sparse lifting of only the products present, acceptance allows
   300 s, report LP size and per-node time in the PR.
6. **Rebase conflicts in `bnb.py`** between A1, A2, C, D. Mitigation: the
   region table in §5; the later merger rebases and re-runs its acceptance
   test on the current `main`.
7. **ex6_2_6 remains open** even after C (bound -8.8e-4 needs a relaxation
   change; the MINLPLib reference gap is open). It is reported honestly as a
   relaxation gap, not re-counted under A.

---

## 10. Status at completion

Status sweep on `main` at 3047be4, after the PRs merged in this loop
(#60–#67: scheduled FBBT, lifted-LP rational rows, symmetry pruning,
pseudo-Boolean variable elimination, gap-weighted pseudocosts, the
"unproven" status after incumbent repair, and the symmetry cost budget).
PASS means solved within max(2 x SCIP time, 10 s). There were two rounds per
bucket, with load recorded: a cold first round (compilation included) and a
warm second round (compile caches reused).

**74 of 79 fast targets PASS warm, and 68 of 79 cold.** Six fail only cold, at
10.0-11.9 s against a 10 s cap: launch, ex8_4_4, smallinvDAXr3b010-011,
ex8_5_3, ex6_2_6 and flay04m. SCIP has no compile step, so the cold count is
the fair comparison. On instances held out from development the pass rate is
lower; see the README's Status section. These targets pass now and were open in
earlier rounds: hadamard_5 (7.0 s), autocorr_bern20-10 and 30-04, kall_diffcircles_8 (18.7 s, cap 64.5),
ex8_4_4 (8.9 s), ex8_4_5, ex6_2_6, flay04m, csched1a, batch and launch.
ex8_4_4 and smallinvDAXr3b010-011 (9.7 s) are within 15% of the 10 s cap.

### Out of reach (documented, not closed)

| instance | dual | primal | best known | why |
|---|---|---|---|---|
| tln5 | 9.72 | 12.6 | 10.3 | The RLT root bound is weak, and the solver reaches 7.9k nodes without closing it (SCIP: 0.22 s). It needs global branching or a primal heuristic for the integer-scaled products. |
| tln4 | 7.75 | 8.3 | 8.3 | The same structure as tln5. The incumbent is optimal but the bound stalls (12.5k nodes). |
| wastewater15m1 | 2008 | 2446.43 | — | The root gap is too large for bilinear McCormick. It needs stronger relaxations for bilinear terms (SCIP: 3.0 s). |
| smallinvDAXr5b150-165 | 88.082 | 88.120 | 88.105 | Timing out at 10.1 s, 359 nodes (SCIP: about 850 nodes). It needs a stronger node bound or a better incumbent. |
| kall_circlespolygons_c1p12 | 0.3211 | 0.3396 | — | The bilinear-area bound is weak, the same class as c1p13/c1r13. Gap-weighted pseudocosts (#65) cut its node count, but it is still over the 13.6 s cap. |

Smaller follow-ups found during review are not blocking and are tracked in
the loop notes:
- `sym_seconds` is not enforced while `lifted_form` is built.
- `_refine` cannot be interrupted.
- The test for gap weights is weak.
- There are node-count regressions under #65 on c52, ex14_1_6, ex9_2_6 and st_robot.
- The slow test `test_integer_neighbourhood_polish` depends on a wall-clock budget.
