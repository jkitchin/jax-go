# Root and search scheduling on m410_CLay0203HRS (cold, 60 s), 2026-11-17

**Question.** m410 timed out cold with incumbent 42098 (optimum 41572.9) and bound 1373.5, after 55–99 nodes; SCIP solves it in 0.6 s. Where does the time go after the root bound, and which scheduling rules buy bound calls?

## Diagnosis
Sources: `benchmarks/profile_root.py` cold, with two additions. `bnb._TRACE` is now recorded per stage, sub-event and search iteration. Each local solve's caller chain is noted.

**Base, load 2.5.**
- The root bound arrives at 22.2 s; the root stages run to 35.9 s.
- The search gets 26 s: 7 iterations, 99 nodes. Of that, 19.1 s goes to local solves (`update_incumbent` 13.6 s, `pool_nlp` 5.5 s). The bound calls take 0.3–1.8 s per 16 boxes.

**What finds the optimum.**
- Warm, the outer-approximation heuristic finds 41572.9 in its first call (1.5 s). The bound then jumps 0 → 41520.9 within about 15 s, over about 300 nodes.
- Cold, its own cut program (`jacfwd` of every row's relax.py relaxation: 76k jaxpr equations; 5–7 s trace, 1.5 s lower, 11–23 s compile) is only started after the root stages. It finished at 57 s, and the call never ran.

**A bug in the root OA call.**
- With the root local share (`_ROOT_LOCAL_SHARE`) spent, `solve_local` returns its starts unsolved. A root OA call's completions were then the MILP points themselves, infeasible, `F = inf`.
- Each no-good cut then excluded a good assignment for good. The MILP proposed the optimal assignment (value 41572.46) at round 8, and it was lost.

**Main-thread eager compiles.** `dive()`'s `jnp.zeros` / `jnp.asarray` compiled tiny XLA programs on the main thread: 0.8–2 s each, 4.7 s in all, next to the background compiles.

**Late rebuilds and waits.**
- `pool_judge` started the no-rows pool kernel 7 s before the limit (a 12 s build).
- The root waited 5.5–9 s for an OBBT compile, for no tightening.

**Size of OA's own program** (jaxpr equations):

| model | equations |
|---|---:|
| m403 | 26k |
| m396 | 27k |
| m406 | 31k |
| m391 | 32k |
| m399 | 34k |
| m394 | 39k |
| m420 | 41k |
| m421 | 53k |
| m400 | 54k |
| m441 | 61k |
| m426 | 67k |
| m472 | 70k |
| m410 | 76k |
| m422 | 77k |
| m473 | 95k |
| m474 | 98k |

## Changes (`jaxglobal/bnb.py`, `jaxglobal/milpbound.py`)

**On by default:**
- **`_OA_POOL`.** With a cut pool (reduced-space LP), `MilpBound(tangents=...)` takes OA's cuts from the pool's tangent kernel (`lp.make_pool_cuts`, compiled behind the root bound kernel anyway). The parts are [cv_f, cv_g, cv_h, −cc_h], and v = b + a·p. There is no own program, and OA can run in the root stage.
- **`root["oa"]`.** OA's completions are not counted in, or cut by, the root local share. OA has its own budget (`_OA_SHARE`).
- **`_OA_WARM`.** An improving completion is solved again from itself with `_WARM[0]`.
- **`milpbound.COMPILE_OPTIONS = {"xla_backend_optimization_level": 0}`.** OA's own program on m410 compiles in 9.3 s instead of 13.7 s; a call takes 5.0 ms instead of 6.4 ms; values agree to 2e-10. `ready()` is split into `lower()` and `compile()`.
- **`_SEARCH_LOCAL_SHARE = (0.5, 0.25)`.** With integer variables and a time limit, `update_incumbent`'s local searches may take at most this share of the search's wall time (without, with an incumbent). The first batch always runs. Continuous problems are excluded: on kall_diffcircles_8 these solves are the incumbent heuristic.
- **`_PLAIN_FIT = 2`.** The no-rows pool kernel is built only with 2× the root bound kernel's build time left.
- **`_ROOT_DIVE_NOWAIT`.** With an incumbent, the root dives skip a dive kernel that is not yet compiled.
- **NumPy dive arguments.**

**Options, off (mixed evidence):**
- **`_OA_EARLY` / `_OA_EARLY_EQNS = 35k`.** OA's own program is traced on the kernel tracer and compiled behind the bound kernel. m406 was solved in 2 of 3 pairs with it (base: 0 of 3), but m420–m422 lost 10–60% of their nodes.
- **`_POOL_NLP_SEARCH`** (pool NLP share of search time, not of total time). cvxnonsep_normcon30 went from 123 nodes to unsolved.
- **`_OBBT_HELD`** (a held or running OBBT compile counts in root OBBT's cost). cvxnonsep_nsig30 went from 115 nodes to more than 2047.
- **`_OA_ROUNDS = 8`.** 32 rounds was tried; m420 searched 1019–1279 nodes, against 3327.

**Diagnostics (`benchmarks/profile_root.py`):**
- The wait wrapper now releases the compile gate. Before, m441 waited 38 s on an OBBT compile that never started.
- Local solves record their caller chain.
- `bnb._TRACE` is recorded.

## Validity
- Only incumbent-only work and its timing change.
- Every candidate still passes `screen`.
- OA's MILPs are never bounds. The tangent cuts are valid minorants on the box, the same cuts the pool adds to node LPs.
- `milp_root` (off by default) keeps its own cut program, at -O0 when OA built it first; its margin is 1e-6 relative.
- check_wrong seeds 0 and 1: no wrong answers.

## Measurements

**m410 cold, final, load 2–4:**
- **Profile run:** root bound 21.5 s. The root OA call (35.4–38.0 s) finds 41572.9. The search then runs 16 s, 293 nodes, optimal at 53.7 s.
- **run.py:** 51.6 s and 53.0 s.
- **Base:** time limit, in each paired run.

**Stage-2b, all 19 cold, 60 s, `--jobs 1`, load 3–6, against `stage2b_round4_cold.csv`:**
- **Solved 9 → 10.** m410 gained.
- **CLay models, faster:** m391 50.4 → 48.7 s, m394 54.4 → 50.7 s, m396 36.9 → 32.9 s, m399 49.0 → 41.1 s, m403 36.6 → 30.5 s, m406 44.0 → 37.7 s, m474 48.9 → 44.2 s.
- **k-means, more nodes:**

  | model | bound | nodes | incumbent |
  |---|---|---|---|
  | m420 | 0.69 → 0.75 | 12.8k → 16.6k | |
  | m421 | 1.48 → 1.60 | 9.2k → 14.1k | |
  | m422 | 2.07 → 2.30 | 6.4k → 11.3k | |
  | m426 | 0 → 0.014 | 765 → 3811 | 0.76 → 0.51 |

- **m400:** 15 → 31 nodes, still no incumbent.
- **Slower:** m6 41.6 → 54.0 s and m34 53.0 → 59.9 s, but at higher load (3.2 / 5.3 against 2.3 / 2.2).
- **m472:** incumbent 250 → 139 (maximization: worse); the A/B pairs had base at 140 as well.

**MINLPLib, 60 s cold, `--jobs 2 --threads-per-job 2`, back to back per half, load 10–21:**
- **int50:** 29 → 27 solved; gained m3. The three losses (clay0203m, flay03m, gear2) were in a half where the new arm ran at load 21 against 14. Rerun as `--jobs 1` pairs at load 8–12, all were solved in both arms with equal node counts.
- **start50:** 31 → 31 solved, no losses, geometric mean time 21.5 → 20.9 s.
- **Small int50 slowdowns in the reruns:** gear2 17 → 23 s, prob02 7.3 → 8.7 s, sporttournament06 9.2 → 11.0 s. They are within the load noise but not ruled out.

## What is left
- **Root dives and their polish** (5–8 s, for 42098 on m410), and the root OBBT compile wait.
- **A cheap OA program for models without a pool:** affine rows from `lp._affine_forms`, relaxing only the nonlinear rows. Then `_OA_EARLY` could pay off without the k-means cost.
- **Overlapping local solves with in-flight bound calls:** not tried, since the cores are already contended.
- **m400:** still no incumbent; it needs a CLay hull incumbent heuristic.
