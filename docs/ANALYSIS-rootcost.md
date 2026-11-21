# Root cost on the larger stage-2c models, 2026-11-20

**Question.** On 10 of the 13 larger stage-2c models (300–1000 variables or constraints), jaxglobal processes zero nodes in 60 s. Where does the time go?

## Profile

`benchmarks/profile_root.py --modes cold --time-limit 60` on 12 of the 13; m29 was left out, as it was killed at 10–12 GB in the benchmark run. m2 and m129 did not return within the profiler's child timeout (300 s).

Seconds from the start of `minimize`:

| model | domain bounding ends | bound kernel trace | bound kernel compile | first bound call | local-callback compiles |
|---|---:|---|---|---|---:|
| m24 | 26.7 | 27.5–30.5 | 31.1–41.0 | 41.0–41.4 | 10.3 s |
| m79 | 89.3 | – | – | – | – |
| m84 | 0.1 | 1.3–21.9 | 23.2–35.8 | 35.8–123.7 | 8.6 s |
| m211 | 4.3 | 5.4–33.8 | 35.0–61.5 | – | 21.9 s |
| m269 | 0.1 | 2.1–7.3 | 8.6–23.9 | 23.9–24.7 | 11.3 s |
| m303 | 4.3 | 5.3–30.3 | 31.2–50.4 | 50.4–124.3 | 91.0 s |
| m305 | 3.4 | 4.4–7.2 | 7.7–17.3 | 17.3–17.9 | 9.3 s |
| m348 | 5.0 | 7.0–50.1 | 51.1–71.7 | – | 9.1 s |
| m363 | (61+) | – | – | – | 3.1 s |
| m388 | 4.1 | 5.8–39.6 | 40.4–52.6 | 52.6–259.5 | 15.8 s |

The profiler's timer around kernel calls blocks on each call. The "first bound call" column is therefore the call's full length, but the profiled solves also overrun by that much. Run without it (`run.py`), m84, m303 and m388 return at about 60 s, with the call dropped at the deadline (`_await`).

## Three costs

1. **Domain bounding** (models with unbounded variables).
   - m24: 26.7 s, of which 22.8 s is the feasible-point search.
   - m363: the whole budget, in the feasible-point search.
   - m79: 89 s, past the limit. Two causes:
     - `objective_only_bounds` evaluates one eager McCormick pass over the gradient per candidate variable (170 candidates, a 3,556-equation gradient). Relaxing the whole gradient once instead traces to 137k equations: 75 s of trace and 24 s of compile, no cheaper.
     - The first local solve compiles the Lagrangian Hessian inside the solve: 165 s at XLA's defaults, and neither the deadline nor pounce's `max_wall_time` can interrupt a compile.
2. **The bound kernel's trace and compile.**
   - The trace alone takes 20–43 s on m84, m211, m303, m348 and m388, and the compile another 12–26 s.
   - On m211 and m348 the kernel is not ready by the limit.
3. **The first bound call.** 88 s (m84), 74 s (m303), 207 s (m388): one LP batch for 16 boxes. This is the same as m47 in stage 2b, now on three of thirteen models.

Smaller items:
- The local-solver callbacks take 8–22 s to compile on most models (91 s on m303).
- The final `_verify`, which evaluates eagerly, ran 6–17 s past the limit on m79, next to a kernel still being traced in the background.

## Changes (commit 18c53bf)

- **`local._TieredJit`.** The Lagrangian Hessian compiles at `-O0` first. When that compile takes over `HESS_SLOW_O0` (3 s), the `-O0` executable is kept; otherwise the Hessian is compiled again with the defaults, as before.
  - m79: 165 s → 8.9 s, 0.7 ms per call.
  - Elsewhere, measured cold: the `-O0` compile takes 0.1–1.5 s and the default one 0.2–3.6 s.
  - The verdict is kept per HLO hash for the process. A persistent-cache hit compiles in no time, which would otherwise mark a slow module fast in the solver's second callback instance.
- **`domain.objective_only_bounds`.** Stops after `_OBJ_ONLY_SECONDS` (5 s) or at the deadline. A variable left free only keeps its bounds.
- **`bnb._verify`.** The GIL switch interval is 1e-5 s while it evaluates.
- **m79:** 99–121 s with no point → 64 s with a feasible point (896,514; best known 64,142).

## Validation

All cold, 60 s.

**Stage 2c** (`--jobs 1`; `stage2c_root.csv` against `stage2c_branch.csv`, the branch before this commit):
- **Solved:** the same 3 (m395, m397, m404), at 47.4 / 42.6 / 42.4 s against 49.0 / 45.1 / 44.2 s.
- **Primals:** 10 of 20 models report one, against 9; m79 is new.
- **m411:** keeps the optimal 6545 (6605 in the run before).
- **Past 61 s:** m24 70.8 → 61.4 s, m79 98.9 → 64.1 s, m363 61.7 → 61.1 s, m414 61.9 → 60.4 s.

**Stage 2b, int50, start50 in the full runs:** 11 → 10, 47 → 46 and 41 → 40, against the arms run before this commit. Those arms ran in the previous container, and part of the int50 run overlapped two of my own m6 test solves.

**Paired reruns** on this container, old settings (`HESS_SLOW_O0` None, `_VERIFY_SWITCH` None, no `objective_only_bounds` budget) against new, models back to back (`s2c_chk{2,3,4}_*.csv`):

| model | old | new |
|---|---|---|
| gbd | 10.5 s | 10.0 s |
| nvs01 | 27.9 s | 27.8 s |
| st_e15 | 18.7 s | 18.3 s |
| st_miqp1 | 15.2 s | 15.0 s |
| tln4 | 57.9 s | 59.0 s |
| m6 | 37.8 s | 37.4 s |
| m391 | 44.2 s | 46.7 s |
| m441 | time limit | time limit |
| ex8_5_3 | time limit | time limit |
| kall_circles_c6b | 50.9 s | 53.4 s |
| kriging_peaks-red030 | 48.3 s | 48.4 s |

Same primal and dual values in both arms. The changes are neutral on these models. The drops in the full runs are this container's speed and load: m441, solved at 60.1 s in the previous container, times out in both arms here.

**Tests:** test_local_callbacks and test_domain pass.

## What is left (largest first)

- **The bound kernel's trace** (20–43 s) and **first call** (74–207 s) on the 200–400-variable models. Both scale with the dense LP the kernel builds per box. This is the issue's step 6, kernel size: which rows and which relaxations the kernel carries for these models, and whether a sparse or row-limited LP would do.
- **A primal fallback.** On these models nothing produces a point before the root bound. A time-boxed multistart of the local solver, which now compiles fast, run while the bound kernel builds, would report one.
- **The feasible-point search in domain bounding** (m24, m363) can take the whole budget. It needs a share of the time limit, not the deadline.

## Follow-up: a primal fallback while the root kernel builds

**Change (`bnb._FALLBACK_SHARE` = 0.25).** While the root bound kernel is still building and there is no incumbent, the main thread, otherwise idle, runs local solves (`local()`): the box center, then rounds of `_FALLBACK_STARTS` = 4 random starts. They stop at the first incumbent, when the kernel is ready, or after 25% of the time limit.
- **Not before** the main thread has waited `_FALLBACK_DELAY` = 15 s for the kernel. Next to a kernel that is nearly built, the solves only slowed its compile: m34 37 → 42 s, smallinvDAXr5b150-165 52 s → time limit, both with root incumbents of their own.
- **Not before** the local solver's callbacks and `screen`'s executables are compiled (`solve.ready`, `screen_ready`). A solve before that compiles them on the main thread, beyond the deadline's reach: m303 returned at 93 s and m84 at 118 s. The 25% counts from that point; the wait itself is idle time.
- **Outside the root's local-solve share** (`_ROOT_LOCAL_SHARE`). Each solve batch gets a share of the time left.
- **Polish:** a fallback incumbent that the root does not improve on is polished at the root's `update_incumbent` stage, with the root's budget.

**A dropped kernel call blocks later device work** (`bnb._device_busy`). XLA:CPU runs work behind a call still in progress: a tiny eager op dispatched behind a 16 s call took 16.3 s. Once m388 had a fallback incumbent, its final `_verify` waited for the ~200 s first bound call that `_await` had dropped at the deadline, and the solve returned at 258 s.
- `_await` now remembers the outputs of a dropped call. While one still runs, `minimize` skips `repair_incumbent` and `_verify`, and returns the screened incumbent with `verified=False` and a message saying why.
- Test: `test_dropped_call_marks_device_busy`.

**Stage 2c** (`stage2c_fb.csv` against `stage2c_root.csv`):
- **Primals:** **14 of 20** models report one, against 10.
  - m2 407,600 (optimum 348,337).
  - m84 **1,062,700**, the best known value; SCIP's incumbent is 1,069,800.
  - m129 1,122,800 (best known 1,086,000; SCIP 1,418,300).
  - m348 199.5 (best known 291.5).
- **Solved:** the same 3, in 48.2 / 43.2 / 45.0 s.
- **Overruns:** at most 63.9 s (m79, m388).

**Stage 2b, int50, start50** (against `s2c_root_cold.csv`, `s2c_int50_root_cold.csv`, `s2c_start50_root_cold.csv`, same container): 10 → 10, 46 → 46, 40 → 40. No model is slower by more than 10 s; in stage 2b the largest changes are m391 −7.6 s and m399 +4.4 s.

**Tests:** test_compile_gate, test_local_callbacks and test_incumbent pass. The exception is `test_integer_neighbourhood_polish`, which fails with a fresh kernel cache on `main` (9d182d4) and on every commit of this branch, and passes on slower runs: its root polish has a budget that grows with the elapsed time, and the test stops after one iteration. That is pre-existing and left for a separate fix.

## Follow-up: the lifted quadratic LP up to 2n columns

m388 and m129 are quadratic (MIQCP), but their lifted forms have 594 and 694 columns, over `quadratic.MAX_COLUMNS` (400). They got the reduced-space kernel, whose first bound call on m388 takes about 200 s.

**Change.** `bnb._quadratic_form` allows `max(MAX_COLUMNS, _QUAD_COLS_PER_VAR * n)` columns, with `_QUAD_COLS_PER_VAR` = 2.

**Which models change.** Across stage 2b, stage 2c, int50 and start50, only these two. The smallinvDAX models (31 variables, 496 columns) stay reduced-space. Note that `quadratic.MAX_COLUMNS` is a default argument, so setting the module constant later changes nothing; the override has to be passed in the call.

**m388:**
- A bound call for 16 boxes takes 1.5–2 s.
- Root bound −1601.9 (optimum −1338.25).
- The root stages leave no time to search: make_local 10 s, root local solves 14 s with no incumbent, probing 10 s.

**Stage 2c** (`stage2c_qcols.csv` against `stage2c_fb.csv`):
- **Finite bounds:** 10 → 12 models (m129 605,793 against SCIP's 684,752 at 60 s; m388 −1601.9).
- **Primals:** 14 → 15.
- **Solved:** the same 3.
- **m348:** returned at 68.4 s in this run, from the fallback path, run-to-run.
