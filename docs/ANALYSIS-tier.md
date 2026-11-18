# Lighter XLA compiles on a cold cache (minlp.org stage 2b), 2026-11-18

**Question.** Issue #87's next step 1 is the time margin: several stage-2b models finish at 46–51 s, and the bound kernel's XLA compile is the largest fixed root cost. Does compiling at LLVM `-O0` buy that margin, either as a tiered root kernel (`_FAST_COMPILE`, off since ANALYSIS-compile) or for the kernels called less often than the bound kernel?

**Setup.**
- Machine: 4-core cloud VM, idle apart from the run (1-min load 2.4–4.6 at each run's start, which is the previous solve's own load).
- `run.py --lib minlporg --instances instances_stage2b.csv --time-limit 60 --jobs 1`, each arm with a fresh `JAXGLOBAL_CACHE_DIR` (cold), arms run one after another.
- Arms differ only in module constants, set through the new `JAXGLOBAL_SET` hook of `run.py`.
- A model counts as solved by the issue's rule (optimal, ≤ 61 s, primal within 1e-3 of SCIP's; m6 against 1.5364151807).

## Arms

| arm | change (all `-O0` = `{"xla_backend_optimization_level": 0}`) |
|---|---|
| base | `main` at 9d182d4 |
| tierA | `_FAST_COMPILE = -O0`: the root bound kernel is compiled at `-O0` first, then at the defaults when the search starts (the existing option) |
| tierB | as tierA, but the `-O2` compile starts as soon as the `-O0` executable is ready (`_TIER_EARLY`, new) |
| lightC | `-O0` for the dive, OBBT, pool-cut and screening kernels (`_LIGHT_COMPILE`, `_LIGHT_KERNELS`, new); the bound kernel unchanged |
| lightD | lightC, plus the local solver's callbacks (`local.COMPILE_OPTIONS`, new) |
| lightE | `-O0` for the root OBBT kernel only |

## Results

Solve time (s) of the ten models `main` solves; all other models time out in every arm.

| arm | solved | m6 | m34 | m391 | m394 | m396 | m399 | m403 | m406 | m410 | m474 | sum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 10 | 39.2 | 35.9 | 46.9 | 49.0 | 30.0 | 43.0 | 29.2 | 36.2 | 50.2 | 43.2 | 402.7 |
| tierA | 9 | 34.9 | 41.2 | 50.2 | 51.7 | 34.3 | 48.8 | 30.2 | 41.5 | 52.8 | **60.8** | 446.5 |
| tierB | 10 | 34.4 | 38.0 | 46.7 | 52.1 | 32.1 | 44.5 | 32.0 | 41.5 | 55.2 | 41.3 | 417.7 |
| lightC | 10 | 40.8 | 37.8 | 54.4 | 56.7 | 28.7 | 42.3 | 29.0 | 36.8 | 47.1 | 42.5 | 416.1 |
| lightD | 9 | 40.4 | 38.4 | 54.5 | **60.5** | 31.4 | 43.8 | 31.7 | 38.2 | 51.5 | 46.0 | 436.5 |
| lightE | 10 | 41.1 | 36.3 | 46.6 | 49.7 | 33.5 | 43.6 | 30.1 | 38.4 | 49.3 | 43.2 | 411.9 |

Raw rows: `benchmarks/results/s2c_{base,tierA,tierB,lightC,lightD,lightE}_cold.csv`. The base arm reproduces round 5 (`stage2b_round5_cold.csv`): the same 10 models, each within 3.5 s.

No arm improves on `main`. The bests per model (m6 34.4 in tierB, m410 47.1 in lightC) are offset by larger losses elsewhere in the same arm.

## Why

**Root timeline, m410 cold, base** (`profile_root.py`; s from the start of `minimize`):

| s | main thread | compiler threads |
|---|---|---|
| 0–2.8 | `bound_domain` (waits for its FBBT compile) | FBBT |
| 2.8–7.0 | cut pool, `make_local` | — |
| 7.0–16.8 | waits for the bound kernel | bound kernel 7.6–16.5; local callbacks 7.0–16.2; screening 7.0–13.3 |
| 16.8–17.9 | first incumbent | dive 16.5–20.3 |
| 17.9–27.7 | root dives: waits 2.4 s for the dive kernel, then 6.5 s of fixed local solves | OBBT 17.9–25.2 |
| 27.7–32.7 | root OBBT, probing, OA heuristic | — |
| 32.7–49.8 | search | — |

From 7 s to 25 s the four cores compile without a break. The bound kernel's compile (8.9 s here) is not stretched much by the compiles beside it: about 8 s alone (ANALYSIS-trace).

1. **The tiered bound kernel (tierA, tierB).**
   - `-O0` shortens the root-bound wait by only 1.2 s on m410 (root-bound stage 9.8 → 8.6 s).
   - The `-O2` recompile then takes about three cores for 10+ s. In tierA it overlaps the start of the search; in tierB it overlaps the root dives and root OBBT. On m410 tierB's OBBT stage grew 0.7 → 4.0 s, 3 s of it a synchronous main-thread compile.
   - Until the `-O2` executable is ready, bound calls run on the slower `-O0` kernel (about 30% slower, ANALYSIS-compile).
   - Net: slower on 9 of 10 models in tierA, 7 of 10 in tierB.
2. **The secondary kernels (lightC, lightD).**
   - Their `-O0` compiles are shorter: on m410, OBBT 7.3 → 4.6 s and dive 3.8 → 2.4–2.9 s, and the search starts about 1.5 s earlier (profiles, single runs).
   - The dive, screening and pool-cut kernels also run in the search. m391 and m394 spend 24–28 s there (6–7k nodes) and lose 7–8 s, more than the root gains anywhere.
   - The local solver's callbacks run in every local solve; lightD loses m394 outright.
3. **OBBT only (lightE).** The root OBBT kernel is called once, but its compile overlaps the root dives' local solves, which are not compile-bound. The arm is within noise of base: sum 411.9 against 402.7 s, single runs.

## Conclusion

Lowering XLA's optimization level does not buy the time margin on this hardware.
- The bound kernel's `-O0` saves about a second at the root, and the recompile costs more.
- Every other kernel worth compiling lighter is also hot in the search.

`_FAST_COMPILE` stays off. The new options (`_TIER_EARLY`, `_LIGHT_COMPILE`/`_LIGHT_KERNELS`, `local.COMPILE_OPTIONS`) are off by default and are kept for re-measuring on other hardware.

The margin on the models near the limit lies elsewhere:
- **m391, m394:** 24–28 s of search, 6.6–7.5k nodes. This is node throughput or tree size, not compile.
- **m410:** 17 s of search (293 nodes) after an 18 s compile-bound root.
- **The CLay models:** the root dives' fixed local solves take 3–6.5 s.
- **Remaining compile-side lever:** fewer or smaller kernels (issue #87 next steps 5 and 6), not cheaper compiles of the same kernels.
