# Stage 2c: 20 new minlp.org instances (held-out and larger), 2026-11-20

**Question.** Do the stage-2b gains generalize to instances not used for tuning, and where does jaxglobal fall short on larger models?

## Selection

The selection was made by rule, not by hand (`benchmarks/minlporg/instances_stage2c.csv`; skipped ones in `skipped_instances_stage2c.csv`).

- **Held-out:** every nonlinear minlp.org instance of at most 300 variables and 300 constraints not used in stages 1, 2 or 2b. That is 12 instances; 7 convert, all constrained layout (m395, m397, m404, m407, m411, m413, m414).
  - The k-means (m427, m428) and pooling-competition (m475, m476) instances have no model file on minlp.org; m453 fails in GAMS.
- **Larger:** instances of 300–1000 variables or constraints, the smallest of each minlp.org problem, excluding the held-out families, taken in size order until 13 converted.
  - m2, m24, m29, m79, m84, m129, m211, m269, m303, m305, m348, m363 and m388.
  - Skipped: m62 and m213 (zip archives), m80 (a data file), m248 (a GAMS error).

**Arms.** Cold, 60 s, `--jobs 1`, on the 4-core VM:
- SCIP 10 through pyscipopt.
- `main` at 9d182d4, in a separate worktree.
- This branch at 0e6a699.

**Solved:** optimal, at most 61 s, and a primal within 1e-3 (relative) of the reference. The reference is SCIP's value when SCIP proves optimality, else minlp.org's best known value.

## Results

| | SCIP | `main` | branch |
|---|---:|---:|---:|
| held-out (7) | 7 | 3 | 3 |
| larger (13) | 4 | 0 | 0 |
| all (20) | 11 | 3 | 3 |

No wrong answers in either jaxglobal arm. Raw rows: `benchmarks/results/stage2c_{scip,main,branch}.csv`.

### Held-out constrained-layout models

- **Solved in both arms:** m395, m397 and m404. The branch is faster on all three: 56.9 → 49.0 s, 58.4 → 45.1 s and 56.4 → 44.2 s.
- **Unsolved in both:** m407, m411, m413 and m414 (SCIP: 1.3–6.3 s).
  - m413 has the optimal incumbent in both arms.
  - m407 and m414 stall at 41386 and 64167 (optimum 40262) in both arms.
  - m411's incumbent is 6545 (optimal) on `main` but 6605 on the branch.

The speed gain carries over to unseen instances of the family the work was tuned on; no solve is gained.

### Larger models: the root is not reached

- **No nodes at all:** 10 of 13 process zero nodes in 60 s in both arms, so no root bound is computed and no root heuristic runs. Most have no primal either. The exceptions:
  - m305 (1,257 nodes on `main`; primal within 0.01% of SCIP's).
  - m269 (7 / 15 nodes).
  - m24 (0 / 7 nodes, primal 5645 against 5924).
- **Memory:**
  - m29 (1,000 variables) is killed at 10.6 / 12.0 GB.
  - m2 peaks at 8.4 GB.
  - m84 and m211 near 5 GB.
- **Time limit overruns:** m79 returns at 121 s on `main` and 99 s on the branch; m24 at 67 / 71 s. A call or build still runs past the deadline at this size.
- **SCIP:** solves m2, m29, m269 and m388 in 0.1–14 s. On the rest it has an incumbent within 1e-3 of the best known value on all but m84, m129 and m211.

## What it says

1. **The stage-2b work did not overfit, but it didn't move the solved count here either.** On the held-out family the branch is 8–13 s faster on what both arms solve. The constrained-layout models that neither arm solves (m407, m411, m413, m414) are the same failure modes as stage 2b's m400: weak hull and big-M bounds, and branching.
2. **Above about 200 variables the bottleneck is the root's setup, not the search.**
   - Kernel trace and compile time, memory, and the first bound call: the issue's step 1 (time margin) and step 6 (row-level pruning, kernel size), now on most of a sample rather than one model (m47).
   - A solver that cannot produce a first bound or a first feasible point within 60 s on 10 of 13 such models will not be competitive there, whatever the search does.
3. **Deadline overruns at this size** (m79 99–121 s): `_await` covers kernel calls, so something else on the root path (a trace, a local solve, the domain bound) runs past the limit.

## Suggested next steps

- **A root-cost profile of the larger set** (`profile_root.py --modes cold`): trace, compile and first-call time, and memory, per kernel. This finds what dominates: the bound kernel's size (dense LP rows), alphaBB, or the OBBT kernel.
- **A fallback that always produces a primal:** a time-boxed multistart local solve before or alongside the kernel build. On 10 of 13 larger models jaxglobal reports nothing at all.
- **Find and bound the m79 / m24 deadline overruns.**
