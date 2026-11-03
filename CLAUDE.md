# Working on jaxglobal

Current state, numbers and priorities: [`docs/STATUS.md`](docs/STATUS.md).
Keep it up to date when a change finishes an item or a measurement is re-run.

## Measurement strategy: smoke tests, not long runs

Use short, targeted runs to decide where work is needed. Do not spend hours
confirming that something slow is slow.

- **Smoke tests first.** Pick a small, varied sample (about 10–20 instances,
  spread over size bands and model families) and run it with the usual 60 s
  limit. If the sample is slow or fails, the full set will be too; that is
  enough to act on.
- **Stop early.** Stop a run once it has shown what it needed to show, and
  report the partial result. A full-set run (hours) is for the final numbers
  of a finished change, not for finding the next problem.
- **Profile, don't wait.** When something is slow, profile one
  representative instance (`benchmarks/profile_root.py`) to see where the
  time goes, rather than re-running more instances.
- **Correctness beats speed.** A wrong answer (`report.classify` says
  `wrong`) takes priority over any speed work.

## Avoid overfitting

The goal is a better general solver, not a list of solved instances.

- **Fix the mechanism, not the instance.** Do not add code that targets a
  named instance or model family. A fix should be stated in terms of a
  general property (scaling, structure, a primitive, a tolerance).
- **After a fix, validate on different problems.** Run a few smoke tests on
  instances that were *not* used to find or develop the fix: other families,
  other size bands, and a few from the held-out set
  (`benchmarks/sets/heldout.txt`). Report them separately from the instance
  that motivated the fix.
- **Check for regressions.** Include a few instances the solver already
  solves, and compare node counts and status before and after.
- **Keep development and evaluation sets apart.** Never tune on the
  held-out set. Name any instance used for development in the PR, so later
  evaluations can exclude it.

## Benchmark conventions

- Runs go through `benchmarks/run.py`; results go to `benchmarks/results/`
  as CSV with the 1-min load per row. Check results against the MINLPLib
  reference bounds with `benchmarks/report.py` (`classify`).
- Report cold (compilation included, `JAXGLOBAL_CACHE=0` or an empty
  `JAXGLOBAL_CACHE_DIR`) and warm separately; cold is the fair comparison
  with SCIP.
- Every number quoted in docs gives its commit, date and machine.
