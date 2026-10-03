# Benchmark results

`baseline.csv` is the phase-0 reference run: 322 continuous MINLPLib instances
(≤ 50 variables), 60 s limit, relative gap 1e-4, 4 concurrent jobs on an Apple
M-series laptop (14 cores). It is the "before" for phase 1 (bound tightening)
and is for internal comparison only. jaxglobal had no FBBT/OBBT at this point
and skips instances with unbounded variables.

Columns: solver, name, status, time (s), primal, dual (model's objective
sense), nodes, message. `python benchmarks/report.py --tag baseline` checks
each claim against the MINLPLib reference bounds and writes `baseline.md`.

Rerunning needs the instances (`python benchmarks/minlplib/fetch.py`). The
comparison solvers are optional: SCIP (`scip` on PATH), Couenne (AMPL build),
BARON through GAMS. Use `--solvers jaxglobal` where they are not installed.
