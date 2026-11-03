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

## November 2026 smoke runs

All on a 4-core cloud VM, 60 s, cold, two at a time, jaxglobal only; see
`docs/STATUS.md` for what they show.

- `scale_int100_cold.csv`: the first 150 of the 328 instances in
  `sets/int100.txt` (by size), stopped on purpose once the pattern was clear.
  `benchmarks/scale_2026_11.sh` resumes it (and runs SCIP);
  `benchmarks/scale_report.py` prints the tables against `int328_stage5.csv`.
  `jit1` in it is the wrong answer fixed afterwards.
- `smoke_repair_after.csv`, `smoke_repair_after2.csv`, `smoke_repair_final.csv`:
  validation of the incumbent-repair fix on instances from other families
  (the first file was run with an earlier version of the fix that differed
  only on `jit1`-like cases; the other two with the final one).
