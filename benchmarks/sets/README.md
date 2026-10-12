# Instance subsets

`start50.txt`: 50 continuous MINLPLib instances (≤ 50 variables) for a quick
re-measurement against the phase-0 `baseline` run, chosen from
`results/baseline.csv` with a fixed seed (2026) by baseline outcome and size
band (1–5, 6–10, 11–20, 21–50 variables):

| group | count | rule |
|---|---:|---|
| gap | 20 | SCIP solved, jaxglobal did not (0/6/8/8 per band; at most 3 per model family) |
| unsupported | 12 | jaxglobal refused them for unbounded variables (3 per band) |
| solved | 12 | jaxglobal solved them (5/5/2/0 per band): regressions and speed |
| wrong | 6 | jaxglobal's baseline answer contradicted the reference (later fixed to time-outs) |

Run with

```
python benchmarks/run.py --solvers jaxglobal --tag start50 --time-limit 60 \
  --names $(grep -v '^#' benchmarks/sets/start50.txt | awk '{print $2}' | paste -sd, -)
```

## Results on `start50`

Checked against the MINLPLib reference bounds with `report.classify`; 60 s,
4 jobs. "Warm" reruns the same instances with the kernels already in the disk
cache (`JAXGLOBAL_CACHE_DIR`), so it measures search without compilation.

| run | solved | wrong | unsolved | unsupported | SGM time (s) |
|---|---:|---:|---:|---:|---:|
| phase-0 baseline (`baseline.csv`) | 12 | 0 | 26 | 12 | 40.5 |
| jax 0.11, loaded machine (`start50.csv`) | 12 | 0 | 36 | 2 | 53.5 |
| cold (`start50_cold.csv`) | 12 | 0 | 36 | 2 | 50.5 |
| warm (`start50_warm.csv`) | 18 | 0 | 30 | 2 | 32.1 |
| FBBT interpreter, cold (`start50_interp_cold.csv`) | 15 | 0 | 32 | 3 | 48.3 |
| FBBT interpreter, warm (`start50_interp_warm.csv`) | 20 | 0 | 27 | 3 | 29.2 |
| + relaxation interpreter, cold (`start50_relax_cold.csv`) | 18 | 0 | 29 | 3 | 47.2 |
| + relaxation interpreter, warm (`start50_relax_warm.csv`) | 21 | 0 | 26 | 3 | 30.5 |
| + scheduled sweeps, cold (`start50_sched_cold.csv`) | 18 | 0 | 29 | 3 | 49.7 |
| + scheduled sweeps, warm (`start50_sched_warm.csv`) | 22 | 0 | 25 | 3 | 28.5 |
| SCIP (`baseline.csv`) | 44 | 1 | 5 | 0 | 1.9 |

Most of the cold time is tracing and compiling the bounding kernels, whose
size grows with the number of scalar operations in the model (a scalar `.nl`
model of 1,166 operations gives a 500,000-equation bound kernel). In the cold
and warm runs five instances with unbounded variables were killed by the
runner while `bound_domain` compiled its FBBT kernel; the time limit now covers
that phase too.

With the FBBT tape interpreter (`jaxglobal/fbbt_interp.py`) FBBT's compile
time no longer grows with the model; new solves include `ex5_3_2` (22
variables) and `pooling_foulds2tp` (36). The relaxation and LP kernels are
still compiled straight-line and now dominate the cold time (kriging still
does not finish compiling within the limit).

The relaxation interpreter (`jaxglobal/relax_interp.py`: McCormick values and
subgradients, the univariate hulls' curvature bounds and alphaBB's interval
Hessian) compiles in 1.5-3.5 s whatever the model, but under `vmap` it is
10-300x slower per call than the straight-line kernels, which compile in a few
seconds up to several hundred rows (the 480-row `st_rv7` constraints: 2 s). It
is therefore used only for functions of at least 600 rows
(`relax_interp.MIN_ROWS`). No `start50` function is that large, so its
`start50` rows run the same code as the FBBT-interpreter rows; the differences
(18 vs 15 cold, 21 vs 20 warm) are run-to-run variation on a loaded machine
(load average up to 217 during the warm run). With the interpreter applied from
50 rows instead, start50 dropped to 12 / 17 solved cold / warm. Outside the set,
`kriging_peaks-red030` (one 1,162-row function), which never finished
compiling before, now solves cold in 40 s; its warm solve (35 s, 603 nodes) is
bound by the interpreter's per-call cost.

The interpreters then sweep step by step instead of row by row
(`fbbt_interp.schedule` / `layout`: groups of independent rows, each step
evaluating every opcode once, vectorized over its rows, and writing its results
as one contiguous block so XLA updates the carried slot array in place). In a
warm `kriging_peaks-red030` solve FBBT was 466 of the bound kernel's 519 ms per
call (16 boxes); stepping brought it to 141 ms and the warm solve from 25.6 s
to 8.1 s (603 nodes either way). Small tapes have only a few rows per step and
were up to 2x slower that way for 256 boxes, so FBBT sweeps step by step from
300 rows (`fbbt_interp.SCHEDULE_MIN_ROWS`) and row by row below; the relaxation
interpreter (600+ rows) always steps. The `start50` difference (22 warm) is
within run-to-run variation (load average ~50 during the runs).
