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

`int50.txt`: 50 MINLPLib instances with binary or integer variables (at most
50 variables), for the integer work (`docs/ROADMAP.md`, *Integers: plan*).
`benchmarks/sets/make_int50.py` selects them (seed 2026) from
`minlplib/instances_int.csv` (`python benchmarks/minlplib/fetch.py --integer`,
220 instances): those with a reference solution closed to 1e-4, 14 / 12 / 12 /
12 from the bands of 1-5, 6-10, 11-20 and 21-50 integer variables, taken
round-robin over binary-only or general-integer and convex or not, at most 3
per model family. SCIP runs through GAMS (`--solvers scip-gams`).

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
| + one LP row per linear part, cold (`start50_dedup_cold.csv`) | 23 | 0 | 24 | 3 | 43.0 |
| + one LP row per linear part, warm (`start50_dedup_warm.csv`) | 25 | 0 | 22 | 3 | 22.6 |
| + one relaxation trace per cut, cold (`start50_cuts_cold.csv`) | 23 | 0 | 24 | 3 | 43.0 |
| + one relaxation trace per cut, warm (`start50_cuts_warm.csv`) | 25 | 0 | 22 | 3 | 24.0 |
| SCIP (`baseline.csv`) | 44 | 1 | 5 | 0 | 1.9 |

The four pooling instances and `nous1` were rerun alone for issue #12 (LP rows
only along each part's nonlinear variables, `results/issue12_before_*.csv`
and `issue12_after_*.csv`, load 18 to 57): 2 of 5 solved, 0 wrong, before and
after; at 60 s warm the `pooling_rt2tp` bound improved from -5528 to -4578
(optimum -4392). The full set was not rerun.

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

A profile of the nine instances that ended without any bound (cold) showed
that, past compilation, the time went to the LPs of the node bound and OBBT,
not to FBBT or the relaxations (tens of ms for 16 boxes). Those LPs have a row
per convex part and cut point (2n + 1 points plus the Kelley points), and the
dense IPM costs rows x n^2 per iteration: about 11,000 rows by 47 columns on
`pooling_adhya1stp`. A linear constraint gives the same cut at every point, so
most rows were copies (`st_rv7`: 1,281 rows, 21 distinct). Linear parts, read
off the scalarized tape (`lp._affine_parts`), now get one row, from the box
center. Measured one at a time, a call on 16 boxes went from 8.1 to 2.2 s on
`wastewater15m1`, 13.3 to 8.7 s on `pooling_adhya1stp` and 0.76 to 0.14 s on
`st_rv7`; root OBBT from 20 to 4.5 s, 33 to 20 s and 1.2 to 0.1 s. On `start50`
every status change was a gain (5 cold, 3 warm), and instances solved before
ran about 1.5x faster. The run started at load average 2 but other jobs
raised it to 40 to 80 by the end, which only slows it down.

The node bound and OBBT kernels traced the relaxation more often than
needed: each cut evaluated it for the values and again inside `jax.jacfwd`,
and the interval bounds of f, g and h took three more evaluations. Cuts now
take the values from the forward-mode pass (`has_aux`), and the interval
bounds, which do not depend on the point, from the first cut (`lp._cuts`).
On `st_rv7` the bounder went from 72k to 44k jaxpr equations (trace 10 to
4.3 s, compile 9.5 to 3.9 s) and OBBT from 64k to 46k, with the same bounds;
`st_rv7` now solves cold (33.6 s) and `pooling_adhya1stp` gets its first bound
at about 31 s instead of none. On `start50` the counts are unchanged: these
instances were not limited by it, and the node counts are identical (the warm
times are within the run's variation, the same on `kriging_peaks-red030`). Each
kernel still evaluates cuts at two call sites (initial and Kelley points;
the two OBBT rounds); merging them would cost runtime.