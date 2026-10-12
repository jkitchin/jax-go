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
| SCIP (`baseline.csv`) | 44 | 1 | 5 | 0 | 1.9 |

Most of the cold time is tracing and compiling the bounding kernels, whose
size grows with the number of scalar operations in the model (a scalar `.nl`
model of 1,166 operations gives a 500,000-equation bound kernel). In the cold
and warm runs five instances with unbounded variables were killed by the
runner while `bound_domain` compiled its FBBT kernel; the time limit now covers
that phase too.
