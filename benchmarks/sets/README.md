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
