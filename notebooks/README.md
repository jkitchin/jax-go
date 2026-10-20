# Notebooks

Executed notebooks with prose, math and figures. Notebooks 01–06 show how jaxglobal works and how it performs; 07–13 write the
same optimization models in Pyomo and as plain Python/JAX functions for jaxglobal, solve both, and compare objective, certified
bound, time and status, with Pyomo's strengths (integers, mature solvers, sparse structure, duals, ecosystem) discussed alongside
jaxglobal's.

## How jaxglobal works

| Notebook | Contents |
|---|---|
| `01-relaxations.ipynb` | From a JAX function to bounds: the jaxpr, interval arithmetic, McCormick relaxations and αBB; how relaxations tighten when a box is split; first- vs second-order convergence; a 2-D relaxation surface. Writes `figures/relaxations.png`. |
| `02-branch-and-bound.ipynb` | Spatial branch and bound and its certificate `L ≤ f* ≤ U`; a hand-run pruning picture; convergence of the gap; what the bounding options buy. Writes `figures/convergence.png`. |
| `03-test-problems.ipynb` | The 14 unconstrained and 7 constrained test problems: formulas, a contour gallery with local-solver traps, feasible-region plots, and result tables against interval bounds and multistart. Writes `results/test_problems_results.json`. |
| `04-equation-based-solvers.ipynb` | The same problems in SCIP, Couenne, BARON and LindoGlobal via a `jnp`→Pyomo shim; what has to be reformulated by hand; who solves what and how fast. Writes `figures/benchmark.png` and updates the jaxglobal column of `results/compare_results.json`. |
| `05-throughput.ipynb` | Throughput of the batched `jit(vmap)` bounding kernel vs batch size, compile cost, and where a full solve spends its time. Writes `results/throughput_results.json`. |
| `06-gaussian-processes.ipynb` | Optimizing over trained Gaussian processes, after Schweidtmann et al. (2021), with the kernels from `jaxglobal.kernels` and the standard deviation, EI and PI from `jaxglobal.gp` (8 to 27 times fewer nodes than the `jnp` formulas, and everything certified): GP-mean minimization of the peaks function for N = 20 to 400 and four kernels, LCB/EI/PI acquisition functions (and why PI needs a margin), a chance constraint built from a second GP, and a short Bayesian-optimization loop. Compares against the `jnp` formulas, which leave the chance constraint and the loop uncertified. Takes about 10 minutes. |

## Comparison with Pyomo

| Notebook | Contents |
|---|---|
| `07-pyomo-overview.ipynb` | Overview, the lines-of-code helper, and a summary table built from the saved results of 08–13 |
| `08-pyomo-basics.ipynb` | Six-hump camel function, Haverly pooling problem |
| `09-control-flow.ipynb` | `where`/`clip`/`max`/`interp` (pipe regimes, steam tables) vs Pyomo GDP and `Piecewise` |
| `10-ode-models.ipynb` | Kinetic parameter estimation: explicit RK4 and `pyomo.dae` vs a Python loop |
| `11-neural-network-surrogates.ipynb` | Optimizing a trained tanh MLP: jaxglobal (with and without `crown`), hand-written Pyomo, OMLT |
| `12-things-pyomo-cannot-express.ipynb` | `fori_loop`, library functions, einsum, parametric families, including what failed |
| `13-many-switches.ipynb` | Motor price lists (one `jnp.where`) vs one binary per rating: scaling with the list length, a counterexample, and switches inside a time loop |

## Layout

* `src/*.py`: the notebooks as jupytext percent-format scripts (edit these).
* `*.ipynb`: generated from `src/` and executed, outputs included.
* `nbutil.py`: helpers for 07–13 (`loc` line counter, uniform solver wrappers, results table, result saving).
* `compare.py`: the `jnp`→Pyomo shim, hand reformulations and solver wrappers used by notebook 04. Run as a script
  (`PYTHONPATH=. python notebooks/compare.py` from the repository root), it fills in `results/compare_results.json` for problems
  not yet in it.
* `export_scip.py`: exports a model with `where`, `maximum`, `tanh`, `relu`, a `fori_loop` and an integer variable to SCIP
  (`jaxglobal.export.to_scip`) and to a `.nl` file (`to_nl`), solves it with SCIP and compares with `minimize`. No hand
  reformulation is needed. Run `PYTHONPATH=. python notebooks/export_scip.py [time_limit]` from the repository root.
* `results/`: stored results. `compare_results.json` (all solvers on all problems; notebook 04 reads it), `test_problems_results.json`
  and `throughput_results.json` (notebooks 03 and 05), and `NN.json`, one per Pyomo comparison notebook, which 07 summarizes.

## Re-running

```
notebooks/run_nb.sh 03-test-problems   # one notebook: jupytext --to ipynb, then nbconvert --execute --inplace
notebooks/run_all.sh                   # all of them, in order, 07 last; deletes results/NN.json first
```

Requirements: `jaxglobal` (imported from this repository), `jupytext`, `jupyter`/`nbconvert`, `matplotlib`; for 04 and 07–13 also
`pyomo`, `pandas` and the solvers SCIP (`/opt/homebrew/bin/scip`), Couenne (`/Applications/AMPL/couenne`) and BARON via GAMS
(paths in `nbutil.py`); 04 also uses LindoGlobal via GAMS (stored results are used unless `RUN_SOLVERS = True`); 11 needs `optax`
and `omlt`. The Pyomo comparisons use relative gap 1e-4 and at most 60 s per solve unless a notebook says otherwise.

Most of the run time is JIT compilation: the first `minimize` call on a new function compiles the bounding kernels and the local
solver, and later calls on the same function reuse them. Timings are wall clock on the machine that ran the notebook.
