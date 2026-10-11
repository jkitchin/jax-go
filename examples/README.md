# Examples

Executed notebooks with prose, math and figures, showing how jaxglobal works and how it performs.

| Notebook | Contents |
|---|---|
| `01-relaxations.ipynb` | From a JAX function to bounds: the jaxpr, interval arithmetic, McCormick relaxations and αBB; how relaxations tighten when a box is split; first- vs second-order convergence; a 2-D relaxation surface. Writes `figures/relaxations.png`. |
| `02-branch-and-bound.ipynb` | Spatial branch and bound and its certificate `L ≤ f* ≤ U`; a hand-run pruning picture; convergence of the gap; what the bounding options buy. Writes `figures/convergence.png`. |
| `03-test-problems.ipynb` | The 14 unconstrained and 7 constrained test problems: formulas, a contour gallery with local-solver traps, feasible-region plots, and result tables against interval bounds and multistart. Writes `test_problems_results.json`. |
| `04-equation-based-solvers.ipynb` | The same problems in SCIP, Couenne, BARON and LindoGlobal via a `jnp`→Pyomo shim; what has to be reformulated by hand; who solves what and how fast. Writes `figures/benchmark.png` and updates the jaxglobal column of `compare_results.json`. |
| `05-throughput.ipynb` | Throughput of the batched `jit(vmap)` bounding kernel vs batch size, compile cost, and where a full solve spends its time. Writes `throughput_results.json`. |
| `06-gaussian-processes.ipynb` | Optimizing over trained Gaussian processes, after Schweidtmann et al. (2021): GP-mean minimization of the peaks function for N = 20 to 400 and four kernels, LCB/EI/PI acquisition functions (and why PI needs a margin), a chance constraint built from a second GP, and a short Bayesian-optimization loop. Reports what is certified and what is not. Takes about 20 minutes, mostly its time-limited solves. |

## Layout

* `src/*.py`: the notebooks as jupytext percent-format scripts (edit these).
* `*.ipynb`: generated from `src/` and executed, outputs included.
* `compare.py`: the `jnp`→Pyomo shim, hand reformulations and solver wrappers used by notebook 04. Run as a script (`python examples/compare.py` from the repository root), it fills in `compare_results.json` for problems not yet in it.
* `compare_results.json`: stored results of all solvers on all problems. Notebook 04 reads it; rerunning the external solvers needs SCIP, Couenne and GAMS with BARON and LindoGlobal (set `RUN_SOLVERS = True`).

## Re-running

```bash
cd examples
./run_nb.sh 03-test-problems   # jupytext --to ipynb, then nbconvert --execute --inplace
```

Each notebook takes one to a few minutes, mostly JIT compilation: the first `minimize` call on a new function compiles the bounding kernels and the local solver, and later calls on the same function reuse them. Timings are wall clock on the machine that ran the notebook.
