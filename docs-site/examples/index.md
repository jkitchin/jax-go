# Example notebooks

Executed notebooks with prose, math and figures, showing how jaxglobal works
and how it performs. They are shown here with their stored outputs; they are
not re-run when the site is built. Timings are wall clock on the machine that
ran the notebook, mostly JIT compilation on a first call.

| notebook | contents |
|---|---|
| [01 Relaxations](01-relaxations.ipynb) | From a JAX function to bounds: the jaxpr, interval arithmetic, McCormick relaxations and αBB; how relaxations tighten when a box is split; first- vs second-order convergence; a 2-D relaxation surface. |
| [02 Branch and bound](02-branch-and-bound.ipynb) | Spatial branch and bound and its certificate `L ≤ f* ≤ U`; a hand-run pruning picture; convergence of the gap; what the bounding options buy. |
| [03 Test problems](03-test-problems.ipynb) | The 14 unconstrained and 7 constrained test problems: formulas, a contour gallery with local-solver traps, feasible-region plots, and result tables against interval bounds and multistart. |
| [04 Equation-based solvers](04-equation-based-solvers.ipynb) | The same problems in SCIP, Couenne, BARON and LindoGlobal via a `jnp`→Pyomo shim; what has to be reformulated by hand; who solves what and how fast. |
| [05 Throughput](05-throughput.ipynb) | Throughput of the batched `jit(vmap)` bounding kernel vs batch size, compile cost, and where a full solve spends its time. |
| [06 Gaussian processes](06-gaussian-processes.ipynb) | Optimizing over trained Gaussian processes with the kernels from `jaxglobal.kernels` and the standard deviation, EI and PI from `jaxglobal.gp`: GP-mean minimization, acquisition functions, a chance constraint built from a second GP, and a short Bayesian-optimization loop. |

## Re-running

The notebooks are generated from jupytext percent-format scripts in
`examples/src/`. From a clone of the repository:

```bash
cd examples
./run_nb.sh 03-test-problems   # jupytext --to ipynb, then nbconvert --execute --inplace
```
