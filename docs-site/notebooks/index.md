# Notebooks

Executed notebooks with prose, math and figures. They are shown here with their
stored outputs; they are not re-run when the site is built. Timings are wall
clock on the machine that ran the notebook, mostly JIT compilation on a first
call.

## How jaxglobal works

| notebook | contents |
|---|---|
| [01 Relaxations](01-relaxations.ipynb) | From a JAX function to bounds: the jaxpr, interval arithmetic, McCormick relaxations and αBB; how relaxations tighten when a box is split; first- vs second-order convergence; a 2-D relaxation surface. |
| [02 Branch and bound](02-branch-and-bound.ipynb) | Spatial branch and bound and its certificate `L ≤ f* ≤ U`; a hand-run pruning picture; convergence of the gap; what the bounding options buy. |
| [03 Test problems](03-test-problems.ipynb) | The 14 unconstrained and 7 constrained test problems: formulas, a contour gallery with local-solver traps, feasible-region plots, and result tables against interval bounds and multistart. |
| [04 Equation-based solvers](04-equation-based-solvers.ipynb) | The same problems in SCIP, Couenne, BARON and LindoGlobal via a `jnp`→Pyomo shim; what has to be reformulated by hand; who solves what and how fast. |
| [05 Throughput](05-throughput.ipynb) | Throughput of the batched `jit(vmap)` bounding kernel vs batch size, compile cost, and where a full solve spends its time. |
| [06 Gaussian processes](06-gaussian-processes.ipynb) | Optimizing over trained Gaussian processes with the kernels from `jaxglobal.kernels` and the standard deviation, EI and PI from `jaxglobal.gp`: GP-mean minimization, acquisition functions, a chance constraint built from a second GP, and a short Bayesian-optimization loop. |

## Comparison with Pyomo

Each of these writes the same optimization model in Pyomo and as a plain
Python/JAX function for jaxglobal, solves both, and compares objective,
certified bound, time and status. Pyomo's strengths (integers, mature solvers,
sparse structure, duals, ecosystem) are discussed alongside jaxglobal's.

| notebook | contents |
|---|---|
| [07 Overview](07-pyomo-overview.ipynb) | Overview, the lines-of-code helper, and a summary table built from the saved results of 08 to 13 |
| [08 Basics](08-pyomo-basics.ipynb) | Six-hump camel function, Haverly pooling problem |
| [09 Control flow](09-control-flow.ipynb) | `where`/`clip`/`max`/`interp` (pipe regimes, steam tables) vs Pyomo GDP and `Piecewise` |
| [10 ODE models](10-ode-models.ipynb) | Kinetic parameter estimation: explicit RK4 and `pyomo.dae` vs a Python loop |
| [11 Neural-network surrogates](11-neural-network-surrogates.ipynb) | Optimizing a trained tanh MLP: jaxglobal (with and without `crown`), hand-written Pyomo, OMLT |
| [12 Things Pyomo cannot express](12-things-pyomo-cannot-express.ipynb) | `fori_loop`, library functions, einsum, parametric families, including what failed |

## Re-running

The notebooks are generated from jupytext percent-format scripts in
`notebooks/src/`. From a clone of the repository:

```bash
notebooks/run_nb.sh 03-test-problems   # jupytext --to ipynb, then nbconvert --execute --inplace
notebooks/run_all.sh                   # all of them, 07 last
```

Notebooks 04 and 07 to 13 also need `pyomo`, `pandas` and the external solvers
SCIP, Couenne and BARON (via GAMS); 11 needs `omlt`. All Pyomo comparisons use
a relative gap of 1e-4.
