# Pyomo comparison notebooks

These executed notebooks write the same optimization model in Pyomo and as a
plain Python/JAX function for jaxglobal, solve both, and compare objective,
certified bound, time and status. Pyomo's strengths (integers, mature solvers,
sparse structure, duals, ecosystem) are discussed alongside jaxglobal's.

| notebook | contents |
|---|---|
| [00 Overview](00-index.ipynb) | Overview, the lines-of-code helper, and a summary table built from the other notebooks' saved results |
| [01 Basics](01-basics.ipynb) | Six-hump camel function, Haverly pooling problem |
| [02 Control flow](02-control-flow.ipynb) | `where`/`clip`/`max`/`interp` (pipe regimes, steam tables) vs Pyomo GDP and `Piecewise` |
| [03 ODE models](03-ode-models.ipynb) | Kinetic parameter estimation: explicit RK4 and `pyomo.dae` vs a Python loop |
| [04 Neural-network surrogates](04-neural-network-surrogates.ipynb) | Optimizing a trained tanh MLP: jaxglobal (with and without `crown`), hand-written Pyomo, OMLT |
| [05 Things Pyomo cannot express](05-things-pyomo-cannot-express.ipynb) | `fori_loop`, library functions, einsum, parametric families, including what failed |

Re-running them needs `pyomo`, `jupytext`, `nbconvert`, `pandas`, `omlt` and
the external solvers SCIP, Couenne and BARON (via GAMS). All solves use a
relative gap of 1e-4 and at most 60 s. Timings are rough wall-clock figures on
a loaded laptop.
