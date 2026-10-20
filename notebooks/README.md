# Pyomo vs jaxglobal notebooks

Executed notebooks that write the same optimization model in Pyomo and as a plain Python/JAX function for jaxglobal, solve both, and compare objective,
certified bound, time and status. Pyomo's strengths (integers, mature solvers, sparse structure, duals, ecosystem) are discussed alongside jaxglobal's.

| Notebook | Contents |
|---|---|
| `00-index.ipynb` | Overview, the lines-of-code helper, and a summary table built from the other notebooks' saved results |
| `01-basics.ipynb` | Six-hump camel function, Haverly pooling problem |
| `02-control-flow.ipynb` | `where`/`clip`/`max`/`interp` (pipe regimes, steam tables) vs Pyomo GDP and `Piecewise` |
| `03-ode-models.ipynb` | Kinetic parameter estimation: explicit RK4 and `pyomo.dae` vs a Python loop |
| `04-neural-network-surrogates.ipynb` | Optimizing a trained tanh MLP: jaxglobal (with and without `crown`), hand-written Pyomo, OMLT |
| `05-things-pyomo-cannot-express.ipynb` | `fori_loop`, library functions, einsum, parametric families, including what failed |
| `06-many-switches.ipynb` | Motor price lists (one `jnp.where`) vs one binary per rating: scaling with the list length, a counterexample, and switches inside a time loop |

## Layout

* `src/*.py`: the notebooks as jupytext percent-format scripts (edit these).
* `*.ipynb`: generated from `src/` and executed, outputs included.
* `nbutil.py`: shared helpers (`loc` line counter, uniform solver wrappers, results table, result saving).
* `results/*.json`: per-notebook results written when a notebook runs; `00-index` reads them.

## Re-running

Requirements: `jaxglobal` (imported from this repository), `pyomo`, `jupytext`, `jupyter`/`nbconvert`, `pandas`, `optax` (notebook 04), `omlt` (04), and the solvers
SCIP (`/opt/homebrew/bin/scip`), Couenne (`/Applications/AMPL/couenne`) and BARON via GAMS. Solver paths are set in `nbutil.py`. All solves use relative gap 1e-4 and at most 60 s.

```
cd notebooks
./run_nb.sh 01-basics        # one notebook: jupytext --to ipynb, then nbconvert --execute --inplace
./run_all.sh                 # all, in order (00 last); deletes results/*.json first
```

Run times on a loaded laptop are several minutes per notebook, mostly JIT compilation and solver time limits. The first jaxglobal call in a process pays the compile cost.
Timings in the notebooks are rough wall-clock figures and will differ on other machines.
