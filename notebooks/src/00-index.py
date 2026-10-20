# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Pyomo and jaxglobal, side by side
#
# This series writes the same optimization problems twice: once as a [Pyomo](https://www.pyomo.org) model [hart2011pyomo, bynum2021pyomo], and once as a plain Python/JAX function for jaxglobal,
# a global optimizer that traces the function and bounds it with McCormick/alphaBB relaxations inside a branch-and-bound [mccormick1976computability]. Every notebook solves both and
# compares objective, certified bound, time and status. The audience is chemical engineers and optimization researchers who already know one of the two styles.
#
# The comparison is meant to be fair. Pyomo has a large ecosystem, mature global solvers (SCIP, BARON, Couenne here), logical (GDP) modelling, duals,
# named constraints and sparse large-scale structure. jaxglobal is a research solver for continuous and integer variables; these notebooks use only continuous ones. Where Pyomo is faster or more capable, the notebooks say so.
#
# | Notebook | Topic |
# |---|---|
# | 01-basics | Six-hump camel function; Haverly pooling problem |
# | 02-control-flow | `where`, `clip`, `max`, `interp`: pipe flow regimes and steam tables; Pyomo GDP and `Piecewise` |
# | 03-ode-models | Kinetic parameter estimation with an ODE; explicit RK4 and `pyomo.dae` |
# | 04-neural-network-surrogates | Minimizing a trained tanh MLP; hand-written Pyomo and OMLT |
# | 05-things-pyomo-cannot-express | `fori_loop`, library functions, einsum, parametric families |
# | 06-many-switches | Motor price lists as one `jnp.where`: when many switches favor jaxglobal, and when they do not |
#
# **Conditions.** All solvers use relative gap 1e-4 (absolute 1e-6) and at most 60 s per solve. Timings are wall-clock on a loaded laptop and include
# the model-writing time for Pyomo solvers and JIT compilation for jaxglobal; treat them as rough. All data are either published problem data or synthetic (labeled in each notebook).
#
# **Reading the tables.** `obj` is the best objective found, `bound` the certified lower bound the solver reports (NaN means the solver did not return one;
# Couenne does not through the `.nl` interface), `status` the solver's termination.
# 
# **References.** Citation keys in square brackets refer to entries in `docs/references.bib` at the repository root.

# %% [markdown]
# ## Counting lines of model code
#
# "Lines of model code" is measured by one helper, `nbutil.loc`, which takes the functions that make up a model and counts source lines excluding blanks, comments and docstrings.
# Its source:

# %%
import inspect
import json
import os

import numpy as np
import pandas as pd

import nbutil
from nbutil import loc

print(inspect.getsource(loc))

# %% [markdown]
# ## Results of all comparisons
#
# Filled in from the results saved by notebooks 01 to 06 (`notebooks/results/*.json`). Run those notebooks first.
# Model size is the Pyomo model's unfixed variables and active constraints (the jaxglobal model has no explicit constraints beyond optional `ineq`/`eq` functions).

# %%
here = os.path.dirname(os.path.abspath(nbutil.__file__))
NAMES = {"01": "01-basics", "02": "02-control-flow", "03": "03-ode-models", "04": "04-neural-network-surrogates", "05": "05-things-pyomo-cannot-express", "06": "06-many-switches"}
cases = []
for nb in sorted(NAMES):
    path = os.path.join(here, "results", f"{nb}.json")
    if os.path.exists(path):
        for case, d in json.load(open(path)).items():
            cases.append((NAMES[nb], case, d))
print(f"{len(cases)} comparisons from {len({c[0] for c in cases})} notebooks")

# %%
size_rows = [
    {
        "notebook": nb,
        "case": case,
        "loc Pyomo": d["loc_pyomo"],
        "loc jaxglobal": d["loc_jax"],
        "Pyomo vars": d["pyomo_vars"],
        "Pyomo constraints": d["pyomo_cons"],
    }
    for nb, case, d in cases
]
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 60)
pd.DataFrame(size_rows).fillna("n/a")

# %% [markdown]
# Pyomo needs more lines in most cases, partly because it must declare variables, bounds and constraints that a function does not need. The line count says little about
# solver quality, and model sizes are for the Pyomo formulation only.

# %%
res_rows = []
for nb, case, d in cases:
    for r in d["rows"]:
        res_rows.append({"notebook": nb[:2], "case": case, "solver": r["solver"], "obj": r["obj"], "bound": r["bound"], "time (s)": round(r["time"], 2) if r["time"] == r["time"] else np.nan, "status": r["status"]})
df = pd.DataFrame(res_rows)
df.style.format({"obj": "{:.6g}", "bound": "{:.6g}"}, na_rep="n/a").hide(axis="index")

# %% [markdown]
# ## Summary
#
# * **Speed.** On these small problems the Pyomo solvers (SCIP, BARON) finish in about a second or a few seconds. jaxglobal takes from a few seconds to about 30 s, a large part of which is JIT compilation of the traced function.
#   Where the same function is solved again, the second solve is much faster (notebook 01).
# * **Agreement.** Where both sides finish, objectives agree to the gap tolerance. Full-space formulations can report an objective a few 1e-5 below the optimum, because their equations hold only to the solver's feasibility tolerance (notebook 04).
# * **Where jaxglobal reads better.** Models that are numerical code: ODE integrators, neural networks, loops, library functions, vectorized algebra. There is no transcription step.
# * **Where Pyomo is better.** Integer and logical structure, a choice of mature solvers, named constraints and duals, sparse large-scale models, and a large ecosystem (`pyomo.dae`, GDP, OMLT).
# * **Current jaxglobal limits found while writing these notebooks.** Compile time grows quickly with the length of an unrolled loop (notebook 03); `lax.scan` and `jnp.interp` are not supported
#   (notebooks 02, 03). `jax.scipy.special.erfc`, and with it the normal CDF, now has a rule (notebook 05).
