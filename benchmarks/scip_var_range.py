"""Range of variables over a model's feasible set, by SCIP: maximize (``hi``) or minimize (``lo``)
each listed variable instead of the objective (10 s each). ``unbounded`` means the feasible set
is unbounded in that direction. Assumes SCIP's variable order is the .nl column order.

Usage:  python benchmarks/scip_var_range.py <name> '[[3, "hi"], [5, "lo"]]'
"""
import sys, json
from pyscipopt import Model
name, idx = sys.argv[1], json.loads(sys.argv[2])
out = {}
for j, side in idx:
    m = Model(); m.hideOutput(); m.readProblem(f"/home/user/jax-go/benchmarks/minlplib/nl/{name}.nl")
    vs = [v for v in m.getVars() if v.name != "objvar"] if False else m.getVars()
    v = vs[j]
    m.setObjective(v, "maximize" if side == "hi" else "minimize")
    m.setParam("limits/time", 10.0)
    m.optimize()
    st = m.getStatus()
    out[f"x{j}{side}"] = (st, m.getObjVal() if st == "optimal" else None, v.name, v.getLbOriginal(), v.getUbOriginal())
print(json.dumps(out))
