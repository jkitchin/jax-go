"""SCIP's dual bound after a given number of nodes (re-solved with each node limit; deterministic).

Usage:  python benchmarks/scip_bound_per_node.py minlplib/nl/wastewater15m1.nl 100,300,1000,3000,10000
"""
import sys
from pyscipopt import Model
for lim in [int(v) for v in sys.argv[2].split(",")]:
    m = Model(); m.hideOutput(); m.readProblem(sys.argv[1])
    m.setParam("limits/nodes", lim); m.setParam("limits/time", 60.0)
    m.optimize()
    print(f"SCIP nodes {m.getNNodes():6d} dual {m.getDualbound():.6g} time {m.getSolvingTime():.2f}", flush=True)
