"""MAiNGO (maingopy) adapter (TESTED on peaks_1x5/1x10/1x20 with a locally built maingopy 0.10.3, macOS arm64) for the NN-surrogate problems: reduced-space McCormick with tanh.

This is the same reduced-space formulation as Schweidtmann & Mitsos [schweidtmann2019deterministic], which
MAiNGO implements with McCormick relaxations of tanh (envelopes via MC++).
Written against the documented maingopy modelling API (maingopy.MAiNGOmodel,
OptimizationVariable, EvaluationContainer, MAiNGO.set_option / solve).
"""
import time

import numpy as np

from . import nn


def make_model(Ws, bs, lo, hi):
    import maingopy

    class ANNModel(maingopy.MAiNGOmodel):
        def __init__(self):
            maingopy.MAiNGOmodel.__init__(self)

        def get_variables(self):
            return [maingopy.OptimizationVariable(maingopy.Bounds(float(l), float(h)), maingopy.VT_CONTINUOUS, f"x{i}")
                    for i, (l, h) in enumerate(zip(lo, hi))]

        def evaluate(self, v):
            h = list(v)
            for W, b in zip(Ws[:-1], bs[:-1]):
                h = [maingopy.tanh(sum(float(W[i, j]) * h[i] for i in range(W.shape[0])) + float(b[j]))
                     for j in range(W.shape[1])]
            W, b = Ws[-1], bs[-1]
            res = maingopy.EvaluationContainer()
            res.objective = sum(float(W[i, 0]) * h[i] for i in range(W.shape[0])) + float(b[0])
            return res

    return ANNModel()


def solve(p, limit=60, eps_abs=1e-6, eps_rel=1e-6):
    """Solve a nn.PROBLEMS entry with MAiNGO; returns the dict used by flagship_nn_smoke."""
    import maingopy

    Ws, bs = nn.load_weights(p.name)
    model = make_model(Ws, bs, p.lo, p.hi)
    g = maingopy.MAiNGO(model)
    g.set_option("maxTime", float(limit))
    g.set_option("epsilonA", eps_abs)
    g.set_option("epsilonR", eps_rel)
    for k in ("BAB_verbosity", "LBP_verbosity", "UBP_verbosity"):
        g.set_option(k, 0)
    g.set_option("loggingDestination", 0)
    t = time.perf_counter()
    status = g.solve()
    dt = time.perf_counter() - t
    return dict(status=str(status), ub=float(g.get_objective_value()), lb=float(g.get_final_LBD()),
                time=dt, nodes=int(g.get_iterations()))
