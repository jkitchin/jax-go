"""Why bound_domain leaves variables unbounded: per model, the variables still infinite after
bound_domain, whether they reach the objective (F), enter nonlinearly (N) or only linearly (L),
the number of constraint rows they reach (r), and whether a feasible point was found.

Usage:  python benchmarks/unsupported_diag.py ex9_1_1,prolog,...
"""
import sys, logging; logging.disable(logging.INFO)
import numpy as np, jax; jax.config.update("jax_enable_x64", True)
sys.path.insert(0, "/home/user/jax-go")
from jaxglobal.nl import read_nl
from jaxglobal.domain import bound_domain
from jaxglobal.lp import _nonlinear_vars
from jaxglobal.fbbt import joint_tape
from jaxglobal.fbbt_interp import scalarize
for name in sys.argv[1].split(","):
    M = read_nl(f"/home/user/jax-go/benchmarks/minlplib/nl/{name}.nl", integers="mask"); n = len(M.lo)
    lo, hi, inf0, xf, ff = bound_domain(M.f, M.ineq, M.eq, M.lo, M.hi, integer=M.integer)
    free = np.flatnonzero(~(np.isfinite(lo) & np.isfinite(hi)))
    # dependence: which outputs each variable reaches, and nonlinearly or not
    with jax.ensure_compile_time_eval():
        tape = joint_tape(M.f, M.ineq, M.eq, np.zeros(n))
    prog = scalarize(tape, n)
    S = len(prog.const_val); dep = np.zeros((S, n), bool); dep[np.arange(n), np.arange(n)] = True
    for a, b, o in zip(prog.a, prog.b, prog.out): dep[o] = dep[a] | dep[b]
    nlv = _nonlinear_vars(M.f, M.ineq, M.eq, n)  # parts x n: nonlinear dependence
    outs = np.concatenate([np.ravel(o) for o in prog.outs])
    in_f = dep[np.ravel(prog.outs[0])].any(0)
    nrows = len(outs) - 1
    print(f"{name:14s} n={n:2d} rows={nrows:2d} feasible point {'yes' if xf is not None else 'no '} f={ff:.4g}; unbounded {len(free)}: ", end="")
    desc = []
    for j in free:
        rows_dep = dep[outs[1:], j].sum() if nrows else 0
        nl = nlv[:, j].any() if nlv is not None else None
        side = ("lo" if not np.isfinite(lo[j]) else "") + ("hi" if not np.isfinite(hi[j]) else "")
        desc.append(f"x{j}[{side}]{'F' if in_f[j] else ''}{'N' if nl else 'L'}r{rows_dep}")
    print(" ".join(desc), flush=True)
