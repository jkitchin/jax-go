"""Structure of a problem read off its tape: which variables enter nonlinearly, and the sparsity
patterns of the constraint Jacobian and the Lagrangian Hessian.

For tapes that ``fbbt_interp.scalarize`` turns into scalar rows (models read from ``.nl``
files), each scalar slot carries a bitmask of the variables it depends on. A row is affine
in its variable inputs when it is a sum, difference or negation, a product with a constant,
or a division by a constant. Every other row is nonlinear: all variables its inputs depend on
enter nonlinearly, and every pair of them may meet in the Hessian. The union of those pairs
over the rows is a superset of the Lagrangian Hessian's pattern (chain and product rules: a
second derivative is nonzero only between variables that meet in some nonlinear row), and
the dependencies of the constraint outputs are the Jacobian's pattern. Exact or a superset,
and no compilation or evaluation.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import numpy as np


class Structure(NamedTuple):
    nonlinear: np.ndarray  # (n,) bool: variables that enter some function nonlinearly
    jac: tuple  # (rows, cols) of the constraint Jacobian, [ineq..., eq...] rows
    hess: tuple  # (rows, cols), rows >= cols: lower triangle of the Lagrangian Hessian


def structure(f, ineq, eq, n):
    """``Structure`` of the problem on ``n`` variables, or None when the tape cannot be scalarized
    (vectorized code) or closes over a value being transformed."""
    from .fbbt import _has_tracer, joint_tape
    from .fbbt_interp import scalarize, use_interpreter

    with jax.ensure_compile_time_eval():
        tape = joint_tape(f, ineq, eq, np.zeros(n))
    if _has_tracer(tape) or not use_interpreter(tape, n, min_rows=0):
        return None
    prog = scalarize(tape, n)
    if prog is None:
        return None
    const = ~np.isnan(prog.const_val)
    deps = [1 << j for j in range(n)] + [0] * (len(prog.const_val) - n)
    nl_bits, hess = 0, [0] * n  # hess[i]: bits of the j that may meet x_i in a nonlinear row
    for o, a, b, out in zip(prog.op.tolist(), prog.a.tolist(), prog.b.tolist(), prog.out.tolist()):
        name, d = prog.opcodes[o][0], deps[a] | deps[b]
        affine = (name in ("add", "add_any", "sub", "neg") or (name == "mul" and (const[a] or const[b]))
                  or (name == "div" and const[b]))  # fmt: skip
        if not affine and d:
            nl_bits |= d
            if not _all_pairs(hess, d):
                for i in _bits(d):
                    hess[i] |= d
        deps[out] = d
    rows, cols, r = [], [], 0
    for slots in prog.outs[1:]:  # the outputs after f are the constraints, ineq then eq
        for s in np.asarray(slots).reshape(-1).tolist():
            for j in _bits(deps[s]):
                rows.append(r)
                cols.append(j)
            r += 1
    hr, hc = [], []
    for i in range(n):
        for j in _bits(hess[i]):
            if j <= i:
                hr.append(i)
                hc.append(j)
    nonlinear = np.array([(nl_bits >> j) & 1 for j in range(n)], bool)
    return Structure(nonlinear, (np.asarray(rows, np.int64), np.asarray(cols, np.int64)),
                     (np.asarray(hr, np.int64), np.asarray(hc, np.int64)))  # fmt: skip


def _bits(d):
    """Indices of the set bits of the int ``d``."""
    out, j = [], 0
    while d:
        if d & 1:
            out.append(j)
        d >>= 1
        j += 1
    return out


def _all_pairs(hess, d):
    """True if every pair of variables in ``d`` is already in ``hess`` (the row adds nothing)."""
    return all(hess[i] & d == d for i in _bits(d))
