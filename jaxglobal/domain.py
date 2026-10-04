"""Finite bounds for variables declared unbounded.

Branch-and-bound needs a finite box. Many models (most of MINLPLib's
"unbounded" instances) only declare some variables free because the
constraints or the objective bound them anyway. ``bound_domain`` derives
finite bounds without changing the problem:

1. FBBT from the declared (possibly infinite) box: constraints such as
   ``x + y = 1`` with ``y in [0, 1]`` bound ``x``.
2. Variables that no function depends on (structurally, on the jaxpr) are
   fixed at a point of their range: the optimum does not depend on them.
3. If variables are still unbounded, a few local solves look for a feasible
   point. Its value ``UB`` is a valid cutoff, so FBBT with ``f <= UB`` keeps
   every point that could be optimal; for example ``min x^2 + y^2`` bounds
   both variables once any feasible point is known.

What remains infinite after that is reported to the caller. All steps only
remove points that are infeasible or worse than a known feasible point, so
the global optimum is preserved.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .fbbt import joint_tape, make_fbbt


def used_variables(f, ineq, eq, n):
    """Boolean mask: variables that some function structurally depends on."""
    from .univariate import _dep_masks

    tape = joint_tape(f, ineq, eq, jnp.zeros(n))
    with jax.ensure_compile_time_eval():
        masks = _dep_masks(tape, n)
        used = np.zeros(n, bool)
        for a in tape.outs:
            if isinstance(a, int) and a in masks:
                m = np.asarray(masks[a]) > 0
                used |= m.reshape(-1, n).any(0)
    return used


def _feasible_points(f, ineq, eq, lo, hi, feas_tol, n_starts, seed):
    """Local solves (scipy SLSQP, infinite bounds allowed) from scattered starts."""
    from scipy.optimize import minimize

    def vec(fn):
        return (lambda x: jnp.atleast_1d(fn(x))) if fn is not None else None

    g, h = vec(ineq), vec(eq)
    fvg = jax.jit(jax.value_and_grad(f))
    cons = []
    if g is not None:
        gj = jax.jit(lambda x: (g(x), jax.jacfwd(g)(x)))
        cons.append({"type": "ineq", "fun": lambda x: -np.asarray(gj(x)[0]), "jac": lambda x: -np.asarray(gj(x)[1])})
    if h is not None:
        hj = jax.jit(lambda x: (h(x), jax.jacfwd(h)(x)))
        cons.append({"type": "eq", "fun": lambda x: np.asarray(hj(x)[0]), "jac": lambda x: np.asarray(hj(x)[1])})
    bounds = [(a if np.isfinite(a) else None, b if np.isfinite(b) else None) for a, b in zip(lo, hi)]

    def viol(x):
        v = 0.0
        if g is not None:
            v = max(v, float(np.max(np.asarray(g(x)), initial=-np.inf)))
        if h is not None:
            v = max(v, float(np.max(np.abs(np.asarray(h(x))), initial=0.0)))
        return v

    rng = np.random.default_rng(seed)
    best_x, best_f = None, np.inf
    for k in range(n_starts):
        scale = 10.0 ** (k % 3)  # 1, 10, 100: the model's scale is unknown
        fin = np.isfinite(lo) & np.isfinite(hi)
        mid = np.where(fin, 0.5 * (np.where(fin, lo, 0.0) + np.where(fin, hi, 0.0)), np.clip(0.0, lo, hi))
        x0 = np.clip(mid + (rng.normal(size=lo.size) * scale if k else 0.0), lo, hi)
        try:
            res = minimize(lambda x: tuple(np.asarray(v, np.float64) for v in fvg(x)), x0, jac=True,
                           method="SLSQP", bounds=bounds, constraints=cons,
                           options={"maxiter": 300, "ftol": 1e-10})  # fmt: skip
            x = np.clip(res.x, lo, hi)
        except Exception:
            continue
        fx = float(f(jnp.asarray(x)))
        if np.isfinite(fx) and viol(jnp.asarray(x)) <= feas_tol and fx < best_f:
            best_x, best_f = x, fx
    return best_x, best_f


def bound_domain(f, ineq, eq, lo, hi, feas_tol=1e-6, rounds=30, n_starts=6, seed=0):
    """Derive finite bounds. Returns ``(lo, hi, infeasible, x_feasible, f_feasible)``.

    ``lo``/``hi`` may still contain infinities when nothing bounds a variable;
    ``x_feasible`` is a feasible point found on the way (or ``None``).
    """
    lo, hi = np.array(lo, np.float64), np.array(hi, np.float64)
    fb = jax.jit(make_fbbt(f, ineq, eq, rounds=rounds, feas_tol=feas_tol))

    def tighten(cutoff):
        nonlocal lo, hi
        a, b, empty = fb(jnp.asarray(lo), jnp.asarray(hi), cutoff)
        lo, hi = np.maximum(lo, np.asarray(a)), np.minimum(hi, np.asarray(b))
        return bool(empty)

    if tighten(np.inf):
        return lo, hi, True, None, np.inf
    free = ~(np.isfinite(lo) & np.isfinite(hi))
    if free.any():
        unused = free & ~used_variables(f, ineq, eq, lo.size)
        pt = np.clip(0.0, lo, hi)
        lo, hi = np.where(unused, pt, lo), np.where(unused, pt, hi)
    xf, ff = None, np.inf
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        xf, ff = _feasible_points(f, ineq, eq, lo, hi, feas_tol, n_starts, seed)
        if xf is not None:
            cutoff = ff + 1e-9 * (1 + abs(ff))
            keep = lo.copy(), hi.copy()
            if tighten(cutoff):  # cannot happen for a truly feasible point; be safe
                lo, hi = keep
            lo, hi = np.minimum(lo, xf), np.maximum(hi, xf)  # keep the incumbent inside
    return lo, hi, False, xf, ff
