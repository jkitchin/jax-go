"""Finite bounds for variables declared unbounded.

Branch-and-bound needs a finite box. Many models (most of MINLPLib's
"unbounded" instances) only declare some variables free because the
constraints or the objective bound them anyway. ``bound_domain`` derives
finite bounds without changing the problem:

1. FBBT from the declared (possibly infinite) box: constraints such as
   ``x + y = 1`` with ``y in [0, 1]`` bound ``x``.
2. Variables that no function depends on (structurally, on the jaxpr) are
   fixed at a point of their range: the optimum does not depend on them.
   Monotone slacks are capped the same way (``monotone_bounds``): a free
   ``s`` used only in rows like ``s >= |x - y|``, linearly, and not raising
   ``f`` as it falls, can be lowered to the largest value any row needs on the
   box without losing feasibility or optimality. Caps and FBBT alternate
   until nothing changes (at most ``_SETTLE_ROUNDS`` rounds).
3. If variables are still unbounded, a few local solves look for a feasible
   point. Its value ``UB`` is a valid cutoff, so FBBT with ``f <= UB`` keeps
   every point that could be optimal; for example ``min x^2 + y^2`` bounds
   both variables once any feasible point is known.

What remains infinite after that is reported to the caller. All steps only
remove points that are infeasible or worse than a known feasible point (or,
for the caps, points with an equally good feasible point inside the cap), so
the global optimum value is preserved.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp
import numpy as np

from .fbbt import joint_tape, make_fbbt

_POOL = []  # one compiler thread, created on first use
_SETTLE_ROUNDS = 5  # rounds of monotone caps + FBBT in bound_domain


class DomainTimeout(Exception):
    """``bound_domain`` ran out of time (its FBBT kernel was still compiling)."""


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


def _row_structure(f, ineq, eq, n):
    """Per tape output row ``[f, ineq..., eq...]``: ``(dep, nl, kind)`` or None if unknown.

    ``dep[r, j]``: row r depends on x_j; ``nl[r, j]``: nonlinearly (as in
    ``lp._nonlinear_vars``: sums, differences, negations and products with or
    quotients by constants keep a variable linear, anything else makes it
    nonlinear). A row affine in x_j therefore has a constant coefficient on it.
    ``kind`` is 0 for the objective, 1 for an inequality, 2 for an equality.
    """
    from .fbbt import _has_tracer
    from .fbbt_interp import scalarize

    with jax.ensure_compile_time_eval():
        tape = joint_tape(f, ineq, eq, np.zeros(n))
    if _has_tracer(tape):
        return None
    prog = scalarize(tape, n)
    if prog is None:
        return None
    S = len(prog.const_val)
    dep = np.zeros((S, n), bool)
    dep[np.arange(n), np.arange(n)] = True
    nl = np.zeros((S, n), bool)
    const = ~np.isnan(np.asarray(prog.const_val, np.float64))
    const[:n] = False
    for o, a, b, out in zip(prog.op, prog.a, prog.b, prog.out):
        name = prog.opcodes[o][0]
        dep[out] = dep[a] | dep[b]
        if name in ("add", "add_any", "sub", "neg") or (name == "mul" and (const[a] or const[b])) or \
                (name == "div" and const[b]):  # fmt: skip
            nl[out] = nl[a] | nl[b]
        else:
            nl[out] = dep[a] | dep[b]
    kinds = [k for k, fn in enumerate((f, ineq, eq)) if fn is not None]
    D, N, K = [], [], []
    for k, s in zip(kinds, prog.outs):
        s = np.atleast_1d(np.asarray(s)).ravel()
        D.append(dep[s])
        N.append(nl[s])
        K.append(np.full(len(s), k))
    return np.concatenate(D), np.concatenate(N), np.concatenate(K)


def _row_upper(f, ineq, eq, lo, hi):
    """Interval upper bound of every output row ``[f, ineq..., eq...]`` over the box (eager FBBT forward sweep)."""
    from .fbbt import Const, _forward

    with jax.ensure_compile_time_eval():
        tape = joint_tape(f, ineq, eq, jnp.zeros(lo.size))
        vals, _ = _forward(tape, (jnp.asarray(lo), jnp.asarray(hi)), {})
        out = []
        for a in tape.outs:
            v = a.val if isinstance(a, Const) else vals[a]
            v = getattr(v, "hi", v)
            out.append(np.atleast_1d(np.asarray(v, np.float64)).ravel())
    return np.concatenate(out)


def monotone_bounds(f, ineq, eq, lo, hi, integer=None, struct=None):
    """Finite bounds for free variables that only ever help by moving toward a bound.

    A variable x_j with ``hi_j = inf`` such that (structurally) no equality
    uses it, every inequality ``g_r(x) <= tol`` that uses it is affine in it with
    coefficient ``a_r <= 0`` and the objective is affine in it with coefficient
    ``>= 0`` (or does not use it) is a slack of the kind ``x_j >= |y - z|``:
    lowering x_j never violates a constraint and never raises ``f``. Each such
    row needs only ``x_j >= (g_r(x) - a_r x_j) / -a_r``, whose right side is at most
    ``U_r = R_r / -a_r`` with ``R_r`` an interval upper bound of the rest of the
    row on the box (x_j itself fixed at 0). So any feasible point stays feasible,
    and no worse, with x_j lowered to ``U = max(lo_j, max_r U_r)``: the cap
    ``hi_j = U`` keeps an optimal point (the same argument as fixing unused
    variables). The mirror case (``lo_j = -inf``, ``a_r >= 0``, objective
    coefficient ``<= 0``) gives ``lo_j = min(hi_j, min_r -R_r / a_r)``. A rule
    whose bound would be infinite (some ``R_r`` infinite, or no row at all and the
    opposite bound infinite) is not applied. Integer caps are rounded outward.

    Returns new ``(lo, hi)``; variables that do not qualify are unchanged.
    """
    lo, hi = np.array(lo, np.float64), np.array(hi, np.float64)
    n = lo.size
    free_hi, free_lo = ~np.isfinite(hi), ~np.isfinite(lo)
    if not (free_hi.any() or free_lo.any()):
        return lo, hi
    st = struct if struct is not None else _row_structure(f, ineq, eq, n)
    if st is None:
        return lo, hi
    dep, nl, kind = st
    # constant coefficients of the rows affine in x_j: a Jacobian at any point of the box
    x0 = np.where(np.isfinite(lo) & np.isfinite(hi), 0.5 * (np.where(np.isfinite(lo), lo, 0) + np.where(np.isfinite(hi), hi, 0)),
                  np.where(np.isfinite(lo), lo + 1.0, np.where(np.isfinite(hi), hi - 1.0, 0.0)))  # fmt: skip
    used = [fn for fn in (f, ineq, eq) if fn is not None]
    J = np.asarray(jax.jacfwd(lambda x: jnp.concatenate([jnp.atleast_1d(fn(x)) for fn in used]))(jnp.asarray(x0)))
    J = J.reshape(-1, n)
    if J.shape[0] != dep.shape[0]:
        return lo, hi
    ints = np.zeros(n, bool) if integer is None else np.asarray(integer, bool)
    for j in np.flatnonzero(free_hi | free_lo):
        rows = np.flatnonzero(dep[:, j])
        if np.any(kind[rows] == 2) or np.any(nl[rows, j]) or not np.all(np.isfinite(J[rows, j])):
            continue
        a = J[rows, j]
        rk = kind[rows]
        for side in (+1, -1):  # +1: cap hi (lowering x_j helps), -1: cap lo (raising helps)
            if side > 0 and not free_hi[j] or side < 0 and not free_lo[j]:
                continue
            if np.any(side * a[rk == 0] < 0) or np.any(side * a[rk == 1] > 0):
                continue  # the objective or some inequality wants x_j to move the other way
            need = rows[(rk == 1) & (a != 0)]
            if len(need):
                L, H = lo.copy(), hi.copy()
                L[j] = H[j] = 0.0
                R = _row_upper(f, ineq, eq, L, H)[need]
                if not np.all(np.isfinite(R)):
                    continue
                cap = np.max(R / -J[need, j]) if side > 0 else np.min(-R / J[need, j])
            else:
                cap = -np.inf if side > 0 else np.inf  # no row needs x_j: the opposite bound will do
            if side > 0:
                v = max(cap, lo[j])
                if np.isfinite(v):
                    hi[j] = np.ceil(v) if ints[j] else v
            else:
                v = min(cap, hi[j])
                if np.isfinite(v):
                    lo[j] = np.floor(v) if ints[j] else v
    return lo, hi


def _feasible_points(f, ineq, eq, lo, hi, feas_tol, n_starts, seed, deadline=None, integer=None):
    """Local solves (scipy SLSQP, infinite bounds allowed) from scattered starts.

    With ``integer`` (a mask) a point counts only when it is integral: each
    solve's integers are rounded and fixed, and the continuous variables are
    solved again (a feasible point of the continuous relaxation can be better
    than every integer one, so its value would cut off the integer optimum).
    """
    from scipy.optimize import minimize

    def vec(fn):
        return (lambda x: jnp.atleast_1d(fn(x))) if fn is not None else None

    def timed(fn):
        """``fn`` that raises once the deadline has passed: one SLSQP solve can run for minutes."""
        if deadline is None:
            return fn

        def call(x):
            if time.perf_counter() > deadline:
                raise DomainTimeout()
            return fn(x)

        return call

    g, h = vec(ineq), vec(eq)
    fvg = jax.jit(jax.value_and_grad(f))
    cons = []
    if g is not None:
        gj = jax.jit(lambda x: (g(x), jax.jacfwd(g)(x)))
        cons.append({"type": "ineq", "fun": timed(lambda x: -np.asarray(gj(x)[0])), "jac": lambda x: -np.asarray(gj(x)[1])})
    if h is not None:
        hj = jax.jit(lambda x: (h(x), jax.jacfwd(h)(x)))
        cons.append({"type": "eq", "fun": timed(lambda x: np.asarray(hj(x)[0])), "jac": lambda x: np.asarray(hj(x)[1])})
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
        if deadline is not None and time.perf_counter() > deadline:
            break
        scale = 10.0 ** (k % 3)  # 1, 10, 100: the model's scale is unknown
        fin = np.isfinite(lo) & np.isfinite(hi)
        mid = np.where(fin, 0.5 * (np.where(fin, lo, 0.0) + np.where(fin, hi, 0.0)), np.clip(0.0, lo, hi))
        x0 = np.clip(mid + (rng.normal(size=lo.size) * scale if k else 0.0), lo, hi)
        try:
            res = minimize(timed(lambda x: tuple(np.asarray(v, np.float64) for v in fvg(x))), x0, jac=True,
                           method="SLSQP", bounds=bounds, constraints=cons,
                           options={"maxiter": 300, "ftol": 1e-10})  # fmt: skip
            x = np.clip(res.x, lo, hi)
            cands = [x]
            if integer is not None and integer.any():
                from .bnb import ROUNDINGS

                cands = []
                fl = np.floor(x)
                for t in ROUNDINGS:  # several roundings: nearest can leave a nonlinear constraint violated
                    if deadline is not None and time.perf_counter() > deadline:
                        break
                    xi = np.clip(fl + (x - fl > t), lo, hi)
                    fixed = [(v, v) if i else bd for v, i, bd in zip(xi, integer, bounds)]
                    try:
                        res = minimize(timed(lambda x: tuple(np.asarray(v, np.float64) for v in fvg(x))),
                                       np.where(integer, xi, x), jac=True, method="SLSQP", bounds=fixed,
                                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})  # fmt: skip
                    except Exception:
                        continue
                    cands.append(np.where(integer, xi, np.clip(res.x, lo, hi)))
        except Exception:
            continue
        for x in cands:
            fx = float(f(jnp.asarray(x)))
            if np.isfinite(fx) and viol(jnp.asarray(x)) <= feas_tol and fx < best_f:
                best_x, best_f = x, fx
    return best_x, best_f


def bound_domain(f, ineq, eq, lo, hi, feas_tol=1e-6, rounds=30, n_starts=6, seed=0, deadline=None, integer=None):
    """Derive finite bounds. Returns ``(lo, hi, infeasible, x_feasible, f_feasible)``.

    ``integer`` (a mask): the feasible point, whose value bounds the objective
    for FBBT, must have those variables integral.

    ``lo``/``hi`` may still contain infinities when nothing bounds a variable;
    ``x_feasible`` is a feasible point found on the way (or ``None``). With a
    ``deadline`` (a ``time.perf_counter()`` value) the FBBT kernel is compiled
    in a background thread and ``DomainTimeout`` is raised if it is not ready
    in time; no new local solve starts after it.
    """
    lo, hi = np.array(lo, np.float64), np.array(hi, np.float64)
    fb = jax.jit(make_fbbt(f, ineq, eq, rounds=rounds, feas_tol=feas_tol))
    if deadline is not None and np.isfinite(deadline):
        if not _POOL:
            _POOL.append(ThreadPoolExecutor(1, thread_name_prefix="jaxglobal-domain"))
        spec = jax.ShapeDtypeStruct(lo.shape, jnp.float64)
        fut = _POOL[0].submit(lambda: fb.trace(spec, spec, jax.ShapeDtypeStruct((), jnp.float64)).lower().compile())
        try:
            fb = fut.result(timeout=max(deadline - time.perf_counter(), 0.0))
        except TimeoutError:
            raise DomainTimeout() from None

    def tighten(cutoff):
        nonlocal lo, hi
        a, b, empty = fb(jnp.asarray(lo), jnp.asarray(hi), jnp.asarray(cutoff, jnp.float64))
        lo, hi = np.maximum(lo, np.asarray(a)), np.minimum(hi, np.asarray(b))
        return bool(empty)

    if tighten(np.inf):
        return lo, hi, True, None, np.inf
    free = ~(np.isfinite(lo) & np.isfinite(hi))
    if free.any():
        unused = free & ~used_variables(f, ineq, eq, lo.size)
        pt = np.clip(0.0, lo, hi)
        lo, hi = np.where(unused, pt, lo), np.where(unused, pt, hi)
    struct = []

    def settle(cutoff):
        """Monotone caps (``monotone_bounds``) and FBBT in turn while they bound more variables."""
        nonlocal lo, hi
        for _ in range(_SETTLE_ROUNDS):
            if np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)):
                return
            if not struct:
                struct.append(_row_structure(f, ineq, eq, lo.size))
            if struct[0] is None:
                return
            nfree = int(np.sum(~np.isfinite(lo)) + np.sum(~np.isfinite(hi)))
            keep = lo.copy(), hi.copy()
            lo, hi = monotone_bounds(f, ineq, eq, lo, hi, integer=integer, struct=struct[0])
            if int(np.sum(~np.isfinite(lo)) + np.sum(~np.isfinite(hi))) == nfree:
                return
            if tighten(cutoff):  # the caps keep a feasible point if there is one: be conservative anyway
                lo, hi = keep
                return

    settle(np.inf)
    xf, ff = None, np.inf
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        xf, ff = _feasible_points(f, ineq, eq, lo, hi, feas_tol, n_starts, seed, deadline, integer)
        if xf is not None:
            cutoff = ff + 1e-9 * (1 + abs(ff))
            keep = lo.copy(), hi.copy()
            if tighten(cutoff):  # cannot happen for a truly feasible point; be safe
                lo, hi = keep
            settle(cutoff)
            lo, hi = np.minimum(lo, xf), np.maximum(hi, xf)  # keep the incumbent inside
    return lo, hi, False, xf, ff
