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

from . import kcache
from .fbbt import joint_tape, make_fbbt

_POOL = []  # one compiler thread, created on first use
_SETTLE_ROUNDS = 5  # rounds of monotone caps + FBBT in bound_domain
_LP_ROUNDS = 4  # rounds of LP bounds + FBBT in bound_domain
# Tapes of at most this many nodes: the structural sweeps below (dependency masks, row upper bounds, the
# Jacobian) run as one jitted program. Applied eagerly, each primitive is its own small XLA compile
# (~100 of them, ~1 s, on a 3-variable model); on larger tapes eager wins, because one program's compile
# grows faster than the number of distinct (primitive, shape) pairs.
_JIT_MAX_NODES = 400


class DomainTimeout(Exception):
    """``bound_domain`` ran out of time (its FBBT kernel was still compiling)."""


def used_variables(f, ineq, eq, n):
    """Boolean mask: variables that some function structurally depends on."""
    from .univariate import _dep_masks

    tape = joint_tape(f, ineq, eq, jnp.zeros(n))

    def outs():
        masks = _dep_masks(tape, n)
        return [jnp.any(masks[a].reshape(-1, n) > 0, axis=0) for a in tape.outs if isinstance(a, int) and a in masks]

    used = np.zeros(n, bool)
    if len(tape.nodes) <= _JIT_MAX_NODES:
        masks = jax.jit(outs)()
    else:
        with jax.ensure_compile_time_eval():
            masks = outs()
    for m in masks:
        used |= np.asarray(m)
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


def _row_upper_fn(f, ineq, eq, n):
    """``(lo, hi) ->`` interval upper bound of every output row ``[f, ineq..., eq...]`` over the box.

    One FBBT forward sweep; on a small tape (``_JIT_MAX_NODES``) compiled once and reused for every box."""
    from .fbbt import Const, _forward, guarded_slots

    tape = joint_tape(f, ineq, eq, jnp.zeros(n))
    guarded = guarded_slots(tape)  # an empty untaken jnp.where arm is unbounded, not crossed

    def upper(lo, hi):
        vals, _ = _forward(tape, (lo, hi), {}, False, guarded)
        out = []
        for a in tape.outs:
            v = a.val if isinstance(a, Const) else vals[a]
            v = getattr(v, "hi", v)
            out.append(jnp.ravel(jnp.atleast_1d(jnp.asarray(v, jnp.float64))))
        return jnp.concatenate(out)

    small = len(tape.nodes) <= _JIT_MAX_NODES
    jit_upper = jax.jit(upper) if small else None

    def call(lo, hi):
        lo, hi = jnp.asarray(lo, jnp.float64), jnp.asarray(hi, jnp.float64)
        if small:
            return np.asarray(jit_upper(lo, hi))
        with jax.ensure_compile_time_eval():
            return np.asarray(upper(lo, hi))

    call.small = small  # monotone_bounds compiles its Jacobian on the same condition
    return call


def _row_upper(f, ineq, eq, lo, hi):
    """Interval upper bound of every output row ``[f, ineq..., eq...]`` over the box (FBBT forward sweep)."""
    return _row_upper_fn(f, ineq, eq, np.asarray(lo).size)(lo, hi)


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
    row_upper = _row_upper_fn(f, ineq, eq, n)
    jac = jax.jacfwd(lambda x: jnp.concatenate([jnp.atleast_1d(fn(x)) for fn in used]))
    J = np.asarray((jax.jit(jac) if row_upper.small else jac)(jnp.asarray(x0)))
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
                R = row_upper(L, H)[need]
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
        fut = kcache.track(_POOL[0].submit(lambda: fb.trace(spec, spec, jax.ShapeDtypeStruct((), jnp.float64)).lower().compile()))
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

    relax = []

    def linear(cutoff):
        """LP bounds (``LinearRelaxation``) and FBBT in turn while they tighten the box."""
        nonlocal lo, hi
        for _ in range(_LP_ROUNDS):
            if np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)):
                return
            if deadline is not None and time.perf_counter() > deadline:
                return
            if not relax:
                relax.append(LinearRelaxation(f, ineq, eq, lo.size, feas_tol))
            if not relax[0].ok:
                return
            keep = lo.copy(), hi.copy()
            lo, hi = relax[0].bounds(lo, hi, cutoff, integer=integer)
            if tighten(cutoff):  # the LP bounds keep every feasible point: be conservative anyway
                lo, hi = keep
                return
            w0, w1 = keep[1] - keep[0], hi - lo
            if not np.any((w1 < w0) & ~(np.isfinite(w0) & (w1 > 0.9 * w0))):
                return  # no new finite bound and no finite width cut by 10% or more

    settle(np.inf)
    xf, ff = None, np.inf
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        # a feasible point first (also a good seed for the search), then the LP bounds with its cutoff
        xf, ff = _feasible_points(f, ineq, eq, lo, hi, feas_tol, n_starts, seed, deadline, integer)
        cutoff = np.inf
        if xf is not None:
            cutoff = ff + 1e-9 * (1 + abs(ff))
            keep = lo.copy(), hi.copy()
            if tighten(cutoff):  # cannot happen for a truly feasible point; be safe
                lo, hi = keep
            settle(cutoff)
        linear(cutoff)
        settle(cutoff)
        if xf is not None:
            lo, hi = np.minimum(lo, xf), np.maximum(hi, xf)  # keep the incumbent inside
    return lo, hi, False, xf, ff


# ---------------------------------------------------------------------------------------------
# LP bounds on free variables (``LinearRelaxation``)
# ---------------------------------------------------------------------------------------------

_LP_MAX_ROWS = 20000  # scalar tape rows; larger models are not relaxed here
_LP_FBBT_ROUNDS = 5  # FBBT rounds for the slot intervals the rows are built from
_LP_OBBT_MAX_N = 100  # on models this small every bound is tightened, not only the infinite ones
_LP_MAX_COEF = 1e8  # facets / tangents with larger coefficients are dropped (only loosens the LP)
_LP_MARGIN = 1e-3  # relative widening of an LP-derived bound (absorbs LP solver tolerances)


def _curvature(name, params, const_a, const_b, cval, L, U):
    """+1 if the univariate row is convex on [L, U], -1 if concave, 0 if unknown.

    ``cval`` is the constant operand of a binary row (``pow``), if any; ``mul`` is ``x * x``.
    """
    if name in ("exp", "exp2", "expm1", "square", "cosh", "abs", "mul"):
        return 1
    if name in ("log", "log1p", "sqrt"):
        return -1
    if name == "rsqrt":
        return 1 if L > 0 else 0
    if name == "integer_pow":
        y = int(params["y"])
        if y > 0 and y % 2 == 0:
            return 1
        if y > 1:
            return 1 if L >= 0 else (-1 if U <= 0 else 0)
        if y < 0:
            return 1 if L > 0 else ((1 if y % 2 == 0 else -1) if U < 0 else 0)
        return 0
    if name == "pow":
        if const_a and not const_b:  # c ** x = exp(x log c)
            return 1 if cval > 0 else 0
        if const_b and not const_a and L >= 0:  # x ** e on x >= 0
            if cval >= 1 or (cval < 0 and L > 0):
                return 1
            if 0 < cval < 1:
                return -1
    return 0


def _tangent_points(L, U):
    """Points of [L, U] (either end may be infinite) at which tangents are taken."""
    if np.isfinite(L) and np.isfinite(U):
        return np.unique(L + (U - L) * np.array([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]))
    if np.isfinite(L):
        return L + max(1.0, abs(L)) * np.array([1e-3, 0.1, 1.0, 10.0, 100.0, 1e3])
    if np.isfinite(U):
        return U - max(1.0, abs(U)) * np.array([1e-3, 0.1, 1.0, 10.0, 100.0, 1e3])
    return np.array([-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0])


def _tangent_fn(name, eqn, n_in, const_a):
    """``(x, c) -> (g, g')`` vectorized, for the variable operand ``x`` of a univariate row."""
    prim, params = (eqn.primitive, eqn.params) if eqn is not None else (None, None)
    if name == "xlogx":
        g = lambda x, c: x * jnp.log(x)  # noqa: E731
    elif name == "mul":
        g = lambda x, c: x * x  # noqa: E731
    elif n_in > 1 and const_a:
        g = lambda x, c: prim.bind(c, x, **params)  # noqa: E731
    elif n_in > 1:
        g = lambda x, c: prim.bind(x, c, **params)  # noqa: E731
    else:
        g = lambda x, c: prim.bind(x, **params)  # noqa: E731
    return jax.jit(jax.vmap(jax.value_and_grad(lambda x, c: jnp.reshape(g(x, c), ()))))


class LinearRelaxation:
    """Linear relaxation of ``[f, ineq, eq]`` on the scalar tape, for bounding free variables.

    Every non-constant scalar slot of the tape (``fbbt_interp.scalarize``) is an LP
    column, bounded by its FBBT interval (infinite ends allowed; the intervals come
    from the tape interpreter with ``all_slots``). Rows:

    * sums, differences, negations, products with and quotients by constants: exact
      linear equalities;
    * ``w = a * b``: the McCormick facets that only use finite bounds, e.g.
      ``w >= aL b + bL a - aL bL`` (from ``(a - aL)(b - bL) >= 0``) needs ``aL`` and
      ``bL`` finite, whatever ``aU``, ``bU`` are; ``w = a / b`` the same facets on
      ``a = w * b``;
    * univariate rows convex (concave) on their input interval: tangents
      ``w >= g(p) + g'(p)(v - p) - eps`` (``<= ... + eps``), valid on the whole
      interval, so also on an unbounded one; e.g. ``-log(t) >= -log(p) - (t - p) / p``;
    * outputs: ``f <= cutoff``, ``ineq <= feas_tol``, ``|eq| <= feas_tol`` (the
      relaxation ``make_fbbt`` uses).

    Every row holds at every point of the box that satisfies the (relaxed)
    constraints and the cutoff, so minimizing / maximizing a variable over the LP
    bounds it over that set. This sees combinations of rows that FBBT cannot:
    ``x = (1 - t)(20 + x + y)`` with ``t >= 0.2`` and ``y <= x / 7`` bounds ``x``,
    while FBBT from ``x in [0, inf)`` does not.
    """

    def __init__(self, f, ineq, eq, n, feas_tol=1e-6):
        from .fbbt import _has_tracer
        from .fbbt_interp import make_propagate

        self.ok = False
        self.n, self.ineq, self.eq, self.feas_tol = n, ineq, eq, feas_tol
        with jax.ensure_compile_time_eval():
            tape = joint_tape(f, ineq, eq, np.zeros(n))
        if _has_tracer(tape):
            return
        run = make_propagate(tape, ineq, eq, n, feas_tol, _LP_FBBT_ROUNDS, scheduled=False)
        if run is None or len(run.program.op) > _LP_MAX_ROWS:
            return
        self.prog = run.program
        self.sweep = jax.jit(lambda lo, hi, c: run(lo, hi, c, all_slots=True))
        self.tangent = {}
        self.ok = True

    def _rows(self, L, H):
        """``(rows_ub, rows_eq)``: lists of ``({column: coef}, rhs)``, rows ``<= rhs`` / ``== rhs``."""
        prog = self.prog
        cval = prog.const_val
        isc = ~np.isnan(cval)
        col = np.cumsum(~isc) - 1
        rows_ub, rows_eq = [], []

        def add(kind, pairs, rhs):
            """Row ``sum c * slot (kind) rhs``; constant slots move to the rhs."""
            coefs = {}
            for c, s in pairs:
                if not np.isfinite(c) or abs(c) > _LP_MAX_COEF:
                    return
                if isc[s]:
                    rhs -= c * cval[s]
                elif c != 0:
                    coefs[int(col[s])] = coefs.get(int(col[s]), 0.0) + c
            if np.isfinite(rhs) and abs(rhs) <= _LP_MAX_COEF**2 and coefs:
                (rows_eq if kind == "==" else rows_ub).append((coefs, float(rhs)))

        def mccormick(w, a, b):
            """Facets of ``w = a * b`` (slots) that use only finite bounds."""
            aL, aU, bL, bU = L[a], H[a], L[b], H[b]
            for ab, bb, sgn in ((aL, bL, 1), (aU, bU, 1), (aU, bL, -1), (aL, bU, -1)):
                if np.isfinite(ab) and np.isfinite(bb):
                    # sgn=+1: w >= ab*b + bb*a - ab*bb ; sgn=-1: w <= ab*b + bb*a - ab*bb
                    add("<=", [(-sgn, w), (sgn * ab, b), (sgn * bb, a)], sgn * ab * bb)

        tangents = {}  # opcode (or "xlogx") -> (rows, points, constants, curvatures)
        logs = {int(out): int(a) for o, a, out in zip(prog.op, prog.a, prog.out) if prog.opcodes[o][0] == "log"}
        for k, (o, a, b, out) in enumerate(zip(prog.op, prog.a, prog.b, prog.out)):
            name, _, same, ca, cb, n_in, eqn = prog.opcodes[o]
            if name == "mul" and not (same or ca or cb) and (logs.get(int(a)) == b or logs.get(int(b)) == a):
                # v * log(v): convex on v > 0 (FBBT and McCormick see an unbounded product near v = 0)
                v = b if logs.get(int(a)) == b else a
                pts = _tangent_points(max(L[v], 0.0), H[v])
                pts = pts[pts > 0]
                t = tangents.setdefault("xlogx", ([], [], [], []))
                t[0].extend([k] * len(pts))
                t[1].extend(pts)
                t[2].extend([0.0] * len(pts))
                t[3].extend([1] * len(pts))
                add("<=", [(-1.0, out)], np.exp(-1.0) * (1 + 1e-12))  # v log v >= -1/e
            if name in ("add", "add_any"):
                add("==", [(1.0, out), (-1.0, a), (-1.0, b)], 0.0)
            elif name == "sub":
                add("==", [(1.0, out), (-1.0, a), (1.0, b)], 0.0)
            elif name == "neg":
                add("==", [(1.0, out), (1.0, a)], 0.0)
            elif name == "mul" and (ca or cb):
                c, v = (cval[a], b) if ca else (cval[b], a)
                add("==", [(1.0, out), (-c, v)], 0.0)
            elif name == "div" and cb:
                if cval[b] != 0:
                    add("==", [(1.0, out), (-1.0 / cval[b], a)], 0.0)
            elif name == "div":
                mccormick(a, out, b)  # a = out * b
            elif name == "mul" and not same:
                mccormick(out, a, b)
            else:
                v = b if (n_in > 1 and ca) else a  # the variable operand
                c = float(cval[a] if ca else cval[b]) if (n_in > 1 and name != "mul") else 0.0
                curv = _curvature(name, eqn.params, ca, cb, c, L[v], H[v])
                if curv:
                    pts = _tangent_points(L[v], H[v])
                    t = tangents.setdefault(o, ([], [], [], []))
                    t[0].extend([k] * len(pts))
                    t[1].extend(pts)
                    t[2].extend([c] * len(pts))
                    t[3].extend([curv] * len(pts))
        for o, (ks, ps, cs, curvs) in tangents.items():
            if o == "xlogx":
                name, ca, n_in = "xlogx", False, 1
            else:
                name, _, same, ca, cb, n_in, eqn = prog.opcodes[o]
            if o not in self.tangent:
                self.tangent[o] = _tangent_fn(name, None if o == "xlogx" else eqn, n_in, ca)
            ps, cs = np.asarray(ps, np.float64), np.asarray(cs, np.float64)
            with jax.ensure_compile_time_eval():
                gv, dv = (np.asarray(t, np.float64) for t in self.tangent[o](jnp.asarray(ps), jnp.asarray(cs)))
            for k, p, gp, dp, curv in zip(ks, ps, gv, dv, curvs):
                if not (np.isfinite(gp) and np.isfinite(dp)):
                    continue
                if o == "xlogx":
                    v = prog.b[k] if logs.get(int(prog.a[k])) == prog.b[k] else prog.a[k]
                else:
                    v = prog.b[k] if (n_in > 1 and ca) else prog.a[k]
                eps = 1e-9 * (1 + abs(gp) + abs(dp) * (1 + abs(p)))
                # convex: out >= gp + dp (v - p) - eps ; concave: out <= gp + dp (v - p) + eps
                add("<=", [(-curv, prog.out[k]), (curv * dp, v)], curv * (dp * p - gp) + eps)
        return rows_ub, rows_eq

    def bounds(self, lo, hi, cutoff=np.inf, integer=None, tighten_all=None):
        """New ``(lo, hi)``: each infinite end replaced by its LP bound where the LP is bounded.

        With ``tighten_all`` (default: ``n <= _LP_OBBT_MAX_N``) finite ends are
        tightened as well (OBBT), which sharpens the intervals the next call
        builds its rows from. An LP optimum ``v`` gives the bound ``v`` widened
        by ``_LP_MARGIN * (1 + |v|)``; integer bounds are rounded outward.
        """
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix

        lo, hi = np.array(lo, np.float64), np.array(hi, np.float64)
        n = self.n
        if not self.ok:
            return lo, hi
        Ls, Hs, empty = self.sweep(jnp.asarray(lo), jnp.asarray(hi), jnp.asarray(cutoff, jnp.float64))
        if bool(empty):
            return lo, hi  # FBBT reports infeasibility itself
        L, H = np.array(Ls, np.float64), np.array(Hs, np.float64)
        L, H = np.where(np.isnan(L), -np.inf, L), np.where(np.isnan(H), np.inf, H)
        prog = self.prog
        isc = ~np.isnan(prog.const_val)
        zs = [(-np.inf, cutoff)] + ([(-np.inf, self.feas_tol)] if self.ineq is not None else []) + \
            ([(-self.feas_tol, self.feas_tol)] if self.eq is not None else [])  # fmt: skip
        for ids, (zl, zh) in zip(prog.outs, zs):
            ids = np.asarray(ids).ravel()
            ids = ids[~isc[ids]]
            L[ids], H[ids] = np.maximum(L[ids], zl), np.minimum(H[ids], zh)
        if np.any(L[~isc] > H[~isc]):
            return lo, hi
        rows_ub, rows_eq = self._rows(L, H)
        ncol = int(np.sum(~isc))

        def mat(rows):
            if not rows:
                return None, None
            r, c, v = [], [], []
            for i, (coefs, _) in enumerate(rows):
                r += [i] * len(coefs)
                c += list(coefs)
                v += list(coefs.values())
            return coo_matrix((v, (r, c)), shape=(len(rows), ncol)).tocsr(), np.array([b for _, b in rows])

        A_ub, b_ub = mat(rows_ub)
        A_eq, b_eq = mat(rows_eq)
        if A_ub is None and A_eq is None:
            return lo, hi
        bnds = [(l_ if np.isfinite(l_) else None, h_ if np.isfinite(h_) else None) for l_, h_ in zip(L[~isc], H[~isc])]
        ints = np.zeros(n, bool) if integer is None else np.asarray(integer, bool)
        if tighten_all is None:
            tighten_all = n <= _LP_OBBT_MAX_N
        new_lo, new_hi = lo.copy(), hi.copy()
        for j in range(n):
            for side in (-1, 1):
                free = not np.isfinite(lo[j] if side < 0 else hi[j])
                if not (free or (tighten_all and hi[j] > lo[j])):
                    continue
                c = np.zeros(ncol)
                c[j] = -side  # columns 0..n-1 are x (slots 0..n-1 are never constants)
                try:
                    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bnds, method="highs")
                except Exception:
                    continue
                if res.status != 0 or not np.isfinite(res.fun):
                    continue
                v = -side * res.fun
                v += side * _LP_MARGIN * (1 + abs(v))
                if side > 0:
                    v = np.ceil(v) if ints[j] else v
                    new_hi[j] = min(new_hi[j], v)
                else:
                    v = np.floor(v) if ints[j] else v
                    new_lo[j] = max(new_lo[j], v)
        bad = new_lo > new_hi  # cannot happen for a nonempty feasible set; keep the old bounds then
        return np.where(bad, lo, new_lo), np.where(bad, hi, new_hi)
