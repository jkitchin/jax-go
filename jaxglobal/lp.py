"""Constrained node bounds: outer-approximation LP + Lagrangian certificate.

For a box B and the problem

    min f(x)  s.t.  g(x) <= 0,  h(x) = 0,  x in B,

the McCormick relaxations give a convex problem

    min cv_f(x)  s.t.  cv_g(x) <= 0,  cv_h(x) <= 0,  -cc_h(x) <= 0,  x in B.

Subgradients from ``jax.jacfwd`` turn each convex function into affine cuts at
a set of points (the box center, 2n axis points, then Kelley points: the LP
solution of the previous round). Moving along a variable that a part is
linear in does not change its cut, so each part gets rows only from the center
and the axis points of the variables it is nonlinear in (a linear constraint
gets one row; a bilinear pooling balance five instead of 2n + 1). The
resulting small dense LP is solved by a fixed-iteration primal-dual
interior-point method written in JAX, so it ``vmap``s over hundreds of boxes.

The bound returned does **not** trust the LP solver. For any multipliers
y >= 0 on the cut rows, weak duality gives

    min_B t  >=  min_{z in box} (c + A^T y) . z - y . b,

which is analytic over a box. We evaluate this at the (clipped) IPM duals, so a
poorly converged or infeasible LP still yields a valid bound. Infeasible LPs
drive y to a Farkas ray, so the bound blows up and the node is pruned.

The certificate is evaluated in floating point, so a bound on its rounding
error (proportional to |A|^T|y| and |y|.|b|) is subtracted; without it, a
diverging dual (|y| ~ 1e170 on degenerate LPs) turns cancellation noise into a
huge, invalid bound. Constraint cuts are relaxed by ``tol`` (the B&B
feasibility tolerance), which keeps the LP interior non-empty when an equality
gives two opposing cuts, and makes the bound valid for the problem B&B
actually solves (constraints satisfied to ``tol``).

The same certificate gives bound tightening for free:

* **duality-based range reduction** (Ryoo & Sahinidis): the certificate is
  ``f >= lb + T (r_j u_j - min(r_j, 0))`` with reduced costs ``r = c + A^T y``,
  so with an incumbent ``f <= cutoff`` each ``u_j`` is confined to an interval
  (each ``|r_j|`` is first shrunk by its own rounding-error bound);
* **optimality-based bound tightening** (``make_obbt``): minimize and maximize
  each variable over the cut LP (2n LPs, one ``vmap``) and keep the
  certified bounds.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from . import memory
from .relax import SAFE_K
from .univariate import make_relaxer

EPS = float(jnp.finfo(jnp.float64).eps)


def _nonlinear_vars(f, ineq, eq, n):
    """``(K, n)`` bool: the variables each convex part [cv_f, cv_g..., cv_h..., -cc_h...] depends on
    nonlinearly, or None if unknown.

    Read off the scalarized tape (``fbbt_interp.scalarize``): each value
    carries the variables it depends on and those it depends on nonlinearly.
    Sums, differences, negations and products with (or quotients by)
    constants pass both along; any other operation makes every variable of its
    inputs nonlinear. A part with no nonlinear variable is affine. Tapes that
    are not mostly scalar are not scalarized (None): all parts then count as
    nonlinear in every variable. Calling a variable nonlinear only keeps
    duplicate rows; calling it linear would only drop valid cuts, never add an
    invalid one.
    """
    from .fbbt import _has_tracer, joint_tape
    from .fbbt_interp import scalarize, use_interpreter

    with jax.ensure_compile_time_eval():
        tape = joint_tape(f, ineq, eq, np.zeros(n))
    if _has_tracer(tape) or not use_interpreter(tape, n, min_rows=0):
        return None
    prog = scalarize(tape, n)
    if prog is None:
        return None
    S = len(prog.const_val)
    dep = np.zeros((S, n), bool)  # the variables of each slot
    dep[np.arange(n), np.arange(n)] = True
    nl = np.zeros((S, n), bool)  # those it is nonlinear in
    const = ~np.isnan(prog.const_val)
    const[:n] = False
    for o, a, b, out in zip(prog.op, prog.a, prog.b, prog.out):
        name = prog.opcodes[o][0]
        dep[out] = dep[a] | dep[b]
        if name in ("add", "add_any", "sub", "neg") or (name == "mul" and (const[a] or const[b])) or \
                (name == "div" and const[b]):  # fmt: skip
            nl[out] = nl[a] | nl[b]
        else:
            nl[out] = dep[a] | dep[b]
    out = [nl[s] for s in prog.outs]
    if eq is not None:
        out.append(out[-1])  # -cc_h
    return np.concatenate(out)


AFFINE_MAX_ENTRIES = 2e7  # ``_affine_forms``: largest (scalar slots x n) coefficient table it builds


def _affine_forms(f, ineq, eq, n):
    """``(A, b)``: each convex part [cv_f, cv_g..., cv_h..., -cc_h...] that ``_nonlinear_vars`` calls
    affine is ``A[k] . x + b[k]`` (``(K, n)``, ``(K,)``; NaN rows for the other parts), or None.

    The coefficients are propagated through the scalarized tape (the operations
    ``_nonlinear_vars`` keeps affine: sums, differences, negations, products with and quotients by
    constants), in float64 as the tape would compute them. An affine part's relaxation is the
    function itself, so these give the cut every point gives it (``make_constrained_bounder`` and
    ``make_obbt`` with ``SPLIT_AFFINE``), without its forward-mode pass. None when the tape is not
    scalarized or the table would exceed ``AFFINE_MAX_ENTRIES``."""
    from .fbbt import _has_tracer, joint_tape
    from .fbbt_interp import scalarize, use_interpreter

    with jax.ensure_compile_time_eval():
        tape = joint_tape(f, ineq, eq, np.zeros(n))
    if _has_tracer(tape) or not use_interpreter(tape, n, min_rows=0):
        return None
    prog = scalarize(tape, n)
    if prog is None:
        return None
    S = len(prog.const_val)
    if S * (n + 1) > AFFINE_MAX_ENTRIES:
        return None
    L = np.zeros((S, n))  # coefficients of each slot's affine form
    L[np.arange(n), np.arange(n)] = 1.0
    c0 = np.where(np.isnan(prog.const_val), 0.0, prog.const_val)  # constant terms (constants: their value)
    ok = np.ones(S, bool)  # the slot is affine (its form is exact)
    const = ~np.isnan(prog.const_val)
    const[:n] = False
    for o, a, b, out in zip(prog.op.tolist(), prog.a.tolist(), prog.b.tolist(), prog.out.tolist()):
        name = prog.opcodes[o][0]
        if name in ("add", "add_any"):
            L[out], c0[out] = L[a] + L[b], c0[a] + c0[b]
        elif name == "sub":
            L[out], c0[out] = L[a] - L[b], c0[a] - c0[b]
        elif name == "neg":
            L[out], c0[out] = -L[a], -c0[a]
        elif name == "mul" and const[a]:
            L[out], c0[out] = prog.const_val[a] * L[b], prog.const_val[a] * c0[b]
        elif name == "mul" and const[b]:
            L[out], c0[out] = L[a] * prog.const_val[b], c0[a] * prog.const_val[b]
        elif name == "div" and const[b]:
            L[out], c0[out] = L[a] / prog.const_val[b], c0[a] / prog.const_val[b]
        else:
            ok[out] = False
            continue
        ok[out] = ok[a] and ok[b]
    outs = [np.asarray(s) for s in prog.outs]
    if eq is not None:
        outs.append(outs[-1])
    sign = np.concatenate([np.ones(sum(len(s) for s in prog.outs))] + ([-np.ones(len(outs[-1]))] if eq is not None else []))
    s = np.concatenate(outs)
    A, b = sign[:, None] * L[s], sign * c0[s]
    good = ok[s] & np.all(np.isfinite(A), axis=1) & np.isfinite(b)
    return np.where(good[:, None], A, np.nan), np.where(good, b, np.nan)


def _affine_of(f, ineq, eq):
    """``_affine_forms`` memoized on the number of variables."""
    cache = {}

    def get(n):
        if n not in cache:
            cache[n] = _affine_forms(f, ineq, eq, n)
        return cache[n]

    return get


def _affine_parts(f, ineq, eq, n):
    """Which of the convex parts are affine in x (``_nonlinear_vars``), or None if unknown."""
    nl = _nonlinear_vars(f, ineq, eq, n)
    return None if nl is None else ~nl.any(axis=1)


def _nonlinear_of(f, ineq, eq):
    """``_nonlinear_vars`` memoized on the number of variables."""
    cache = {}

    def get(n):
        if n not in cache:
            cache[n] = _nonlinear_vars(f, ineq, eq, n)
        return cache[n]

    return get


def axis_vars(nl, n):
    """Indices of the variables that get axis points (``cut_points``): those some convex part is
    nonlinear in (``_nonlinear_vars`` ``nl``), all ``n`` when ``nl`` is unknown (None).

    Along a variable every part is linear in, each part's cut at the axis point is its cut at
    the center (see ``_cut_rows``), so ``_cut_rows`` dropped every row of such a point: it is
    not evaluated at all. That matters for memory as much as for time: the forward-mode cuts
    hold ``(points, slots, n)`` tangents per box (m34_convex2, 100 variables of which 40 enter
    nonlinearly: 81 points instead of 201; docs/ANALYSIS-memory.md)."""
    if nl is None or nl.shape[1] != n:
        return np.arange(n)
    return np.flatnonzero(nl.any(axis=0))


def cut_points(c, w, ax):
    """The initial cut points of a box with center ``c`` and widths ``w``: the center, then
    ``c + e_j w_j / 4`` and then ``c - e_j w_j / 4`` for the variables ``j`` in ``ax``
    (``axis_vars``)."""
    E = jnp.eye(c.size, dtype=c.dtype)[np.asarray(ax, dtype=np.int64)]
    return jnp.concatenate([c[None], c + jnp.concatenate([E, -E]) * (w / 4)])


def _point_bytes_of(f, ineq, eq):
    """``memory.cut_point_bytes`` of the relaxations of ``f``, ``ineq``, ``eq``, memoized on ``n``."""
    cache = {}

    def get(n):
        if n not in cache:
            cache[n] = memory.cut_point_bytes((f, ineq, eq), n)
        return cache[n]

    return get


def _cut_rows(shape, nl):
    """Flat indices of the LP rows to keep from cuts of shape ``(P, K)`` (P points, K parts;
    point-major rows), or None for all of them.

    The points are the center, the axis points ``c + e_j w_j / 4`` then
    ``c - e_j w_j / 4`` of the variables ``j`` of ``axis_vars(nl)`` (``cut_points``), then
    Kelley points. Part k keeps its cut at the center,
    at the axis points of the variables it is nonlinear in (``nl[k]``) and at
    every Kelley point unless it is affine. Moving along a variable a part is
    linear in leaves its subgradient and its cut unchanged (McCormick, the
    univariate hulls and alphaBB all keep the linear terms exact), so the
    dropped rows repeat the center's cut.
    """
    P, K = shape
    if nl is None or nl.shape[0] != K:
        return None
    ax = axis_vars(nl, nl.shape[1])
    keep = np.ones((P, K), bool)
    axes = np.concatenate([nl.T[ax], nl.T[ax]])[: max(P - 1, 0)]  # (2 |ax|, K) for the axis points
    keep[1 : 1 + len(axes)] = axes
    keep[1 + 2 * len(ax) :] = nl.any(axis=1)
    if keep.all():
        return None
    return np.flatnonzero(keep.reshape(-1))


def ipm(c, A, b, iters=40, sigma=0.2, cert=None, mu_tol=1e-10, box=False, target=jnp.inf):
    """Primal-dual interior point for ``min c.z s.t. A z <= b`` (dense, tiny).

    Infeasible start: z = 0.5 (the LP lives in the unit box), slacks and duals 1.
    Iteration stops (the state is frozen) once the duality measure falls below
    ``mu_tol``: past that point the normal equations are too ill-conditioned
    and the duals drift. If ``cert(y)`` is given, the best value it takes over
    all iterates is returned as well, so a late bad step can never lose a good
    certificate.

    ``box=True`` adds the rows ``0 <= z <= 1`` implicitly (``_ipm_box``, with
    predictor-corrector steps and an early stop), and ``y`` is then
    the duals of the rows of ``A`` only; ``target``: stop once the certificate
    reaches it (box mode only).

    Returns ``(z, y_best, best_cert)`` where ``y_best`` attains ``best_cert``.
    """
    if box:
        return _ipm_box(c, A, b, iters, sigma, cert, mu_tol, target)
    m, n = A.shape
    eye = jnp.eye(n)
    cert = cert or (lambda y: -jnp.inf)

    def step_len(v, dv):
        r = jnp.where(dv < 0, -v / jnp.where(dv < 0, dv, -1.0), jnp.inf)
        return jnp.minimum(1.0, 0.99 * jnp.min(r))

    def body(_, st):
        z, s, y, best, yb = st
        rp = A @ z + s - b
        rd = c + A.T @ y
        mu = s @ y / m
        M = (A.T * (y / s)) @ A + 1e-12 * eye
        rhs = -rd - A.T @ ((sigma * mu - s * y + y * rp) / s)
        dz = jnp.linalg.solve(M, rhs)
        ds = -rp - A @ dz
        dy = (sigma * mu - s * y - y * ds) / s
        ap, ad = step_len(s, ds), step_len(y, dy)
        new = (z + ap * dz, s + ap * ds, y + ad * dy)
        ok = jnp.all(jnp.isfinite(new[0])) & jnp.all(jnp.isfinite(new[2])) & (mu > mu_tol)
        z, s, y = jax.tree.map(lambda a, b_: jnp.where(ok, a, b_), new, (z, s, y))
        cy = cert(y)
        cy = jnp.where(jnp.isnan(cy), -jnp.inf, cy)
        better = cy > best
        return z, s, y, jnp.where(better, cy, best), jnp.where(better, y, yb)

    z0 = jnp.full(n, 0.5)
    s0 = jnp.maximum(b - A @ z0, 1.0)
    y0 = jnp.ones(m)
    c0 = cert(y0)
    st = (z0, s0, y0, jnp.where(jnp.isnan(c0), -jnp.inf, c0), y0)
    z, s, y, best, yb = jax.lax.fori_loop(0, iters, body, st)
    return z, yb, best


def _ipm_box(c, A, b, iters, sigma, cert, mu_tol, target, pc=True):
    """``ipm`` for ``min c.z s.t. A z <= b, 0 <= z <= 1`` with the box rows implicit.

    The rows ``z <= 1`` and ``-z <= 0`` are unit vectors, so their share of the normal
    matrix ``A^T (y/s) A`` is the diagonal ``yh/sh + yl/sl``. With ``pc=False`` the Newton
    steps, step lengths and iterates are those of ``ipm`` with the box rows appended to
    ``A`` (up to rounding), at the cost of ``A`` alone (on the reduced-space LP the box rows
    were about a third of all rows). ``pc=True`` (the default) takes Mehrotra
    predictor-corrector steps instead of ``ipm``'s fixed ``sigma``: an affine step sets
    ``sigma = (mu_aff / mu)^3`` and a second-order correction, both solved with one
    Cholesky factorization (Jacobi-scaled, as in ``quadratic._ipm``). On flay04m's node LPs
    that cut the mean iteration count from about 23 to 10. The fixed trip count becomes a ``while_loop`` that stops once
    the iterate is frozen (``mu <= mu_tol`` or a failed step, after which ``ipm``'s
    remaining iterations change nothing) or the best certificate reaches ``target`` (the
    caller prunes there); under ``vmap`` it runs until every box has stopped. ``cert``
    gets the duals of the ``A`` rows only; the stopping rule only decides when to stop,
    ``best`` is a certificate whatever it says."""
    m, n = A.shape
    eye = jnp.eye(n)
    cert = cert or (lambda y: -jnp.inf)
    mt = m + 2 * n

    def step_len(v, dv):
        r = jnp.where(dv < 0, -v / jnp.where(dv < 0, dv, -1.0), jnp.inf)
        return jnp.min(r, initial=jnp.inf)

    def cond(st):
        k, frozen, best = st[0], st[1], st[-2]
        return (k < iters) & ~frozen & ~(best >= target)

    def body(st):
        k, _, z, s, y, sh, yh, sl, yl, best, yb = st
        rp, rph, rpl = A @ z + s - b, z + sh - 1.0, sl - z
        rd = c + A.T @ y + yh - yl
        mu = (s @ y + sh @ yh + sl @ yl) / mt
        M = (A.T * (y / s)) @ A + jnp.diag(yh / sh + yl / sl) + 1e-12 * eye
        t, th, tl = ((sigma * mu - s * y + y * rp) / s, (sigma * mu - sh * yh + yh * rph) / sh,
                     (sigma * mu - sl * yl + yl * rpl) / sl)  # fmt: skip
        dz = jnp.linalg.solve(M, -rd - (A.T @ t + th - tl))
        ds, dsh, dsl = -rp - A @ dz, -rph - dz, -rpl + dz
        dy, dyh, dyl = ((sigma * mu - s * y - y * ds) / s, (sigma * mu - sh * yh - yh * dsh) / sh,
                        (sigma * mu - sl * yl - yl * dsl) / sl)  # fmt: skip
        ap = jnp.minimum(1.0, 0.99 * jnp.minimum(step_len(s, ds), jnp.minimum(step_len(sh, dsh), step_len(sl, dsl))))
        ad = jnp.minimum(1.0, 0.99 * jnp.minimum(step_len(y, dy), jnp.minimum(step_len(yh, dyh), step_len(yl, dyl))))
        new = (z + ap * dz, s + ap * ds, y + ad * dy, sh + ap * dsh, yh + ad * dyh, sl + ap * dsl, yl + ad * dyl)
        ok = jnp.all(jnp.isfinite(new[0])) & jnp.all(jnp.isfinite(new[2])) & jnp.all(jnp.isfinite(new[4])) & \
            jnp.all(jnp.isfinite(new[6])) & (mu > mu_tol)  # fmt: skip
        z, s, y, sh, yh, sl, yl = jax.tree.map(lambda a, b_: jnp.where(ok, a, b_), new, (z, s, y, sh, yh, sl, yl))
        cy = cert(y)
        cy = jnp.where(jnp.isnan(cy), -jnp.inf, cy)
        better = cy > best
        # a step not taken leaves the state as it was, so every later one would repeat it: stop
        return k + 1, ~ok, z, s, y, sh, yh, sl, yl, jnp.where(better, cy, best), jnp.where(better, y, yb)

    def body_pc(st):
        # Mehrotra predictor-corrector: an affine-scaling step picks sigma = (mu_aff / mu)^3 and a
        # second-order correction; both solves share one (Jacobi-scaled) Cholesky factorization
        k, _, z, s, y, sh, yh, sl, yl, best, yb = st
        rp, rph, rpl = A @ z + s - b, z + sh - 1.0, sl - z
        rd = c + A.T @ y + yh - yl
        mu = (s @ y + sh @ yh + sl @ yl) / mt
        M = (A.T * (y / s)) @ A + jnp.diag(yh / sh + yl / sl) + 1e-12 * eye
        dm = jnp.sqrt(jnp.diag(M))
        fac = jax.scipy.linalg.cho_factor(M / dm[:, None] / dm[None, :], lower=True)

        def direction(rc, rch, rcl):
            t, th, tl = (rc + y * rp) / s, (rch + yh * rph) / sh, (rcl + yl * rpl) / sl
            dz = jax.scipy.linalg.cho_solve(fac, (-rd - (A.T @ t + th - tl)) / dm) / dm
            ds, dsh, dsl = -rp - A @ dz, -rph - dz, -rpl + dz
            return dz, ds, dsh, dsl, (rc - y * ds) / s, (rch - yh * dsh) / sh, (rcl - yl * dsl) / sl

        def lengths(d):
            dz, ds, dsh, dsl, dy, dyh, dyl = d
            return (jnp.minimum(step_len(s, ds), jnp.minimum(step_len(sh, dsh), step_len(sl, dsl))),
                    jnp.minimum(step_len(y, dy), jnp.minimum(step_len(yh, dyh), step_len(yl, dyl))))

        aff = direction(-s * y, -sh * yh, -sl * yl)
        pa, da = lengths(aff)
        pa, da = jnp.minimum(1.0, pa), jnp.minimum(1.0, da)
        _, ds_, dsh_, dsl_, dy_, dyh_, dyl_ = aff
        mu_aff = ((s + pa * ds_) @ (y + da * dy_) + (sh + pa * dsh_) @ (yh + da * dyh_)
                  + (sl + pa * dsl_) @ (yl + da * dyl_)) / mt
        sg = jnp.clip((mu_aff / mu) ** 3, 0.0, 1.0)
        sg = jnp.where(jnp.isfinite(sg), sg, sigma)
        d = direction(sg * mu - s * y - ds_ * dy_, sg * mu - sh * yh - dsh_ * dyh_, sg * mu - sl * yl - dsl_ * dyl_)
        ap, ad = lengths(d)
        ap, ad = jnp.minimum(1.0, 0.99 * ap), jnp.minimum(1.0, 0.99 * ad)
        dz, ds, dsh, dsl, dy, dyh, dyl = d
        new = (z + ap * dz, s + ap * ds, y + ad * dy, sh + ap * dsh, yh + ad * dyh, sl + ap * dsl, yl + ad * dyl)
        ok = jnp.all(jnp.isfinite(new[0])) & jnp.all(jnp.isfinite(new[2])) & jnp.all(jnp.isfinite(new[4])) & \
            jnp.all(jnp.isfinite(new[6])) & (mu > mu_tol)  # fmt: skip
        z, s, y, sh, yh, sl, yl = jax.tree.map(lambda a, b_: jnp.where(ok, a, b_), new, (z, s, y, sh, yh, sl, yl))
        cy = cert(y)
        cy = jnp.where(jnp.isnan(cy), -jnp.inf, cy)
        better = cy > best
        return k + 1, ~ok, z, s, y, sh, yh, sl, yl, jnp.where(better, cy, best), jnp.where(better, y, yb)

    z0 = jnp.full(n, 0.5)
    s0 = jnp.maximum(b - A @ z0, 1.0)
    y0 = jnp.ones(m)
    ones = jnp.ones(n)
    c0 = cert(y0)
    st = (0, jnp.array(False), z0, s0, y0, ones, ones, ones, ones, jnp.where(jnp.isnan(c0), -jnp.inf, c0), y0)
    st = jax.lax.while_loop(cond, body_pc if pc else body, st)
    return st[2], st[-1], st[-2]


def _cut_const(v, J, lo, p, coef, safe):
    """Value of the cut at u = 0, lowered (when ``safe``) by the rounding of ``v + J (lo - p)`` and of ``coef . u``.

    Lowering a cut's constant only weakens it, so the cut stays a valid minorant.
    """
    const = v + J @ (lo - p)
    if safe:
        n = lo.size
        const = const - SAFE_K * (n + 3) * EPS * (jnp.abs(v) + jnp.abs(J) @ jnp.abs(lo - p) + jnp.abs(coef).sum(axis=-1))
    return const


def _cuts(parts, data, p, lo, hi, safe):
    """Cuts of the convex parts at ``p``: ``(const, coef, bounds)`` in unit-box coordinates
    (``const`` -inf and ``coef`` 0 where not finite), ``bounds`` from ``parts.bounds``.

    The values come out of the forward-mode pass as its auxiliary output, so the
    relaxation is traced (and compiled) once for values and derivatives.
    """

    def fn(x):
        v, bounds = parts.bounds(x, lo, hi, data)
        return v, (v, bounds)

    J, (v, bounds) = jax.jacfwd(fn, has_aux=True)(p)
    v, J = jax.lax.optimization_barrier((v, J))  # one evaluation for coef and const (see lifted.node_cuts)
    coef = J * (hi - lo)  # d cut / d u
    const = _cut_const(v, J, lo, p, coef, safe)  # value of the cut at u = 0
    ok = jnp.isfinite(const) & jnp.all(jnp.isfinite(coef), axis=1)
    return jnp.where(ok, const, -jnp.inf), jnp.where(ok[:, None], coef, 0.0), bounds


def _row_margin(A, b, mag, live, safe):
    """Slack added to normalized rows ``A u <= b`` for the rounding of the division by the row norm."""
    if not safe:
        return b
    m = SAFE_K * EPS * (jnp.abs(b) + jnp.abs(A).sum(axis=1) + mag)
    return jnp.where(live, b + m, b)


def _safe_costs(r, A, y):
    """Reduced costs shrunk toward 0 by their rounding error (|A| <= 1)."""
    err = (A.shape[0] + 2) * EPS * (1.0 + jnp.abs(A).T @ y)
    return jnp.sign(r) * jnp.maximum(jnp.abs(r) - err, 0.0)


def reduce_ranges(lo, hi, lb, r, cutoff):
    """Duality-based range reduction from ``f(x) >= lb + sum_j (r_j x_j - min_box r_j x_j)``.

    Every point of the box with ``f(x) <= cutoff`` satisfies
    ``r_j x_j - min(r_j lo_j, r_j hi_j) <= cutoff - lb`` for each ``j``.
    Returns the reduced box (possibly empty when ``lb > cutoff``).
    """
    gap = cutoff - lb
    gap = gap + 1e-9 * (1 + jnp.abs(cutoff))  # floating-point slack
    use = jnp.isfinite(gap) & jnp.isfinite(lb) & (gap >= 0)
    rs = jnp.where(r != 0, r, 1.0)
    new_hi = jnp.where(use & (r > 0), lo + gap / rs, hi)
    new_lo = jnp.where(use & (r < 0), hi + gap / rs, lo)
    ok = jnp.isfinite(new_lo) & jnp.isfinite(new_hi)
    return jnp.where(ok, jnp.maximum(lo, new_lo), lo), jnp.where(ok, jnp.minimum(hi, new_hi), hi)


def _as_vec(fn):
    return None if fn is None else (lambda x: jnp.atleast_1d(fn(x)))


def _parts_fn(f, ineq, eq, univariate, alphabb=False, multivariate=False, safe=False, convex=True, exact=False):
    """``prepare(lo, hi)`` and ``parts(x, lo, hi, data)`` = [cv_f, cv_g..., cv_h..., -cc_h...].

    With ``convex`` each part is ``max(relaxation, function)`` where the function is
    certified convex on the box (``convexity``), so its cuts are outer approximation.
    ``exact``: the boxes lie where the certificates' conditions hold, so a function that is
    certified throughout is relaxed by itself alone (``convexity.exact_certificate``)."""
    # f and g only enter through their convex side (and interval bounds, which may be infinite)
    rs = [make_relaxer(_as_vec(fn), univariate, alphabb, multivariate=multivariate, safe=safe, convex=convex,
                       sides=sides, conditional=exact) if fn is not None else None
          for fn, sides in ((f, ("cv",)), (ineq, ("cv",)), (eq, ("cv", "cc")))]  # fmt: skip

    def prepare(lo, hi):
        return tuple(r.prepare(lo, hi) if r is not None else None for r in rs)

    def parts(x, lo, hi, data):
        return parts_bounds(x, lo, hi, data)[0]

    def parts_bounds(x, lo, hi, data, skip=None):
        """``(parts, bounds)``: ``bounds`` are the interval bounds ``(f, g, h)`` on the box, each
        an MC's ``(lo, hi)`` or None; they do not depend on ``x``. ``skip``: per function (f, g, h),
        None to relax it, or its number of outputs to leave it out (zeros in its parts' places and
        None bounds: for callers that use none of its parts, see ``_SubParts``)."""
        rf, rg, rh = rs
        skip = skip or (None, None, None)
        out, bounds = [], [None, None, None]
        if skip[0] is None:
            mf = rf(x, lo, hi, data[0])
            out.append(jnp.atleast_1d(mf.cv))
            bounds[0] = (mf.lo, mf.hi)
        else:
            out.append(jnp.zeros(skip[0], x.dtype))
        if rg is not None:
            if skip[1] is None:
                mg = rg(x, lo, hi, data[1])
                out.append(mg.cv)
                bounds[1] = (mg.lo, mg.hi)
            else:
                out.append(jnp.zeros(skip[1], x.dtype))
        if rh is not None:
            if skip[2] is None:
                mh = rh(x, lo, hi, data[2])
                out += [mh.cv, -mh.cc]
                bounds[2] = (mh.lo, mh.hi)
            else:
                out.append(jnp.zeros(2 * skip[2], x.dtype))
        return jnp.concatenate(out), tuple(bounds)

    parts.bounds = parts_bounds
    return prepare, parts, rs


# One cut pass for the first LP and the Kelley rounds in the reduced-space bound
# (``make_constrained_bounder``): the relaxation's forward-mode pass and the LP appear once in the
# kernel instead of twice, which halves XLA's compile time on large tapes (m410_CLay0203HRS,
# m34_convex2: 41k -> 15k HLO lines, 24 s -> 12 s). Round 0 takes ceil(P0 / SHARED_CHUNK) passes of
# SHARED_CHUNK points; each Kelley round one pass, whose first point is the LP point (the rest are
# discarded: the price of the shared pass). "auto": on tapes of at least SHARED_MIN_EQNS equations
# (``fbbt_interp.tape_size``), where compile time is what matters; True / False: always / never.
SHARED_CUTS = "auto"
SHARED_CHUNK = 32
SHARED_MIN_EQNS = 700


# Affine parts linearized once per box, at the center, instead of at every cut point
# (``make_constrained_bounder``; the LP is the same). "auto": when at least half of the parts are
# affine (the extra center pass costs about one point's work; m34_convex2: 240 of 242 parts affine,
# node bound 3.9 s -> 0.9 s per 16 boxes) on tapes of at least SHARED_MIN_EQNS equations (where it
# was measured); True / False: whenever there are both kinds / never.
SPLIT_AFFINE = "auto"


# With ``SPLIT_AFFINE``, the affine parts' cuts are taken from their coefficients (``_affine_forms``)
# instead of a forward-mode pass of the whole relaxation at the box center: an affine part's
# relaxation is the function itself, so its cut is the same row (up to float rounding, which the
# relaxation's pass has too). On m34_convex2 that pass was 9.4k of the node bound kernel's 35k HLO
# lines. Not with ``safe`` (the relaxation's pass rounds outward; this does not).
AFFINE_EXACT = True


def _exact_affine(affine_of, n, afp, safe):
    """``(A, b)`` of the affine parts ``afp`` (``_affine_forms``), or None (``AFFINE_EXACT`` off,
    ``safe``, or a form unknown)."""
    if not AFFINE_EXACT or safe:
        return None
    forms = affine_of(n)
    if forms is None:
        return None
    A, b = forms[0][afp], forms[1][afp]
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(b))):
        return None
    return A, b


def _part_sizes(f, ineq, eq, n):
    """Outputs of ``(f, ineq, eq)`` (0 for a missing one): the parts are 1 + m_g + 2 m_h."""
    x = jax.ShapeDtypeStruct((n,), jnp.float64)

    def size(fn):
        return 0 if fn is None else int(np.prod(jax.eval_shape(lambda z: jnp.atleast_1d(fn(z)), x).shape))

    return size(f), size(ineq), size(eq)


def _affine_bounds(A, b, sizes, skip, lo, hi):
    """Interval bounds ``(lo, hi)`` over the box of the functions ``skip`` leaves out (their parts
    all affine, ``A x + b``), in ``parts.bounds``' layout; None for the others."""
    out, start = [], 0
    for i, m in enumerate(sizes):
        rows = slice(start, start + m)
        start += m if i < 2 else 2 * m
        if skip[i] is None or m == 0:
            out.append(None)
            continue
        Ar, br = jnp.asarray(A[rows]), jnp.asarray(b[rows])
        p, q = jnp.where(Ar != 0, Ar * lo, 0.0), jnp.where(Ar != 0, Ar * hi, 0.0)
        out.append((br + jnp.sum(jnp.minimum(p, q), axis=1), br + jnp.sum(jnp.maximum(p, q), axis=1)))
    return tuple(out)


def _affine_cuts(A, b, lo, w):
    """The cuts of affine parts ``A x + b`` in unit-box coordinates, shaped as one point's:
    ``(1, Ka)`` constants (the value at ``lo``) and ``(1, Ka, n)`` coefficients."""
    A = jnp.asarray(A)
    const, coef = jnp.asarray(b) + A @ lo, A * w
    ok = jnp.isfinite(const) & jnp.all(jnp.isfinite(coef), axis=1)  # (as ``_cuts``: an infinite box edge)
    return jnp.where(ok, const, -jnp.inf)[None], jnp.where(ok[:, None], coef, 0.0)[None]


class _SubParts:
    """``parts`` restricted to the parts ``idx`` (``parts.bounds`` interface of ``_cuts``).

    ``sizes``: the outputs of (f, g, h) (``_part_sizes``); a function none of whose parts is in
    ``idx`` is then not relaxed at all (its bounds are None). XLA does not drop an unused
    function's interpreted relaxation from the forward-mode pass by itself: on m34_convex2
    (nonlinear parts: f and one row of g) the equality rows' relaxation was 5.3k of the node
    bound kernel's 25.7k HLO lines."""

    def __init__(self, parts, idx, sizes=None):
        self.parts, self.idx = parts, np.asarray(idx)
        self.skip = None
        if sizes is not None:
            ends = np.cumsum([sizes[0], sizes[1], 2 * sizes[2]])
            starts = ends - np.array([sizes[0], sizes[1], 2 * sizes[2]])
            self.skip = tuple(None if np.any((self.idx >= a) & (self.idx < b)) else int(m)
                              for a, b, m in zip(starts, ends, sizes))  # fmt: skip

    def bounds(self, x, lo, hi, data):
        # contiguous runs as static slices (not a gather), so XLA drops the other parts' work
        v, b = self.parts.bounds(x, lo, hi, data, self.skip) if self.skip is not None else self.parts.bounds(x, lo, hi, data)
        idx = self.idx
        cuts = np.flatnonzero(np.diff(idx) != 1) + 1
        runs = [(int(r[0]), int(r[-1]) + 1) for r in np.split(idx, cuts)]
        return jnp.concatenate([v[a:b_] for a, b_ in runs]), b


def _tape_eqns(f, ineq, eq, n):
    """Equations of the joint tape of ``f``, ``ineq``, ``eq`` (0 if it cannot be built)."""
    from .fbbt import _has_tracer, joint_tape
    from .fbbt_interp import tape_size

    try:
        with jax.ensure_compile_time_eval():
            tape = joint_tape(f, ineq, eq, np.zeros(n))
        if _has_tracer(tape):
            return 0
        return int(tape_size(tape)[1])
    except Exception:  # noqa: BLE001 -- only a layout choice depends on it
        return 0


def _shared_cuts(f, ineq, eq):
    """``(shared(n), large(n))``: ``SHARED_CUTS`` resolved for a problem, and whether its tape has
    at least ``SHARED_MIN_EQNS`` equations (memoized on ``n``)."""
    cache = {}

    def get(n):
        if n not in cache:
            cache[n] = _tape_eqns(f, ineq, eq, n) >= SHARED_MIN_EQNS
        return cache[n]

    def shared(n):
        return bool(SHARED_CUTS) if SHARED_CUTS != "auto" else get(n)

    return shared, get


def _n_parts(f, ineq, eq, n):
    """Number of convex parts [cv_f, cv_g..., cv_h..., -cc_h...] (``_parts_fn``)."""
    x = jax.ShapeDtypeStruct((n,), jnp.float64)

    def size(fn):
        return int(np.prod(jax.eval_shape(lambda z: jnp.atleast_1d(fn(z)), x).shape))

    return size(f) + (size(ineq) if ineq is not None else 0) + (2 * size(eq) if eq is not None else 0)


KELLEY_STALL = 1e-4  # Kelley rounds stop once a round raises the bound by at most this share of the gap


def _kelley_go(prev, best, cutoff, first=False):
    """Whether another Kelley round is worth it for a box whose bound went from ``prev`` to
    ``best`` in the last round: not when the box is already pruned (``best >= cutoff``), nor
    when the round raised a finite bound by at most ``KELLEY_STALL`` of the gap to ``cutoff``
    (of ``1 + |best|`` without a finite cutoff). ``first``: after the first LP (no previous
    round to compare with), only the pruning test applies. Any number of rounds gives a
    valid bound (each round's certificate is one), so this only trades bound for time; under
    ``vmap`` the rounds continue while any box of the batch still wants one."""
    prev, best, cutoff = jnp.asarray(prev), jnp.asarray(best), jnp.asarray(cutoff)
    go = ~(best >= cutoff)
    if first:
        return go
    gap = jnp.where(jnp.isfinite(cutoff), cutoff - best, 1.0 + jnp.abs(best))
    stalled = jnp.isfinite(prev) & jnp.isfinite(best) & (best - prev <= KELLEY_STALL * gap)
    return go & ~stalled


def make_constrained_bounder(f, ineq=None, eq=None, n_kelley=4, ipm_iters=40, tol=0.0, univariate=False,
                             alphabb=False, multivariate=False, safe=False, exact=False, boxes=memory.BATCH):
    """Return ``bound(lo, hi, cutoff, pool=None, root=None) -> (lb, x_lp, infeasible, lo', hi')`` for one box.

    ``[lo', hi']`` is the box after duality-based range reduction against
    ``f <= cutoff`` (the incumbent value); it is the input box when
    ``cutoff`` is infinite.

    ``pool`` ``(R, n + 2)``: extra cuts ``a . x + b <= t`` (objective, last
    column > 0) or ``a . x + b <= tol`` (constraint), rows ``[a, b, obj]``,
    valid on the box ``root`` ``(2, n)`` (``make_pool_cuts``: tangents of the
    root box's relaxation). They are added to every LP when ``[lo, hi]`` lies in
    ``root``; ``b = -inf`` marks an empty slot. ``exact``: see ``_parts_fn``.

    ``boxes``: the boxes per kernel call (the ``vmap`` around ``bound``) that the memory budget
    of the initial cut points assumes (``memory.point_chunk``: over the budget, the points are
    linearized a chunk at a time).
    """
    prepare, parts, _ = _parts_fn(f, ineq, eq, univariate, alphabb, multivariate, safe, exact=exact)
    nonlinear = _nonlinear_of(f, ineq, eq)
    point_bytes = _point_bytes_of(f, ineq, eq)
    shared_of, large_of = _shared_cuts(f, ineq, eq)
    affine_of = _affine_of(f, ineq, eq)
    sizes_of = functools.lru_cache(maxsize=None)(lambda n: _part_sizes(f, ineq, eq, n))

    def bound(lo, hi, cutoff=jnp.inf, pool=None, root=None):
        n = lo.size
        nl = nonlinear(n)
        w = hi - lo
        c = 0.5 * (lo + hi)
        data = prepare(lo, hi)

        # Affine parts (``_nonlinear_vars``: no nonlinear variable) have the same cut everywhere, and
        # ``_cut_rows`` keeps only their cut at the center. With ``SPLIT_AFFINE`` they are linearized
        # once, at the center, and the cut points (axis and Kelley points) linearize the nonlinear
        # parts only: the same LP, without the forward-mode pass of every linear row at every point
        # (m34_convex2: 2 of 242 parts are nonlinear).
        split = bool(SPLIT_AFFINE) and nl is not None and nl.shape[1] == n and nl.any() and not nl.any(axis=1).all()
        if split and SPLIT_AFFINE == "auto":
            split = int((~nl.any(axis=1)).sum()) >= int(nl.any(axis=1).sum()) and large_of(n)
        if split:
            nlp = np.flatnonzero(nl.any(axis=1))
            afp = np.flatnonzero(~nl.any(axis=1))
            nl_p = nl[nlp]
            obj_p, obj_a = bool(nlp[0] == 0), bool(afp[0] == 0)
            exact = _exact_affine(affine_of, n, afp, safe)
            if exact is not None:
                # the affine parts' cuts from their coefficients (``AFFINE_EXACT``), and the cut points
                # relax only the functions with a nonlinear part; the others' interval bounds are their
                # affine forms' ranges on the box
                sizes = sizes_of(n)
                sub = _SubParts(parts, nlp, sizes)
                aff = _affine_cuts(*exact, lo, w)
                aff_b = _affine_bounds(*affine_of(n), sizes, sub.skip, lo, hi)

                def cuts_at(p):
                    """Cuts of the nonlinear parts at p (``_cuts``) and the interval bounds."""
                    ck, gk, b = _cuts(sub, data, p, lo, hi, safe)
                    return ck, gk, tuple(a if a is not None else e for a, e in zip(b, aff_b))
            else:
                sub = _SubParts(parts, nlp)
                ca, ga, bounds_c = _cuts(parts, data, c, lo, hi, safe)
                aff = (ca[afp][None], ga[afp][None])  # (1, Ka), (1, Ka, n)

                def cuts_at(p):
                    """Cuts of the nonlinear parts at p (``_cuts``); the bounds are the center's."""
                    ck, gk, _ = _cuts(sub, data, p, lo, hi, safe)
                    return ck, gk, bounds_c
        else:
            nl_p, obj_p, obj_a, aff = nl, True, False, None

            def cuts_at(p):
                """Affine minorants of each convex part at p, in unit-box coordinates u, and the interval
                bounds of f, g, h (``_cuts``)."""
                return _cuts(parts, data, p, lo, hi, safe)

        pts0 = cut_points(c, w, axis_vars(nl, n))
        P0 = pts0.shape[0]
        chunk = memory.point_chunk(point_bytes(n), P0, boxes, "node bound cut points")

        def obj_range(oc, og):
            """Largest finite lower / upper bound over the box of the objective cuts ``oc + og . u``."""
            cmin = oc + jnp.sum(jnp.minimum(og, 0), axis=-1)
            cmax = oc + jnp.sum(jnp.maximum(og, 0), axis=-1)
            if safe:  # rounding of the sums over coef
                cmin = cmin - SAFE_K * n * EPS * jnp.sum(jnp.abs(og), axis=-1)
                cmax = cmax + SAFE_K * n * EPS * jnp.sum(jnp.abs(og), axis=-1)
            return (jnp.max(jnp.where(jnp.isfinite(cmin), cmin, -jnp.inf)),
                    jnp.max(jnp.where(jnp.isfinite(cmax), cmax, -jnp.inf)))  # fmt: skip

        ninf = jnp.array(-jnp.inf)
        a_cmin, a_cmax = obj_range(aff[0][0, 0], aff[1][0, 0]) if obj_a else (ninf, ninf)

        def point_info(const0, coef0, bounds):
            """``(infeasible, flo, fhi, cmin, cmax)`` of a set of cut points: the interval bounds (the
            same at every point) and the largest finite lower / upper bound of the objective's cuts
            over the box (-inf if none)."""
            (flo, fhi), bg, bh = jax.tree.map(lambda b: b[0], bounds)
            flo, fhi = flo[0], fhi[0]
            infeasible = jnp.array(False)
            if bg is not None:
                infeasible |= jnp.any(bg[0] > tol)
            if bh is not None:
                infeasible |= jnp.any(bh[0] > tol) | jnp.any(bh[1] < -tol)
            cmin, cmax = obj_range(const0[:, 0], coef0[:, 0]) if obj_p else (ninf, ninf)
            return infeasible, flo, fhi, jnp.maximum(cmin, a_cmin), jnp.maximum(cmax, a_cmax)

        def t_range(flo, fhi, cmin, cmax):
            """``(tlo, T, bad_t)``: the objective-variable range [tlo, tlo + T] from intervals and cuts."""
            tlo = jnp.maximum(flo, cmin)
            thi = jnp.minimum(fhi, cmax)
            thi = jnp.where(jnp.isfinite(thi) & (thi > tlo), thi, tlo + 1.0)
            return tlo, thi - tlo, ~jnp.isfinite(tlo)

        def interval_info(const0, coef0, bounds):
            """``(infeasible, tlo, T, bad_t)`` from the interval bounds and the initial cuts."""
            infeasible, *rest = point_info(const0, coef0, bounds)
            return (infeasible, *t_range(*rest))

        if pool is not None:
            # pool cuts in unit-box coordinates; valid only where the box lies in their root box
            pa, pb0, pobj = pool[:, :n], pool[:, n], pool[:, n + 1] > 0
            inside = jnp.all(lo >= root[0]) & jnp.all(hi <= root[1])
            pcoef = pa * w
            pconst = _cut_const(pb0, pa, lo, jnp.zeros(n), pcoef, safe)  # a . lo + b
            pok = inside & jnp.isfinite(pconst) & jnp.all(jnp.isfinite(pcoef), axis=1)
            pconst = jnp.where(pok, pconst, -jnp.inf)
            pcoef = jnp.where(pok[:, None], pcoef, 0.0)

        def block(const, coef, obj, T, tlo_s):
            """Rows of cuts ``(P, K)`` / ``(P, K, n)`` (point-major); part 0 is the objective if ``obj``."""
            tcol = jnp.zeros(const.shape + (1,))
            if obj:
                tcol = tcol.at[..., 0, 0].set(-T)
                bvec = jnp.concatenate([tlo_s - const[..., :1], tol - const[..., 1:]], axis=-1)
                mag = jnp.concatenate([jnp.abs(tlo_s) + jnp.abs(const[..., :1]), abs(tol) + jnp.abs(const[..., 1:])], axis=-1)
            else:
                bvec, mag = tol - const, abs(tol) + jnp.abs(const)
            A = jnp.concatenate([coef, tcol], axis=-1).reshape(-1, n + 1)
            return A, bvec.reshape(-1), mag.reshape(-1)

        def rows(const, coef, T, tlo_s):
            """LP rows over z = (u, tau) with t = tlo + T tau.

            objective cut:  const0 + coef0.u <= t   -> coef0.u - T tau <= tlo - const0
            constraint cut: const_i + coef_i.u <= tol -> coef_i.u        <= tol - const_i

            the affine parts' center cuts (``split``), then the cut points' rows, then the pool's
            rows (the same two forms).
            """
            A, bvec, mag = block(const, coef, obj_p, T, tlo_s)
            keep = _cut_rows(const.shape, nl_p)
            if keep is not None:
                A, bvec, mag = A[keep], bvec[keep], mag[keep]
            if aff is not None:
                Aa, ba, ma = block(aff[0], aff[1], obj_a, T, tlo_s)
                A, bvec, mag = jnp.concatenate([Aa, A]), jnp.concatenate([ba, bvec]), jnp.concatenate([ma, mag])
            if pool is not None:
                pool_A = jnp.concatenate([pcoef, jnp.where(pobj, -T, 0.0)[:, None]], axis=1)
                pool_b = jnp.where(pobj, tlo_s, tol) - pconst
                pool_mag = jnp.where(pobj, jnp.abs(tlo_s), abs(tol)) + jnp.abs(pconst)
                A, bvec, mag = (jnp.concatenate([A, pool_A]), jnp.concatenate([bvec, pool_b]),
                                jnp.concatenate([mag, pool_mag]))  # fmt: skip
            live = jnp.isfinite(bvec)
            nrm = jnp.linalg.norm(A, axis=1)
            live &= nrm > 0
            nrm = jnp.where(live, nrm, 1.0)
            A = jnp.where(live[:, None], A / nrm[:, None], 0.0)
            bvec = jnp.where(live, bvec / nrm, 1.0)  # dead rows: 0 <= 1
            bvec = _row_margin(A, bvec, jnp.where(live, mag / nrm, 0.0), live, safe)
            return A, bvec

        cobj = jnp.zeros(n + 1).at[n].set(1.0)

        def dual(A, bvec, y, T, tlo_s):
            # Certificate over u in [0,1]^n, tau in [0, inf): the cap tau <= 1
            # used inside the LP is NOT relied on (later Kelley cuts may exceed
            # thi). Objective-row duals are rescaled so the tau coefficient
            # 1 - sum(y_obj * T / |row|) stays >= 0; any y >= 0 is then valid.
            # Returns the bound and the reduced costs of u.
            y = jnp.where(jnp.isfinite(y), jnp.maximum(y, 0.0), 0.0)
            s_obj = -(A[:, n] @ y)
            y = y * jnp.where(s_obj > 1.0, 1.0 / s_obj, 1.0) ** (A[:, n] < 0)
            coef = cobj + A.T @ y
            val = jnp.sum(jnp.minimum(coef[:n], 0.0)) - y @ bvec
            # rounding error of the two sums (|A| <= 1 after row scaling)
            err = (A.shape[0] + n + 2) * EPS * (jnp.sum(1.0 + jnp.abs(A).T @ y) + jnp.abs(bvec) @ y)
            if safe:  # the certificate is evaluated in floats: widen the error bound, round tlo + T * val
                err = SAFE_K * err
                return tlo_s + T * (val - err) - SAFE_K * EPS * (jnp.abs(tlo_s) + T * (jnp.abs(val) + err)), _safe_costs(coef[:n], A[:, :n], y)
            return tlo_s + T * (val - err), _safe_costs(coef[:n], A[:, :n], y)

        def solve(const, coef, best, rlo, rhi, T, tlo_s, bad_t):
            """One LP over the current cuts: (best', rlo', rhi', u)."""
            A, bvec = rows(const, coef, T, tlo_s)
            nA = A.shape[0]
            z, y, lb = ipm(cobj, A, bvec, ipm_iters, cert=lambda y: dual(A, bvec, y[:nA], T, tlo_s)[0], box=True,
                           target=cutoff)  # fmt: skip
            lb = jnp.where(jnp.isnan(lb) | bad_t, -jnp.inf, lb)
            best = jnp.maximum(best, lb)
            # range reduction from this round's certificate (r is per unit of u)
            _, r = dual(A, bvec, y[:nA], T, tlo_s)
            klo, khi = reduce_ranges(lo, hi, lb, T * r / jnp.where(w > 0, w, 1.0), cutoff)
            rlo, rhi = jnp.maximum(rlo, klo), jnp.minimum(rhi, khi)
            return best, rlo, rhi, jnp.clip(z[:n], 0.0, 1.0)

        # Kelley rounds over a fixed-size cut buffer: slots for cuts not generated yet hold
        # const = -inf, which rows() turns into dead rows (0 <= 1), so every round has the
        # same LP shape. The rounds stop early once a round no longer raises the bound
        # (``KELLEY_STALL``) or the box is pruned (see ``_kelley_go``).
        if not shared_of(n):
            const0, coef0, bounds = memory.map_points(cuts_at, pts0, chunk)  # (P0, Kp), (P0, Kp, n)
            infeasible, tlo, T, bad_t = interval_info(const0, coef0, bounds)
            tlo_s = jnp.where(bad_t, 0.0, tlo)
            const = jnp.concatenate([const0, jnp.full((n_kelley,) + const0.shape[1:], -jnp.inf)])
            coef = jnp.concatenate([coef0, jnp.zeros((n_kelley,) + coef0.shape[1:])])
            best0 = jnp.where(bad_t, -jnp.inf, tlo)
            best, rlo, rhi, u = solve(const, coef, best0, lo, hi, T, tlo_s, bad_t)

            def kelley(carry):
                # Kelley: cuts at the last LP solution, then the LP again
                k, const, coef, best, rlo, rhi, u, _ = carry
                ck, gk, _ = cuts_at(lo + w * u)
                const, coef = const.at[P0 + k].set(ck), coef.at[P0 + k].set(gk)
                prev = best
                best, rlo, rhi, u = solve(const, coef, best, rlo, rhi, T, tlo_s, bad_t)
                return k + 1, const, coef, best, rlo, rhi, u, _kelley_go(prev, best, cutoff)

            init = (0, const, coef, best, rlo, rhi, u, _kelley_go(best0, best, cutoff, first=True))
            _, _, _, best, rlo, rhi, u, _ = jax.lax.while_loop(lambda c: (c[0] < n_kelley) & c[7], kelley, init)
        else:
            # One loop for the first LP and the Kelley rounds, so the relaxation's forward-mode
            # pass and the LP appear once in the kernel instead of twice (see ``SHARED_CUTS``).
            Kp = len(nlp) if split else _n_parts(f, ineq, eq, n)
            C = P0 if SHARED_CHUNK is None else min(P0, SHARED_CHUNK)
            if chunk is not None:
                C = min(C, chunk)
            nC = -(-P0 // C)
            pts0p = jnp.concatenate([pts0, jnp.broadcast_to(c, (nC * C - P0, n))])  # padding: the center again
            const = jnp.full((P0 + n_kelley, Kp), -jnp.inf)
            coef = jnp.zeros((P0 + n_kelley, Kp, n))
            slot = jnp.arange(C)
            drop = P0 + n_kelley  # out of range: dropped by the scatter

            def round_(carry):
                k, const, coef, info, best, rlo, rhi, u, _ = carry
                first = k == 0
                p_kelley = jnp.broadcast_to(lo + w * u, (C, n))

                def chunk_(cc):
                    j, const, coef, pinfo = cc
                    pts = jnp.where(first, jax.lax.dynamic_slice(pts0p, (j * C, 0), (C, n)), p_kelley)
                    ck, gk, bounds = jax.vmap(cuts_at)(pts)
                    # round 0 fills slots 0..P0-1 chunk by chunk, round k >= 1 slot P0 + k - 1
                    s0 = j * C + slot
                    idx = jnp.where(first, jnp.where(s0 < P0, s0, drop), jnp.where(slot == 0, P0 + k - 1, drop))
                    const = const.at[idx].set(ck, mode="drop")
                    coef = coef.at[idx].set(gk, mode="drop")
                    e, flo, fhi, cmn, cmx = point_info(ck, gk, bounds)
                    pinfo = (pinfo[0] | e, flo, fhi, jnp.maximum(pinfo[3], cmn), jnp.maximum(pinfo[4], cmx))
                    return j + 1, const, coef, pinfo

                pinfo0 = (jnp.array(False), ninf, ninf, ninf, ninf)
                trips = jnp.where(first, nC, 1)
                _, const, coef, pinfo = jax.lax.while_loop(lambda cc: cc[0] < trips, chunk_,
                                                           (0, const, coef, pinfo0))
                new = (pinfo[0], *t_range(*pinfo[1:]))
                info = jax.tree.map(lambda a, b: jnp.where(first, a, b), new, info)
                _, tlo, T, bad_t = info
                tlo_s = jnp.where(bad_t, 0.0, tlo)
                prev = jnp.where(first, jnp.where(bad_t, -jnp.inf, tlo), best)
                best, rlo, rhi, u = solve(const, coef, prev, rlo, rhi, T, tlo_s, bad_t)
                go = jnp.where(first, _kelley_go(prev, best, cutoff, first=True), _kelley_go(prev, best, cutoff))
                return k + 1, const, coef, info, best, rlo, rhi, u, go

            info0 = (jnp.array(False), jnp.array(0.0), jnp.array(1.0), jnp.array(False))
            init = (0, const, coef, info0, jnp.array(-jnp.inf), lo, hi, jnp.full(n, 0.5), jnp.array(True))
            out = jax.lax.while_loop(lambda c: (c[0] <= n_kelley) & c[8], round_, init)
            _, _, _, (infeasible, _, _, _), best, rlo, rhi, u, _ = out

        lb = jnp.where(infeasible, jnp.inf, best)
        return lb, lo + w * u, infeasible, rlo, rhi

    return bound


def make_pool_cuts(f, ineq=None, eq=None, univariate=False, alphabb=False, multivariate=False, safe=False,
                   exact=False):
    """Return ``cuts(p, lo, hi) -> (a, b)``: the cuts ``a[k] . x + b[k]`` of each convex part
    [cv_f, cv_g..., cv_h..., -cc_h...] of the relaxation on the box ``[lo, hi]``, taken at ``p``
    (``a`` ``(K, n)``, ``b`` ``(K,)``; ``b = -inf`` where not finite).

    A tangent of a convex minorant on ``[lo, hi]`` minorizes the part on every sub-box, so
    these cuts, taken once on the root box, are valid at every node of the tree (the cut
    pool of ``make_constrained_bounder``). With ``safe``, ``b`` is lowered by the rounding of
    ``v - a . p``. ``exact``: see ``_parts_fn``.
    """
    prepare, parts, _ = _parts_fn(f, ineq, eq, univariate, alphabb, multivariate, safe, exact=exact)

    def cuts(p, lo, hi):
        data = prepare(lo, hi)

        def fn(x):
            v = parts(x, lo, hi, data)
            return v, v

        J, v = jax.jacfwd(fn, has_aux=True)(p)
        b = v - J @ p
        if safe:
            b = b - SAFE_K * (p.size + 3) * EPS * (jnp.abs(v) + jnp.abs(J) @ jnp.abs(p))
        ok = jnp.isfinite(b) & jnp.all(jnp.isfinite(J), axis=1)
        return jnp.where(ok[:, None], J, 0.0), jnp.where(ok, b, -jnp.inf)

    return cuts


def pool_parts(f, ineq, eq, n, safe=False):
    """Return ``parts_ok(lo, hi) -> (mask, exact)`` from the convexity certificates of
    ``f``, ``ineq``, ``eq`` (``convexity``), or None when no nonlinear part is ever certified.

    ``mask`` ``(K,)``: the convex parts [cv_f, cv_g..., cv_h..., -cc_h...] that are nonlinear
    and certified convex on the box, for which the root box's tangents are worth pooling (on
    them the relaxation is the function itself, so a tangent at a point near the optimum is
    as tight as a cut can be there; elsewhere a root-box tangent is a weak cut, valid but
    rarely useful). ``exact``: every certified verdict's conditions hold on the box, so
    on its sub-boxes a certified function can be relaxed by itself alone (``_parts_fn``).
    ``parts_ok(lo, hi, convex=True)`` adds a third value: every part is affine or certified
    convex on the box (the continuous relaxation of the problem is a convex program there).
    """
    from . import convexity

    if not convexity.ENABLED:
        return None
    x = jnp.zeros(n)
    certs = []  # (cert, side, outputs, all affine when cert is None) per block of parts
    for fn, sides in ((f, ("cv",)), (ineq, ("cv",)), (eq, ("cv", "cc"))):
        if fn is None:
            continue
        flat = lambda x, fn=fn: jnp.reshape(fn(x), (-1,))  # noqa: E731
        try:
            cert = convexity.certificate_of(flat, x)
        except Exception:
            cert = None
        if cert is not None and safe and cert.size > convexity.SAFE_MAX_NODES:
            cert = None
        m = int(np.prod(jax.eval_shape(flat, x).shape))
        aff = None
        if cert is None:  # no verdict: unknown, or every output affine (``classify``)
            try:
                aff = convexity.affine_of(flat, x)
            except Exception:
                aff = None
            aff = bool(aff is not None and aff.all())
        for side in sides:
            certs.append((cert, side, m, aff))
    if not any(c is not None and (np.asarray(getattr(c, side + "_mask")) & ~np.asarray(c.affine)).any()
               for c, side, _, _ in certs):  # fmt: skip
        return None

    def parts_ok(lo, hi, convex=False):
        lo, hi = jnp.asarray(lo, jnp.float64), jnp.asarray(hi, jnp.float64)
        out, exact, cvx = [], True, True
        for cert, side, m, aff in certs:
            if cert is None:
                out.append(np.zeros(m, bool))
                cvx &= aff
                continue
            ok_cv, ok_cc = convexity.prepare(cert, lo, hi)
            ok = np.asarray(ok_cv if side == "cv" else ok_cc, bool)
            static = np.asarray(getattr(cert, side + "_mask"), bool)
            exact &= bool(np.all(ok | ~static))
            affine = np.asarray(cert.affine, bool)
            cvx &= bool(np.all(ok | affine))
            out.append(ok & ~affine)
        return (np.concatenate(out), exact, cvx) if convex else (np.concatenate(out), exact)

    return parts_ok


def make_obbt(f, ineq=None, eq=None, rounds=2, ipm_iters=40, tol=0.0, univariate=False, alphabb=False,
              multivariate=False, safe=False, own_cuts=True, exact=False, boxes=1, variables=None):
    """Return ``obbt(lo, hi, cutoff) -> (lo', hi', infeasible)`` for one box.

    Optimality-based bound tightening: each variable is minimized and
    maximized over the outer-approximation LP

        cut_f(u) <= cutoff,   cut_g(u) <= tol,   cut_h(u) <= tol,   -cut_cc_h(u) <= tol,

    with cuts at the box center and 2n axis points (rows as in the node bound,
    ``_cut_rows``) and, in later rounds, at LP solutions of the previous rounds:
    with ``own_cuts`` each LP gets the cuts at its own earlier solutions, else
    every LP gets the cuts at all 2n of them (2n times the rows, for about the
    same tightening). The 2n LPs are one ``vmap``. As in the
    node bound, the new variable bounds are weak-duality certificates, so an
    inexact LP solve can only make them looser, never invalid. ``exact``: see ``_parts_fn``.
    ``boxes``: the boxes per call the memory budget assumes (``make_constrained_bounder``).
    ``variables``: indices of the variables to tighten, or a function of n giving them (None: all;
    the others keep their bounds).
    """
    prepare, parts, _ = _parts_fn(f, ineq, eq, univariate, alphabb, multivariate, safe, exact=exact)
    nonlinear = _nonlinear_of(f, ineq, eq)
    point_bytes = _point_bytes_of(f, ineq, eq)
    _, large_of = _shared_cuts(f, ineq, eq)
    affine_of = _affine_of(f, ineq, eq)
    sizes_of = functools.lru_cache(maxsize=None)(lambda n: _part_sizes(f, ineq, eq, n))

    def obbt(lo, hi, cutoff=jnp.inf):
        n = lo.size
        nl = nonlinear(n)
        w = hi - lo
        c = 0.5 * (lo + hi)
        data = prepare(lo, hi)

        # ``SPLIT_AFFINE`` as in the node bound: the affine parts are linearized once, at the center
        # (``_cut_rows`` keeps only that cut of theirs), and the cut points of both rounds linearize
        # the nonlinear parts only. The same LP rows (affine ones first), without the forward-mode
        # pass of every affine part at every point: on m34_convex2 (240 of 242 parts affine) the
        # kernel compiled in about half the time.
        split = bool(SPLIT_AFFINE) and nl is not None and nl.shape[1] == n and nl.any() and not nl.any(axis=1).all()
        if split and SPLIT_AFFINE == "auto":
            split = int((~nl.any(axis=1)).sum()) >= int(nl.any(axis=1).sum()) and large_of(n)
        if split:
            nlp, afp = np.flatnonzero(nl.any(axis=1)), np.flatnonzero(~nl.any(axis=1))
            nl_p, obj_p, obj_a = nl[nlp], bool(nlp[0] == 0), bool(afp[0] == 0)
            exact = _exact_affine(affine_of, n, afp, safe)
            if exact is not None:  # (``AFFINE_EXACT``; the cut points relax only the functions they need)
                sub = _SubParts(parts, nlp, sizes_of(n))
                aff = _affine_cuts(*exact, lo, w)
            else:
                sub = _SubParts(parts, nlp)
                ca, ga, _ = _cuts(parts, data, c, lo, hi, safe)
                aff = (ca[afp][None], ga[afp][None])  # (1, Ka), (1, Ka, n)

            def cuts_at(p):
                return _cuts(sub, data, p, lo, hi, safe)[:2]
        else:
            nl_p, obj_p, aff = nl, True, None

            def cuts_at(p):
                return _cuts(parts, data, p, lo, hi, safe)[:2]

        def block(const, coef, obj, nlk):
            """Rows ``A u <= rhs`` (and their magnitudes) of cuts ``(P, K)`` / ``(P, K, n)``; part 0 is
            the objective (``f <= cutoff``) if ``obj``; the rows ``_cut_rows`` keeps for ``nlk``."""
            if obj:
                rhs = jnp.concatenate([cutoff - const[:, :1], tol - const[:, 1:]], axis=1).reshape(-1)
                mag = jnp.concatenate([abs(cutoff) + jnp.abs(const[:, :1]), abs(tol) + jnp.abs(const[:, 1:])], axis=1).reshape(-1)
            else:
                rhs, mag = (tol - const).reshape(-1), (abs(tol) + jnp.abs(const)).reshape(-1)
            A = coef.reshape(-1, n)
            keep = _cut_rows(const.shape, nlk) if nlk is not None else None
            if keep is not None:  # first: the rows are normalized one by one, so this changes no kept row
                A, rhs, mag = A[keep], rhs[keep], mag[keep]
            return A, rhs, mag

        def rows(const, coef):
            A, rhs, mag = block(const, coef, obj_p, nl_p)
            if aff is not None:
                Aa, ra, ma = block(aff[0], aff[1], obj_a, None)
                A, rhs, mag = jnp.concatenate([Aa, A]), jnp.concatenate([ra, rhs]), jnp.concatenate([ma, mag])
            nrm = jnp.linalg.norm(A, axis=1)
            live = jnp.isfinite(rhs) & (nrm > 0)
            nrm = jnp.where(live, nrm, 1.0)
            A = jnp.where(live[:, None], A / nrm[:, None], 0.0)
            b = jnp.where(live, rhs / nrm, 1.0)
            return A, _row_margin(A, b, jnp.where(live, mag / nrm, 0.0), live, safe)

        sel = variables(n) if callable(variables) else variables
        idx = np.arange(n) if sel is None else np.asarray(sel, dtype=np.int64)
        m = idx.size
        E = jnp.eye(n)[idx]
        C = jnp.concatenate([E, -E])

        def solve(cvec, A, bvec):
            nA = A.shape[0]

            def cert(y):
                y = jnp.where(jnp.isfinite(y[:nA]), jnp.maximum(y[:nA], 0.0), 0.0)
                val = jnp.sum(jnp.minimum(cvec + A.T @ y, 0.0)) - y @ bvec
                err = (nA + n + 2) * EPS * (jnp.sum(1.0 + jnp.abs(A).T @ y) + jnp.abs(bvec) @ y)
                if safe:
                    err = SAFE_K * err
                return val - err

            z, _, best = ipm(cvec, A, bvec, ipm_iters, cert=cert, box=True)
            return jnp.clip(z, 0.0, 1.0), best

        pts0 = cut_points(c, w, axis_vars(nl, n))
        const, coef = memory.map_points(cuts_at, pts0, memory.point_chunk(point_bytes(n), pts0.shape[0], boxes,
                                                                          "OBBT cut points"))
        ulo, uhi = jnp.zeros(m), jnp.ones(m)
        own_c = own_g = None  # per LP (own_cuts): cuts at its own earlier solutions
        for k in range(rounds):
            if not own_cuts or k == 0:
                A, bvec = rows(const, coef)
                Z, B = jax.vmap(solve, (0, None, None))(C, A, bvec)
            else:
                Z, B = jax.vmap(lambda cv, oc, og: solve(cv, *rows(jnp.concatenate([const, oc]),
                                                                    jnp.concatenate([coef, og]))))(C, own_c, own_g)
            ulo = jnp.maximum(ulo, jnp.where(jnp.isfinite(B[:m]), B[:m], 0.0))
            uhi = jnp.minimum(uhi, jnp.where(jnp.isfinite(B[m:]), -B[m:], 1.0))
            if k < rounds - 1:  # cuts at this round's 2n LP solutions
                ck, gk = memory.map_points(cuts_at, lo + w * Z, memory.point_chunk(point_bytes(n), Z.shape[0], boxes,
                                                                                   "OBBT round cut points"))
                if own_cuts:
                    own_c = ck[:, None] if k == 0 else jnp.concatenate([own_c, ck[:, None]], axis=1)
                    own_g = gk[:, None] if k == 0 else jnp.concatenate([own_g, gk[:, None]], axis=1)
                else:
                    const, coef = jnp.concatenate([const, ck]), jnp.concatenate([coef, gk])

        infeasible = jnp.any(ulo > uhi + 1e-9)
        ulo, uhi = jnp.zeros(n).at[idx].set(ulo - 1e-9), jnp.ones(n).at[idx].set(uhi + 1e-9)  # floating-point slack
        new_lo = jnp.where(w > 0, jnp.maximum(lo, lo + w * ulo), lo)
        new_hi = jnp.where(w > 0, jnp.minimum(hi, lo + w * uhi), hi)
        return jnp.minimum(new_lo, new_hi), jnp.maximum(new_lo, new_hi), infeasible

    return obbt
