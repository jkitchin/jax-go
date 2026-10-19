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

import jax
import jax.numpy as jnp
import numpy as np

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


def _cut_rows(shape, nl):
    """Flat indices of the LP rows to keep from cuts of shape ``(P, K)`` (P points, K parts;
    point-major rows), or None for all of them.

    The points are the center, the 2n axis points ``c + e_j w_j / 4`` then
    ``c - e_j w_j / 4``, then Kelley points. Part k keeps its cut at the center,
    at the axis points of the variables it is nonlinear in (``nl[k]``) and at
    every Kelley point unless it is affine. Moving along a variable a part is
    linear in leaves its subgradient and its cut unchanged (McCormick, the
    univariate hulls and alphaBB all keep the linear terms exact), so the
    dropped rows repeat the center's cut.
    """
    P, K = shape
    if nl is None or nl.shape[0] != K:
        return None
    n = nl.shape[1]
    keep = np.ones((P, K), bool)
    axes = np.concatenate([nl.T, nl.T])[: max(P - 1, 0)]  # (2n, K) for the axis points
    keep[1 : 1 + len(axes)] = axes
    keep[1 + 2 * n :] = nl.any(axis=1)
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

    def parts_bounds(x, lo, hi, data):
        """``(parts, bounds)``: ``bounds`` are the interval bounds ``(f, g, h)`` on the box, each
        an MC's ``(lo, hi)`` or None; they do not depend on ``x``."""
        rf, rg, rh = rs
        mf = rf(x, lo, hi, data[0])
        out, bounds = [jnp.atleast_1d(mf.cv)], [(mf.lo, mf.hi), None, None]
        if rg is not None:
            mg = rg(x, lo, hi, data[1])
            out.append(mg.cv)
            bounds[1] = (mg.lo, mg.hi)
        if rh is not None:
            mh = rh(x, lo, hi, data[2])
            out += [mh.cv, -mh.cc]
            bounds[2] = (mh.lo, mh.hi)
        return jnp.concatenate(out), tuple(bounds)

    parts.bounds = parts_bounds
    return prepare, parts, rs


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
                             alphabb=False, multivariate=False, safe=False, exact=False):
    """Return ``bound(lo, hi, cutoff, pool=None, root=None) -> (lb, x_lp, infeasible, lo', hi')`` for one box.

    ``[lo', hi']`` is the box after duality-based range reduction against
    ``f <= cutoff`` (the incumbent value); it is the input box when
    ``cutoff`` is infinite.

    ``pool`` ``(R, n + 2)``: extra cuts ``a . x + b <= t`` (objective, last
    column > 0) or ``a . x + b <= tol`` (constraint), rows ``[a, b, obj]``,
    valid on the box ``root`` ``(2, n)`` (``make_pool_cuts``: tangents of the
    root box's relaxation). They are added to every LP when ``[lo, hi]`` lies in
    ``root``; ``b = -inf`` marks an empty slot. ``exact``: see ``_parts_fn``.
    """
    prepare, parts, _ = _parts_fn(f, ineq, eq, univariate, alphabb, multivariate, safe, exact=exact)
    nonlinear = _nonlinear_of(f, ineq, eq)

    def bound(lo, hi, cutoff=jnp.inf, pool=None, root=None):
        n = lo.size
        nl = nonlinear(n)
        w = hi - lo
        c = 0.5 * (lo + hi)
        data = prepare(lo, hi)

        def cuts_at(p):
            """Affine minorants of each convex part at p, in unit-box coords u, and the interval
            bounds of f, g, h (``_cuts``)."""
            return _cuts(parts, data, p, lo, hi, safe)

        axis = jnp.concatenate([jnp.eye(n), -jnp.eye(n)]) * (w / 4)
        pts0 = jnp.concatenate([c[None], c + axis])
        const0, coef0, bounds = jax.vmap(cuts_at)(pts0)  # (P0, 1+m), (P0, 1+m, n)

        # --- interval information (constant on the box: any point's) ---------
        (flo, fhi), bg, bh = jax.tree.map(lambda b: b[0], bounds)
        flo, fhi = flo[0], fhi[0]
        infeasible = jnp.array(False)
        if bg is not None:
            infeasible |= jnp.any(bg[0] > tol)
        if bh is not None:
            infeasible |= jnp.any(bh[0] > tol) | jnp.any(bh[1] < -tol)

        # objective-variable range [tlo, thi] from intervals and the cuts
        cmin = const0[:, 0] + jnp.sum(jnp.minimum(coef0[:, 0], 0), axis=1)
        cmax = const0[:, 0] + jnp.sum(jnp.maximum(coef0[:, 0], 0), axis=1)
        if safe:  # rounding of the sums over coef
            cmin = cmin - SAFE_K * n * EPS * jnp.sum(jnp.abs(coef0[:, 0]), axis=1)
            cmax = cmax + SAFE_K * n * EPS * jnp.sum(jnp.abs(coef0[:, 0]), axis=1)
        tlo = jnp.maximum(flo, jnp.max(jnp.where(jnp.isfinite(cmin), cmin, -jnp.inf)))
        thi = jnp.minimum(fhi, jnp.max(jnp.where(jnp.isfinite(cmax), cmax, -jnp.inf)))
        thi = jnp.where(jnp.isfinite(thi) & (thi > tlo), thi, tlo + 1.0)
        T = thi - tlo
        bad_t = ~jnp.isfinite(tlo)
        tlo_s = jnp.where(bad_t, 0.0, tlo)

        if pool is not None:
            # pool cuts in unit-box coordinates; valid only where the box lies in their root box
            pa, pb0, pobj = pool[:, :n], pool[:, n], pool[:, n + 1] > 0
            inside = jnp.all(lo >= root[0]) & jnp.all(hi <= root[1])
            pcoef = pa * w
            pconst = _cut_const(pb0, pa, lo, jnp.zeros(n), pcoef, safe)  # a . lo + b
            pok = inside & jnp.isfinite(pconst) & jnp.all(jnp.isfinite(pcoef), axis=1)
            pconst = jnp.where(pok, pconst, -jnp.inf)
            pcoef = jnp.where(pok[:, None], pcoef, 0.0)
            pool_A = jnp.concatenate([pcoef, jnp.where(pobj, -T, 0.0)[:, None]], axis=1)
            pool_b = jnp.where(pobj, tlo_s, tol) - pconst
            pool_mag = jnp.where(pobj, jnp.abs(tlo_s), abs(tol)) + jnp.abs(pconst)

        def rows(const, coef):
            """LP rows over z = (u, tau) with t = tlo + T tau.

            objective cut:  const0 + coef0.u <= t   -> coef0.u - T tau <= tlo - const0
            constraint cut: const_i + coef_i.u <= tol -> coef_i.u        <= tol - const_i

            followed by the pool's rows (the same two forms).
            """
            tcol = jnp.zeros(const.shape + (1,)).at[..., 0, 0].set(-T)
            A = jnp.concatenate([coef, tcol], axis=-1).reshape(-1, n + 1)
            bvec = jnp.concatenate([tlo_s - const[..., :1], tol - const[..., 1:]], axis=-1).reshape(-1)
            mag = jnp.concatenate([jnp.abs(tlo_s) + jnp.abs(const[..., :1]), abs(tol) + jnp.abs(const[..., 1:])], axis=-1).reshape(-1)
            keep = _cut_rows(const.shape, nl)
            if keep is not None:
                A, bvec, mag = A[keep], bvec[keep], mag[keep]
            if pool is not None:
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

        def dual(A, bvec, y):
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

        # Kelley rounds over a fixed-size cut buffer: slots for cuts not generated yet hold
        # const = -inf, which rows() turns into dead rows (0 <= 1), so every round has the
        # same LP shape. The rounds stop early once a round no longer raises the bound
        # (``KELLEY_STALL``) or the box is pruned (see ``_kelley_go``).
        P0 = const0.shape[0]
        const = jnp.concatenate([const0, jnp.full((n_kelley,) + const0.shape[1:], -jnp.inf)])
        coef = jnp.concatenate([coef0, jnp.zeros((n_kelley,) + coef0.shape[1:])])

        def solve(const, coef, best, rlo, rhi):
            """One LP over the current cuts: (best', rlo', rhi', u)."""
            A, bvec = rows(const, coef)
            nA = A.shape[0]
            z, y, lb = ipm(cobj, A, bvec, ipm_iters, cert=lambda y: dual(A, bvec, y[:nA])[0], box=True,
                           target=cutoff)  # fmt: skip
            lb = jnp.where(jnp.isnan(lb) | bad_t, -jnp.inf, lb)
            best = jnp.maximum(best, lb)
            # range reduction from this round's certificate (r is per unit of u)
            _, r = dual(A, bvec, y[:nA])
            klo, khi = reduce_ranges(lo, hi, lb, T * r / jnp.where(w > 0, w, 1.0), cutoff)
            rlo, rhi = jnp.maximum(rlo, klo), jnp.minimum(rhi, khi)
            return best, rlo, rhi, jnp.clip(z[:n], 0.0, 1.0)

        best0 = jnp.where(bad_t, -jnp.inf, tlo)
        best, rlo, rhi, u = solve(const, coef, best0, lo, hi)

        def kelley(carry):
            # Kelley: cuts at the last LP solution, then the LP again
            k, const, coef, best, rlo, rhi, u, _ = carry
            ck, gk, _ = cuts_at(lo + w * u)
            const, coef = const.at[P0 + k].set(ck), coef.at[P0 + k].set(gk)
            prev = best
            best, rlo, rhi, u = solve(const, coef, best, rlo, rhi)
            return k + 1, const, coef, best, rlo, rhi, u, _kelley_go(prev, best, cutoff)

        init = (0, const, coef, best, rlo, rhi, u, _kelley_go(best0, best, cutoff, first=True))
        _, _, _, best, rlo, rhi, u, _ = jax.lax.while_loop(lambda c: (c[0] < n_kelley) & c[7], kelley, init)

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
              multivariate=False, safe=False, own_cuts=True, exact=False):
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
    """
    prepare, parts, _ = _parts_fn(f, ineq, eq, univariate, alphabb, multivariate, safe, exact=exact)
    nonlinear = _nonlinear_of(f, ineq, eq)

    def obbt(lo, hi, cutoff=jnp.inf):
        n = lo.size
        nl = nonlinear(n)
        w = hi - lo
        c = 0.5 * (lo + hi)
        data = prepare(lo, hi)

        def cuts_at(p):
            return _cuts(parts, data, p, lo, hi, safe)[:2]

        def rows(const, coef):
            rhs = jnp.concatenate([cutoff - const[:, :1], tol - const[:, 1:]], axis=1).reshape(-1)
            A = coef.reshape(-1, n)
            nrm = jnp.linalg.norm(A, axis=1)
            live = jnp.isfinite(rhs) & (nrm > 0)
            nrm = jnp.where(live, nrm, 1.0)
            mag = jnp.concatenate([abs(cutoff) + jnp.abs(const[:, :1]), abs(tol) + jnp.abs(const[:, 1:])], axis=1).reshape(-1)
            A = jnp.where(live[:, None], A / nrm[:, None], 0.0)
            b = jnp.where(live, rhs / nrm, 1.0)
            b = _row_margin(A, b, jnp.where(live, mag / nrm, 0.0), live, safe)
            keep = _cut_rows(const.shape, nl)
            return (A, b) if keep is None else (A[keep], b[keep])

        C = jnp.concatenate([jnp.eye(n), -jnp.eye(n)])

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

        axis = jnp.concatenate([jnp.eye(n), -jnp.eye(n)]) * (w / 4)
        const, coef = jax.vmap(cuts_at)(jnp.concatenate([c[None], c + axis]))
        ulo, uhi = jnp.zeros(n), jnp.ones(n)
        own_c = own_g = None  # per LP (own_cuts): cuts at its own earlier solutions
        for k in range(rounds):
            if not own_cuts or k == 0:
                A, bvec = rows(const, coef)
                Z, B = jax.vmap(solve, (0, None, None))(C, A, bvec)
            else:
                Z, B = jax.vmap(lambda cv, oc, og: solve(cv, *rows(jnp.concatenate([const, oc]),
                                                                    jnp.concatenate([coef, og]))))(C, own_c, own_g)
            ulo = jnp.maximum(ulo, jnp.where(jnp.isfinite(B[:n]), B[:n], 0.0))
            uhi = jnp.minimum(uhi, jnp.where(jnp.isfinite(B[n:]), -B[n:], 1.0))
            if k < rounds - 1:  # cuts at this round's 2n LP solutions
                ck, gk = jax.vmap(cuts_at)(lo + w * Z)
                if own_cuts:
                    own_c = ck[:, None] if k == 0 else jnp.concatenate([own_c, ck[:, None]], axis=1)
                    own_g = gk[:, None] if k == 0 else jnp.concatenate([own_g, gk[:, None]], axis=1)
                else:
                    const, coef = jnp.concatenate([const, ck]), jnp.concatenate([coef, gk])

        infeasible = jnp.any(ulo > uhi + 1e-9)
        ulo, uhi = ulo - 1e-9, uhi + 1e-9  # floating-point slack
        new_lo = jnp.where(w > 0, jnp.maximum(lo, lo + w * ulo), lo)
        new_hi = jnp.where(w > 0, jnp.minimum(hi, lo + w * uhi), hi)
        return jnp.minimum(new_lo, new_hi), jnp.maximum(new_lo, new_hi), infeasible

    return obbt
