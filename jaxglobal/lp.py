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


def ipm(c, A, b, iters=40, sigma=0.2, cert=None, mu_tol=1e-10):
    """Primal-dual interior point for ``min c.z s.t. A z <= b`` (dense, tiny).

    Infeasible start: z = 0.5 (the LP lives in the unit box), slacks and duals 1.
    Iteration stops (the state is frozen) once the duality measure falls below
    ``mu_tol``: past that point the normal equations are too ill-conditioned
    and the duals drift. If ``cert(y)`` is given, the best value it takes over
    all iterates is returned as well, so a late bad step can never lose a good
    certificate.

    Returns ``(z, y_best, best_cert)`` where ``y_best`` attains ``best_cert``.
    """
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


def _parts_fn(f, ineq, eq, univariate, alphabb=False, multivariate=False, safe=False):
    """``prepare(lo, hi)`` and ``parts(x, lo, hi, data)`` = [cv_f, cv_g..., cv_h..., -cc_h...]."""
    rs = [make_relaxer(_as_vec(fn), univariate, alphabb, multivariate=multivariate, safe=safe) if fn is not None else None for fn in (f, ineq, eq)]

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


def make_constrained_bounder(f, ineq=None, eq=None, n_kelley=4, ipm_iters=40, tol=0.0, univariate=False,
                             alphabb=False, multivariate=False, safe=False):
    """Return ``bound(lo, hi, cutoff) -> (lb, x_lp, infeasible, lo', hi')`` for one box.

    ``[lo', hi']`` is the box after duality-based range reduction against
    ``f <= cutoff`` (the incumbent value); it is the input box when
    ``cutoff`` is infinite.
    """
    prepare, parts, _ = _parts_fn(f, ineq, eq, univariate, alphabb, multivariate, safe)
    nonlinear = _nonlinear_of(f, ineq, eq)

    def bound(lo, hi, cutoff=jnp.inf):
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

        def rows(const, coef):
            """LP rows over z = (u, tau) with t = tlo + T tau.

            objective cut:  const0 + coef0.u <= t   -> coef0.u - T tau <= tlo - const0
            constraint cut: const_i + coef_i.u <= tol -> coef_i.u        <= tol - const_i
            """
            tcol = jnp.zeros(const.shape + (1,)).at[..., 0, 0].set(-T)
            A = jnp.concatenate([coef, tcol], axis=-1).reshape(-1, n + 1)
            bvec = jnp.concatenate([tlo_s - const[..., :1], tol - const[..., 1:]], axis=-1).reshape(-1)
            live = jnp.isfinite(bvec)
            nrm = jnp.linalg.norm(A, axis=1)
            live &= nrm > 0
            nrm = jnp.where(live, nrm, 1.0)
            mag = jnp.concatenate([jnp.abs(tlo_s) + jnp.abs(const[..., :1]), abs(tol) + jnp.abs(const[..., 1:])], axis=-1).reshape(-1)
            A = jnp.where(live[:, None], A / nrm[:, None], 0.0)
            bvec = jnp.where(live, bvec / nrm, 1.0)  # dead rows: 0 <= 1
            bvec = _row_margin(A, bvec, jnp.where(live, mag / nrm, 0.0), live, safe)
            keep = _cut_rows(const.shape, nl)
            return (A, bvec) if keep is None else (A[keep], bvec[keep])

        box_A = jnp.concatenate([jnp.eye(n + 1), -jnp.eye(n + 1)])
        box_b = jnp.concatenate([jnp.ones(n + 1), jnp.zeros(n + 1)])
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

        # Kelley rounds as one lax.scan over a fixed-size cut buffer: slots for
        # cuts not generated yet hold const = -inf, which rows() turns into
        # dead rows (0 <= 1), so every round has the same LP shape.
        P0 = const0.shape[0]
        const = jnp.concatenate([const0, jnp.full((n_kelley,) + const0.shape[1:], -jnp.inf)])
        coef = jnp.concatenate([coef0, jnp.zeros((n_kelley,) + coef0.shape[1:])])

        def kelley(carry, k):
            const, coef, best, rlo, rhi, _ = carry
            A, bvec = rows(const, coef)
            nA = A.shape[0]
            z, y, lb = ipm(
                cobj, jnp.concatenate([A, box_A]), jnp.concatenate([bvec, box_b]), ipm_iters,
                cert=lambda y: dual(A, bvec, y[:nA])[0],
            )  # fmt: skip
            lb = jnp.where(jnp.isnan(lb) | bad_t, -jnp.inf, lb)
            best = jnp.maximum(best, lb)
            # range reduction from this round's certificate (r is per unit of u)
            _, r = dual(A, bvec, y[:nA])
            klo, khi = reduce_ranges(lo, hi, lb, T * r / jnp.where(w > 0, w, 1.0), cutoff)
            rlo, rhi = jnp.maximum(rlo, klo), jnp.minimum(rhi, khi)
            u = jnp.clip(z[:n], 0.0, 1.0)
            # Kelley: add cuts at the LP solution (not after the last round)
            def add(cc):
                ck, gk, _ = cuts_at(lo + w * u)
                return cc[0].at[P0 + k].set(ck), cc[1].at[P0 + k].set(gk)

            const, coef = jax.lax.cond(k < n_kelley, add, lambda cc: cc, (const, coef))
            return (const, coef, best, rlo, rhi, u), None

        init = (const, coef, jnp.where(bad_t, -jnp.inf, tlo), lo, hi, jnp.full(n, 0.5))
        (_, _, best, rlo, rhi, u), _ = jax.lax.scan(kelley, init, jnp.arange(n_kelley + 1))

        lb = jnp.where(infeasible, jnp.inf, best)
        return lb, lo + w * u, infeasible, rlo, rhi

    return bound


def make_obbt(f, ineq=None, eq=None, rounds=2, ipm_iters=40, tol=0.0, univariate=False, alphabb=False,
              multivariate=False, safe=False, own_cuts=True):
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
    inexact LP solve can only make them looser, never invalid.
    """
    prepare, parts, _ = _parts_fn(f, ineq, eq, univariate, alphabb, multivariate, safe)
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

        box_A = jnp.concatenate([jnp.eye(n), -jnp.eye(n)])
        box_b = jnp.concatenate([jnp.ones(n), jnp.zeros(n)])
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

            z, _, best = ipm(cvec, jnp.concatenate([A, box_A]), jnp.concatenate([bvec, box_b]), ipm_iters, cert=cert)
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
