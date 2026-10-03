"""Constrained node bounds: outer-approximation LP + Lagrangian certificate.

For a box B and the problem

    min f(x)  s.t.  g(x) <= 0,  h(x) = 0,  x in B,

the McCormick relaxations give a convex problem

    min cv_f(x)  s.t.  cv_g(x) <= 0,  cv_h(x) <= 0,  -cc_h(x) <= 0,  x in B.

Subgradients from ``jax.jacfwd`` turn each convex function into affine cuts at
a set of points (the box center, 2n axis points, then Kelley points: the LP
solution of the previous round). The resulting small dense LP is solved by a
fixed-iteration primal-dual interior-point method written in JAX, so it
``vmap``s over hundreds of boxes.

The bound returned does **not** trust the LP solver. For any multipliers
y >= 0 on the cut rows, weak duality gives

    min_B t  >=  min_{z in box} (c + A^T y) . z - y . b,

which is analytic over a box. We evaluate this at the (clipped) IPM duals, so a
poorly converged or infeasible LP still yields a valid bound. Infeasible LPs
drive y to a Farkas ray, so the bound blows up and the node is pruned.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .relax import relax


def ipm(c, A, b, iters=40, sigma=0.2, cert=None, mu_tol=1e-10):
    """Primal-dual interior point for ``min c.z s.t. A z <= b`` (dense, tiny).

    Infeasible start: z = 0.5 (the LP lives in the unit box), slacks and duals 1.
    Iteration stops (the state is frozen) once the duality measure falls below
    ``mu_tol``: past that point the normal equations are too ill-conditioned
    and the duals drift. If ``cert(y)`` is given, the best value it takes over
    all iterates is returned as well, so a late bad step can never lose a good
    certificate.

    Returns ``(z, y, best_cert)``.
    """
    m, n = A.shape
    eye = jnp.eye(n)
    cert = cert or (lambda y: -jnp.inf)

    def step_len(v, dv):
        r = jnp.where(dv < 0, -v / jnp.where(dv < 0, dv, -1.0), jnp.inf)
        return jnp.minimum(1.0, 0.99 * jnp.min(r))

    def body(_, st):
        z, s, y, best = st
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
        return z, s, y, jnp.maximum(best, jnp.where(jnp.isnan(cy), -jnp.inf, cy))

    z0 = jnp.full(n, 0.5)
    s0 = jnp.maximum(b - A @ z0, 1.0)
    y0 = jnp.ones(m)
    z, s, y, best = jax.lax.fori_loop(0, iters, body, (z0, s0, y0, cert(y0)))
    return z, y, best


def _as_vec(fn):
    return None if fn is None else (lambda x: jnp.atleast_1d(fn(x)))


def make_constrained_bounder(f, ineq=None, eq=None, n_kelley=4, ipm_iters=40):
    """Return ``bound(lo, hi) -> (lb, x_lp, infeasible)`` for one box."""
    rf = relax(f)
    rg = relax(_as_vec(ineq)) if ineq is not None else None
    rh = relax(_as_vec(eq)) if eq is not None else None

    def convex_parts(x, lo, hi):
        """[cv_f, cv_g..., cv_h..., -cc_h...] -- every entry convex in x."""
        parts = [jnp.atleast_1d(rf(x, lo, hi).cv)]
        if rg is not None:
            parts.append(rg(x, lo, hi).cv)
        if rh is not None:
            mh = rh(x, lo, hi)
            parts += [mh.cv, -mh.cc]
        return jnp.concatenate(parts)

    def bound(lo, hi):
        n = lo.size
        w = hi - lo
        c = 0.5 * (lo + hi)

        # --- interval information (constant on the box) ---------------------
        mf = rf(c, lo, hi)
        flo, fhi = mf.lo, mf.hi
        infeasible = jnp.array(False)
        if rg is not None:
            infeasible |= jnp.any(rg(c, lo, hi).lo > 0)
        if rh is not None:
            mh = rh(c, lo, hi)
            infeasible |= jnp.any(mh.lo > 0) | jnp.any(mh.hi < 0)

        def cuts_at(p):
            """Affine minorants of each convex part at p, in unit-box coords u."""
            v = convex_parts(p, lo, hi)
            J = jax.jacfwd(convex_parts)(p, lo, hi)
            const = v + J @ (lo - p)  # value of the cut at u = 0
            coef = J * w  # d cut / d u
            ok = jnp.isfinite(const) & jnp.all(jnp.isfinite(coef), axis=1)
            return jnp.where(ok, const, -jnp.inf), jnp.where(ok[:, None], coef, 0.0)

        axis = jnp.concatenate([jnp.eye(n), -jnp.eye(n)]) * (w / 4)
        pts0 = jnp.concatenate([c[None], c + axis])
        const0, coef0 = jax.vmap(cuts_at)(pts0)  # (P0, 1+m), (P0, 1+m, n)

        # objective-variable range [tlo, thi] from intervals and the cuts
        cmin = const0[:, 0] + jnp.sum(jnp.minimum(coef0[:, 0], 0), axis=1)
        cmax = const0[:, 0] + jnp.sum(jnp.maximum(coef0[:, 0], 0), axis=1)
        tlo = jnp.maximum(flo, jnp.max(jnp.where(jnp.isfinite(cmin), cmin, -jnp.inf)))
        thi = jnp.minimum(fhi, jnp.max(jnp.where(jnp.isfinite(cmax), cmax, -jnp.inf)))
        thi = jnp.where(jnp.isfinite(thi) & (thi > tlo), thi, tlo + 1.0)
        T = thi - tlo
        bad_t = ~jnp.isfinite(tlo)
        tlo_s = jnp.where(bad_t, 0.0, tlo)

        def rows(const, coef):
            """LP rows over z = (u, tau) with t = tlo + T tau.

            objective cut:  const0 + coef0.u <= t   -> coef0.u - T tau <= tlo - const0
            constraint cut: const_i + coef_i.u <= 0 ->  coef_i.u         <= -const_i
            """
            tcol = jnp.zeros(const.shape + (1,)).at[..., 0, 0].set(-T)
            A = jnp.concatenate([coef, tcol], axis=-1).reshape(-1, n + 1)
            bvec = jnp.concatenate([tlo_s - const[..., :1], -const[..., 1:]], axis=-1).reshape(-1)
            live = jnp.isfinite(bvec)
            nrm = jnp.linalg.norm(A, axis=1)
            live &= nrm > 0
            nrm = jnp.where(live, nrm, 1.0)
            A = jnp.where(live[:, None], A / nrm[:, None], 0.0)
            bvec = jnp.where(live, bvec / nrm, 1.0)  # dead rows: 0 <= 1
            return A, bvec

        box_A = jnp.concatenate([jnp.eye(n + 1), -jnp.eye(n + 1)])
        box_b = jnp.concatenate([jnp.ones(n + 1), jnp.zeros(n + 1)])
        cobj = jnp.zeros(n + 1).at[n].set(1.0)

        def lagrangian_bound(A, bvec, y):
            # Certificate over u in [0,1]^n, tau in [0, inf): the cap tau <= 1
            # used inside the LP is NOT relied on (later Kelley cuts may exceed
            # thi). Objective-row duals are rescaled so the tau coefficient
            # 1 - sum(y_obj * T / |row|) stays >= 0; any y >= 0 is then valid.
            y = jnp.where(jnp.isfinite(y), jnp.maximum(y, 0.0), 0.0)
            s_obj = -(A[:, n] @ y)
            y = y * jnp.where(s_obj > 1.0, 1.0 / s_obj, 1.0) ** (A[:, n] < 0)
            coef = cobj + A.T @ y
            val = jnp.sum(jnp.minimum(coef[:n], 0.0)) - y @ bvec
            return tlo_s + T * val

        const, coef = const0, coef0
        best = jnp.where(bad_t, -jnp.inf, tlo)
        u = jnp.full(n, 0.5)
        for k in range(n_kelley + 1):
            A, bvec = rows(const, coef)
            nA = A.shape[0]
            z, _, lb = ipm(
                cobj, jnp.concatenate([A, box_A]), jnp.concatenate([bvec, box_b]), ipm_iters,
                cert=lambda y, A=A, bvec=bvec, nA=nA: lagrangian_bound(A, bvec, y[:nA]),
            )  # fmt: skip
            best = jnp.maximum(best, jnp.where(jnp.isnan(lb) | bad_t, -jnp.inf, lb))
            u = jnp.clip(z[:n], 0.0, 1.0)
            if k < n_kelley:  # Kelley: add cuts at the LP solution
                ck, gk = cuts_at(lo + w * u)
                const = jnp.concatenate([const, ck[None]])
                coef = jnp.concatenate([coef, gk[None]])

        lb = jnp.where(infeasible, jnp.inf, best)
        return lb, lo + w * u, infeasible

    return bound
