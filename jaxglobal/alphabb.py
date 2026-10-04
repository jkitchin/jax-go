"""Whole-function alphaBB with scaled-Gershgorin alpha from an interval Hessian.

For a smooth ``g`` (each output separately) on the box ``[l, u]``,

    g(x) + sum_i a_i (x_i - l_i)(x_i - u_i)      is convex and <= g,
    g(x) - sum_i b_i (x_i - l_i)(x_i - u_i)      is concave and >= g,

when ``H + 2 diag(a)`` (``-H + 2 diag(b)``) is positive semidefinite for every
Hessian ``H`` of ``g`` on the box. With an interval Hessian ``[H]`` and box
widths ``d``, the scaled Gershgorin rule (Adjiman et al. 1998) gives

    a_i = max(0, -(H_ii^L - sum_{j != i} max|H_ij| d_j / d_i) / 2),
    b_i = max(0,  (H_ii^U + sum_{j != i} max|H_ij| d_j / d_i) / 2).

The interval Hessian is the jaxpr of ``jacfwd(jacfwd(g))`` evaluated by the
relaxation interpreter in interval mode, once per box. The relaxer takes the
pointwise max (min) with the McCormick relaxation: the McCormick rules are
tighter on wide boxes, while alphaBB's gap shrinks with the square of the box
width, so it takes over deep in the tree. Only functions built entirely from
smooth primitives qualify: at a kink the Hessian says nothing.

Quadratic forms are detected for free: when the interval Hessian is a point,
``g`` is exactly quadratic on the box with Hessian ``Q = V diag(lambda) V^T``,
and the eigenvalue relaxation

    g(x) + sum_{k} max(-lambda_k, 0) / 2 (t_k - t_k^L)(t_k - t_k^U),   t = V^T x,

shifts only the directions of negative curvature, by exactly what they need
(each concave ``lambda_k t_k^2`` is replaced by its secant over the range of
``t_k`` on the box). It is combined with the diagonal alphaBB by max.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .fbbt import Const, make_tape, tape_cache
from .relax import MC, Ctx, eval_jaxpr_mc, is_mc

INTERVAL_CTX = Ctx(interval_only=True)


def is_smooth(tape):
    """True when every equation that depends on the variables is a smooth primitive."""
    from .univariate import AFFINE, SMOOTH_ELEMENTWISE

    var = {0}
    for node in tape.nodes:
        if not any(not isinstance(a, Const) and a in var for a in node.ins):
            continue
        name = node.eqn.primitive.name
        if not (name in AFFINE or name in SMOOTH_ELEMENTWISE or name == "dot_general"):
            return False
        var.update(node.outs)
    return True


def make_alphabb(fn, eigen=True):
    """Return ``prepare(lo, hi) -> (a, b)`` and ``apply(x, lo, hi, data) -> (cv, cc)``.

    ``fn`` maps a 1-D array to an array; outputs are flattened to ``(m,)``,
    ``a`` and ``b`` have shape ``(m, n)``. ``prepare`` returns ``None`` when
    ``fn`` is not smooth (then ``apply`` is never needed).
    """

    def flat(x):
        return jnp.reshape(fn(x), (-1,))

    @tape_cache
    def smoothness(x):
        tape = make_tape(fn, x)
        with jax.ensure_compile_time_eval():
            return tape, is_smooth(tape)

    def prepare(lo, hi):
        _, smooth = smoothness(lo)
        if not smooth:
            return None
        try:
            closed = jax.make_jaxpr(jax.jacfwd(jax.jacfwd(flat)))(lo)
            (H,) = eval_jaxpr_mc(closed.jaxpr, closed.consts, [MC(lo, hi, lo, hi)], INTERVAL_CTX)
        except NotImplementedError:
            return None
        H = H if is_mc(H) else MC(H, H, H, H)
        n = lo.size
        m = jnp.size(flat(lo))
        hlo = jnp.broadcast_to(H.lo, (m, n, n))
        hhi = jnp.broadcast_to(H.hi, (m, n, n))
        hlo = jnp.where(jnp.isnan(hlo), -jnp.inf, hlo)
        hhi = jnp.where(jnp.isnan(hhi), jnp.inf, hhi)
        d = hi - lo
        dpos = d > 0
        ratio = jnp.where(dpos[None, :] & dpos[:, None], d[None, :] / jnp.where(dpos, d, 1.0)[:, None], 0.0)
        off = ~jnp.eye(n, dtype=bool)
        mag = jnp.maximum(jnp.abs(hlo), jnp.abs(hhi))
        mag = jnp.where(off[None] & (ratio[None] > 0), mag * ratio[None], 0.0)  # (m, i, j)
        rad = jnp.sum(mag, axis=2)
        diag_lo = jnp.diagonal(hlo, axis1=1, axis2=2)
        diag_hi = jnp.diagonal(hhi, axis1=1, axis2=2)
        a = jnp.maximum(0.0, -0.5 * (diag_lo - rad))
        b = jnp.maximum(0.0, 0.5 * (diag_hi + rad))
        # a zero-width coordinate contributes nothing whatever its alpha
        a = jnp.where(dpos[None, :], a, 0.0)
        b = jnp.where(dpos[None, :], b, 0.0)

        # exact quadratic: a point interval Hessian Q; shift only the negative
        # (positive) curvature eigen-directions, by exactly |lambda|
        quad = jnp.all((hlo == hhi) & jnp.isfinite(hlo), axis=(1, 2)) & eigen
        Q = jnp.where(quad[:, None, None], 0.5 * (hlo + hhi), 0.0)
        lam, V = jnp.linalg.eigh(Q)  # V[:, :, k] is the k-th eigenvector
        tol = 1e-9 * jnp.max(jnp.abs(lam), axis=1, keepdims=True)
        wcv = jnp.maximum(-lam, 0.0) + tol  # margin keeps Q + V diag(w) V^T PSD under rounding
        wcc = jnp.maximum(lam, 0.0) + tol
        Vt = jnp.swapaxes(V, 1, 2)  # (m, k, n)
        tlo = jnp.sum(jnp.minimum(Vt * lo, Vt * hi), axis=2)
        thi = jnp.sum(jnp.maximum(Vt * lo, Vt * hi), axis=2)
        return a, b, (quad, Vt, wcv, wcc, tlo, thi)

    def apply(x, lo, hi, data):
        a, b, (quad, Vt, wcv, wcc, tlo, thi) = data
        q = (x - lo) * (x - hi)  # <= 0 on the box
        fx = flat(x)
        t = Vt @ x  # (m, k)
        qt = jnp.minimum((t - tlo) * (t - thi), 0.0)
        ecv = jnp.where(quad, fx + 0.5 * jnp.sum(wcv * qt, axis=1), -jnp.inf)
        ecc = jnp.where(quad, fx - 0.5 * jnp.sum(wcc * qt, axis=1), jnp.inf)
        # infinite alpha (unbounded Hessian) means no information; keep gradients finite
        fa, fb = jnp.all(jnp.isfinite(a), axis=1), jnp.all(jnp.isfinite(b), axis=1)
        cv = fx + jnp.sum(jnp.where(jnp.isfinite(a), a, 0.0) * q, axis=1) + jnp.where(fa, 0.0, -jnp.inf)
        cc = fx - jnp.sum(jnp.where(jnp.isfinite(b), b, 0.0) * q, axis=1) + jnp.where(fb, 0.0, jnp.inf)
        cv = jnp.maximum(jnp.where(jnp.isnan(cv), -jnp.inf, cv), jnp.where(jnp.isnan(ecv), -jnp.inf, ecv))
        cc = jnp.minimum(jnp.where(jnp.isnan(cc), jnp.inf, cc), jnp.where(jnp.isnan(ecc), jnp.inf, ecc))
        return cv, cc

    return prepare, apply
