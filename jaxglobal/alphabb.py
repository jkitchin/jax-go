"""Whole-function alphaBB with scaled-Gershgorin alpha from an interval Hessian.

For a smooth ``g`` (each output separately) on the box ``[l, u]``,

    g(x) + sum_i a_i (x_i - l_i)(x_i - u_i)      is convex and <= g,
    g(x) - sum_i b_i (x_i - l_i)(x_i - u_i)      is concave and >= g,

when ``H + 2 diag(a)`` (``-H + 2 diag(b)``) is positive semidefinite for every
Hessian ``H`` of ``g`` on the box. With an interval Hessian ``[H]`` and box
widths ``d``, the scaled Gershgorin rule (Adjiman et al. 1998 [adjiman1998global]) gives

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
from jax.extend import core as jcore

from .fbbt import Const, _has_tracer, make_tape, tape_cache
from .relax import MC, SAFE_K, Ctx, eval_jaxpr_mc, is_mc

INTERVAL_CTX = Ctx(interval_only=True)
SAFE_INTERVAL_CTX = Ctx(interval_only=True, safe=True)


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


# Which interval Hessian (``interpret="auto"``). The traced one is ``jacfwd(jacfwd(fn))``
# evaluated in interval mode, straight-line code whose operations on (n, n) arrays make
# XLA's compile time explode on large dense Hessians (hadamard_5: minutes, see
# ``hessian_size``); the interpreted one (``relax_interp.make_taylor2``) compiles in
# well under a second whatever the tape, but runs 1.5-6x slower per box and carries
# every slot's interval Hessian, ``taylor2_bytes``. The interpreter is used when the
# traced Hessian is large and the interpreter's memory fits ``TAYLOR_BYTES``.
HESSIAN_OPS = 15_000  # hessian_size: above either, the traced Hessian is not compiled
HESSIAN_VOLUME = 200_000
TAYLOR_BATCH = 16  # boxes per kernel call the memory budget assumes (minimize's batch_sizes)
TAYLOR_BYTES = 512 * 2**20  # taylor2_bytes(slots, n, TAYLOR_BATCH) allowed
TAYLOR_BUDGET = TAYLOR_BYTES  # former name (it was slots * n**2)


def taylor2_bytes(slots, d, batch=TAYLOR_BATCH):
    """Estimated working memory of ``make_taylor2(tape, n, ictx, d)`` over ``batch`` boxes.

    The interpreter's carry holds ``(lo, hi)`` of every slot's value, gradient and
    Hessian, ``2 + 2d + 2d^2`` float64 per slot; XLA keeps a second copy of it
    (``memory_analysis().temp_size_in_bytes`` is 1.97-2.0 times the carry on the
    batch, ramsey, portfol_buyin, cvxnonsep_nsig30 and hadamard_5 tapes of MINLPLib;
    ``tests/test_alphabb_budget.py`` checks it on a synthetic one), hence the 2.
    """
    return 2 * slots * (2 + 2 * d + 2 * d * d) * 8 * batch


def _live_eqns(jaxpr):
    """The equations of ``jaxpr`` that its outputs depend on (what ``jit`` keeps after dead-code
    elimination), with those of the jaxprs nested in their parameters."""
    live = {v for v in jaxpr.outvars if isinstance(v, jcore.Var)}
    keep = []
    for e in reversed(jaxpr.eqns):
        if e.effects or any(v in live for v in e.outvars):
            keep.append(e)
            live.update(v for v in e.invars if isinstance(v, jcore.Var))
    for e in reversed(keep):
        yield e
        for p in e.params.values():
            for q in p if isinstance(p, (tuple, list)) else (p,):
                inner = getattr(q, "jaxpr", q)
                if isinstance(inner, jcore.Jaxpr):
                    yield from _live_eqns(inner)


def _has_tracer_consts(closed):
    return any(isinstance(c, jax.core.Tracer) for c in closed.consts)


# elementwise primitives: XLA fuses them, so on (n, n) arrays they cost compile time
# like any other operation (``ops``), not like a reduction or a re-layout (``volume``)
ELEMENTWISE = frozenset(
    "abs acos add add_any and asin atan atan2 broadcast_in_dim convert_element_type copy copy_p "
    "cos cosh div eq exp expm1 ge gt integer_pow is_finite jit le log log1p logistic lt max min "
    "mul ne neg nextafter not or pow rsqrt select_n sign sin sinh sqrt square stop_gradient sub "
    "tan tanh".split()
)


def hessian_size(closed, n):
    """``(ops, volume)`` of a traced interval Hessian ``closed``: its live operations, and
    its non-elementwise operations (reductions, stacks, transposes) on arrays of at least
    ``n^2`` elements, times ``n^2``.

    XLA's compile time grows faster than linearly in both (CPU, 16 boxes, measured on
    the start50/int50 tapes): ``kriging_peaks-red030`` (n = 2) has 50k small
    operations and compiles in 160 s; ``ex6_2_6`` with ``safe`` (outward rounding
    multiplies the operations) 25k, 77 s; ``hadamard_5`` (n = 26) 16k, 500 s;
    ``cvxnonsep_nsig30`` 11k with a volume of 326k, 4-14 s, ``launch``'s objective
    387k, 11 s. Every tape below both thresholds compiled in under 2.5 s.
    """
    ops = big = 0
    for e in _live_eqns(closed.jaxpr):
        ops += 1
        if e.primitive.name not in ELEMENTWISE:
            big += any(getattr(v.aval, "size", 1) >= n * n for v in e.outvars)
    return ops, big * n * n


def _too_big(ops, volume):
    return ops > HESSIAN_OPS or volume > HESSIAN_VOLUME


def make_alphabb(fn, eigen=True, safe=False, interpret="auto"):
    """Return ``prepare(lo, hi) -> (a, b)`` and ``apply(x, lo, hi, data) -> (cv, cc)``.

    With ``safe`` the interval Hessian is outward-rounded, alpha is enlarged by
    the rounding of the Gershgorin sums, and ``apply`` lowers (raises) cv (cc)
    by the rounding of ``f(x)`` (an interval evaluation at the point, so the
    value is rigorous while the subgradient stays that of the plain ``f``) and
    of the shift terms.

    ``fn`` maps a 1-D array to an array; outputs are flattened to ``(m,)``,
    ``a`` and ``b`` have shape ``(m, n)``. ``prepare`` returns ``None`` when
    ``fn`` is not smooth (then ``apply`` is never needed).

    ``interpret`` (``"auto"``, True, False): get the interval Hessian from the
    second-order tape interpreter ``relax_interp.make_taylor2`` instead of
    tracing ``jacfwd(jacfwd(fn))``. ``"auto"`` interprets when the traced
    Hessian's ``hessian_size`` exceeds ``HESSIAN_OPS`` or ``HESSIAN_VOLUME`` and the
    interpreter's ``taylor2_bytes`` fit ``TAYLOR_BYTES``. Both are interval
    enclosures of the Hessian on the box, so either gives a valid alpha.
    ``prepare.path(lo)`` says which one is used (``"interpret"``,
    ``"traced"``, or None when ``fn`` is not smooth).
    """

    ictx = SAFE_INTERVAL_CTX if safe else INTERVAL_CTX
    eps = float(jnp.finfo(jnp.float64).eps)

    def flat(x):
        return jnp.reshape(fn(x), (-1,))

    @tape_cache
    def smoothness(x):
        tape = make_tape(fn, x)
        with jax.ensure_compile_time_eval():
            return tape, is_smooth(tape)

    def traced(lo):
        """``jaxpr(lo, hi) -> (H.lo, H.hi)``: the traced interval Hessian, or None (no interval rule)."""

        def hess(a, b):
            closed = jax.make_jaxpr(jax.jacfwd(jax.jacfwd(flat)))(a)
            (H,) = eval_jaxpr_mc(closed.jaxpr, closed.consts, [MC(a, b, a, b)], ictx)
            H = H if is_mc(H) else MC(H, H, H, H)
            return H.lo, H.hi

        try:
            return jax.make_jaxpr(hess)(lo, lo)
        except NotImplementedError:
            return None

    hessians = {}

    def hessian(tape, lo):
        """``(path, how)`` for ``tape`` (cached): ``("interpret", taylor2)``, ``("traced", jaxpr)``, or ``(None, None)``."""
        if id(tape) in hessians:
            return hessians[id(tape)][1]
        local = jnp.ndim(lo) != 1 or _has_tracer(tape)  # belongs to one trace: traced, not cached
        n = jnp.size(lo)
        closed = None if interpret is True and not local else traced(lo)
        run = None
        if not local and interpret is not False:
            big = closed is None or _too_big(*hessian_size(closed, n))
            if interpret is True or big:
                from .relax_interp import make_taylor2

                with jax.ensure_compile_time_eval():
                    run = make_taylor2(tape, n, ictx, n)
                if run is not None and interpret is not True and taylor2_bytes(run.program.const_val.size, n) > TAYLOR_BYTES:
                    run = None
            if run is None and closed is None and interpret is True:
                closed = traced(lo)
        how = ("interpret", run) if run is not None else ("traced", closed) if closed is not None else (None, None)
        if not local and not (closed is not None and _has_tracer_consts(closed)):
            hessians[id(tape)] = (tape, how)
        return how

    def prepare(lo, hi):
        tape, smooth = smoothness(lo)
        if not smooth:
            return None
        n = lo.size
        m = jnp.size(flat(lo))
        path, how = hessian(tape, lo)
        if path == "interpret":
            ids = jnp.asarray(how.program.outs[0])
            r = how(lo, hi, jnp.eye(n, dtype=lo.dtype))
            H = MC(r[4][ids], r[5][ids], r[4][ids], r[5][ids])
        elif path == "traced":
            Hlo, Hhi = jax.core.eval_jaxpr(how.jaxpr, how.consts, lo, hi)
            H = MC(Hlo, Hhi, Hlo, Hhi)
        else:
            return None
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
        if safe:  # rounding of the Gershgorin sums: a slightly too small alpha would break convexity
            pad = SAFE_K * eps * (n + 2)
            a = a * (1 + pad) + pad * (jnp.abs(diag_lo) + rad)
            b = b * (1 + pad) + pad * (jnp.abs(diag_hi) + rad)
        # a zero-width coordinate contributes nothing whatever its alpha
        a = jnp.where(dpos[None, :], a, 0.0)
        b = jnp.where(dpos[None, :], b, 0.0)

        # exact quadratic: a point interval Hessian Q; shift only the negative
        # (positive) curvature eigen-directions, by exactly |lambda|
        if safe:  # outward-rounded Hessian of a quadratic is a few ulps wide; the eigen margin below covers it
            point = (hhi - hlo) <= 1e-12 * (jnp.abs(hlo) + jnp.abs(hhi))
        else:
            point = hlo == hhi
        quad = jnp.all(point & jnp.isfinite(hlo) & jnp.isfinite(hhi), axis=(1, 2)) & eigen
        Q = jnp.where(quad[:, None, None], 0.5 * (hlo + hhi), 0.0)
        lam, V = jnp.linalg.eigh(Q)  # V[:, :, k] is the k-th eigenvector
        tol = 1e-9 * jnp.max(jnp.abs(lam), axis=1, keepdims=True)
        wcv = jnp.maximum(-lam, 0.0) + tol  # margin keeps Q + V diag(w) V^T PSD under rounding
        wcc = jnp.maximum(lam, 0.0) + tol
        Vt = jnp.swapaxes(V, 1, 2)  # (m, k, n)
        tlo = jnp.sum(jnp.minimum(Vt * lo, Vt * hi), axis=2)
        thi = jnp.sum(jnp.maximum(Vt * lo, Vt * hi), axis=2)
        if safe:  # the range of t = V^T x must contain the true one
            tm = SAFE_K * eps * (n + 1) * jnp.sum(jnp.abs(Vt) * jnp.maximum(jnp.abs(lo), jnp.abs(hi)), axis=2)
            tlo, thi = tlo - tm, thi + tm
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
        if safe:
            # rigorous value of f(x); gradients stay those of the plain evaluation
            closed = jax.make_jaxpr(flat)(x)
            (iv,) = eval_jaxpr_mc(closed.jaxpr, closed.consts, [MC(x, x, x, x)], SAFE_INTERVAL_CTX)
            iv = iv if is_mc(iv) else MC(iv, iv, iv, iv)
            flo, fhi = jnp.broadcast_to(iv.lo, fx.shape), jnp.broadcast_to(iv.hi, fx.shape)
            n = x.size
            pad = SAFE_K * eps * (n + 2)
            sa = jnp.sum(jnp.abs(jnp.where(jnp.isfinite(a), a, 0.0) * q), axis=1)
            sb = jnp.sum(jnp.abs(jnp.where(jnp.isfinite(b), b, 0.0) * q), axis=1)
            tsc = jnp.maximum(jnp.abs(tlo), jnp.abs(thi))
            se = jnp.sum(0.5 * wcv * 4 * tsc**2, axis=1)
            sc = jnp.sum(0.5 * wcc * 4 * tsc**2, axis=1)
            sg = jax.lax.stop_gradient
            cv = cv + sg(flo - fx) - sg(pad * (jnp.abs(fx) + sa))
            cc = cc + sg(fhi - fx) + sg(pad * (jnp.abs(fx) + sb))
            ecv = ecv + sg(flo - fx) - sg(pad * (jnp.abs(fx) + se))
            ecc = ecc + sg(fhi - fx) + sg(pad * (jnp.abs(fx) + sc))
        cv = jnp.maximum(jnp.where(jnp.isnan(cv), -jnp.inf, cv), jnp.where(jnp.isnan(ecv), -jnp.inf, ecv))
        cc = jnp.minimum(jnp.where(jnp.isnan(cc), jnp.inf, cc), jnp.where(jnp.isnan(ecc), jnp.inf, ecc))
        return cv, cc

    def path(lo):
        """Which interval Hessian ``prepare`` uses on boxes like ``lo``: ``"interpret"``, ``"traced"`` or None."""
        tape, smooth = smoothness(lo)
        return hessian(tape, lo)[0] if smooth else None

    prepare.path = path
    return prepare, apply
