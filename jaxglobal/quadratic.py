"""Lifted LP bounds for quadratic problems (QCQPs: pooling, blending, bilinear models).

When ``f``, ``g`` and ``h`` are all quadratic, every product ``x_i x_j`` (and
square ``x_i^2``) that appears in them can get its own LP column ``w_ij``, as
in BARON, SCIP and Couenne. Every function is then linear in ``v = (x, w)``,
and each column ``w_ij`` is linked to its factors by its McCormick facets,
which are the convex and concave envelopes of the product over the box:

    w >= l_j x_i + l_i x_j - l_i l_j,      w >= u_j x_i + u_i x_j - u_i u_j,
    w <= l_j x_i + u_i x_j - u_i l_j,      w <= u_j x_i + l_i x_j - l_i u_j,

and, for a square, the secant above and tangents (at the bounds, the
midpoint and later at the LP solution) below. Sharing one column per product
across all rows is what makes this tighter than the reduced-space cuts of
``lp.py``, which relax each function separately. RLT rows (each linear equality
multiplied by a variable, Sherali & Adams) are added where their products are
columns already. On a pooling model with a
balance ``sum_k q_k y_k = p`` the reduced-space relaxation forgets that every
row containing ``q_k y_k`` refers to the same product.

The coefficients are read off the scalarized tape (``fbbt_interp.scalarize``)
in numpy once per problem size, so the LP is built from fixed index arrays:
its shape and its compile time do not depend on the tape. It is solved by an
interior-point method that keeps the rows sparse (``_ipm``; each McCormick row has
three nonzeros) and bounded by the weak-duality certificate of ``lp.py``, so an
inexact solve only loosens the bound, never invalidates it. Problems that are
not quadratic, or whose tape cannot be scalarized, return ``None`` from
``quadratic_form`` and use the reduced-space bounder.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg
import numpy as np

from .lp import EPS, reduce_ranges
from .relax import SAFE_K

MAX_COLUMNS = 400  # beyond this the IPM's N x N solve per iteration costs more than it gains


class QuadraticForm(NamedTuple):
    E: np.ndarray  # (R, N) rows [f, g..., h...] linear in v = (x, w)
    c: np.ndarray  # (R,) constants
    pairs: np.ndarray  # (P, 2) i <= j: the product of column n + p
    n_g: int
    n_h: int
    rlt: np.ndarray  # (T, N) RLT rows: a linear equality times x_v, linear in v (= 0)
    rlt_var: np.ndarray  # (T,) the v of each


def quadratic_form(f, ineq, eq, n):
    """The problem as ``QuadraticForm``, or None if some function is not quadratic, the tape
    cannot be scalarized, or the lifted LP would have more than ``MAX_COLUMNS`` columns."""
    from .fbbt import _has_tracer, joint_tape
    from .fbbt_interp import scalarize

    with jax.ensure_compile_time_eval():
        tape = joint_tape(f, ineq, eq, np.zeros(n))
    if _has_tracer(tape):
        return None
    prog = scalarize(tape, n)
    if prog is None:
        return None
    cval = prog.const_val
    polys = {}

    def poly(s):  # {(): c, (i,): a, (i, j): q} with i <= j
        if s < n:
            return {(s,): 1.0}
        if not np.isnan(cval[s]):
            return {(): float(cval[s])}
        return polys[s]

    def axpy(p, q, a=1.0):
        r = dict(p)
        for k, v in q.items():
            r[k] = r.get(k, 0.0) + a * v
        return r

    def mul(p, q):
        r = {}
        for k1, v1 in p.items():
            for k2, v2 in q.items():
                k = tuple(sorted(k1 + k2))
                if len(k) > 2:
                    return None
                r[k] = r.get(k, 0.0) + v1 * v2
        return r

    for o, a, b, out in zip(prog.op.tolist(), prog.a.tolist(), prog.b.tolist(), prog.out.tolist()):
        name, eqn = prog.opcodes[o][0], prog.opcodes[o][6]
        pa, pb = poly(a), poly(b)
        if name in ("add", "add_any"):
            r = axpy(pa, pb)
        elif name == "sub":
            r = axpy(pa, pb, -1.0)
        elif name == "neg":
            r = {k: -v for k, v in pa.items()}
        elif name == "mul":
            r = mul(pa, pb)
        elif name == "div" and set(pb) == {()}:
            r = {k: v / pb[()] for k, v in pa.items()}
        elif name == "square" or (name == "integer_pow" and eqn.params["y"] == 2):
            r = mul(pa, pa)
        elif name == "integer_pow" and eqn.params["y"] == 1:
            r = pa
        else:
            return None
        if r is None:
            return None
        polys[out] = r

    rows = [[poly(int(s)) for s in np.asarray(slots).reshape(-1).tolist()] for slots in prog.outs]
    flat = [p for group in rows for p in group]
    pairs = sorted({k for p in flat for k in p if len(k) == 2 and p[k] != 0.0})
    if not pairs or n + len(pairs) > MAX_COLUMNS:
        return None
    col = {k: n + i for i, k in enumerate(pairs)}
    E, c = np.zeros((len(flat), n + len(pairs))), np.zeros(len(flat))
    for r, p in enumerate(flat):
        for k, v in p.items():
            if len(k) == 0:
                c[r] += v
            else:
                E[r, k[0] if len(k) == 1 else col[k]] += v
    n_g = len(rows[1]) if ineq is not None else 0
    n_h = len(rows[-1]) if eq is not None else 0
    # RLT: a linear equality a.x + c = 0 times a variable x_v of some product gives
    # sum_k a_k w_kv + c x_v = 0, linear in v when every w_kv is a column already
    # (no new columns: the LP keeps its size, and on pooling models these rows carry
    # the strengthening of the full RLT system).
    rlt, rlt_var = [], []
    prod_vars = sorted({i for k in pairs for i in k})
    for p in (rows[-1] if eq is not None else []):
        if any(len(k) == 2 for k in p) or not any(len(k) == 1 and v for k, v in p.items()):
            continue
        for v in prod_vars:
            terms = [(tuple(sorted((k[0], v))), a) for k, a in p.items() if len(k) == 1 and a != 0.0]
            if all(k in col for k, _ in terms):
                row = np.zeros(n + len(pairs))
                for k, a in terms:
                    row[col[k]] += a
                row[v] += p.get((), 0.0)
                rlt.append(row)
                rlt_var.append(v)
    rlt = np.asarray(rlt).reshape(-1, n + len(pairs))
    return QuadraticForm(E, c, np.asarray(pairs, np.int64).reshape(-1, 2), n_g, n_h, rlt,
                         np.asarray(rlt_var, np.int64))  # fmt: skip




# --- the LP as fixed sparsity pattern + per-box values --------------------------------------
#
# Every row of the lifted LP has a sparsity pattern known from Q alone: a McCormick row touches
# (w_p, x_i, x_j), a function or RLT row the nonzeros of its row of E, a square tangent
# (w_p, x_i). Only the values (facets) and right-hand sides depend on the box, so rows are kept
# as (row, col) index arrays and a vector of values, and the IPM forms its normal matrix
# sum_r d_r a_r a_r^T from the (static) pairs of nonzeros within each row instead of a dense
# product: its cost is the number of such pairs plus one N x N solve, not rows x N^2.


class _Pattern(NamedTuple):
    rows: np.ndarray  # (nnz,) row of each nonzero
    cols: np.ndarray  # (nnz,) column of each nonzero
    m: int  # number of rows
    pa: np.ndarray  # pairs (pa, pb) of nonzeros in the same row, all ordered pairs
    pb: np.ndarray


def _pattern(rows, cols, m):
    rows, cols = np.concatenate(rows).astype(np.int64), np.concatenate(cols).astype(np.int64)
    order = np.argsort(rows, kind="stable")
    starts = np.searchsorted(rows[order], np.arange(m + 1))
    pa, pb = [], []
    for r in range(m):
        idx = order[starts[r]:starts[r + 1]]
        pa.append(np.repeat(idx, idx.size))
        pb.append(np.tile(idx, idx.size))
    return _Pattern(rows, cols, m, np.concatenate(pa), np.concatenate(pb))


class _Layout(NamedTuple):
    """Row blocks of the LP: McCormick (4P), g, h+, h-, RLT+, RLT-, square tangents
    (n_tan P), and an optional objective-cutoff row (OBBT)."""

    pattern: _Pattern
    static_vals: np.ndarray  # values of the E-based rows (in pattern order after the McCormick block)
    n_tan: int
    cutoff_row: bool


def _layout(Q, n, n_tan, cutoff_row):
    P = len(Q.pairs)
    I, J = Q.pairs[:, 0], Q.pairs[:, 1]
    ar = np.arange(P)
    rows, cols, vals = [], [], []
    for k in range(4):  # McCormick: values (w, x_i, x_j) per box
        rows += [k * P + ar] * 3
        cols += [n + ar, I, J]
    m = 4 * P
    g = Q.E[1:1 + Q.n_g]
    h = Q.E[1 + Q.n_g:1 + Q.n_g + Q.n_h]
    blocks = [g, h, -h, Q.rlt, -Q.rlt] + ([Q.E[:1]] if cutoff_row else [])
    for B in blocks[:5]:
        r, c = np.nonzero(B)
        rows.append(m + r)
        cols.append(c)
        vals.append(B[r, c])
        m += B.shape[0]
    if n_tan:  # square tangents at LP points: values (w, x_i) per round
        for t in range(n_tan):
            rows += [m + ar] * 2
            cols += [n + ar, I]
            m += P
    if cutoff_row:
        r, c = np.nonzero(Q.E[:1])
        rows.append(m + r)
        cols.append(c)
        vals.append(Q.E[0, c])
        m += 1
    static = np.concatenate(vals) if vals else np.zeros(0)
    return _Layout(_pattern(rows, cols, m), static, n_tan, cutoff_row)


def _segsum(v, idx, size):
    return jax.ops.segment_sum(v, jnp.asarray(idx), size)


def _lp(Q, L, lo, hi, tol, safe, cutoff=None):
    """Column bounds and values ``(vals, b)`` of the rows ``A v <= b`` of the lifted LP over the
    box, in ``L``'s layout (tangent rows dead: ``b = inf``), and ``tangents(x, k)``, which
    returns the values and right-hand sides with the tangents of round ``k`` set at ``x``."""
    n, P = lo.size, len(Q.pairs)
    N = n + P
    I, J = jnp.asarray(Q.pairs[:, 0]), jnp.asarray(Q.pairs[:, 1])
    sq = jnp.asarray(Q.pairs[:, 0] == Q.pairs[:, 1])
    li, ui, lj, uj = lo[I], hi[I], lo[J], hi[J]
    corners = jnp.stack([li * lj, li * uj, ui * lj, ui * uj])
    straddle = (li < 0) & (ui > 0)
    wlo = jnp.where(sq, jnp.where(straddle, 0.0, jnp.minimum(li * li, ui * ui)), corners.min(0))
    whi = corners.max(0)
    vlo, vhi = jnp.concatenate([lo, wlo]), jnp.concatenate([hi, whi])
    vmag = jnp.maximum(jnp.abs(vlo), jnp.abs(vhi))

    # McCormick: (w, x_i, x_j) coefficients and rhs, 4 rows per product. For a square
    # (x_j = x_i): tangents at l and u, the secant, and the tangent at the midpoint.
    mid = 0.5 * (li + ui)
    one = jnp.ones(P)
    cw = jnp.stack([-one, -one, one, jnp.where(sq, -1.0, 1.0)])
    ci = jnp.stack([jnp.where(sq, 2 * li, lj), jnp.where(sq, 2 * ui, uj), jnp.where(sq, -(li + ui), -lj),
                    jnp.where(sq, 2 * mid, -uj)])  # fmt: skip
    cj = jnp.stack([jnp.where(sq, 0.0, li), jnp.where(sq, 0.0, ui), jnp.where(sq, 0.0, -ui), jnp.where(sq, 0.0, -li)])
    rhs = jnp.stack([li * lj, ui * uj, -ui * lj, jnp.where(sq, mid * mid, -li * uj)])
    mc_vals = jnp.stack([cw, ci, cj], axis=1).reshape(-1)  # pattern order: k, then (w, i, j), then p
    c = Q.c
    g = c[1:1 + Q.n_g]
    h = c[1 + Q.n_g:1 + Q.n_g + Q.n_h]
    V = jnp.asarray(Q.rlt_var)
    rt = tol * jnp.maximum(jnp.abs(lo[V]), jnp.abs(hi[V]))  # |h(x) x_v| <= tol |x_v| when |h| <= tol
    b = [rhs.reshape(-1), jnp.asarray(tol - g), jnp.asarray(tol - h), jnp.asarray(tol + h), rt, rt]
    if L.n_tan:
        b.append(jnp.full(L.n_tan * P, jnp.inf))
    if L.cutoff_row:
        b.append(jnp.atleast_1d(cutoff - c[0]))
    b = jnp.concatenate(b)
    n_static = L.static_vals.size - (np.count_nonzero(Q.E[0]) if L.cutoff_row else 0)
    tan_at = 12 * P + n_static
    pieces = [mc_vals, jnp.asarray(L.static_vals[:n_static])]
    if L.n_tan:
        pieces.append(jnp.zeros(L.n_tan * 2 * P))
    if L.cutoff_row:
        pieces.append(jnp.asarray(L.static_vals[n_static:]))
    vals = jnp.concatenate(pieces)
    R, C = L.pattern.rows, L.pattern.cols
    absAv = _segsum(jnp.abs(vals) * vmag[C], R, L.pattern.m)

    def margin(b, absAv):  # rounding of the facet coefficients and of A v over the column ranges
        return b + SAFE_K * EPS * (N + 3) * (jnp.abs(b) + abs(tol) + absAv) if safe else b

    b = margin(b, absAv)
    t0 = 4 * P + Q.n_g + 2 * Q.n_h + 2 * len(Q.rlt_var)  # first tangent row

    def tangents(vals, b, x, k):
        """Tangents ``-w + 2 p x_i <= p^2`` at ``p = x_i`` for the squares, as round ``k``'s rows."""
        p = jnp.clip(x[I], li, ui)
        tv = jnp.stack([jnp.where(sq, -1.0, 0.0), jnp.where(sq, 2 * p, 0.0)]).reshape(-1)
        tb = jnp.where(sq, p * p, 0.0)
        tb = margin(tb, jnp.where(sq, vmag[n:] + 2 * jnp.abs(p) * vmag[:n][I], 0.0))
        vals = jax.lax.dynamic_update_slice(vals, tv, (tan_at + k * 2 * P,))
        b = jax.lax.dynamic_update_slice(b, jnp.where(sq, tb, jnp.inf), (t0 + k * P,))
        return vals, b

    return vlo, vhi, vals, b, tangents


def _scale(vlo, vhi, safe):
    scale = jnp.maximum(vhi - vlo, 1e-9 * (1 + jnp.abs(vlo)))
    if safe:  # vlo + scale >= vhi in floats
        ceps = SAFE_K * EPS
        scale = scale * (1 + 2 * ceps) + ceps * (jnp.abs(vlo) + jnp.abs(vhi))
    return scale


def _unit(pat, vals, b, vlo, scale, safe):
    """Rows in unit coordinates ``v = vlo + scale * u``, normalized (dead rows: ``0 <= 1``)."""
    N = vlo.size
    R, C = pat.rows, pat.cols
    mb = jnp.abs(b) + _segsum(jnp.abs(vals) * jnp.abs(vlo)[C], R, pat.m)
    b = b - _segsum(vals * vlo[C], R, pat.m)
    vals = vals * scale[C]
    nrm = jnp.sqrt(_segsum(vals * vals, R, pat.m))
    live = jnp.isfinite(b) & (nrm > 0)
    nrm = jnp.where(live, nrm, 1.0)
    vn = jnp.where(live[R], vals / nrm[R], 0.0)
    bn = jnp.where(live, b / nrm, 1.0)
    if safe:  # rounding of the change of variables and of the row scaling
        ceps = SAFE_K * EPS
        rs = _segsum(jnp.abs(vals), R, pat.m)
        bn = jnp.where(live, bn + ceps * (N + 3) * (jnp.abs(bn) + (mb + rs) / nrm), bn)
    return vn, bn


def _certificate(pat, vals, b, y, cvec, safe):
    """Weak-duality bound of ``min cvec.u, A u <= b, u in [0, 1]^N`` at duals ``y`` (clipped
    to >= 0), less its rounding error, and the reduced costs shrunk by theirs."""
    N = cvec.size
    y = jnp.where(jnp.isfinite(y), jnp.maximum(y, 0.0), 0.0)
    coef = cvec + _segsum(vals * y[pat.rows], pat.cols, N)
    aty = _segsum(jnp.abs(vals) * y[pat.rows], pat.cols, N)
    val = jnp.sum(jnp.minimum(coef, 0.0)) - y @ b
    err = (pat.m + N + 2) * EPS * (jnp.sum(jnp.abs(cvec) + aty) + jnp.abs(b) @ y)
    if safe:
        err = SAFE_K * err
    cerr = (pat.m + 2) * EPS * (1.0 + aty)  # as lp._safe_costs (rows have norm <= 1)
    return val - err, jnp.sign(coef) * jnp.maximum(jnp.abs(coef) - cerr, 0.0)


def _ipm(c, pat, vals, b, iters=40, sigma=0.2, cert=None, mu_tol=1e-10, target=jnp.inf, gap_tol=1e-9):
    """``lp.ipm`` for ``min c.u, A u <= b, 0 <= u <= 1`` with ``A`` in the sparse pattern
    ``pat``: the box rows enter the normal matrix as a diagonal, the others through the pairs of
    nonzeros of each row. Returns ``(u, y_best, best_cert)`` (``y`` of the ``A`` rows).

    It stops before ``iters`` iterations once the best certificate reaches ``target`` (the box
    is pruned) or is within ``gap_tol`` (relative) of the objective at a primal point whose rows
    hold to ``gap_tol``; under ``vmap`` the loop runs until every box has stopped."""
    m, N = pat.m, c.size
    R, C = jnp.asarray(pat.rows), jnp.asarray(pat.cols)
    key = jnp.asarray(pat.cols[pat.pa] * N + pat.cols[pat.pb])
    vv = vals[jnp.asarray(pat.pa)] * vals[jnp.asarray(pat.pb)]
    rpa = jnp.asarray(pat.rows[pat.pa])
    cert = cert or (lambda y: -jnp.inf)

    def Av(z):
        return jax.ops.segment_sum(vals * z[C], R, m)

    def ATy(y):
        return jax.ops.segment_sum(vals * y[R], C, N)

    def step_len(v, dv):
        r = jnp.where(dv < 0, -v / jnp.where(dv < 0, dv, -1.0), jnp.inf)
        return jnp.min(r)

    def done(st):
        z, s, y, sh, yh, sl, yl, best, yb, k = st
        zc = jnp.clip(z, 0.0, 1.0)
        feas = jnp.max(Av(zc) - b, initial=-jnp.inf) <= gap_tol
        closed = feas & (c @ zc - best <= gap_tol * (1.0 + jnp.abs(best)))
        mu = (s @ y + sh @ yh + sl @ yl) / (m + 2 * N)
        return (k >= iters) | (best >= target) | closed | (mu <= mu_tol)

    def body(st):
        z, s, y, sh, yh, sl, yl, best, yb, k = st
        rp, rph, rpl = Av(z) + s - b, z + sh - 1.0, sl - z
        rd = c + ATy(y) + yh - yl
        mu = (s @ y + sh @ yh + sl @ yl) / (m + 2 * N)
        M = jnp.zeros(N * N).at[key].add(vv * (y / s)[rpa]).reshape(N, N)
        M = M + jnp.diag(yh / sh + yl / sl + 1e-12)
        t, th, tl = ((sigma * mu - s * y + y * rp) / s, (sigma * mu - sh * yh + yh * rph) / sh,
                     (sigma * mu - sl * yl + yl * rpl) / sl)  # fmt: skip
        dz = jax.scipy.linalg.cho_solve(jax.scipy.linalg.cho_factor(M, lower=True), -rd - (ATy(t) + th - tl))  # M is SPD
        ds, dsh, dsl = -rp - Av(dz), -rph - dz, -rpl + dz
        dy, dyh, dyl = ((sigma * mu - s * y - y * ds) / s, (sigma * mu - sh * yh - yh * dsh) / sh,
                        (sigma * mu - sl * yl - yl * dsl) / sl)  # fmt: skip
        ap = jnp.minimum(1.0, 0.99 * jnp.minimum(step_len(s, ds), jnp.minimum(step_len(sh, dsh), step_len(sl, dsl))))
        ad = jnp.minimum(1.0, 0.99 * jnp.minimum(step_len(y, dy), jnp.minimum(step_len(yh, dyh), step_len(yl, dyl))))
        new = (z + ap * dz, s + ap * ds, y + ad * dy, sh + ap * dsh, yh + ad * dyh, sl + ap * dsl, yl + ad * dyl)
        ok = jnp.all(jnp.isfinite(new[0])) & jnp.all(jnp.isfinite(new[2])) & (mu > mu_tol)
        z, s, y, sh, yh, sl, yl = jax.tree.map(lambda a, b_: jnp.where(ok, a, b_), new, (z, s, y, sh, yh, sl, yl))
        cy = cert(y)
        cy = jnp.where(jnp.isnan(cy), -jnp.inf, cy)
        better = cy > best
        return z, s, y, sh, yh, sl, yl, jnp.where(better, cy, best), jnp.where(better, y, yb), k + 1

    z0 = jnp.full(N, 0.5)
    s0 = jnp.maximum(b - Av(z0), 1.0)
    y0 = jnp.ones(m)
    ones = jnp.ones(N)
    c0 = cert(y0)
    st = (z0, s0, y0, ones, ones, ones, ones, jnp.where(jnp.isnan(c0), -jnp.inf, c0), y0, 0)
    z, _, _, _, _, _, _, best, yb, _ = jax.lax.while_loop(lambda st: ~done(st), body, st)
    return z, yb, best


def make_quadratic_bounder(Q, n_kelley=4, ipm_iters=40, tol=0.0, safe=False):
    """Return ``bound(lo, hi, cutoff) -> (lb, x_lp, infeasible, lo', hi')`` for one box
    (the contract of ``lp.make_constrained_bounder``) from the lifted LP of ``Q``."""
    squares = bool(np.any(Q.pairs[:, 0] == Q.pairs[:, 1]))
    n_kelley = n_kelley if squares else 0  # the bilinear facets are already the envelopes
    layouts = {}

    def bound(lo, hi, cutoff=jnp.inf):
        n = lo.size
        if n not in layouts:
            layouts[n] = _layout(Q, n, n_kelley, False)
        L = layouts[n]
        pat = L.pattern
        vlo, vhi, vals0, b0, tangents = _lp(Q, L, lo, hi, tol, safe)
        finite = jnp.all(jnp.isfinite(vlo) & jnp.isfinite(vhi))
        vlo, vhi = jnp.where(finite, vlo, 0.0), jnp.where(finite, vhi, 1.0)
        vals0, b0 = jnp.where(finite, vals0, 0.0), jnp.where(finite, b0, 1.0)
        scale = _scale(vlo, vhi, safe)
        Ef, cf = jnp.asarray(Q.E[0]), Q.c[0]
        cu = Ef * scale
        cn = jnp.maximum(jnp.linalg.norm(cu), 1e-300)
        chat = cu / cn
        f0 = cf + Ef @ vlo
        fmag = abs(cf) + jnp.abs(Ef) @ jnp.abs(vlo)
        N = vlo.size

        def lower(val):
            lbv = f0 + cn * val
            if safe:
                lbv = lbv - SAFE_K * EPS * (N + 3) * (fmag + jnp.abs(f0) + cn * jnp.abs(val))
            return lbv

        def round_(carry, k):
            vals, b, best, rlo, rhi, _ = carry
            A, bu = _unit(pat, vals, b, vlo, scale, safe)
            target = (cutoff - f0) / cn + 1e-9 * (1 + jnp.abs(cutoff) / cn)  # unit-space value that prunes
            z, y, val = _ipm(chat, pat, A, bu, ipm_iters, target=target,
                             cert=lambda y: _certificate(pat, A, bu, y, chat, safe)[0])  # fmt: skip
            lb = jnp.where(jnp.isnan(val) | ~finite, -jnp.inf, lower(val))
            best = jnp.maximum(best, lb)
            _, r = _certificate(pat, A, bu, y, chat, safe)
            klo, khi = reduce_ranges(vlo[:n], vlo[:n] + scale[:n], lb, cn * r[:n] / scale[:n], cutoff)
            rlo, rhi = jnp.maximum(rlo, klo), jnp.minimum(rhi, khi)
            v = vlo + scale * jnp.clip(z, 0.0, 1.0)
            if n_kelley:  # Kelley: tangents of the squares at the LP solution (not after the last round)
                vals, b = jax.lax.cond(k < n_kelley, lambda: tangents(vals, b, v, jnp.minimum(k, n_kelley - 1)),
                                       lambda: (vals, b))  # fmt: skip
            return (vals, b, best, rlo, rhi, v[:n]), None

        init = (vals0, b0, jnp.array(-jnp.inf, lo.dtype), lo, hi, 0.5 * (lo + hi))
        (_, _, best, rlo, rhi, x), _ = jax.lax.scan(round_, init, jnp.arange(n_kelley + 1))
        infeasible = best == jnp.inf
        rlo, rhi = jnp.where(finite, rlo, lo), jnp.where(finite, rhi, hi)
        return best, jnp.clip(x, lo, hi), infeasible, rlo, rhi

    return bound


def make_quadratic_obbt(Q, rounds=2, ipm_iters=40, tol=0.0, safe=False):
    """Return ``obbt(lo, hi, cutoff) -> (lo', hi', infeasible)``: each variable minimized and
    maximized over the lifted LP with ``f <= cutoff`` (2n LPs, one ``vmap``), the facets
    rebuilt on the tightened box between rounds. The new bounds are certificates."""
    layouts = {}

    def obbt(lo, hi, cutoff=jnp.inf):
        n = lo.size
        if n not in layouts:
            layouts[n] = _layout(Q, n, 0, True)
        L = layouts[n]
        pat = L.pattern
        empty = jnp.array(False)
        for _ in range(rounds):
            vlo, vhi, vals, b, _ = _lp(Q, L, lo, hi, tol, safe, cutoff=cutoff)
            finite = jnp.all(jnp.isfinite(vlo) & jnp.isfinite(vhi))
            vlo, vhi = jnp.where(finite, vlo, 0.0), jnp.where(finite, vhi, 1.0)
            vals, b = jnp.where(finite, vals, 0.0), jnp.where(finite, b, 1.0)
            scale = _scale(vlo, vhi, safe)
            A, bu = _unit(pat, vals, b, vlo, scale, safe)
            N = vlo.size
            C = jnp.concatenate([jnp.eye(n, N), -jnp.eye(n, N)])

            def solve(cvec, A=A, bu=bu):
                _, _, val = _ipm(cvec, pat, A, bu, ipm_iters, cert=lambda y: _certificate(pat, A, bu, y, cvec, safe)[0])
                return val

            B = jax.vmap(solve)(C)
            ulo = jnp.where(jnp.isfinite(B[:n]), jnp.maximum(B[:n], 0.0), 0.0)
            uhi = jnp.where(jnp.isfinite(B[n:]), jnp.minimum(-B[n:], 1.0), 1.0)
            empty |= finite & jnp.any(ulo > uhi + 1e-9)
            s = scale[:n]
            nlo = jnp.where(finite, jnp.maximum(lo, vlo[:n] + s * (ulo - 1e-9)), lo)
            nhi = jnp.where(finite, jnp.minimum(hi, vlo[:n] + s * (uhi + 1e-9)), hi)
            lo, hi = jnp.minimum(nlo, nhi), jnp.maximum(nlo, nhi)
        return lo, hi, empty

    return obbt
