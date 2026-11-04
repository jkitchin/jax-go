"""The lifted quadratic/polynomial LP bound with one compiled kernel for many problems.

Third increment of the kernel redesign (after ``fbbt_generic`` and ``bound_generic``). For about
two thirds of the benchmark models ``minimize`` bounds nodes with ``quadratic.make_quadratic_bounder``:
an LP over ``v = (x, w)`` with a column per product of two earlier columns, McCormick facets per
product, the problem's rows (linear in ``v``), RLT rows, square tangents (Kelley rounds) and
bound-factor RLT rows, solved by a sparse interior-point method and certified by weak duality.
The LP's structure (the ``QuadraticForm``) is already arrays; here it is a kernel argument:

* the sparse pattern is laid out in fixed blocks, each padded to a bucket: McCormick (4 rows and
  12 nonzeros per product), static rows (inequalities, equalities both ways, RLT rows both ways,
  binary-product cuts: one block, each row tagged with what its right-hand side is), square
  tangents (``n_kelley`` rounds), bound-factor RLT rows; padded rows are dead (``b = inf``),
  padded nonzeros are 0, padded columns appear in no row;
* the box-dependent values (column ranges level by level, facets, right-hand sides with the
  tolerance and the cleared denominators, the RLT values) are computed from data and gathered by
  tag; ``quadratic._ipm``, ``_unit`` and ``_certificate`` take the pattern as values.

Left out (v1): the binary-cut separation pool and the objective lattice (``_snap``): both only
raise the bound on binary polynomial models; without them the bound is still a certificate.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import fbbt_generic as fg
from .lp import EPS, reduce_ranges
from .quadratic import _Pattern, _certificate, _ipm, _pattern, _scale, _unit, square_needs_lower
from .relax import SAFE_K

F64 = jnp.float64
# static-row kinds: what the right-hand side is
K_G, K_HP, K_HM, K_RLT, K_CUT, K_DEAD = 0, 1, 2, 3, 4, 5


def _b(v, least=8):
    return fg._bucket(v, least)


class Data(NamedTuple):
    # pattern (padded): nonzeros in block order [McCormick 12 Pb | static Sb | tangents nt 2 Pb | bound-factor 2 Qb]
    rows: np.ndarray  # (nnz,)
    cols: np.ndarray  # (nnz,)
    pa: np.ndarray  # (pp,) pairs of nonzeros in a row (normal matrix assembly)
    pb: np.ndarray
    static_vals: np.ndarray  # (Sb,)
    # columns and products
    n_real: np.ndarray  # () real variables (the others of the nb are padding, fixed at 0)
    I: np.ndarray  # (Pb,) factor columns of each product (padded numbering)
    J: np.ndarray
    sq: np.ndarray  # (Pb,) a square
    level: np.ndarray  # (Pb,) 1 + the larger level of its factors; 0: padding
    pair_live: np.ndarray  # (Pb,)
    need: np.ndarray  # (Pb,) a square that gets tangents (``square_needs_lower``)
    max_level: np.ndarray  # ()
    Ef: np.ndarray  # (Nc,) objective row
    cf: np.ndarray  # () objective constant
    # static rows (Rb): kind, constant, RLT variable, cut rhs
    s_kind: np.ndarray
    s_c: np.ndarray
    s_var: np.ndarray
    s_rhs: np.ndarray
    s_den: np.ndarray  # (Rb,) a cleared equality row (its tolerance scales with the denominator)
    # denominators: terms of d per static row (both signs of a cleared h row get its terms)
    d_row: np.ndarray  # (Db,)
    d_coef: np.ndarray  # (Db,)
    d_pow: np.ndarray  # (Db, nb)
    binary: np.ndarray  # (Bb,) variables the binary cuts assume in [0, 1] (padding: a padding variable)
    has_binary: np.ndarray  # ()
    # bound-factor RLT: pairs (g row, k) and their nonzeros
    q_c: np.ndarray  # (T2b,) the constant of the pair's g row
    q_var: np.ndarray  # (T2b,) its k
    q_row: np.ndarray  # (Qb,) pair of each nonzero
    q_w: np.ndarray  # (Qb,)
    q_a: np.ndarray  # (Qb,)
    q_k: np.ndarray  # (Qb,)


jax.export.register_namedtuple_serialization(Data, serialized_name="jaxglobal.quadratic_generic.Data")


class Generic(NamedTuple):
    data: Data
    n: int
    nb: int
    Pb: int
    m: int  # rows of the padded pattern


def encode(Q, n, n_kelley=4):
    """``Q`` (a ``QuadraticForm`` on ``n`` variables) as padded kernel data. The separation pool
    and the lattice are dropped (see the module docstring)."""
    P = len(Q.pairs)
    N0 = Q.E.shape[1]
    nb, Pb = _b(n), _b(P)
    Nc = nb + Pb

    def col(c):  # original column -> padded numbering
        c = np.asarray(c, np.int64)
        return np.where(c < n, c, c - n + nb)

    I, J = np.zeros(Pb, np.int64), np.zeros(Pb, np.int64)
    I[:P], J[:P] = col(Q.pairs[:, 0]), col(Q.pairs[:, 1])
    sq = np.zeros(Pb, bool)
    sq[:P] = Q.pairs[:, 0] == Q.pairs[:, 1]
    level = np.zeros(Pb, np.int64)
    level[:P] = Q.level if Q.level.size == P else 1
    live = np.zeros(Pb, bool)
    live[:P] = True
    need = np.zeros(Pb, bool)
    need[:P] = square_needs_lower(Q)
    ar = np.arange(Pb)
    rows, cols = [], []
    for k in range(4):  # McCormick: (w, x_i, x_j) per product
        rows += [k * Pb + ar] * 3
        cols += [nb + ar, I, J]
    m = 4 * Pb
    # static rows: g, h+, h-, RLT+, RLT-, binary cuts
    g = Q.E[1:1 + Q.n_g]
    h = Q.E[1 + Q.n_g:1 + Q.n_g + Q.n_h]
    cg, ch = Q.c[1:1 + Q.n_g], Q.c[1 + Q.n_g:1 + Q.n_g + Q.n_h]
    cuts = Q.cuts.reshape(-1, N0) if Q.cuts.size else np.zeros((0, N0))
    blocks = [(g, K_G, cg, None), (h, K_HP, ch, None), (-h, K_HM, ch, None), (Q.rlt, K_RLT, None, Q.rlt_var),
              (-Q.rlt, K_RLT, None, Q.rlt_var), (cuts, K_CUT, Q.cut_rhs, None)]  # fmt: skip
    R0 = sum(B.shape[0] for B, *_ in blocks)
    Rb = _b(R0)
    s_kind, s_c, s_var, s_den = np.full(Rb, K_DEAD, np.int64), np.zeros(Rb), np.zeros(Rb, np.int64), np.zeros(Rb, bool)
    srow, scol, sval, r = [], [], [], 0
    den_rows = set(int(t) for t in Q.den_row)
    h_static = {}  # h index -> its two static rows
    for B, kind, cvec, var in blocks:
        rr, cc = np.nonzero(B)
        srow.append(r + rr)
        scol.append(col(cc))
        sval.append(B[rr, cc])
        for i in range(B.shape[0]):
            s_kind[r + i] = kind
            if cvec is not None:
                s_c[r + i] = cvec[i]
            if var is not None:
                s_var[r + i] = col(var[i])
            if kind in (K_HP, K_HM):
                h_static.setdefault(i, []).append(r + i)
                s_den[r + i] = i in den_rows
        r += B.shape[0]
    srow, scol, sval = np.concatenate(srow), np.concatenate(scol), np.concatenate(sval)
    Sb = _b(srow.size + 1)  # (at least one zero nonzero: the padded pairs point at it)
    static_vals = np.zeros(Sb)
    static_vals[: sval.size] = sval
    pad_row = np.flatnonzero(s_kind == K_DEAD)
    pad_r = int(pad_row[0]) if pad_row.size else Rb - 1
    SENTINEL = -1  # padded nonzeros: no row while the normal-matrix pairs are built (``_pattern``), then
    # a dead row (all of them in one row would make that row's pairs quadratic in the padding)
    rows.append(np.concatenate([m + srow, np.full(Sb - srow.size, SENTINEL)]))
    cols.append(np.concatenate([scol, np.zeros(Sb - scol.size, np.int64)]))
    m += Rb
    for _ in range(n_kelley):  # square tangents: (w, x_i) per product, values set per round
        rows += [m + ar] * 2
        cols += [nb + ar, I]
        m += Pb
    T2 = len(Q.qr_g)
    T2b, Qb = _b(T2), _b(len(Q.qr_row))
    q_row = np.zeros(Qb, np.int64)
    q_row[: len(Q.qr_row)] = Q.qr_row
    qcol = np.zeros(Qb, np.int64)
    qcol[: len(Q.qr_col)] = col(Q.qr_col)
    nq = len(Q.qr_row)
    for s in range(2):  # bound-factor RLT: lower factors, then upper factors
        rows.append(np.where(np.arange(Qb) < nq, m + s * T2b + q_row, SENTINEL))
        cols.append(qcol)
    m += 2 * T2b
    allrows = np.concatenate(rows)
    pat = _pattern([np.where(allrows < 0, m, allrows)], [np.concatenate(cols)], m)  # (row m: not paired)
    pat = pat._replace(rows=np.where(allrows < 0, 4 * Pb + pad_r, allrows))
    q_c, q_var = np.zeros(T2b), np.zeros(T2b, np.int64)
    q_c[:T2], q_var[:T2] = Q.c[1 + Q.qr_g], col(Q.qr_var)
    q_w, q_a, q_k = np.zeros(Qb), np.zeros(Qb), np.zeros(Qb)
    q_w[: len(Q.qr_w)], q_a[: len(Q.qr_a)], q_k[: len(Q.qr_k)] = Q.qr_w, Q.qr_a, Q.qr_k
    # denominators of cleared equality rows, one term list per static row (both signs)
    dr, dc, dp = [], [], []
    for t in range(Q.den_row.size):
        for srow_i in h_static.get(int(Q.den_row[t]), []):
            dr.append(srow_i)
            dc.append(Q.den_coef[t])
            p = np.zeros(nb, np.int64)
            p[:n] = Q.den_pow[t]
            dp.append(p)
    Db = _b(len(dr), least=1)
    d_row, d_coef, d_pow = np.zeros(Db, np.int64), np.zeros(Db), np.zeros((Db, nb), np.int64)
    if dr:
        d_row[: len(dr)], d_coef[: len(dr)], d_pow[: len(dr)] = dr, dc, np.stack(dp)
    Bb = _b(Q.binary.size, least=1)
    binary = np.full(Bb, nb - 1 if nb > n else 0, np.int64)
    binary[: Q.binary.size] = col(Q.binary)
    Ef = np.zeros(Nc)
    Ef[col(np.arange(N0))] = Q.E[0]
    # pad the normal-matrix pairs to a bucket with a pair of zero nonzeros (the static block's padding)
    zero = 12 * Pb + sval.size
    pp = _b(pat.pa.size)
    pa, pbb = np.full(pp, zero, np.int64), np.full(pp, zero, np.int64)
    pa[: pat.pa.size], pbb[: pat.pb.size] = pat.pa, pat.pb
    data = Data(pat.rows, pat.cols, pa, pbb, static_vals, np.asarray(n), I, J, sq, level, live, need,
                np.asarray(int(level.max(initial=0))), Ef, np.asarray(float(Q.c[0])), s_kind, s_c, s_var,
                _cut_rhs(s_kind, Q.cut_rhs, Rb), s_den, d_row, d_coef, d_pow, binary,
                np.asarray(Q.binary.size > 0), q_c, q_var, q_row, q_w, q_a, q_k)  # fmt: skip
    return Generic(data, n, nb, Pb, m)


def _cut_rhs(s_kind, rhs, Rb):
    out = np.zeros(Rb)
    at = np.flatnonzero(s_kind == K_CUT)
    out[at] = rhs[: at.size]
    return out


def _bound_one(lo, hi, cutoff, tol, d: Data, m, n_kelley, ipm_iters, safe):
    """``make_quadratic_bounder``'s ``bound`` (no pool, no lattice) on the padded box."""
    nb, Pb = lo.shape[0], d.I.shape[0]
    N = nb + Pb
    I, J, sq = d.I, d.J, d.sq
    pat = _Pattern(d.rows, d.cols, m, d.pa, d.pb)

    def product_range(li, ui, lj, uj, sq):
        corners = jnp.stack([li * lj, li * uj, ui * lj, ui * uj])
        straddle = (li < 0) & (ui > 0)
        wlo = jnp.where(sq, jnp.where(straddle, 0.0, jnp.minimum(li * li, ui * ui)), corners.min(0))
        whi = corners.max(0)
        if safe:
            ceps = SAFE_K * EPS
            wlo, whi = wlo - ceps * jnp.abs(wlo), whi + ceps * jnp.abs(whi)
        return wlo, whi

    vlo = jnp.concatenate([lo, jnp.zeros(Pb)])
    vhi = jnp.concatenate([hi, jnp.zeros(Pb)])

    def lev(k, c):  # column ranges level by level (a product's factors are earlier columns)
        vlo, vhi = c
        wlo, whi = product_range(vlo[I], vhi[I], vlo[J], vhi[J], sq)
        at = d.level == k
        return vlo.at[nb:].set(jnp.where(at, wlo, vlo[nb:])), vhi.at[nb:].set(jnp.where(at, whi, vhi[nb:]))

    vlo, vhi = jax.lax.fori_loop(1, d.max_level + 1, lev, (vlo, vhi))
    li, ui, lj, uj = vlo[I], vhi[I], vlo[J], vhi[J]
    vmag = jnp.maximum(jnp.abs(vlo), jnp.abs(vhi))

    mid = 0.5 * (li + ui)
    one = jnp.ones(Pb)
    cw = jnp.stack([-one, -one, one, jnp.where(sq, -1.0, 1.0)])
    ci = jnp.stack([jnp.where(sq, 2 * li, lj), jnp.where(sq, 2 * ui, uj), jnp.where(sq, -(li + ui), -lj),
                    jnp.where(sq, 2 * mid, -uj)])  # fmt: skip
    cj = jnp.stack([jnp.where(sq, 0.0, li), jnp.where(sq, 0.0, ui), jnp.where(sq, 0.0, -ui), jnp.where(sq, 0.0, -li)])
    rhs = jnp.stack([li * lj, ui * uj, -ui * lj, jnp.where(sq, mid * mid, -li * uj)])
    rhs = jnp.where(d.pair_live[None, :], rhs, jnp.inf)
    mc_vals = jnp.stack([cw, ci, cj], axis=1).reshape(-1)

    # static rows' right-hand sides by kind
    ok = d.has_binary & jnp.all((vlo[d.binary] >= 0.0) & (vhi[d.binary] <= 1.0))
    rt = tol * jnp.maximum(jnp.abs(vlo[d.s_var]), jnp.abs(vhi[d.s_var]))
    term = jnp.abs(d.d_coef) * jnp.prod(vmag[:nb][None, :] ** d.d_pow, axis=1)
    dmag = jax.ops.segment_sum(term, d.d_row, d.s_kind.shape[0])
    dmag = jnp.where(jnp.isnan(dmag), jnp.inf, dmag)
    th = jnp.where(d.s_den, tol * dmag * (1 + 1e-12), tol)
    k = d.s_kind
    sb = jnp.where(k == K_G, tol - d.s_c, jnp.where(k == K_HP, th - d.s_c, jnp.where(k == K_HM, th + d.s_c,
         jnp.where(k == K_RLT, rt, jnp.where(k == K_CUT, jnp.where(ok, d.s_rhs, jnp.inf), jnp.inf)))))  # fmt: skip
    T2b, Qb = d.q_c.shape[0], d.q_row.shape[0]
    cq = d.q_c - tol
    ql, qu = vlo[d.q_var], vhi[d.q_var]
    base = d.q_w + cq[d.q_row] * d.q_k
    qr_vals = jnp.concatenate([base - ql[d.q_row] * d.q_a, -base + qu[d.q_row] * d.q_a])
    b = jnp.concatenate([rhs.reshape(-1), sb, jnp.full(n_kelley * Pb, jnp.inf), cq * ql, -cq * qu])
    vals = jnp.concatenate([mc_vals, d.static_vals, jnp.zeros(n_kelley * 2 * Pb), qr_vals])
    absAv = jax.ops.segment_sum(jnp.abs(vals) * vmag[d.cols], d.rows, m)

    def margin(b, absAv):
        return b + SAFE_K * EPS * (N + 3) * (jnp.abs(b) + jnp.abs(tol) + absAv) if safe else b

    b = margin(b, absAv)
    t0 = 4 * Pb + d.s_kind.shape[0]
    q0 = t0 + n_kelley * Pb
    if not safe:  # quadratic._lp's plain-mode margin of the bound-factor rows
        qm = EPS * (N + 3) * (jnp.abs(b[q0:q0 + 2 * T2b]) + jnp.abs(tol) + absAv[q0:q0 + 2 * T2b])
        b = b.at[q0:q0 + 2 * T2b].add(qm)
    tan_at = 12 * Pb + d.static_vals.shape[0]

    def tangents(vals, b, x, k):
        p = jnp.clip(x[I], li, ui)
        tv = jnp.stack([jnp.where(sq, -1.0, 0.0), jnp.where(sq, 2 * p, 0.0)]).reshape(-1)
        tb = margin(jnp.where(sq, p * p, 0.0), jnp.where(sq, vmag[nb:] + 2 * jnp.abs(p) * vmag[I], 0.0))
        vals = jax.lax.dynamic_update_slice(vals, tv, (tan_at + k * 2 * Pb,))
        b = jax.lax.dynamic_update_slice(b, jnp.where(sq & d.pair_live, tb, jnp.inf), (t0 + k * Pb,))
        return vals, b

    finite = jnp.all(jnp.isfinite(vlo) & jnp.isfinite(vhi))
    vlo, vhi = jnp.where(finite, vlo, 0.0), jnp.where(finite, vhi, 1.0)
    vals, b = jnp.where(finite, vals, 0.0), jnp.where(finite, b, 1.0)
    scale = _scale(vlo, vhi, safe)
    Ef, cf = d.Ef, d.cf
    cu = Ef * scale
    cn = jnp.maximum(jnp.linalg.norm(cu), 1e-300)
    chat = cu / cn
    f0 = cf + Ef @ vlo
    fmag = jnp.abs(cf) + jnp.abs(Ef) @ jnp.abs(vlo)
    need = d.need

    def lower(val):
        lbv = f0 + cn * val
        if safe:
            lbv = lbv - SAFE_K * EPS * (N + 3) * (fmag + jnp.abs(f0) + cn * jnp.abs(val))
        return lbv

    def round_(carry, k):
        vals, b, best, rlo, rhi, vprev, active = carry
        A, bu = _unit(pat, vals, b, vlo, scale, safe)
        target = (cutoff - f0) / cn + 1e-9 * (1 + jnp.abs(cutoff) / cn)
        z, y, val = _ipm(chat, pat, A, bu, jnp.where(active, ipm_iters, 0), target=target,
                         cert=lambda y: _certificate(pat, A, bu, y, chat, safe)[0])  # fmt: skip
        lb = jnp.where(jnp.isnan(val) | ~finite | ~active, -jnp.inf, lower(val))
        best = jnp.maximum(best, lb)
        _, r = _certificate(pat, A, bu, y, chat, safe)
        klo, khi = reduce_ranges(vlo[:nb], vlo[:nb] + scale[:nb], lb, cn * r[:nb] / scale[:nb], cutoff)
        rlo = jnp.where(active, jnp.maximum(rlo, klo), rlo)
        rhi = jnp.where(active, jnp.minimum(rhi, khi), rhi)
        v = jnp.where(active, vlo + scale * jnp.clip(z, 0.0, 1.0), vprev)
        cut = jnp.array(False)
        if n_kelley:
            gap = jnp.where(need, v[I] * v[J] - v[nb:], -jnp.inf)
            cut = (k < n_kelley) & jnp.any(gap > 1e-9 * (1.0 + scale[nb:]))
            vals, b = jax.lax.cond(k < n_kelley, lambda: tangents(vals, b, v, jnp.minimum(k, n_kelley - 1)),
                                   lambda: (vals, b))  # fmt: skip
        active = active & (best < cutoff) & cut
        return (vals, b, best, rlo, rhi, v, active), None

    init = (vals, b, jnp.array(-jnp.inf), lo, hi, vlo + 0.5 * scale, jnp.array(True))
    (_, _, best, rlo, rhi, x, _), _ = jax.lax.scan(round_, init, jnp.arange(n_kelley + 1))
    x = jnp.clip(x[:nb], lo, hi)
    infeasible = best == jnp.inf
    rlo, rhi = jnp.where(finite, rlo, lo), jnp.where(finite, rhi, hi)
    return best, x, infeasible, rlo, rhi, _value_at(x, d, tol)


def _value_at(x, d: Data, tol):
    """``f(x)`` if ``x`` satisfies the rows to ``tol``, else inf (an incumbent candidate; the caller
    re-checks it). The rows are linear in ``v = (x, products)``: the products level by level."""
    nb, Pb = x.shape[0], d.I.shape[0]
    v = jnp.concatenate([x, jnp.zeros(Pb)])
    v = jax.lax.fori_loop(1, d.max_level + 1,
                          lambda k, v: v.at[nb:].set(jnp.where(d.level == k, v[d.I] * v[d.J], v[nb:])), v)  # fmt: skip
    Sb, Rb = d.static_vals.shape[0], d.s_kind.shape[0]
    at = 12 * Pb  # the static block's nonzeros
    srow = d.rows[at:at + Sb] - 4 * Pb
    rowv = jax.ops.segment_sum(d.static_vals * v[d.cols[at:at + Sb]], jnp.clip(srow, 0, Rb - 1), Rb) + d.s_c
    vmag = jnp.abs(v)
    dmag = jax.ops.segment_sum(jnp.abs(d.d_coef) * jnp.prod(vmag[:nb][None, :] ** d.d_pow, axis=1), d.d_row, Rb)
    th = jnp.where(d.s_den, tol * dmag * (1 + 1e-12), tol)
    k = d.s_kind
    viol = jnp.where(k == K_G, rowv - tol, jnp.where(k == K_HP, jnp.abs(rowv) - th, -1.0))
    f = d.cf + d.Ef @ v
    return jnp.where(jnp.all(viol <= 0) & jnp.isfinite(f), f, jnp.inf)


@functools.lru_cache(maxsize=None)
def kernel(m, n_kelley=4, ipm_iters=40, safe=False):
    """Batched ``(lo (B, nb), hi (B, nb), cutoff, tol, data) -> (lb, x, infeasible, lo', hi')``;
    ``m`` (rows of the padded pattern) is a static size."""

    def run(lo, hi, cutoff, tol, data):
        one = functools.partial(_bound_one, m=m, n_kelley=n_kelley, ipm_iters=ipm_iters, safe=safe)
        return jax.vmap(one, in_axes=(0, 0, None, None, None))(lo, hi, cutoff, tol, data)

    return jax.jit(run)


_EXES = {}


def executable(m, n_kelley, ipm_iters, safe, args):
    """The compiled kernel for ``args``' shapes (``fbbt_generic.executable``'s caching)."""
    from . import kcache

    specs = jax.tree_util.tree_map(lambda a: jax.ShapeDtypeStruct(jnp.shape(a), jnp.result_type(a)), args)
    key = (m, n_kelley, ipm_iters, bool(safe), tuple((s.shape, str(s.dtype)) for s in jax.tree_util.tree_leaves(specs)))
    if key in _EXES:
        return _EXES[key]
    fn = kernel(m, n_kelley, ipm_iters, safe)
    lowered = None
    if kcache.cache_root() is not None:
        import hashlib

        import jaxlib

        dev = jax.devices()[0]
        fp = "quadratic_generic-" + hashlib.sha256(repr((kcache._source_hash(), jax.__version__, jaxlib.__version__,
                                                         dev.platform, dev.device_kind, key)).encode()).hexdigest()[:32]  # fmt: skip
        exp = kcache.load(fp)
        if exp is None:
            try:
                exp = jax.export.export(fn)(*specs)
            except Exception:
                exp = None
            kcache.save(fp, exp)
        if exp:
            lowered = jax.jit(exp.call).trace(*specs).lower()
    if lowered is None:
        lowered = fn.trace(*specs).lower()
    _EXES[key] = lowered.compile()
    return _EXES[key]


class GenericQuadraticBounder:
    """``bound(lo, hi, cutoff) -> (lb, x_lp, infeasible, lo', hi')`` on batches ``(B, n)``."""

    def __init__(self, g: Generic, tol, safe=False, n_kelley=4, ipm_iters=40):
        self.g, self.safe, self.n_kelley, self.ipm_iters = g, bool(safe), int(n_kelley), int(ipm_iters)
        self.tol = jnp.asarray(tol, F64)
        self.data = Data(*(jnp.asarray(v) for v in g.data))

    def __call__(self, lo, hi, cutoff=jnp.inf):
        n, nb = self.g.n, self.g.nb
        lo, hi = jnp.asarray(lo, F64), jnp.asarray(hi, F64)
        z = jnp.zeros(lo.shape[:-1] + (nb - n,), F64)
        args = (jnp.concatenate([lo, z], -1), jnp.concatenate([hi, z], -1), jnp.asarray(cutoff, F64), self.tol, self.data)
        lb, x, inf, rlo, rhi, fx = executable(self.g.m, self.n_kelley, self.ipm_iters, self.safe, args)(*args)
        return lb, x[..., :n], inf, rlo[..., :n], rhi[..., :n], fx


def make_generic_quadratic_bounder(Q, n, tol=0.0, safe=False, n_kelley=4, ipm_iters=40):
    """The shared-kernel lifted-LP bounder for the ``QuadraticForm`` ``Q`` on ``n`` variables."""
    return GenericQuadraticBounder(encode(Q, n, n_kelley), tol, safe, n_kelley, ipm_iters)
