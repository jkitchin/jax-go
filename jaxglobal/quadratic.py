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

MAX_RLT_SUPPORT = 12  # ineq_rlt: linear inequalities with at most this many variables
# ineq_rlt: rows with at most this many variables take the pair budget first. Wider rows only
# fill what is left: on trim loss (tln4, tln5) a pattern's capacity row sum_i a_i y_i <= W times
# its multiplicity m (m * y_i are the columns) gives sum_i a_i w_i <= W m, which links the demand
# rows sum_m w_mi >= d_i to the number of rolls cut (root bound 2.42 -> 7.66 on tln4, opt 8.3).
SMALL_RLT_SUPPORT = 3
RLT_PAIRS = 3.0  # ineq_rlt: at most this many (row, k) pairs per product column (two LP rows each)
MAX_COLUMNS = 400  # beyond this the IPM's N x N solve per iteration costs more than it gains
# polynomial_form: the cap when some monomial has degree > 2. The reduced-space fallback traces
# and compiles one relaxation per term, which on these problems takes minutes
# (autocorr_bern20-10, 813 columns: 233 s of XLA compile before the first node).
MAX_POLY_COLUMNS = 1000
MAX_DEGREE = 6  # polynomial_form: highest monomial degree lifted
MAX_TERMS = 20000  # polynomial_form: give up when an intermediate polynomial has more terms
# polynomial_form: give up when a factor (a.x + b) that gets multiplied out has |b| / max|a| above
# this. Expanding (x - c)^k around 0 replaces one convex term by monomials whose McCormick gaps
# grow with c while the function's own variation does not; the reduced-space bounder relaxes
# the factored term directly. Measured on min (x0-c)^4 + (x1-c-1)^2 (x0-c+2)^2 (offset ~c):
# 15 / 63 / 255 / 511 / 1023 nodes at c = 0 / 2 / 5 / 10 / 20, time limit from c = 50 (safe) or
# 100 (plain), where the reduced-space path takes 1 node. start50/int50 offsets: alkyl 8.5,
# ex1252a 3.5, all other polynomial-path instances 0. Not applied at degree 2.
OFFSET_LIMIT = 10.0
MAX_STATIC_CUTS = 6000  # _binary_cuts: static rows kept (in order of generation)
MAX_POOL = 20000  # _binary_cuts: rows in the separation pool
N_SEP = 1  # make_quadratic_bounder: rounds of pool separation (each one more IPM solve)


class Lattice(NamedTuple):
    """The objective's values on binary points lie on ``offset + step * k`` (``_lattice``)."""

    step: float
    offset: float
    j: int  # the epigraph variable t (f = s t + c_0), or -1: f itself is the polynomial
    s: float  # f's coefficient of t (0.0 when j = -1)
    alpha: float  # t's coefficient in its inequality row (0.0 when j = -1)
    vars: np.ndarray  # the variables of the polynomial: the lattice holds where they are in [0, 1]
    mag: float  # |offset| + sum of |coefficients| (scaled to f): bounds |f| at those points


class QuadraticForm(NamedTuple):
    E: np.ndarray  # (R, N) rows [f, g..., h...] linear in v = (x, w)
    c: np.ndarray  # (R,) constants
    pairs: np.ndarray  # (P, 2) a <= b: column n + p is v_a * v_b (a, b < n + p; both < n when quadratic)
    n_g: int
    n_h: int
    rlt: np.ndarray  # (T, N) RLT rows: a linear equality times x_v, linear in v (= 0)
    rlt_var: np.ndarray  # (T,) the v of each
    level: np.ndarray = np.zeros(0, np.int64)  # (P,) 1 + the larger level of the factors (x: level 0)
    # bound-factor RLT rows (``ineq_rlt``): a linear inequality g = a.x + c <= 0 times x_k - l_k >= 0
    # and u_k - x_k >= 0, one entry per nonzero of the product row sum_m a_m w_mk + c x_k
    qr_g: np.ndarray = np.zeros(0, np.int64)  # (T2,) the g row of each pair
    qr_var: np.ndarray = np.zeros(0, np.int64)  # (T2,) its k
    qr_row: np.ndarray = np.zeros(0, np.int64)  # (nnz,) the pair of each entry
    qr_col: np.ndarray = np.zeros(0, np.int64)  # (nnz,) its column
    qr_w: np.ndarray = np.zeros(0)  # (nnz,) coefficient of the products sum_m a_m w_mk
    qr_a: np.ndarray = np.zeros(0)  # (nnz,) coefficient a_m of x_m in g
    qr_k: np.ndarray = np.zeros(0)  # (nnz,) 1.0 on the column x_k (carries c)
    # products of binaries (``_binary_cuts``; valid on a box only where every variable of
    # ``binary`` is within [0, 1]): static rows ``cuts v <= cut_rhs``, and a pool of rows
    # ``sum_k pool_vals[r, k] v[pool_cols[r, k]] <= pool_rhs[r]`` separated at the LP point
    cuts: np.ndarray = np.zeros((0, 0))  # (Cs, N)
    cut_rhs: np.ndarray = np.zeros(0)  # (Cs,)
    pool_cols: np.ndarray = np.zeros((0, 1), np.int64)  # (C, W)
    pool_vals: np.ndarray = np.zeros((0, 1))  # (C, W)
    pool_rhs: np.ndarray = np.zeros(0)  # (C,)
    binary: np.ndarray = np.zeros(0, np.int64)  # integer variables the cuts assume binary
    # objective lattice (``_lattice``): a ``Lattice`` when every value
    # of the objective at an integer feasible point is offset + step * k (k integer), or None
    lattice: tuple = None
    # complement symmetry (``_flip``): an integer variable that may be fixed to 1 on a root box
    # where every integer variable is binary, or -1
    flip: int = -1
    # equality rows cleared of a denominator d (``quadratic_form``): d = sum_t den_coef[t] *
    # prod_i x_i ** den_pow[t, i] over the terms t with den_row[t] = the row (index into h)
    den_row: np.ndarray = np.zeros(0, np.int64)
    den_coef: np.ndarray = np.zeros(0)
    den_pow: np.ndarray = np.zeros((0, 0), np.int64)


def quadratic_form(f, ineq, eq, n, degree=2, max_columns=MAX_COLUMNS, integer=None, max_poly_columns=None):
    """The problem as ``QuadraticForm``, or None if some function is not a polynomial of degree
    at most ``degree``, the tape cannot be scalarized, or the lifted LP would have more than
    ``max_columns`` columns. With ``degree=2`` the columns are the products ``x_i x_j``; above,
    every monomial of degree >= 2 gets a column that is the product of two earlier columns
    (``_lift``), so McCormick facets apply recursively. With ``integer`` (indices of the integer
    variables) and ``degree > 2``, products of binaries get valid cuts (``_binary_cuts``) and an
    integral objective a lattice (``_lattice``)."""
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
    exact = [True]  # every constant and coefficient so far an integer below 2^50: the dicts are exact

    def small_int(v):
        return float(v).is_integer() and abs(v) < 2.0**50

    def poly(s):  # {(): c, (i,): a, (i, j): q, ...}: monomials as sorted tuples of variables
        if s < n:
            return {(s,): 1.0}
        if not np.isnan(cval[s]):
            exact[0] = exact[0] and small_int(cval[s])
            return {(): float(cval[s])}
        return polys[s]

    def axpy(p, q, a=1.0):
        r = dict(p)
        for k, v in q.items():
            r[k] = r.get(k, 0.0) + a * v
            exact[0] = exact[0] and small_int(r[k])
        return r

    def mul(p, q):
        if p is None or q is None or len(p) * len(q) > MAX_TERMS:
            return None
        r = {}
        for k1, v1 in p.items():
            for k2, v2 in q.items():
                k = tuple(sorted(k1 + k2))
                if len(k) > degree:
                    return None
                r[k] = r.get(k, 0.0) + v1 * v2
                exact[0] = exact[0] and small_int(v1 * v2) and small_int(r[k])
        return r

    def power(p, y):
        r = {(): 1.0}
        for _ in range(y):
            r = mul(r, p)
        return r

    def offset(p):  # |constant term| / largest |coefficient| of a non-constant term (0: no offset)
        lin = max((abs(v) for k, v in p.items() if k), default=0.0)
        return abs(p.get((), 0.0)) / lin if lin > 0 else 0.0

    # Rational values: slot s is polys[s] / dens[s] (no entry: denominator 1). They are only kept
    # for equality rows, which are cleared of their denominator below (``den_*``).
    dens = {}

    def den(s):
        return dens.get(s) if s >= n and np.isnan(cval[s]) else None

    def dmul(p, q):  # product of two denominators (None: 1)
        return q if p is None else p if q is None else mul(p, q)

    worst_offset = 0.0
    for o, a, b, out in zip(prog.op.tolist(), prog.a.tolist(), prog.b.tolist(), prog.out.tolist()):
        name, eqn = prog.opcodes[o][0], prog.opcodes[o][6]
        pa, pb = poly(a), poly(b)
        da = den(a)
        db = den(b) if name in ("add", "add_any", "sub", "mul", "div") else None
        if degree > 2 and name in ("mul", "square", "integer_pow"):
            for p in (pa, pb) if name == "mul" else (pa,):
                if any(len(k) >= 1 for k in p):  # a product of a non-constant factor (x - c): expanded
                    worst_offset = max(worst_offset, offset(p))
        d = None
        if name in ("add", "add_any", "sub"):
            sgn = -1.0 if name == "sub" else 1.0
            if da == db:  # (also both 1)
                r, d = axpy(pa, pb, sgn), da
            else:  # pa / da + pb / db = (pa db + pb da) / (da db)
                ta, tb, d = mul(pa, db or {(): 1.0}), mul(pb, da or {(): 1.0}), dmul(da, db)
                if ta is None or tb is None or d is None:
                    return None
                r = axpy(ta, tb, sgn)
        elif name == "neg":
            r, d = {k: -v for k, v in pa.items()}, da
        elif name == "mul":
            r, d = mul(pa, pb), dmul(da, db)
            if (da is not None or db is not None) and d is None:
                return None
        elif name == "div" and set(pb) == {()} and db is None:
            exact[0] = exact[0] and abs(pb[()]) == 1.0
            r, d = {k: v / pb[()] for k, v in pa.items()}, da
        elif name == "div":  # (pa / da) / (pb / db) = (pa db) / (da pb)
            r, d = mul(pa, db or {(): 1.0}), dmul(da, pb)
            if d is None:
                return None
        elif name == "square":
            r = mul(pa, pa)
            d = None if da is None else mul(da, da)
            if da is not None and d is None:
                return None
        elif name == "integer_pow" and 1 <= eqn.params["y"] <= degree:
            r = power(pa, eqn.params["y"])
            d = None if da is None else power(da, eqn.params["y"])
            if da is not None and d is None:
                return None
        else:
            return None
        if r is None or len(r) > MAX_TERMS or (d is not None and len(d) > MAX_TERMS):
            return None
        polys[out] = r
        if d is not None:
            dens[out] = d

    if worst_offset > OFFSET_LIMIT:
        return None
    out_slots = [np.asarray(slots).reshape(-1).tolist() for slots in prog.outs]
    rows = [[poly(int(s)) for s in slots] for slots in out_slots]
    # An equality h = p / d = 0 becomes p = 0: at every point where h is defined (d != 0) the two
    # agree, so the polynomial row is a relaxation (it adds points with d = 0, where h is not
    # defined and so not feasible). With the tolerance |h| <= tol it holds to |p| <= tol |d|
    # (``den_*``: the LP bounds |d| over the box). Objective and inequality rows stay polynomial
    # (clearing an inequality needs the sign of d).
    if any(den(int(s)) is not None for group in out_slots[:len(out_slots) - (eq is not None)] for s in group):
        return None
    den_row, den_coef, den_mono = [], [], []
    if eq is not None:
        for r, s in enumerate(out_slots[-1]):
            d = den(int(s))
            for k, v in (d or {}).items():
                if v == 0.0:
                    continue
                den_row.append(r)
                den_coef.append(v)
                den_mono.append(k)
    flat = [p for group in rows for p in group]
    monos = {k for p in flat for k in p if len(k) >= 2 and p[k] != 0.0}
    if not monos:
        return None
    factors = _lift(monos)
    cap = max_columns if max_poly_columns is None or all(len(k) <= 2 for k in monos) else max_poly_columns
    if factors is None or n + len(factors) > cap:
        return None
    order = sorted(factors, key=lambda k: (len(k), k))  # factors before products (degree 2: sorted pairs)
    col = {k: n + i for i, k in enumerate(order)}

    def column(k):
        return k[0] if len(k) == 1 else col[k]

    pairs = np.asarray([sorted((column(factors[k][0]), column(factors[k][1]))) for k in order], np.int64)
    level = np.zeros(n + len(order), np.int64)
    for p, (a, b) in enumerate(pairs):
        level[n + p] = 1 + max(level[a], level[b])
    E, c = np.zeros((len(flat), n + len(order))), np.zeros(len(flat))
    for r, p in enumerate(flat):
        for k, v in p.items():
            if len(k) == 0:
                c[r] += v
            elif len(k) == 1 or k in col:  # a product whose coefficient cancelled to 0.0 has no column
                E[r, column(k)] += v
    n_g = len(rows[1]) if ineq is not None else 0
    n_h = len(rows[-1]) if eq is not None else 0
    # RLT: a linear equality a.x + c = 0 times a variable x_v of some product gives
    # sum_k a_k w_kv + c x_v = 0, linear in v when every w_kv is a column already
    # (no new columns: the LP keeps its size, and on pooling models these rows carry
    # the strengthening of the full RLT system).
    rlt, rlt_var = [], []
    prod_vars = sorted({i for k in col for i in k})
    cleared = set(den_row)
    for r, p in enumerate(rows[-1] if eq is not None else []):
        if r in cleared or any(len(k) >= 2 for k in p) or not any(len(k) == 1 and v for k, v in p.items()):
            continue
        for v in prod_vars:
            terms = [(tuple(sorted((k[0], v))), a) for k, a in p.items() if len(k) == 1 and a != 0.0]
            if all(k in col for k, _ in terms):
                row = np.zeros(n + len(order))
                for k, a in terms:
                    row[col[k]] += a
                row[v] += p.get((), 0.0)
                rlt.append(row)
                rlt_var.append(v)
    rlt = np.asarray(rlt).reshape(-1, n + len(order))
    Q = QuadraticForm(E, c, pairs.reshape(-1, 2), n_g, n_h, rlt, np.asarray(rlt_var, np.int64), level[n:])
    if den_row:
        pw = np.zeros((len(den_row), n), np.int64)
        for t, k in enumerate(den_mono):
            for i in k:
                pw[t, i] += 1
        Q = Q._replace(den_row=np.asarray(den_row, np.int64), den_coef=np.asarray(den_coef, float), den_pow=pw)
    if integer is not None and degree > 2:
        ints = set(np.asarray(integer, np.int64).reshape(-1).tolist())
        Q = Q._replace(**_binary_cuts(col, n, ints), lattice=_lattice(Q, col, n, ints))
        if exact[0] and not den_row:  # (the symmetry argument is about the rows as polynomials)
            Q = Q._replace(flip=_flip(rows[0][0], rows[1] if ineq is not None else [],
                                      rows[-1] if eq is not None else [], ints))
    return ineq_rlt(Q, n) if degree == 2 else Q


def ineq_rlt(Q, n, max_support=None, max_pairs=None):
    """``Q`` with bound-factor RLT rows (Sherali & Adams): a linear inequality
    ``g(x) = a.x + c <= 0`` (with support of 2 to ``max_support`` variables) times a bound factor
    ``x_k - l_k >= 0`` or ``u_k - x_k >= 0`` gives

        sum_m a_m w_mk + c x_k - l_k a.x <= c l_k,      l_k a.x - ... (the same with u_k, negated),

    linear in ``v`` when every product ``x_m x_k`` is a column already. The bound factors are
    the node's, so the rows are rebuilt per box like the McCormick facets. They are what links a
    difference ``d = x_j - x_i`` that a linear row orders (``x_i <= x_j``) to the products in
    ``d^2 = w_jj - 2 w_ij + w_ii``: their sum over ``k = i, j`` is ``d (u_j - l_i - d) >= 0``,
    the secant of ``d^2`` over ``[0, u_j - l_i]`` instead of over ``[l_i - u_j, u_j - l_i]``,
    which the McCormick facets of the three products alone give (circle packing with ordered
    identical circles, kall_circles_*). At most ``max_pairs`` pairs (default: ``RLT_PAIRS``
    per product column): those of rows with at most ``SMALL_RLT_SUPPORT`` variables first, so a
    wide row never displaces one of theirs, and within each group those with ``k`` in the row's
    support first."""
    P = len(Q.pairs)
    max_pairs = int(RLT_PAIRS * P) if max_pairs is None else max_pairs
    max_support = MAX_RLT_SUPPORT if max_support is None else max_support
    col = {(int(a), int(b)): n + p for p, (a, b) in enumerate(Q.pairs) if a < n and b < n}
    partners = {}
    for a, b in col:
        partners.setdefault(a, set()).add(b)
        partners.setdefault(b, set()).add(a)
    cand = []  # (wide row, k not in the row's support, row, k): short rows, then a factor of the row's own variables
    for r in range(Q.n_g):
        row = Q.E[1 + r]
        if np.any(row[n:] != 0):
            continue
        supp = [int(m) for m in np.flatnonzero(row[:n])]
        if not 2 <= len(supp) <= max_support:
            continue
        common = set.intersection(*(partners.get(m, set()) | ({m} if (m, m) in col else set()) for m in supp))
        cand += [(len(supp) > SMALL_RLT_SUPPORT, k not in supp, r, k) for k in sorted(common)]
    gi, gk, er, ec, ew, ea, ek = [], [], [], [], [], [], []
    for _, _, r, k in sorted(cand)[:max_pairs]:
        row = Q.E[1 + r]
        ent = {}
        for m in np.flatnonzero(row[:n]):
            m = int(m)
            ent.setdefault(col[tuple(sorted((m, k)))], [0.0, 0.0, 0.0])[0] += row[m]
            ent.setdefault(m, [0.0, 0.0, 0.0])[1] += row[m]
        ent.setdefault(k, [0.0, 0.0, 0.0])[2] = 1.0
        t = len(gi)
        gi.append(r)
        gk.append(k)
        for cc, (w, a, kk) in sorted(ent.items()):
            er.append(t)
            ec.append(cc)
            ew.append(w)
            ea.append(a)
            ek.append(kk)
    if not gi:
        return Q
    i64 = lambda v: np.asarray(v, np.int64)  # noqa: E731
    return Q._replace(qr_g=i64(gi), qr_var=i64(gk), qr_row=i64(er), qr_col=i64(ec), qr_w=np.asarray(ew),
                      qr_a=np.asarray(ea), qr_k=np.asarray(ek))  # fmt: skip


def _flip(f, gs, hs, ints):
    """An integer variable ``i`` that may be fixed to 1 by complement symmetry, or -1.

    Let ``sigma`` map ``x_k`` to ``1 - x_k`` for every integer ``k`` and leave the others. When the
    polynomials (exact: integer coefficients, ``quadratic_form``) satisfy ``f o sigma = f``,
    ``g o sigma = g`` and ``h o sigma = +-h`` as polynomials, ``sigma`` maps the feasible set of a
    box whose integer variables are all binary onto itself and keeps ``f``. Of ``x`` and
    ``sigma(x)`` one has ``x_i = 1``, so fixing ``x_i = 1`` keeps an optimal point and the optimal
    value: a bound of the restricted problem is a bound of the problem. ``i`` is the integer
    variable in the most monomials (an epigraph objective ``t`` has its polynomial in a row)."""

    def clean(p):
        return {k: int(v) for k, v in p.items() if v != 0.0}

    def sigma(p):
        out = {}
        for k, v in p.items():
            keep = [i for i in k if i not in ints]
            flip = [i for i in k if i in ints]
            for mask in range(1 << len(flip)):  # prod (1 - x_i): the subsets of flip, signed
                chosen = [i for b, i in enumerate(flip) if mask >> b & 1]
                m = tuple(sorted(keep + chosen))
                out[m] = out.get(m, 0) + (-1) ** len(chosen) * v
        return {k: v for k, v in out.items() if v != 0}

    count = {}
    for p in [f, *gs, *hs]:
        for k in clean(p):
            for i in set(k) & ints:
                count[i] = count.get(i, 0) + 1
    if not count:
        return -1
    for p in [f, *gs]:
        q = clean(p)
        if sigma(q) != q:
            return -1
    for p in hs:
        q = clean(p)
        sq = sigma(q)
        if sq != q and sq != {k: -v for k, v in q.items()}:
            return -1
    return max(sorted(count), key=lambda i: count[i])


def polynomial_form(f, ineq, eq, n, degree=MAX_DEGREE, max_columns=MAX_COLUMNS, integer=None,
                    max_poly_columns=None):
    """``quadratic_form`` of a polynomial problem of degree up to ``degree`` (sparse lifted LP:
    one column per monomial present and per intermediate product, no other monomials). At most
    ``max_columns`` columns, or ``max_poly_columns`` when some monomial has degree above 2 (by
    default ``MAX_POLY_COLUMNS`` with the default ``max_columns``, else ``max_columns``)."""
    if max_poly_columns is None:
        max_poly_columns = MAX_POLY_COLUMNS if max_columns == MAX_COLUMNS else max_columns
    return quadratic_form(f, ineq, eq, n, degree=degree, max_columns=max_columns, integer=integer,
                          max_poly_columns=max_poly_columns)


def _binary_cuts(col, n, ints):
    """Valid rows for the columns that are products of binary variables.

    For binaries a product column is the AND of its variables, ``w_T = prod_{i in T} x_i`` (as a
    set: ``x^2 = x``), so for any columns or variables ``S, S_1, ..., S_k`` (``v_S`` their
    values) and ``U = T minus the union of the S_i``:

    * ``w_T <= v_S`` when ``S`` is a proper subset of ``T`` (static rows; only the ``S`` not
      inside another such subset, the rest are implied);
    * ``v_S + v_S' - w_T <= 1`` when ``S`` and ``S'`` cover ``T`` (static rows);
    * flower inequalities (Del Pia & Khajavirad 2018) ``sum_{U} x_i + v_S - w_T <= |U|`` for a
      column ``S`` meeting ``T`` in at least two variables, ``S`` not necessarily inside ``T``
      (a pool separated at the LP point).

    Each holds at every binary point: if every variable of ``T`` is 1 then ``w_T = 1`` and the
    left-hand sides are at most ``v_S <= 1`` resp. ``|U| + v_S - 1``; otherwise ``w_T = 0`` and a
    variable at 0 lies in ``S`` (``v_S = 0``) or in ``U`` (one ``x_i`` term is 0). They are not
    valid for fractional values in [0, 1] (except the first), so a box where some variable of
    ``binary`` is outside [0, 1] gets none of them (``_lp``).
    """
    sets = {T: frozenset(T) for T in col if all(i in ints for i in T)}
    if not sets:
        return {}
    cand = {(i,): frozenset((i,)) for i in sorted({i for T in sets.values() for i in T})}
    cand.update(sets)
    index = {T: (T[0] if len(T) == 1 else col[T]) for T in cand}
    N = n + len(col)
    static, pool, seen = [], [], set()

    def add(dst, d, rhs):
        key = (tuple(sorted(d.items())), rhs)
        if key not in seen and any(d.values()):
            seen.add(key)
            dst.append((d, rhs))

    for T, sT in sorted(sets.items(), key=lambda kv: (len(kv[0]), kv[0])):
        if len(sT) < 2:
            continue
        subs = [S for S, s in cand.items() if s < sT]
        maximal = [S for S in subs if not any(cand[S] < cand[R] for R in subs)]
        for S in maximal:
            add(static, {index[T]: 1.0, index[S]: -1.0}, 0.0)
        for a in range(len(subs)):
            for b in range(a + 1, len(subs)):
                if cand[subs[a]] | cand[subs[b]] == sT:
                    add(static, {index[T]: -1.0, index[subs[a]]: 1.0, index[subs[b]]: 1.0}, 1.0)
        for S, s in sets.items():
            if S != T and len(s & sT) >= 2 and s != sT:
                d = {index[T]: -1.0, index[S]: 1.0}
                for i in sT - s:
                    d[i] = d.get(i, 0.0) + 1.0
                add(pool, d, float(len(sT - s)))
    static, pool = static[:MAX_STATIC_CUTS], pool[:MAX_POOL]
    cuts = np.zeros((len(static), N))
    for r, (d, _) in enumerate(static):
        for k, v in d.items():
            cuts[r, k] += v
    W = max((len(d) for d, _ in pool), default=1)
    pc, pv = np.zeros((len(pool), W), np.int64), np.zeros((len(pool), W))
    for r, (d, _) in enumerate(pool):
        for k, (j, v) in enumerate(sorted(d.items())):
            pc[r, k], pv[r, k] = j, v
    binary = sorted({i for T in cand for i in T})
    return dict(cuts=cuts, cut_rhs=np.asarray([b for _, b in static]), pool_cols=pc, pool_vals=pv,
                pool_rhs=np.asarray([b for _, b in pool]), binary=np.asarray(binary, np.int64))  # fmt: skip


def binary_only(Q, n):
    """Whether every one of the ``n`` variables is a binary of the product columns (``_binary_cuts``)
    or the epigraph variable of the objective lattice (``_lattice``). ``bnb`` then skips FBBT on the
    nodes: with the binary cuts and reduced-cost fixing in the LP it seldom tightens these boxes,
    and on the expanded polynomials it costs more than the LP (autocorr_bern30-04: 0.19 s per 16
    boxes against 0.15 s for the bound)."""
    if Q is None or not Q.binary.size:
        return False
    rest = set(range(n)) - set(np.asarray(Q.binary).tolist())
    if Q.lattice is not None and Q.lattice.j >= 0:
        rest.discard(int(Q.lattice.j))
    return not rest


def _rational_step(vals, max_den=1000):
    """The largest ``g > 0`` with every ``v / g`` an integer (``vals`` nonzero), or None."""
    from fractions import Fraction
    from math import gcd

    fr = []
    for v in vals:
        q = Fraction(v).limit_denominator(max_den)
        if float(q) != v:  # v is the double nearest q (``_snap`` covers that rounding)
            return None
        fr.append(q)
    num = 0
    den = 1
    for q in fr:
        den = den * q.denominator // gcd(den, q.denominator)
    for q in fr:
        num = gcd(num, int(q * den))
    return num / den if num else None


def _lattice(Q, col, n, ints):
    """A ``Lattice`` when the minimum of the objective over any box whose integer variables are
    binary is a lattice point ``offset + step * k`` (or the box bound of the objective, ``_snap``).

    Two shapes qualify. (a) ``f`` is a combination of integer variables and products of integer
    variables, with ``j = -1``: its value at every integer point is ``c_0 + sum a_T m_T``, on the
    lattice of step ``gcd(a)``. (b) An epigraph ``f = s t + c_0`` with ``t`` a continuous variable
    that appears in exactly one other row, an inequality ``alpha t + P(x) <= 0`` with ``s alpha < 0``
    and ``P`` of shape (a), and in no product. For fixed integer ``x``, the best ``t`` is
    ``-P(x) / alpha`` unless the box bound of ``t`` stops it first, so the minimum over a box is a
    lattice point of ``-s P / alpha + c_0`` or the box bound ``s t_hi + c_0`` (``s < 0``) resp.
    ``s t_lo + c_0``; ``j = t``, ``s`` its coefficient.

    ``_snap`` uses it only on boxes where the variables of the polynomial are in [0, 1], where
    every monomial is 0 or 1 and ``mag`` bounds the size of ``f`` (and of its rounding errors)."""
    E, c = Q.E, Q.c
    N = E.shape[1]
    intcol = np.zeros(N, bool)
    intcol[[i for i in ints if i < n]] = True
    for T, k in col.items():
        intcol[k] = all(i in ints for i in T)

    inv = {k: T for T, k in col.items()}

    def lattice(row, const, j=-1, s=0.0, alpha=0.0):
        nz = np.flatnonzero(row)
        if not nz.size or not np.all(intcol[nz]):
            return None
        g = _rational_step(row[nz].tolist())
        if g is None:
            return None
        vs = sorted({i for k in nz.tolist() for i in (inv[k] if k in inv else (k,))})
        mag = abs(float(const)) + float(np.sum(np.abs(row[nz])))
        return Lattice(g, float(const), j, s, alpha, np.asarray(vs, np.int64), mag)

    f = E[0]
    nz = np.flatnonzero(f)
    if nz.size and np.all(intcol[nz]):
        return lattice(f, c[0])
    if nz.size != 1 or intcol[nz[0]] or nz[0] >= n:
        return None
    j = int(nz[0])
    s = float(f[j])
    if np.any(Q.pairs == j):
        return None
    rows = [r for r in range(1, E.shape[0]) if E[r, j] != 0]
    if len(rows) != 1 or not 1 <= rows[0] <= Q.n_g:
        return None
    r = rows[0]
    alpha = float(E[r, j])
    if s * alpha >= 0:
        return None
    P = E[r].copy()
    P[j] = 0.0
    scale = -s / alpha
    return lattice(P * scale, c[r] * scale + c[0], j, s, alpha)


def _lift(monos):
    """``{monomial: (A, B)}`` for every monomial of degree >= 2 in ``monos`` and the
    intermediate monomials they need: ``A * B`` is the monomial (sorted tuples; a factor of
    degree 1 is a variable). Each monomial is split into a contiguous prefix and suffix
    (in sorted order), choosing the split that adds the fewest new columns given the
    columns made so far (ties: the most balanced). Degree 2: ``(i, j) -> ((i,), (j,))``."""
    made = {}

    def build(m, memo):  # (number of new columns, best split) for m given ``made``
        if len(m) == 1 or m in made:
            return 0, None
        if m in memo:
            return memo[m]
        d, best = len(m), None
        for k in sorted(range(1, d), key=lambda k: abs(d - 2 * k)):
            cost = 1 + build(m[:k], memo)[0] + build(m[k:], memo)[0]
            if best is None or cost < best[0]:
                best = (cost, (m[:k], m[k:]))
        memo[m] = best
        return best

    def add(m, memo):
        if len(m) == 1 or m in made:
            return
        a, b = build(m, memo)[1]
        add(a, memo)
        add(b, memo)
        made[m] = (a, b)

    for m in sorted(monos, key=lambda k: (len(k), k)):
        add(m, {})
    return made


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
    """Row blocks of the LP: McCormick (4P), g, h+, h-, RLT+, RLT-, binary cuts (Cs), square
    tangents (n_tan P), bound-factor RLT rows (2 T2: lower factors, then upper), separated pool
    rows (n_sep rounds of K rows of width W), and an optional objective-cutoff row (OBBT)."""

    pattern: _Pattern
    static_vals: np.ndarray  # values of the E-based rows (in pattern order after the McCormick block)
    n_tan: int
    cutoff_row: bool
    n_sep: int = 0
    K: int = 0  # pool rows separated per round
    sep_row: int = 0  # first row of the pool slots
    sep_at: int = 0  # first pattern entry (and value) of the pool slots


def _sep_size(Q, n_sep):
    """Rows separated from ``Q``'s pool per round (0: no pool or no rounds)."""
    C = Q.pool_rhs.size
    return min(C, max(16, (n_sep > 0) * (len(Q.pairs) + Q.E.shape[1]) // 2)) if n_sep and C else 0


def _layout(Q, n, n_tan, cutoff_row, n_sep=0):
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
    cuts = Q.cuts.reshape(-1, Q.E.shape[1]) if Q.cuts.size else np.zeros((0, Q.E.shape[1]))
    for B in (g, h, -h, Q.rlt, -Q.rlt, cuts):
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
    T2 = len(Q.qr_g)
    if T2:  # bound-factor RLT rows: lower factors, then upper factors (values per box)
        for s in range(2):
            rows.append(m + s * T2 + Q.qr_row)
            cols.append(Q.qr_col)
        m += 2 * T2
    K = _sep_size(Q, n_sep)
    sep_row, sep_at = m, sum(np.size(r) for r in rows)
    if K:  # separated pool rows: W slots per row, columns and values set per round
        W = Q.pool_cols.shape[1]
        rows.append(m + np.repeat(np.arange(n_sep * K), W))
        cols.append(np.zeros(n_sep * K * W, np.int64))
        m += n_sep * K
    if cutoff_row:
        r, c = np.nonzero(Q.E[:1])
        rows.append(m + r)
        cols.append(c)
        vals.append(Q.E[0, c])
        m += 1
    static = np.concatenate(vals) if vals else np.zeros(0)
    return _Layout(_pattern(rows, cols, m), static, n_tan, cutoff_row, n_sep if K else 0, K, sep_row, sep_at)


def _binary_ok(Q, lo, hi):
    """True where every variable the binary cuts assume binary is within [0, 1] on the box."""
    B = jnp.asarray(Q.binary)
    return jnp.all((lo[B] >= 0.0) & (hi[B] <= 1.0)) if Q.binary.size else jnp.array(False)


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

    def product_range(li, ui, lj, uj, sq):  # interval of v_a v_b (of v_a^2 for a square)
        corners = jnp.stack([li * lj, li * uj, ui * lj, ui * uj])
        straddle = (li < 0) & (ui > 0)
        wlo = jnp.where(sq, jnp.where(straddle, 0.0, jnp.minimum(li * li, ui * ui)), corners.min(0))
        whi = corners.max(0)
        if safe:  # the products rounded to nearest: widen by a few ulps so the true product is inside
            ceps = SAFE_K * EPS
            wlo, whi = wlo - ceps * jnp.abs(wlo), whi + ceps * jnp.abs(whi)
        return wlo, whi

    # column ranges level by level: a product's factors are x (level 0) or earlier products
    level = Q.level if Q.level.size == P else np.ones(P, np.int64)
    vlo, vhi = jnp.concatenate([lo, jnp.zeros(P, lo.dtype)]), jnp.concatenate([hi, jnp.zeros(P, hi.dtype)])
    for lev in range(1, int(level.max(initial=0)) + 1):
        at = np.flatnonzero(level == lev)
        a, bb = Q.pairs[at, 0], Q.pairs[at, 1]
        wlo, whi = product_range(vlo[a], vhi[a], vlo[bb], vhi[bb], jnp.asarray(a == bb))
        vlo, vhi = vlo.at[n + at].set(wlo), vhi.at[n + at].set(whi)
    li, ui, lj, uj = vlo[I], vhi[I], vlo[J], vhi[J]
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
    ok = _binary_ok(Q, lo, hi)
    th = jnp.full(Q.n_h, tol, lo.dtype)
    if Q.den_row.size:  # a cleared row p = h d holds to |p| <= tol max |d| over the box
        xmag = vmag[:n]
        term = jnp.abs(jnp.asarray(Q.den_coef)) * jnp.prod(xmag[None, :] ** jnp.asarray(Q.den_pow), axis=1)
        dmag = _segsum(term, Q.den_row, Q.n_h)
        dmag = jnp.where(jnp.isnan(dmag), jnp.inf, dmag)  # (an unbounded variable: the row is dropped)
        cl = np.zeros(Q.n_h, bool)
        cl[Q.den_row] = True
        th = jnp.where(jnp.asarray(cl), tol * dmag * (1 + 1e-12), th)
    b = [rhs.reshape(-1), jnp.asarray(tol - g), th - h, th + h, rt, rt,
         jnp.where(ok, jnp.asarray(Q.cut_rhs, lo.dtype), jnp.inf)]  # fmt: skip
    if L.n_tan:
        b.append(jnp.full(L.n_tan * P, jnp.inf))
    T2 = len(Q.qr_g)
    if T2:  # (a.x + c')(x_k - l_k) <= 0 and (a.x + c')(u_k - x_k) <= 0 with c' = c - tol (g <= tol)
        cq = jnp.asarray(c[1 + Q.qr_g]) - tol
        kq = jnp.asarray(Q.qr_var)
        rq = jnp.asarray(Q.qr_row)
        base = jnp.asarray(Q.qr_w) + cq[rq] * jnp.asarray(Q.qr_k)
        qa = jnp.asarray(Q.qr_a)
        ql, qu = lo[kq], hi[kq]
        qr_vals = jnp.concatenate([base - ql[rq] * qa, -base + qu[rq] * qa])
        b += [cq * ql, -cq * qu]
    if L.K:
        b.append(jnp.full(L.n_sep * L.K, jnp.inf))
    if L.cutoff_row:
        b.append(jnp.atleast_1d(cutoff - c[0]))
    b = jnp.concatenate(b)
    n_static = L.static_vals.size - (np.count_nonzero(Q.E[0]) if L.cutoff_row else 0)
    tan_at = 12 * P + n_static
    pieces = [mc_vals, jnp.asarray(L.static_vals[:n_static])]
    if L.n_tan:
        pieces.append(jnp.zeros(L.n_tan * 2 * P))
    if T2:
        pieces.append(qr_vals)
    if L.K:
        pieces.append(jnp.zeros(L.n_sep * L.K * Q.pool_cols.shape[1]))
    if L.cutoff_row:
        pieces.append(jnp.asarray(L.static_vals[n_static:]))
    vals = jnp.concatenate(pieces)
    R, C = L.pattern.rows, L.pattern.cols
    absAv = _segsum(jnp.abs(vals) * vmag[C], R, L.pattern.m)

    def margin(b, absAv):  # rounding of the facet coefficients and of A v over the column ranges
        return b + SAFE_K * EPS * (N + 3) * (jnp.abs(b) + abs(tol) + absAv) if safe else b

    b = margin(b, absAv)
    t0 = 4 * P + Q.n_g + 2 * Q.n_h + 2 * len(Q.rlt_var) + Q.cut_rhs.size  # first tangent row
    if T2 and not safe:
        # The bound-factor rows need a rounding margin in plain mode too. When x_k is fixed
        # (l_k = u_k, e.g. an integer at a node), both rows of a pair reduce to the equality
        # a.(w_k - l_k x) = 0, which McCormick also implies. Rounded, these equalities can be
        # inconsistent by a few ulps. The weak-duality certificate of that empty LP then
        # diverges and prunes a box that holds feasible points.
        q0 = t0 + L.n_tan * P
        qm = EPS * (N + 3) * (jnp.abs(b[q0:q0 + 2 * T2]) + abs(tol) + absAv[q0:q0 + 2 * T2])
        b = b.at[q0:q0 + 2 * T2].add(qm)

    def tangents(vals, b, x, k):
        """Tangents ``-w + 2 p x_i <= p^2`` at ``p = x_i`` for the squares, as round ``k``'s rows."""
        p = jnp.clip(x[I], li, ui)
        tv = jnp.stack([jnp.where(sq, -1.0, 0.0), jnp.where(sq, 2 * p, 0.0)]).reshape(-1)
        tb = jnp.where(sq, p * p, 0.0)
        tb = margin(tb, jnp.where(sq, vmag[n:] + 2 * jnp.abs(p) * vmag[I], 0.0))
        vals = jax.lax.dynamic_update_slice(vals, tv, (tan_at + k * 2 * P,))
        b = jax.lax.dynamic_update_slice(b, jnp.where(sq, tb, jnp.inf), (t0 + k * P,))
        return vals, b

    return vlo, vhi, vals, b, tangents


def _separate(Q, L, cols, vals, b, v, k, vlo, vhi, ok, tol, safe):
    """Round ``k``'s pool rows: the ``L.K`` rows of ``Q``'s pool most violated (relative to their
    norm) at the lifted point ``v``, written into the round's slots of the pattern columns, values
    and right-hand sides; a slot whose row holds at ``v`` (or every slot, on a box where the cuts
    are not valid, ``ok`` False) is dead (``b = inf``). Also returns whether some slot is live
    (a row that cuts ``v`` off)."""
    P, W, K = len(Q.pairs), Q.pool_cols.shape[1], L.K
    pc, pv, pb = jnp.asarray(Q.pool_cols), jnp.asarray(Q.pool_vals, v.dtype), jnp.asarray(Q.pool_rhs, v.dtype)
    nrm = jnp.sqrt(jnp.sum(pv * pv, axis=1))
    viol = (jnp.sum(pv * v[pc], axis=1) - pb) / nrm
    top, idx = jax.lax.top_k(viol, K)
    live = ok & (top > 1e-9)
    c_new, v_new = pc[idx].reshape(-1), pv[idx].reshape(-1)
    b_new = pb[idx]
    if safe:  # as _lp's margin
        vmag = jnp.maximum(jnp.abs(vlo), jnp.abs(vhi))
        absAv = jnp.sum(jnp.abs(pv[idx]) * vmag[pc[idx]], axis=1)
        b_new = b_new + SAFE_K * EPS * (vlo.size + 3) * (jnp.abs(b_new) + abs(tol) + absAv)
    n_static = L.static_vals.size - (np.count_nonzero(Q.E[0]) if L.cutoff_row else 0)
    # row blocks: McCormick, g, h+, h-, RLT+, RLT-, binary cuts, tangents, bound-factor RLT
    # (2 T2 rows, 2 nnz values), then the pool slots (``_layout``)
    T2, nq = len(Q.qr_g), len(Q.qr_row)
    at0 = 12 * P + n_static + L.n_tan * 2 * P + 2 * nq  # first value (and pattern entry) of the slots
    row0 = 4 * P + Q.n_g + 2 * Q.n_h + 2 * len(Q.rlt_var) + Q.cut_rhs.size + L.n_tan * P + 2 * T2
    assert (at0, row0) == (L.sep_at, L.sep_row), "pool slots moved: _layout and _separate disagree"
    at, r0 = at0 + k * K * W, row0 + k * K
    cols = jax.lax.dynamic_update_slice(cols, c_new.astype(cols.dtype), (at,))
    vals = jax.lax.dynamic_update_slice(vals, v_new, (at,))
    b = jax.lax.dynamic_update_slice(b, jnp.where(live, b_new, jnp.inf), (r0,))
    return cols, vals, b, jnp.any(live)


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
    # rows that hold on all of [0, 1]^N (cuts whose columns are fixed, say) are dropped: they
    # leave no interior for the IPM, which then stalls. Dropping a row only relaxes the LP.
    slack = bn - _segsum(jnp.maximum(vn, 0.0), R, pat.m)
    idle = live & (slack >= -1e-9)
    vn = jnp.where(idle[R], 0.0, vn)
    bn = jnp.where(idle, 1.0, bn)
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


def _ipm(c, pat, vals, b, iters=40, sigma=0.2, cert=None, mu_tol=1e-10, target=jnp.inf, gap_tol=1e-9,
         settled=None):
    """``lp.ipm`` for ``min c.u, A u <= b, 0 <= u <= 1`` with ``A`` in the sparse pattern
    ``pat``: the box rows enter the normal matrix as a diagonal, the others through the pairs of
    nonzeros of each row. Returns ``(u, y_best, best_cert)`` (``y`` of the ``A`` rows).

    It stops before ``iters`` iterations once the best certificate reaches ``target`` (the box
    is pruned) or is within ``gap_tol`` (relative) of the objective at a primal point whose rows
    hold to ``gap_tol``, or once ``settled(best, c.u)`` holds at such a point (the bound cannot
    improve in the units the caller rounds it to, ``_snap``); under ``vmap`` the loop runs until
    every box has stopped. The stopping rules only decide when to stop: ``best`` is a
    certificate whatever they say."""
    m, N = pat.m, c.size
    R, C = jnp.asarray(pat.rows), jnp.asarray(pat.cols)
    key = C[pat.pa] * N + C[pat.pb]  # the columns may be traced (separated rows)
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
        z, s, y, sh, yh, sl, yl, best, yb, k, stuck = st
        zc = jnp.clip(z, 0.0, 1.0)
        feas = jnp.max(Av(zc) - b, initial=-jnp.inf) <= gap_tol
        closed = feas & (c @ zc - best <= gap_tol * (1.0 + jnp.abs(best)))
        if settled is not None:
            closed = closed | (feas & settled(best, c @ zc))
        mu = (s @ y + sh @ yh + sl @ yl) / (m + 2 * N)
        return (k >= iters) | (best >= target) | closed | (mu <= mu_tol) | stuck

    def body(st):
        z, s, y, sh, yh, sl, yl, best, yb, k, _ = st
        rp, rph, rpl = Av(z) + s - b, z + sh - 1.0, sl - z
        rd = c + ATy(y) + yh - yl
        mu = (s @ y + sh @ yh + sl @ yl) / (m + 2 * N)
        M = jnp.zeros(N * N).at[key].add(vv * (y / s)[rpa]).reshape(N, N)
        M = M + jnp.diag(yh / sh + yl / sl + 1e-12)
        t, th, tl = ((sigma * mu - s * y + y * rp) / s, (sigma * mu - sh * yh + yh * rph) / sh,
                     (sigma * mu - sl * yl + yl * rpl) / sl)  # fmt: skip
        dm = jnp.sqrt(jnp.diag(M))  # Jacobi scaling: the factorization survives a wider spread of y/s
        Ms = M / dm[:, None] / dm[None, :]
        dz = jax.scipy.linalg.cho_solve(jax.scipy.linalg.cho_factor(Ms, lower=True), (-rd - (ATy(t) + th - tl)) / dm) / dm
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
        # a failed step (the factorization broke down) leaves the iterate as it was: stop there
        return z, s, y, sh, yh, sl, yl, jnp.where(better, cy, best), jnp.where(better, y, yb), k + 1, ~ok

    z0 = jnp.full(N, 0.5)
    s0 = jnp.maximum(b - Av(z0), 1.0)
    y0 = jnp.ones(m)
    ones = jnp.ones(N)
    c0 = cert(y0)
    st = (z0, s0, y0, ones, ones, ones, ones, jnp.where(jnp.isnan(c0), -jnp.inf, c0), y0, 0, jnp.array(False))
    z, _, _, _, _, _, _, best, yb, _, _ = jax.lax.while_loop(lambda st: ~done(st), body, st)
    return z, yb, best


def squares_bounded_below(Q):
    """Whether a tangent (a lower bound ``w_p >= 2 p x - p^2``) on some square column can change
    the lifted LP's value; see ``square_needs_lower``."""
    return bool(square_needs_lower(Q).any())


def square_needs_lower(Q):
    """``(P,)`` mask of the square columns that get Kelley tangents (lower bounds on ``w``).

    A square column ``w`` does not need them when it enters the objective with coefficient
    <= 0 and every inequality row ``g <= 0`` with coefficient <= 0, appears in no equality or
    equality-RLT row and is no factor of another column. Then raising ``w`` to its secant (the
    one row that bounds it above, ``w <= (l + u) x - l u``) keeps those rows feasible and does
    not raise the objective, and the secant is above ``x^2`` and so above every tangent: some
    LP optimum has all of that column's tangent rows slack, and Kelley rounds only re-solve the
    same LP. This is the case of reverse-convex distance constraints
    ``(x_i - x_j)^2 + (y_i - y_j)^2 >= d^2`` (circle packing), where only the secants of the
    squares bind, and of concave quadratic objectives (st_*).

    The bound-factor rows of ``ineq_rlt`` are not counted, although they hold squares with
    both signs: on the circle models they come from (kall_circles_c6b, c7a, diffcircles_8)
    Kelley rounds for them cost a third of the node rate and did not raise the bound at equal
    node counts. Leaving a tangent out only loosens the LP, so either choice is sound."""
    P = len(Q.pairs)
    n = Q.E.shape[1] - P
    sq = Q.pairs[:, 0] == Q.pairs[:, 1]
    cols = n + np.arange(P)
    g = Q.E[1:1 + Q.n_g][:, cols]
    h = Q.E[1 + Q.n_g:1 + Q.n_g + Q.n_h][:, cols]
    used = (Q.E[0, cols] > 0) | (g > 0).any(0) | (h != 0).any(0) | np.isin(cols, Q.pairs)
    if Q.rlt.size:
        used |= (Q.rlt[:, cols] != 0).any(0)
    return sq & used


def _snap(Q, lb, lo, hi, tol=0.0):
    """``lb`` raised to the objective lattice of ``Q`` (``_lattice``), where it applies. With a
    feasibility tolerance ``tol`` the epigraph row only holds to ``alpha t + P <= tol``, which lets
    ``f`` go ``|s| tol / |alpha|`` below a lattice point: the lattice is shifted down by that.
    Boxes where a variable of the polynomial is outside [0, 1] keep ``lb``."""
    if Q.lattice is None:
        return lb
    g, b0, j, s, alpha, vs, mag = Q.lattice
    b0 = b0 - (abs(s) * abs(tol) / abs(alpha) if alpha else 0.0)
    t = (lb - b0) / g
    k = jnp.ceil(t - 1e-9 * (1.0 + jnp.abs(t)))  # (lb - b0) / g rounded up, less its float error
    lat = b0 + g * k
    # the lattice point, rounded down, less the error of the coefficients (each the double nearest
    # a multiple of g: relative error EPS / 2) summed over monomials in [0, 1]
    lat = lat - 4 * EPS * (abs(b0) + jnp.abs(g * k)) - (len(vs) + 2) * EPS * mag
    binary = jnp.all((lo[vs] >= 0.0) & (hi[vs] <= 1.0)) if len(vs) else True
    lat = jnp.where(jnp.isfinite(lb) & binary, lat, lb)
    if j < 0:  # every value of f at an integer point is a lattice point
        return jnp.maximum(lb, lat)
    # epigraph f = s t + c_0: the minimum over the box is a lattice point or the box bound fb of f;
    # if fb < lb it cannot be the box bound (the minimum is >= lb), so it is a lattice point
    c0 = float(Q.c[0])
    fb = s * (hi[j] if s < 0 else lo[j]) + c0
    err = 4 * EPS * (jnp.abs(fb) + abs(c0))
    return jnp.where(fb + err < lb, jnp.maximum(lb, lat), jnp.maximum(lb, jnp.minimum(lat, fb - err)))


def make_quadratic_bounder(Q, n_kelley=4, ipm_iters=40, tol=0.0, safe=False, n_sep=N_SEP):
    """Return ``bound(lo, hi, cutoff) -> (lb, x_lp, infeasible, lo', hi')`` for one box
    (the contract of ``lp.make_constrained_bounder``) from the lifted LP of ``Q``.

    Rounds after the first add rows at the previous round's LP point: tangents of the squares
    (``n_kelley`` rounds) and the most violated rows of the binary-cut pool (``n_sep`` rounds,
    ``_separate``). The bound is the best certificate over the rounds, raised to the objective
    lattice (``_snap``) when ``Q`` has one."""
    need = square_needs_lower(Q)
    n_kelley = n_kelley if need.any() else 0  # else the tangents are redundant
    n_sep = n_sep if Q.pool_rhs.size else 0
    layouts = {}

    def bound(lo, hi, cutoff=jnp.inf):
        n = lo.size
        if n not in layouts:
            layouts[n] = _layout(Q, n, n_kelley, False, n_sep)
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
        ok = _binary_ok(Q, lo, hi)
        n_rounds = max(n_kelley, L.n_sep)
        prune_at = cutoff
        if Q.lattice is not None:  # f >= the lattice point below the cutoff, plus a margin, snaps to >= cutoff
            lat = Q.lattice
            g, lb0 = lat.step, lat.offset - 2 * abs(tol) * (abs(lat.s / lat.alpha) if lat.alpha else 0.0)
            below = lb0 + g * (jnp.ceil((cutoff - lb0) / g - 1e-9) - 1.0)
            prune_at = jnp.where(jnp.isfinite(cutoff), jnp.minimum(cutoff, below + 1e-6 * g), cutoff)

        def lower(val):
            lbv = f0 + cn * val
            if safe:
                lbv = lbv - SAFE_K * EPS * (N + 3) * (fmag + jnp.abs(f0) + cn * jnp.abs(val))
            return lbv

        settled = None
        if Q.lattice is not None:  # the lattice point is reached: more iterations cannot raise _snap
            def settled(val, primal):
                return _snap(Q, lower(val), lo, hi, tol) >= _snap(Q, f0 + cn * primal, lo, hi, tol)

        I, J = jnp.asarray(Q.pairs[:, 0]), jnp.asarray(Q.pairs[:, 1])
        needj = jnp.asarray(need)

        def round_(carry, k):
            """One LP. A round after the first runs only while the previous round added a row
            that cuts its LP point off (a tangent of a square that needs one, ``need``, or a
            violated pool row) and the box is not pruned: otherwise that point stays feasible,
            hence optimal, with the new rows (no IPM iterations)."""
            vals, b, cols, best, rlo, rhi, vprev, active = carry
            pk = pat._replace(cols=cols)
            A, bu = _unit(pk, vals, b, vlo, scale, safe)
            target = (prune_at - f0) / cn + 1e-9 * (1 + jnp.abs(prune_at) / cn)  # unit-space value that prunes
            z, y, val = _ipm(chat, pk, A, bu, jnp.where(active, ipm_iters, 0), target=target, settled=settled,
                             cert=lambda y: _certificate(pk, A, bu, y, chat, safe)[0])  # fmt: skip
            lb = jnp.where(jnp.isnan(val) | ~finite | ~active, -jnp.inf, lower(val))
            best = jnp.maximum(best, lb)
            _, r = _certificate(pk, A, bu, y, chat, safe)
            klo, khi = reduce_ranges(vlo[:n], vlo[:n] + scale[:n], lb, cn * r[:n] / scale[:n], cutoff)
            rlo = jnp.where(active, jnp.maximum(rlo, klo), rlo)
            rhi = jnp.where(active, jnp.minimum(rhi, khi), rhi)
            v = jnp.where(active, vlo + scale * jnp.clip(z, 0.0, 1.0), vprev)
            cut = jnp.array(False)  # a row added for the next round cuts v off
            if n_kelley:  # Kelley: tangents of the squares at the LP solution (not after the last round)
                gap = jnp.where(needj, v[I] * v[J] - v[n:], -jnp.inf)  # x^2 - w: > 0 cuts the point off
                cut = (k < n_kelley) & jnp.any(gap > 1e-9 * (1.0 + scale[n:]))
                vals, b = jax.lax.cond(k < n_kelley, lambda: tangents(vals, b, v, jnp.minimum(k, n_kelley - 1)),
                                       lambda: (vals, b))  # fmt: skip
            if L.n_sep:  # pool rows violated at the LP solution
                kk = jnp.minimum(k, L.n_sep - 1)
                cols, vals, b, live = jax.lax.cond(
                    k < L.n_sep, lambda: _separate(Q, L, cols, vals, b, v, kk, vlo, vhi, ok & finite, tol, safe),
                    lambda: (cols, vals, b, jnp.array(False)))  # fmt: skip
                cut = cut | live
            active = active & (best < prune_at) & cut
            return (vals, b, cols, best, rlo, rhi, v, active), None

        init = (vals0, b0, jnp.asarray(pat.cols), jnp.array(-jnp.inf, lo.dtype), lo, hi, vlo + 0.5 * scale,
                jnp.array(True))  # fmt: skip
        (_, _, _, best, rlo, rhi, x, _), _ = jax.lax.scan(round_, init, jnp.arange(n_rounds + 1))
        x = x[:n]
        best = _snap(Q, best, lo, hi, tol)
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
