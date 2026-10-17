"""Structural convexity certificates, so convex functions are relaxed by themselves.

McCormick relaxes a convex function operation by operation and can be far from
the function: a 30-fold product of powers ``-0.2 x_1^0.028 ... x_30^0.004``
(a convex signomial on the positive orthant, cvxnonsep_nsig30), a norm
``sqrt(c + sum x_i^2)`` or ``(sum 2^(x_i + x_j))^2`` all get relaxations whose
cuts are far below the function. When a function is convex on the box it is
its own (tightest) convex underestimator, and the LP's cuts at the center, the
axis points and the Kelley points become outer approximation.

The certificate is **structural**, never numerical sampling. ``classify`` walks
the scalarized tape (``fbbt_interp.scalarize``) once at trace time and gives
each scalar slot

* its affine form (if affine), its quadratic form (if of degree <= 2), and its
  monomial form ``c * prod_k a_k ^ p_k`` over affine *atoms* ``a_k`` (valid
  where every atom is positive);
* DCP-style curvature and sign verdicts, each a set of *conditions* under which
  it holds. A condition is a sign requirement on one affine atom over the box
  (``a_k > 0``, ``>= 0``, ``< 0``, ``<= 0``); conditions are evaluated in the
  kernel from ``lo``/``hi`` (``prepare``), and where one fails the ordinary
  relaxation is used unchanged.

Rules (``u``, ``v`` operands; "cvx & nn" = convex and nonnegative etc.):

* sums, differences, negation, scaling by constants: curvature and sign follow
  the signs of the weights;
* ``exp``, ``exp2``, ``expm1``, ``c^u`` (``c > 0``): convex increasing (decreasing
  for ``c < 1``), so convex of convex (of concave for ``c < 1``);
* ``log``: concave increasing on ``u > 0``; ``log`` of a positive-weighted sum
  of ``exp(affine)`` (log-sum-exp) is convex;
* ``sqrt``: concave increasing on ``u >= 0``; ``sqrt`` of ``c0 + sum w_k g_k^2``
  (``c0, w_k >= 0``, ``g_k`` affine, or convex and nonnegative) is a norm,
  hence convex;
* ``u^2`` (``square``, ``integer_pow`` even, ``u*u``): convex if ``u`` is affine,
  cvx & nn, or concave & nonpositive; odd powers ``>= 3``: convex on cvx & nn;
* ``u^p`` with constant non-integer ``p``: convex for ``p > 1`` on cvx & nn, concave
  for ``0 < p < 1`` on concave & nn, convex for ``p < 0`` on concave & pos;
* ``c / u``: convex for ``c > 0`` on concave & pos;
* ``max`` of convex is convex, ``min`` of concave is concave, ``abs(affine)``
  is convex;
* **monomials** ``c * prod a_k^{p_k}`` (all atoms > 0 on the box): with
  ``s = sum p_k``, the product is concave if every ``p_k >= 0`` and
  ``s <= 1``; convex if every ``p_k <= 0``, or if exactly one ``p_k > 0`` and
  ``s >= 1``. Multiplying by ``c < 0`` swaps the two. These are the classical
  conditions for ``prod x_k^{p_k}`` on the positive orthant, composed with
  affine maps (which preserve curvature);
* **quadratics** (output level only): ``f = c + b.x + x^T Q x`` with
  ``lambda_min(Q) >= -tol |Q|``; the relaxation is
  ``f(x) + d sum_i (x_i - lo_i)(x_i - hi_i)`` with ``d >= max(0, -lambda_min) +``
  a margin for the eigenvalue error, i.e. alphaBB with a uniform alpha that
  covers the eigensolver's error. It is a valid underestimator on any box.

Everything else gives no verdict (the slot is "unknown"), which only means
the McCormick/alphaBB relaxation is kept.

``wrap`` combines the certificate with a relaxer: ``cv = max(cv, f)`` where an
output is certified convex on the box (and ``cc = min(cc, f)`` where it is
concave), the same way whole-function alphaBB is combined. Both are valid
convex underestimators, so their max is one. With ``safe``, ``f(x)`` is
replaced by its outward-rounded interval value at the point, lowered further
by a bound on the rounding of the tangent ``J (y - x)`` over the box.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

ENABLED = True  # module switch for ablations (``profile_root.py --set convexity.ENABLED=False``)
# Safe mode bounds f(x)'s rounding with an outward-rounded interval evaluation of the whole tape
# at the point; XLA's compile time for that grows badly with the tape (a 1644-node quadratic
# does not compile in 5 minutes), so above this size safe mode keeps the base relaxation.
SAFE_MAX_NODES = 600
MAX_QUAD_TERMS = 20_000  # stop tracking a slot's quadratic form beyond this many terms
MAX_QUAD_VARS = 400  # eigen test only up to this many variables
QUAD_TOL = 1e-9  # relative eigenvalue margin of the quadratic rule
MAX_ATOM_CELLS = 2_000_000  # atoms x n of the dense condition matrix

Conds = Optional[frozenset]  # None: no verdict; frozenset(): holds unconditionally
E = frozenset()


def _both(*cs):
    out = E
    for c in cs:
        if c is None:
            return None
        out = out | c
    return out


def _first(*cs):
    best = None
    for c in cs:
        if c is not None and (best is None or len(c) < len(best)):
            best = c
    return best


@dataclass
class Info:
    aff: Optional[tuple] = None  # (c0, {i: a})
    quad: Optional[tuple] = None  # (c0, {i: b}, {(i, j): q} with i <= j)
    mono: Optional[tuple] = None  # (c, {atom: p}); valid where every atom is > 0
    cvx: Conds = None
    ccv: Conds = None
    pos: Conds = None  # > 0
    nn: Conds = None  # >= 0
    neg: Conds = None  # < 0
    np_: Conds = None  # <= 0
    sos: Conds = None  # c0 + sum w_k g_k^2 with c0, w_k >= 0, g_k affine or cvx & nn
    lse: bool = False  # sum of w_k exp(affine_k), w_k > 0
    const: Optional[float] = field(default=None)


def _const(v):
    v = float(v)
    i = Info(aff=(v, {}), quad=(v, {}, {}), mono=(v, {}), cvx=E, ccv=E, const=v)
    if v > 0:
        i.pos = i.nn = E
        i.sos = E
    elif v < 0:
        i.neg = i.np_ = E
    else:
        i.nn = i.np_ = E
        i.sos = E
    return i


class _Atoms:
    """Registry of the affine atoms that conditions and monomials refer to."""

    def __init__(self):
        self.key = {}
        self.forms = []

    def get(self, aff):
        c0, d = aff
        k = (c0, tuple(sorted(d.items())))
        if k not in self.key:
            self.key[k] = len(self.forms)
            self.forms.append(aff)
        return self.key[k]


def _signs_of_affine(i, atoms):
    if i.const is not None:
        return
    a = atoms.get(i.aff)
    i.pos, i.nn, i.neg, i.np_ = frozenset({("gt", a)}), frozenset({("ge", a)}), frozenset({("lt", a)}), \
        frozenset({("le", a)})  # fmt: skip


def _scale_quad(q, s):
    c0, b, Q = q
    return (c0 * s, {k: v * s for k, v in b.items()}, {k: v * s for k, v in Q.items()})


def _add_quad(p, q):
    if p is None or q is None:
        return None
    b = dict(p[1])
    for k, v in q[1].items():
        b[k] = b.get(k, 0.0) + v
    Q = dict(p[2])
    for k, v in q[2].items():
        Q[k] = Q.get(k, 0.0) + v
    if len(Q) > MAX_QUAD_TERMS:
        return None
    return (p[0] + q[0], b, Q)


def _deg(q):
    return 2 if q[2] else (1 if q[1] else 0)


def _mul_quad(p, q):
    """Product of two quadratic forms if it has degree <= 2."""
    if p is None or q is None or _deg(p) + _deg(q) > 2:
        return None
    c0p, bp, Qp = p
    c0q, bq, Qq = q
    b = {}
    for k, v in bp.items():
        b[k] = b.get(k, 0.0) + v * c0q
    for k, v in bq.items():
        b[k] = b.get(k, 0.0) + v * c0p
    Q = {}
    for k, v in Qp.items():
        Q[k] = Q.get(k, 0.0) + v * c0q
    for k, v in Qq.items():
        Q[k] = Q.get(k, 0.0) + v * c0p
    for i, vi in bp.items():
        for j, vj in bq.items():
            key = (min(i, j), max(i, j))
            Q[key] = Q.get(key, 0.0) + vi * vj
    if len(Q) > MAX_QUAD_TERMS:
        return None
    return (c0p * c0q, b, Q)


def _as_mono(i, atoms):
    if i.mono is not None:
        return i.mono
    if i.aff is not None:
        return (1.0, {atoms.get(i.aff): 1.0})
    return None


def _mono_mul(m1, m2, sign=1.0):
    c = m1[0] * (m2[0] if sign > 0 else 1.0 / m2[0])
    d = dict(m1[1])
    for k, p in m2[1].items():
        d[k] = d.get(k, 0.0) + sign * p
    return (c, d)


def _apply_mono(i, m):
    """Set ``i``'s monomial form and the curvature/sign it implies."""
    i.mono = m
    c, d = m
    if not d:
        return
    cond = frozenset(("gt", a) for a in d)
    p = np.array(list(d.values()), float)
    s = float(p.sum())
    nz = p[p != 0]
    if nz.size == 0:
        base_cvx = base_ccv = True
    else:
        base_ccv = bool(np.all(nz >= 0) and s <= 1.0)
        base_cvx = bool(np.all(nz <= 0) or (np.sum(nz > 0) == 1 and s >= 1.0))
    if c > 0:
        cv, cc = base_cvx, base_ccv
        i.pos = _first(i.pos, cond)
        i.nn = _first(i.nn, cond)
    elif c < 0:
        cv, cc = base_ccv, base_cvx
        i.neg = _first(i.neg, cond)
        i.np_ = _first(i.np_, cond)
    else:
        cv = cc = True
    if cv:
        i.cvx = _first(i.cvx, cond)
    if cc:
        i.ccv = _first(i.ccv, cond)


def _neg(u):
    o = Info()
    if u.aff is not None:
        o.aff = (-u.aff[0], {k: -v for k, v in u.aff[1].items()})
    if u.quad is not None:
        o.quad = _scale_quad(u.quad, -1.0)
    if u.mono is not None:
        o.mono = (-u.mono[0], u.mono[1])
    o.cvx, o.ccv = u.ccv, u.cvx
    o.pos, o.nn, o.neg, o.np_ = u.neg, u.np_, u.pos, u.nn
    if u.const is not None:
        o.const = -u.const
    return o


def _scale(u, c):
    if c == 0:
        return _const(0.0)
    o = Info()
    if u.aff is not None:
        o.aff = (u.aff[0] * c, {k: v * c for k, v in u.aff[1].items()})
    if u.quad is not None:
        o.quad = _scale_quad(u.quad, c)
    if u.mono is not None:
        o.mono = (u.mono[0] * c, u.mono[1])
    if u.const is not None:
        o.const = u.const * c
    if c > 0:
        o.cvx, o.ccv, o.pos, o.nn, o.neg, o.np_ = u.cvx, u.ccv, u.pos, u.nn, u.neg, u.np_
        o.sos, o.lse = u.sos, u.lse
    else:
        o.cvx, o.ccv, o.pos, o.nn, o.neg, o.np_ = u.ccv, u.cvx, u.neg, u.np_, u.pos, u.nn
    return o


def _add(u, v):
    if u.const is not None and v.const is not None:
        return _const(u.const + v.const)
    o = Info()
    if u.aff is not None and v.aff is not None:
        d = dict(u.aff[1])
        for k, a in v.aff[1].items():
            d[k] = d.get(k, 0.0) + a
        o.aff = (u.aff[0] + v.aff[0], {k: a for k, a in d.items() if a != 0.0})
    o.quad = _add_quad(u.quad, v.quad)
    if u.const == 0.0:
        o.mono = v.mono
    elif v.const == 0.0:
        o.mono = u.mono
    o.cvx = _both(u.cvx, v.cvx)
    o.ccv = _both(u.ccv, v.ccv)
    o.nn = _both(u.nn, v.nn)
    o.np_ = _both(u.np_, v.np_)
    o.pos = _first(_both(u.pos, v.nn), _both(u.nn, v.pos))
    o.neg = _first(_both(u.neg, v.np_), _both(u.np_, v.neg))
    o.sos = _both(u.sos, v.sos)
    o.lse = (u.lse and v.lse) or (u.lse and v.const == 0.0) or (v.lse and u.const == 0.0)
    return o


def _square(u, atoms):
    o = Info()
    o.nn = E
    o.quad = _mul_quad(u.quad, u.quad)
    if u.aff is not None:
        o.cvx = E
        o.sos = E
    else:
        c1 = _both(u.cvx, u.nn)
        c2 = _both(u.ccv, u.np_)
        o.cvx = _first(c1, c2)
        o.sos = c1
    if u.const is not None:
        return _const(u.const**2)
    m = _as_mono(u, atoms)
    if m is not None:
        _apply_mono(o, (m[0] ** 2, {k: 2 * p for k, p in m[1].items()}))
        o.nn = E
    return o


def _int_pow(u, k, atoms):
    if u.const is not None:
        return _const(u.const**k)
    if k == 0:
        return _const(1.0)
    if k == 1:
        return u
    if k == 2:
        return _square(u, atoms)
    o = Info()
    m = _as_mono(u, atoms)
    if m is not None:
        _apply_mono(o, (m[0] ** k, {a: k * p for a, p in m[1].items()}))
    if k % 2 == 0:
        o.nn = E
        o.cvx = _first(o.cvx, E if u.aff is not None else None, _both(u.cvx, u.nn), _both(u.ccv, u.np_))
    elif k > 0:  # odd >= 3: convex increasing on t >= 0, concave increasing on t <= 0
        o.cvx = _first(o.cvx, _both(u.cvx, u.nn))
        o.ccv = _first(o.ccv, _both(u.ccv, u.np_))
        o.nn = _first(o.nn, u.nn)
        o.np_ = _first(o.np_, u.np_)
        o.pos = _first(o.pos, u.pos)
        o.neg = _first(o.neg, u.neg)
    else:  # t^k, k < 0: convex decreasing on t > 0
        o.cvx = _first(o.cvx, _both(u.ccv, u.pos))
        o.pos = _first(o.pos, u.pos)
        o.nn = _first(o.nn, u.pos)
    return o


def _pow(u, p, atoms):
    """``u ** p`` for a constant non-integer ``p`` (NaN for u < 0)."""
    if float(p).is_integer():
        return _int_pow(u, int(p), atoms)
    o = Info()
    m = _as_mono(u, atoms)
    if m is not None and m[0] > 0:
        _apply_mono(o, (m[0] ** p, {a: q * p for a, q in m[1].items()}))
    if p > 1:
        o.cvx = _first(o.cvx, _both(u.cvx, u.nn))
    elif 0 < p < 1:
        o.ccv = _first(o.ccv, _both(u.ccv, u.nn))
    elif p < 0:
        o.cvx = _first(o.cvx, _both(u.ccv, u.pos))
    o.nn = _first(o.nn, u.nn)
    o.pos = _first(o.pos, u.pos)
    return o


def _exp_of(u, slope):
    """``exp(slope * u)`` (slope != 0)."""
    o = Info(pos=E, nn=E)
    o.cvx = u.cvx if slope > 0 else u.ccv
    if u.aff is not None:
        o.lse = True
    return o


def _unary(name, params, u, atoms):
    if name == "neg":
        return _neg(u)
    if name in ("exp", "exp2"):
        return _exp_of(u, 1.0)
    if name == "expm1":
        o = Info(cvx=u.cvx)
        o.nn, o.pos, o.np_, o.neg = u.nn, u.pos, u.np_, u.neg
        return o
    if name == "log":
        o = Info()
        if u.lse:
            o.cvx = E
        o.ccv = _both(u.ccv, u.pos)
        if u.mono is not None and u.aff is None:
            pass
        return o
    if name == "sqrt":
        o = Info()
        o.nn = _first(u.nn, E)  # where defined
        o.pos = u.pos
        o.ccv = _both(u.ccv, u.nn)
        if u.sos is not None:
            o.cvx = u.sos
        m = _as_mono(u, atoms)
        if m is not None and m[0] > 0:
            _apply_mono(o, (math.sqrt(m[0]), {a: 0.5 * q for a, q in m[1].items()}))
            o.ccv = _first(o.ccv, _both(u.ccv, u.nn))
            if u.sos is not None:
                o.cvx = _first(o.cvx, u.sos)
        return o
    if name == "rsqrt":
        return _pow(u, -0.5, atoms)
    if name == "square":
        return _square(u, atoms)
    if name == "integer_pow":
        return _int_pow(u, int(params["y"]), atoms)
    if name == "abs":
        o = Info(nn=E)
        if u.aff is not None:
            o.cvx = E
        o.cvx = _first(o.cvx, _both(u.cvx, u.nn), _both(u.ccv, u.np_))
        o.ccv = _first(_both(u.ccv, u.nn), _both(u.cvx, u.np_))
        return o
    return Info()


def _binary(name, u, v, same, atoms):
    if name in ("add", "add_any"):
        return _add(u, v)
    if name == "sub":
        return _add(u, _neg(v))
    if name == "mul":
        if u.const is not None:
            return _scale(v, u.const)
        if v.const is not None:
            return _scale(u, v.const)
        if same:
            return _square(u, atoms)
        o = Info()
        o.quad = _mul_quad(u.quad, v.quad)
        m1, m2 = _as_mono(u, atoms), _as_mono(v, atoms)
        if m1 is not None and m2 is not None:
            _apply_mono(o, _mono_mul(m1, m2))
        return o
    if name == "div":
        if v.const is not None:
            return _scale(u, 1.0 / v.const) if v.const != 0 else Info()
        o = Info()
        m1, m2 = _as_mono(u, atoms), _as_mono(v, atoms)
        if m1 is not None and m2 is not None and m2[0] != 0:
            _apply_mono(o, _mono_mul(m1, m2, -1.0))
        if u.const is not None and u.const > 0:  # c / t: convex decreasing on t > 0
            o.cvx = _first(o.cvx, _both(v.ccv, v.pos))
            o.pos = _first(o.pos, v.pos)
            o.nn = _first(o.nn, v.pos)
        elif u.const is not None and u.const < 0:
            o.ccv = _first(o.ccv, _both(v.ccv, v.pos))
            o.neg = _first(o.neg, v.pos)
            o.np_ = _first(o.np_, v.pos)
        return o
    if name == "pow":
        if v.const is not None:
            return _pow(u, v.const, atoms)
        if u.const is not None:
            c = u.const
            if c > 0 and c != 1:
                return _exp_of(v, math.log(c))
            if c == 1:
                return _const(1.0)
        return Info()
    if name == "max":
        o = Info(cvx=_both(u.cvx, v.cvx))
        o.nn = _first(u.nn, v.nn)
        o.pos = _first(u.pos, v.pos)
        o.np_ = _both(u.np_, v.np_)
        o.neg = _both(u.neg, v.neg)
        return o
    if name == "min":
        o = Info(ccv=_both(u.ccv, v.ccv))
        o.np_ = _first(u.np_, v.np_)
        o.neg = _first(u.neg, v.neg)
        o.nn = _both(u.nn, v.nn)
        o.pos = _both(u.pos, v.pos)
        return o
    return Info()


class Certificate:
    """Static per-output verdicts of ``classify``.

    ``cv_mask``/``cc_mask`` ``(m,)``: outputs with a convex (concave) verdict;
    ``cv_need``/``cc_need`` ``(m, C)``: the conditions each verdict needs;
    ``cond_kind`` ``(C,)`` (0: > 0, 1: >= 0, 2: < 0, 3: <= 0) and ``cond_atom``
    ``(C,)`` index the rows of ``atom_A`` ``(K, n)``, ``atom_c`` ``(K,)``;
    ``cv_delta``/``cc_delta`` ``(m,)`` the uniform alpha of the quadratic rule
    (0 for exact verdicts), applied to the variables ``cv_vars``/``cc_vars``
    ``(m, n)``.
    """

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __repr__(self):
        return f"Certificate(convex={self.cv_mask.tolist()}, concave={self.cc_mask.tolist()})"


def _quad_shift(quad, n, sign):
    """Uniform alpha making ``sign * quad`` convex, or None if it is not (nearly) convex."""
    _, _, Q = quad
    if not Q:
        return None
    idx = sorted({i for k in Q for i in k})
    if len(idx) > MAX_QUAD_VARS:
        return None
    pos = {v: j for j, v in enumerate(idx)}
    M = np.zeros((len(idx), len(idx)))
    for (i, j), q in Q.items():
        q = sign * q
        if i == j:
            M[pos[i], pos[i]] += q
        else:
            M[pos[i], pos[j]] += 0.5 * q
            M[pos[j], pos[i]] += 0.5 * q
    scale = float(np.abs(M).sum(axis=1).max())  # >= |lambda| for every eigenvalue
    if not np.all(np.isfinite(M)) or scale == 0:
        return None
    lam = float(np.linalg.eigvalsh(M).min())
    # eigvalsh is backward stable: each computed eigenvalue is within ~ k eps ||M|| of a true one
    margin = QUAD_TOL * scale + 64 * len(idx) * float(np.finfo(float).eps) * scale
    if lam < -QUAD_TOL * scale:
        return None
    delta = max(0.0, -lam) + margin
    vars_ = np.zeros(n, bool)
    vars_[idx] = True
    return delta, vars_


def classify(prog):
    """Per-output convexity verdicts of a scalarized ``Program``, or None if there are none.

    Affine outputs are skipped (their relaxation is already exact).
    """
    n = prog.n
    atoms = _Atoms()
    info = {}
    for i in range(n):
        u = Info(aff=(0.0, {i: 1.0}), quad=(0.0, {i: 1.0}, {}), cvx=E, ccv=E)
        _signs_of_affine(u, atoms)
        info[i] = u
    for s, v in enumerate(prog.const_val):
        if not np.isnan(v):
            info[s] = _const(v)
    for o, a, b, out in zip(prog.op, prog.a, prog.b, prog.out):
        name, params_repr, same, ca, cb, n_in, eqn = prog.opcodes[o]
        u = info[int(a)]
        if n_in == 1:
            r = _unary(name, eqn.params, u, atoms)
        else:
            r = _binary(name, u, info[int(b)], same, atoms)
        if r.aff is not None and r.const is None:
            if r.aff[1]:
                r.cvx = r.ccv = E
                _signs_of_affine(r, atoms)
            else:
                r = _const(r.aff[0])
        info[int(out)] = r
    outs = np.concatenate([np.asarray(o).reshape(-1) for o in prog.outs]).astype(int)
    m = outs.size
    verdicts = []
    for s in outs:
        r = info[int(s)]
        if r.aff is not None:
            verdicts.append((None, None, None, None))
            continue
        cv = (r.cvx, 0.0, None) if r.cvx is not None else None
        cc = (r.ccv, 0.0, None) if r.ccv is not None else None
        if cv is None and r.quad is not None:
            q = _quad_shift(r.quad, n, 1.0)
            if q is not None:
                cv = (E, q[0], q[1])
        if cc is None and r.quad is not None:
            q = _quad_shift(r.quad, n, -1.0)
            if q is not None:
                cc = (E, q[0], q[1])
        verdicts.append((cv, cc, None, None))
    if all(v[0] is None and v[1] is None for v in verdicts):
        return None
    conds = sorted({c for v in verdicts for side in v[:2] if side is not None for c in side[0]})
    cidx = {c: k for k, c in enumerate(conds)}
    used = sorted({a for _, a in conds})
    aidx = {a: k for k, a in enumerate(used)}
    if len(used) * n > MAX_ATOM_CELLS:
        return None
    A = np.zeros((len(used), n))
    c0 = np.zeros(len(used))
    for a in used:
        cc_, d = atoms.forms[a]
        c0[aidx[a]] = cc_
        for i, w in d.items():
            A[aidx[a], i] = w
    kinds = {"gt": 0, "ge": 1, "lt": 2, "le": 3}
    C = len(conds)
    cert = dict(
        cv_mask=np.zeros(m, bool), cc_mask=np.zeros(m, bool),
        cv_need=np.zeros((m, C), bool), cc_need=np.zeros((m, C), bool),
        cv_delta=np.zeros(m), cc_delta=np.zeros(m), cv_vars=np.zeros((m, n), bool), cc_vars=np.zeros((m, n), bool),
        cond_kind=np.array([kinds[k] for k, _ in conds], int), cond_atom=np.array([aidx[a] for _, a in conds], int),
        atom_A=A, atom_c=c0,
    )  # fmt: skip
    for k, (cv, cc, _, _) in enumerate(verdicts):
        for side, v in (("cv", cv), ("cc", cc)):
            if v is None:
                continue
            cert[f"{side}_mask"][k] = True
            for c in v[0]:
                cert[f"{side}_need"][k, cidx[c]] = True
            cert[f"{side}_delta"][k] = v[1]
            if v[2] is not None:
                cert[f"{side}_vars"][k] = v[2]
    return Certificate(**cert)


def certificate_of(fn, x):
    """``classify`` of ``fn``'s scalarized tape at the shape of ``x`` (None if it does not scalarize)."""
    from .fbbt import _has_tracer, make_tape
    from .fbbt_interp import scalarize

    if jnp.ndim(x) != 1:
        return None
    with jax.ensure_compile_time_eval():
        tape = make_tape(fn, x)
        if _has_tracer(tape):
            return None
        prog = scalarize(tape, x.shape[0])
        if prog is None:
            return None
        cert = classify(prog)
        if cert is not None:
            cert.size = len(tape.nodes)
        return cert


def conditions(cert, lo, hi):
    """Truth of each condition of ``cert`` on the box (in-kernel)."""
    A = jnp.asarray(cert.atom_A)
    c = jnp.asarray(cert.atom_c)
    eps = float(jnp.finfo(jnp.float64).eps)
    tl = jnp.where(A == 0, 0.0, jnp.minimum(A * lo, A * hi))
    th = jnp.where(A == 0, 0.0, jnp.maximum(A * lo, A * hi))
    n = lo.shape[0]
    # rounding of the sums; zero when the atom is a single variable with unit coefficient and c = 0
    ml = 4 * (n + 2) * eps * (jnp.abs(c) + jnp.sum(jnp.abs(tl), axis=1))
    mh = 4 * (n + 2) * eps * (jnp.abs(c) + jnp.sum(jnp.abs(th), axis=1))
    single = jnp.asarray((np.count_nonzero(cert.atom_A, axis=1) == 1) & (cert.atom_c == 0)
                         & (np.abs(cert.atom_A).max(axis=1, initial=0) == 1))  # fmt: skip
    ml = jnp.where(single, 0.0, ml)
    mh = jnp.where(single, 0.0, mh)
    alo = c + jnp.sum(tl, axis=1) - ml
    ahi = c + jnp.sum(th, axis=1) + mh
    alo = jnp.where(jnp.isnan(alo), -jnp.inf, alo)
    ahi = jnp.where(jnp.isnan(ahi), jnp.inf, ahi)
    k = jnp.asarray(cert.cond_kind)
    la, ha = alo[jnp.asarray(cert.cond_atom)], ahi[jnp.asarray(cert.cond_atom)]
    return jnp.select([k == 0, k == 1, k == 2], [la > 0, la >= 0, ha < 0], ha <= 0)


def prepare(cert, lo, hi):
    """``(ok_cv, ok_cc)`` per output on the box."""
    if cert.cond_kind.size:
        t = conditions(cert, lo, hi)
        ok_cv = jnp.all(jnp.where(jnp.asarray(cert.cv_need), t[None, :], True), axis=1)
        ok_cc = jnp.all(jnp.where(jnp.asarray(cert.cc_need), t[None, :], True), axis=1)
    else:
        ok_cv = ok_cc = jnp.ones(cert.cv_mask.shape, bool)
    finite = jnp.all(jnp.isfinite(lo)) & jnp.all(jnp.isfinite(hi))
    ok_cv = ok_cv & jnp.asarray(cert.cv_mask) & finite
    ok_cc = ok_cc & jnp.asarray(cert.cc_mask) & finite
    return ok_cv, ok_cc


def values(flat, cert, x, lo, hi, data, safe):
    """``(cv, cc)`` candidates: ``f(x) -/+ delta * sum (x - lo)(hi - x)`` where certified, -inf/+inf elsewhere."""
    from .relax import MC, SAFE_K, eval_jaxpr_mc, is_mc

    ok_cv, ok_cc = data
    fx = flat(x)
    q = (x - lo) * (x - hi)  # <= 0 on the box
    scv = jnp.asarray(cert.cv_delta) * jnp.sum(jnp.where(jnp.asarray(cert.cv_vars), q[None, :], 0.0), axis=1)
    scc = jnp.asarray(cert.cc_delta) * jnp.sum(jnp.where(jnp.asarray(cert.cc_vars), q[None, :], 0.0), axis=1)
    cv, cc = fx + scv, fx - scc
    if safe:
        from .alphabb import SAFE_INTERVAL_CTX

        sg = jax.lax.stop_gradient
        # the corrections are constants to the LP's jacfwd: compute them from a stopped x so no
        # tangents are traced through the interval evaluation and the Jacobian
        xs = sg(x)
        closed = jax.make_jaxpr(flat)(xs)
        (iv,) = eval_jaxpr_mc(closed.jaxpr, closed.consts, [MC(xs, xs, xs, xs)], SAFE_INTERVAL_CTX)
        iv = iv if is_mc(iv) else MC(iv, iv, iv, iv)
        flo, fhi = jnp.broadcast_to(iv.lo, fx.shape), jnp.broadcast_to(iv.hi, fx.shape)
        J = sg(jax.jacfwd(flat)(xs))
        eps = float(jnp.finfo(jnp.float64).eps)
        n = x.size
        # the cuts use the float gradient: bound its effect over the box by a relative error of the tangent
        pad = 16 * SAFE_K * eps * (n + 2)
        tang = jnp.abs(J) @ (hi - lo)
        sh = jnp.abs(scv) + jnp.abs(scc)
        cv = cv + sg(flo - fx) - sg(pad * (jnp.abs(fx) + tang + sh))
        cc = cc + sg(fhi - fx) + sg(pad * (jnp.abs(fx) + tang + sh))
    cv = jnp.where(ok_cv & jnp.isfinite(cv), cv, -jnp.inf)
    cc = jnp.where(ok_cc & jnp.isfinite(cc), cc, jnp.inf)
    return cv, cc


def wrap(fn, base, safe=False):
    """``base`` relaxer of ``fn`` tightened by the convexity certificate (a ``univariate.Relaxer``)."""
    from .fbbt import tape_cache
    from .relax import MC, tighten
    from .univariate import Relaxer

    if not ENABLED:
        return base

    def flat(x):
        return jnp.reshape(fn(x), (-1,))

    get = tape_cache(lambda x: (_Holder(certificate_of(flat, x)),))

    def cert_of(x):
        cert = get(x)[0].cert
        if safe and cert is not None and cert.size > SAFE_MAX_NODES:
            return None
        return cert

    def prep(lo, hi):
        cert = cert_of(lo)
        return base.prepare(lo, hi), (prepare(cert, lo, hi) if cert is not None else None)

    def call(x, lo, hi, data):
        m = base.call(x, lo, hi, data[0])
        if data[1] is None:
            return m
        cert = cert_of(x)
        shp = jnp.shape(m.cv)
        vcv, vcc = values(flat, cert, x, lo, hi, data[1], safe)
        vcv, vcc = jnp.reshape(vcv, shp), jnp.reshape(vcc, shp)
        cv = jnp.where(jnp.isnan(m.cv), vcv, jnp.maximum(m.cv, vcv))
        cc = jnp.where(jnp.isnan(m.cc), vcc, jnp.minimum(m.cc, vcc))
        return tighten(MC(m.lo, m.hi, cv, cc))

    return Relaxer(prep, call)


class _Holder:
    """Lets ``tape_cache`` (which checks ``out[0]`` for tracers) hold a certificate."""

    nodes = ()
    outs = ()

    def __init__(self, cert):
        self.cert = cert
