"""The constrained node bound with one compiled kernel for many problems (the tape is data).

Second increment of the kernel redesign (``fbbt_generic`` is the first). ``lp.make_constrained_bounder``
relaxes ``f``, ``ineq`` and ``eq`` by McCormick (plus, by default, univariate hulls, alphaBB and
convexity certificates), takes subgradient cuts at the box center, the 2n axis points and the
Kelley points, and certifies an interior-point LP over them by weak duality. Everything after the
relaxation depends on the problem only through array shapes; the relaxation is a pass over the
scalarized tape. Here the tape is an argument:

* the McCormick relaxation is ``relax_interp``'s, evaluated row by row with a ``lax.switch`` over
  a fixed opcode table (``fbbt_generic``'s, plus the fused ``x log x`` row), carrying forward-mode
  subgradients with respect to every variable (``relax_interp._branches`` with gradients: the
  same rules, so the same values and subgradients);
* the parts ``[cv_f, cv_g..., cv_h..., -cc_h...]`` are output slots with a sign, padded to a
  bucket; ``_cut_rows``'s selection (no cut along a variable a part is linear in) is a mask, so
  dropped cuts are dead rows of the LP instead of missing ones;
* the LP, its certificate, the Kelley rounds and the duality-based range reduction are
  ``make_constrained_bounder``'s (``lp.ipm``, ``lp._cut_const``, ``lp.reduce_ranges``,
  ``lp._kelley_go``), on the padded variables (a padding variable is fixed at 0: its cut
  coefficients are 0).

This is McCormick only (no univariate hulls, alphaBB, convexity certificates or cut pool), so its
bounds can be weaker than the default bounder's; any certificate is a valid bound. The kernel
depends only on the buckets, ``safe``, ``n_kelley`` and the opcode table, so it is compiled once
per bucket and kept on disk like ``fbbt_generic``'s.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import fbbt_generic as fg
from .fbbt_interp import Program, _live_rows, scalarize
from .lp import EPS, _cut_const, _kelley_go, _safe_costs, ipm, reduce_ranges
from .relax import SAFE_K, Ctx
from .relax_interp import _branches, fuse_xlogx

F64 = jnp.float64


@functools.lru_cache(maxsize=None)
def table(tier):
    """``fbbt_generic.table(tier)`` plus the fused ``x log x`` row (``relax_interp.fuse_xlogx``)."""
    mul = fg._canonical_eqns()[("mul", None)]
    return fg.table(tier) + (("xlogx", None, False, False, False, 1, mul),)


def _kind_of(opcode, tier):
    """Index in ``table(tier)`` of a ``scalarize`` (or ``fuse_xlogx``) opcode, or None."""
    t = table(tier)
    if opcode[0] == "xlogx":
        return len(t) - 1
    k = fg._kind_of(opcode)
    if k is None:
        return None
    key = fg.kinds()[k][:6]
    idx = {e[:6]: i for i, e in enumerate(t)}
    return idx.get(key)


class Data(NamedTuple):
    op: np.ndarray  # (K,) opcode per row (``table(tier)``)
    a: np.ndarray
    b: np.ndarray
    out: np.ndarray
    rows: np.ndarray  # () real rows
    cval: np.ndarray  # (S,)
    is_c: np.ndarray  # (S,)
    part_slot: np.ndarray  # (P,) slot of each part [cv_f, cv_g..., cv_h..., -cc_h...]
    part_sign: np.ndarray  # (P,) +1: cv of the slot, -1: -cc of the slot
    part_live: np.ndarray  # (P,) a real part (padding parts give dead rows)
    nl: np.ndarray  # (P, N) the variables each part is nonlinear in (``lp._nonlinear_vars``)


jax.export.register_namedtuple_serialization(Data, serialized_name="jaxglobal.bound_generic.Data")


class Generic(NamedTuple):
    data: Data
    tier: str
    n: int
    N: int


def encode(prog: Program, n, ineq, eq, nl):
    """``prog`` (scalarized, live rows) as a ``Generic`` bounder program, or None if an opcode is
    outside the table. ``nl``: ``lp._nonlinear_vars`` (``(parts, n)``), or None (all nonlinear)."""
    prog = fuse_xlogx(prog)
    for tier in ("core", "full"):
        kind = [_kind_of(oc, tier) for oc in prog.opcodes]
        if all(k is not None for k in kind):
            break
    else:
        return None
    K, S0 = len(prog.op), len(prog.const_val)
    N = fg._bucket(n, least=8)
    shift = N - n

    def renum(s):
        s = np.asarray(s, np.int64)
        return np.where(s < n, s, s + shift)

    Kb = fg._bucket(K, least=32)
    S = 2 * Kb if S0 + shift <= 2 * Kb else fg._bucket(S0 + shift)
    cval, is_c = np.zeros(S), np.ones(S, bool)
    is_c[:N] = False
    cv = np.asarray(prog.const_val, np.float64)
    other = np.arange(n, S0)
    is_c[other + shift] = ~np.isnan(cv[other])
    cval[other + shift] = np.nan_to_num(cv[other], nan=0.0)
    op, a, b, out = (np.zeros(Kb, np.int32) for _ in range(4))
    op[:K] = np.asarray(kind, np.int32)[prog.op]
    a[:K], b[:K], out[:K] = renum(prog.a), renum(prog.b), renum(prog.out)
    # parts: f (one slot), g, h (cv), h (-cc)
    groups = [renum(np.ravel(o)) for o in prog.outs]
    if groups[0].size != 1:
        return None  # a vector objective: not a node bound
    slots, signs = [groups[0]], [np.ones(1)]
    gi = 1
    if ineq is not None:
        slots.append(groups[gi])
        signs.append(np.ones(groups[gi].size))
        gi += 1
    if eq is not None:
        slots += [groups[gi], groups[gi]]
        signs += [np.ones(groups[gi].size), -np.ones(groups[gi].size)]
    slots, signs = np.concatenate(slots), np.concatenate(signs)
    P = fg._bucket(slots.size, least=4)
    part_slot, part_sign, part_live = np.zeros(P, np.int32), np.ones(P), np.zeros(P, bool)
    part_slot[: slots.size], part_sign[: slots.size], part_live[: slots.size] = slots, signs, True
    nlm = np.zeros((P, N), bool)
    if nl is None or nl.shape != (slots.size, n):
        nlm[: slots.size, :n] = True
    else:
        nlm[: slots.size, :n] = nl
    return Generic(Data(op, a, b, out, np.asarray(K, np.int32), cval, is_c, part_slot, part_sign, part_live, nlm),
                   tier, n, N)  # fmt: skip


@functools.lru_cache(maxsize=None)
def _rules(safe, tier):
    t = table(tier)
    stand = Program(None, None, None, None, None, 0, [(k[0], None) + k[2:] for k in t], [])
    return _branches(stand, Ctx(safe=safe), True)


def _sweep(x, lo, hi, d: Data, safe, tier):
    """McCormick of every slot at ``x`` on the box: ``(V, G)``, ``V`` (S, 4) lo, hi, cv, cc and
    ``G`` (S, 2, N) the subgradients of cv and cc."""
    fwd = _rules(safe, tier)
    N = lo.shape[0]
    cval, is_c = d.cval, d.is_c
    S = cval.shape[0]
    base = jnp.where(is_c, cval, 0.0)
    V = jnp.stack([base.at[:N].set(lo), base.at[:N].set(hi), base.at[:N].set(x), base.at[:N].set(x)], axis=1)
    eye = jnp.eye(N, dtype=F64)
    G = jnp.zeros((S, 2, N), F64).at[:N, 0].set(eye).at[:N, 1].set(eye)

    def step(k, carry):
        V, G = carry
        a, b, o = d.a[k], d.b[k], d.out[k]
        va, vb = V[a], V[b]
        T = jnp.concatenate([G[a], G[b]])  # (4, N): d(cv_a, cc_a, cv_b, cc_b) / dx
        r, J = jax.lax.switch(d.op[k], fwd, va[0], va[1], va[2], va[3], vb[0], vb[1], vb[2], vb[3], T)
        return V.at[o].set(r), G.at[o].set(J)

    return jax.lax.fori_loop(0, d.rows, step, (V, G))


def _bound_one(lo, hi, cutoff, tol, d: Data, safe, tier, n_kelley, ipm_iters):
    """``make_constrained_bounder``'s ``bound`` (no pool) on the padded box: ``(lb, x_lp, infeasible,
    lo', hi')``."""
    N = lo.shape[0]
    w = hi - lo
    c = 0.5 * (lo + hi)
    sign, live = d.part_sign, d.part_live

    def cuts_at(p):
        V, G = _sweep(p, lo, hi, d, safe, tier)
        Vp, Gp = V[d.part_slot], G[d.part_slot]
        v = jnp.where(sign > 0, Vp[:, 2], -Vp[:, 3])
        J = jnp.where((sign > 0)[:, None], Gp[:, 0], -Gp[:, 1])
        coef = J * w
        const = _cut_const(v, J, lo, p, coef, safe)
        ok = live & jnp.isfinite(const) & jnp.all(jnp.isfinite(coef), axis=1)
        plo = jnp.where(sign > 0, Vp[:, 0], -Vp[:, 1])  # a lower bound of each part on the box
        return jnp.where(ok, const, -jnp.inf), jnp.where(ok[:, None], coef, 0.0), (Vp[0, 0], Vp[0, 1], plo)

    axis = jnp.concatenate([jnp.eye(N), -jnp.eye(N)]) * (w / 4)
    pts0 = jnp.concatenate([c[None], c + axis])
    const0, coef0, bounds = jax.vmap(cuts_at)(pts0)
    flo, fhi, plo = bounds[0][0], bounds[1][0], bounds[2][0]
    infeasible = jnp.any(live[1:] & (plo[1:] > tol))  # (a constraint part's lower bound above tol)

    # _cut_rows as a mask: part k keeps the center, the axis points of the variables it is
    # nonlinear in and (unless affine) the Kelley points; a dropped row is a dead row
    P0 = const0.shape[0]
    keep0 = jnp.concatenate([jnp.ones((1, d.nl.shape[0]), bool), d.nl.T, d.nl.T])
    const0 = jnp.where(keep0, const0, -jnp.inf)
    keep_k = jnp.any(d.nl, axis=1)

    cmin = const0[:, 0] + jnp.sum(jnp.minimum(coef0[:, 0], 0), axis=1)
    cmax = const0[:, 0] + jnp.sum(jnp.maximum(coef0[:, 0], 0), axis=1)
    if safe:
        cmin = cmin - SAFE_K * N * EPS * jnp.sum(jnp.abs(coef0[:, 0]), axis=1)
        cmax = cmax + SAFE_K * N * EPS * jnp.sum(jnp.abs(coef0[:, 0]), axis=1)
    tlo = jnp.maximum(flo, jnp.max(jnp.where(jnp.isfinite(cmin), cmin, -jnp.inf)))
    thi = jnp.minimum(fhi, jnp.max(jnp.where(jnp.isfinite(cmax), cmax, -jnp.inf)))
    thi = jnp.where(jnp.isfinite(thi) & (thi > tlo), thi, tlo + 1.0)
    T = thi - tlo
    bad_t = ~jnp.isfinite(tlo)
    tlo_s = jnp.where(bad_t, 0.0, tlo)

    def rows(const, coef):
        tcol = jnp.zeros(const.shape + (1,)).at[..., 0, 0].set(-T)
        A = jnp.concatenate([coef, tcol], axis=-1).reshape(-1, N + 1)
        bvec = jnp.concatenate([tlo_s - const[..., :1], tol - const[..., 1:]], axis=-1).reshape(-1)
        mag = jnp.concatenate([jnp.abs(tlo_s) + jnp.abs(const[..., :1]), jnp.abs(tol) + jnp.abs(const[..., 1:])],
                              axis=-1).reshape(-1)  # fmt: skip
        ok = jnp.isfinite(bvec)
        nrm = jnp.linalg.norm(A, axis=1)
        ok &= nrm > 0
        nrm = jnp.where(ok, nrm, 1.0)
        A = jnp.where(ok[:, None], A / nrm[:, None], 0.0)
        bvec = jnp.where(ok, bvec / nrm, 1.0)
        if safe:
            m = SAFE_K * EPS * (jnp.abs(bvec) + jnp.abs(A).sum(axis=1) + jnp.where(ok, mag / nrm, 0.0))
            bvec = jnp.where(ok, bvec + m, bvec)
        return A, bvec

    cobj = jnp.zeros(N + 1).at[N].set(1.0)

    def dual(A, bvec, y):
        y = jnp.where(jnp.isfinite(y), jnp.maximum(y, 0.0), 0.0)
        s_obj = -(A[:, N] @ y)
        y = y * jnp.where(s_obj > 1.0, 1.0 / s_obj, 1.0) ** (A[:, N] < 0)
        coef = cobj + A.T @ y
        val = jnp.sum(jnp.minimum(coef[:N], 0.0)) - y @ bvec
        err = (A.shape[0] + N + 2) * EPS * (jnp.sum(1.0 + jnp.abs(A).T @ y) + jnp.abs(bvec) @ y)
        if safe:
            err = SAFE_K * err
            return (tlo_s + T * (val - err) - SAFE_K * EPS * (jnp.abs(tlo_s) + T * (jnp.abs(val) + err)),
                    _safe_costs(coef[:N], A[:, :N], y))  # fmt: skip
        return tlo_s + T * (val - err), _safe_costs(coef[:N], A[:, :N], y)

    const = jnp.concatenate([const0, jnp.full((n_kelley,) + const0.shape[1:], -jnp.inf)])
    coef = jnp.concatenate([coef0, jnp.zeros((n_kelley,) + coef0.shape[1:])])

    def solve(const, coef, best, rlo, rhi):
        A, bvec = rows(const, coef)
        nA = A.shape[0]
        z, y, lb = ipm(cobj, A, bvec, ipm_iters, cert=lambda y: dual(A, bvec, y[:nA])[0], box=True, target=cutoff)
        lb = jnp.where(jnp.isnan(lb) | bad_t, -jnp.inf, lb)
        best = jnp.maximum(best, lb)
        _, r = dual(A, bvec, y[:nA])
        klo, khi = reduce_ranges(lo, hi, lb, T * r / jnp.where(w > 0, w, 1.0), cutoff)
        return best, jnp.maximum(rlo, klo), jnp.minimum(rhi, khi), jnp.clip(z[:N], 0.0, 1.0)

    best0 = jnp.where(bad_t, -jnp.inf, tlo)
    best, rlo, rhi, u = solve(const, coef, best0, lo, hi)

    def kelley(carry):
        k, const, coef, best, rlo, rhi, u, _ = carry
        ck, gk, _ = cuts_at(lo + w * u)
        ck = jnp.where(keep_k, ck, -jnp.inf)
        const, coef = const.at[P0 + k].set(ck), coef.at[P0 + k].set(gk)
        prev = best
        best, rlo, rhi, u = solve(const, coef, best, rlo, rhi)
        return k + 1, const, coef, best, rlo, rhi, u, _kelley_go(prev, best, cutoff)

    init = (0, const, coef, best, rlo, rhi, u, _kelley_go(best0, best, cutoff, first=True))
    _, _, _, best, rlo, rhi, u, _ = jax.lax.while_loop(lambda c: (c[0] < n_kelley) & c[7], kelley, init)
    lb = jnp.where(infeasible, jnp.inf, best)
    x = lo + w * u
    return lb, x, infeasible, rlo, rhi, _value_at(x, d, tol, safe, tier)


def _value_at(x, d: Data, tol, safe, tier):
    """``f(x)`` if ``x`` satisfies the constraints to ``tol``, else inf: an incumbent candidate (the
    caller re-checks it). The relaxation's sweep on the point box ``[x, x]``, where every
    interval rule gives the function's value."""
    V, _ = _sweep(x, x, x, d, safe, tier)
    Vp = V[d.part_slot]
    val = jnp.where(d.part_sign > 0, Vp[:, 0], -Vp[:, 1])  # f, g..., h..., -h...
    feas = jnp.all(jnp.where(d.part_live & (jnp.arange(val.shape[0]) > 0), val <= tol, True))
    f = val[0]
    return jnp.where(feas & jnp.isfinite(f), f, jnp.inf)


@functools.lru_cache(maxsize=None)
def kernel(safe=False, tier="full", n_kelley=4, ipm_iters=40):
    """Batched ``(lo (B, N), hi (B, N), cutoff, tol, data) -> (lb, x, infeasible, lo', hi', f(x))``
    (``f(x)`` inf where ``x`` is not feasible)."""

    def run(lo, hi, cutoff, tol, data):
        one = functools.partial(_bound_one, safe=safe, tier=tier, n_kelley=n_kelley, ipm_iters=ipm_iters)
        return jax.vmap(one, in_axes=(0, 0, None, None, None))(lo, hi, cutoff, tol, data)

    return jax.jit(run)


_EXES = {}


def executable(safe, tier, n_kelley, ipm_iters, args):
    """The compiled kernel for ``args``' shapes (``fbbt_generic.executable``'s caching)."""
    from . import kcache

    specs = jax.tree_util.tree_map(lambda a: jax.ShapeDtypeStruct(jnp.shape(a), jnp.result_type(a)), args)
    key = (bool(safe), tier, n_kelley, ipm_iters,
           tuple((s.shape, str(s.dtype)) for s in jax.tree_util.tree_leaves(specs)))  # fmt: skip
    if key in _EXES:
        return _EXES[key]
    fn = kernel(safe, tier, n_kelley, ipm_iters)
    lowered = None
    if kcache.cache_root() is not None:
        import hashlib

        import jaxlib

        dev = jax.devices()[0]
        fp = "bound_generic-" + hashlib.sha256(repr((kcache._source_hash(), jax.__version__, jaxlib.__version__,
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


class GenericBounder:
    """``bound(lo, hi, cutoff) -> (lb, x_lp, infeasible, lo', hi', fx)`` on batches ``(B, n)``;
    ``fx``: ``f(x_lp)`` when ``x_lp`` is feasible, else inf."""

    def __init__(self, g: Generic, tol, safe=False, n_kelley=4, ipm_iters=40):
        self.g, self.safe, self.n_kelley, self.ipm_iters = g, bool(safe), int(n_kelley), int(ipm_iters)
        self.tol = jnp.asarray(tol, F64)
        self.data = Data(*(jnp.asarray(v) for v in g.data))

    def __call__(self, lo, hi, cutoff=jnp.inf):
        n, N = self.g.n, self.g.N
        lo, hi = jnp.asarray(lo, F64), jnp.asarray(hi, F64)
        z = jnp.zeros(lo.shape[:-1] + (N - n,), F64)
        args = (jnp.concatenate([lo, z], -1), jnp.concatenate([hi, z], -1), jnp.asarray(cutoff, F64), self.tol, self.data)
        lb, x, inf, rlo, rhi, fx = executable(self.safe, self.g.tier, self.n_kelley, self.ipm_iters, args)(*args)
        return lb, x[..., :n], inf, rlo[..., :n], rhi[..., :n], fx


def make_generic_bounder(f, ineq=None, eq=None, n=None, tol=0.0, safe=False, n_kelley=4, ipm_iters=40):
    """The shared-kernel McCormick bounder for the problem, or None (tape not scalar, opcode
    outside the table, vector objective)."""
    from .fbbt import _has_tracer, joint_tape
    from .lp import _nonlinear_vars

    with jax.ensure_compile_time_eval():
        tape = joint_tape(f, ineq, eq, np.zeros(n))
    if _has_tracer(tape):
        return None
    prog = scalarize(tape, n)
    if prog is None:
        return None
    prog = _live_rows(prog)
    if len(prog.op) == 0:
        return None
    g = encode(prog, n, ineq, eq, _nonlinear_vars(f, ineq, eq, n))
    if g is None:
        return None
    return GenericBounder(g, tol, safe, n_kelley, ipm_iters)
