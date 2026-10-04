"""Lifted (extended-space) LP relaxation built from the jaxpr.

The reduced-space bounders in ``bnb.py`` / ``lp.py`` linearize the composed
McCormick relaxation of each whole function at a few points. Here every
intermediate value of the jaxpr that is not an affine function of the
variables gets its own LP variable ``w`` instead, as mature global solvers do
(BARON, SCIP, Couenne):

* affine primitives (add, sub, slice, reshape, mul or ``dot_general`` by a
  constant, ``select_n`` on a constant predicate, ...) are composed exactly:
  every value is carried as an affine expression ``E v + c`` of the LP
  variables ``v = (x, w)``;
* every other primitive ``w = op(e_1, ..., e_r)`` is linked to its inputs by
  cuts from **its own** McCormick rule, evaluated with point relaxations
  ``MC(lo_i, hi_i, p_i, p_i)`` of the inputs: ``cv(p)`` is convex and
  ``cc(p)`` concave in ``p``, so for any ``p`` in the input box

      w >= cv(p) + J_cv (e - p),        w <= cc(p) + J_cc (e - p)

  hold at every feasible point. For a bilinear product, cuts near the four
  corners of the input box are exactly the four McCormick facets.

Input and output ranges ``[lo_i, hi_i]`` come from FBBT on the same tape
(``fbbt.propagate``), so the facets use tightened intermediate bounds. Cuts are
generated at a fixed pattern of points per operation, then (Kelley) at the LP
solution. The LP is solved with the same dense IPM as ``lp.py`` and the bound is
the same weak-duality certificate (with its rounding-error correction), so it
is valid for any dual vector. Reduced costs of the ``x`` columns give
duality-based range reduction.
"""

from __future__ import annotations

import itertools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .fbbt import Const, joint_tape, output_bounds, propagate
from .lp import EPS, _parts_fn, _safe_costs, ipm, reduce_ranges
from .relax import MC, RULES, STRUCTURAL, Ctx, Unknown, is_mc


AFFINE = set(STRUCTURAL) | {"neg", "sub"}


def _is_affine(eqn, ins_mc):
    name = eqn.primitive.name
    if name in AFFINE:
        return True
    if name in ("mul", "dot_general"):
        return sum(ins_mc) == 1
    if name == "div":
        return ins_mc[0] and not ins_mc[1]
    if name == "select_n":
        return not ins_mc[0]
    if name == "integer_pow":
        return eqn.params["y"] == 1
    return False


class Lift(NamedTuple):
    node: object  # fbbt.Node
    mc_pos: tuple  # positions of the MC inputs
    in_slots: tuple  # their slots
    out_slots: tuple
    offset: int  # first LP column of the outputs
    size: int  # total number of output elements


def _pattern(r):
    """Relative points in [0, 1]^r at which cuts are generated first."""
    if r == 1:
        return np.array([[0.5], [0.02], [0.98], [0.25], [0.75]])
    if r <= 3:
        corners = np.array(list(itertools.product([0.02, 0.98], repeat=r)))
        return np.concatenate([np.full((1, r), 0.5), corners])
    ax = np.concatenate([np.eye(r), -np.eye(r)]) * 0.45
    return np.concatenate([np.full((1, r), 0.5), 0.5 + ax])


def make_lifted_bounder(f, ineq=None, eq=None, n_kelley=3, ipm_iters=40, fbbt_rounds=3, tol=0.0,
                        univariate=True, alphabb=True, multivariate=False):
    """Return ``bound(lo, hi, cutoff) -> (lb, x_lp, infeasible, lo', hi')`` for one box.

    Same contract as ``lp.make_constrained_bounder`` (``ineq``/``eq`` may be
    ``None``), using the lifted LP. With ``univariate``/``alphabb`` the LP also
    gets the reduced-space cuts of the whole functions (McCormick tightened by
    univariate hull envelopes and alphaBB) on the ``x`` columns, so it is at
    least as tight as both formulations.
    """
    prepare_parts, parts, (rel_f, _, _) = _parts_fn(f, ineq, eq, univariate, alphabb, multivariate)
    n_f = 1

    def bound(lo, hi, cutoff=jnp.inf):
        n = lo.size
        dt = lo.dtype
        tape = joint_tape(f, ineq, eq, lo)
        ob = output_bounds(tape, ineq, eq, jnp.asarray(cutoff, dt), tol)
        vals, empty = propagate(tape, lo, hi, ob, fbbt_rounds)

        # --- structure: which nodes are lifted, and the LP columns ---------
        lifts, N = [], n
        mc_slot = {s for s, v in vals.items() if is_mc(v)}
        var_slot = {s for s, v in vals.items() if isinstance(v, (MC, Unknown))}
        for node in tape.nodes:
            ins_mc = tuple(not isinstance(a, Const) and a in mc_slot for a in node.ins)
            ins_var = tuple(not isinstance(a, Const) and a in var_slot for a in node.ins)
            outs_mc = [s for s in node.outs if s in mc_slot]
            if not any(ins_mc) or not outs_mc or _is_affine(node.eqn, ins_var):
                continue
            size = sum(int(np.prod(tape.shapes[s][0])) for s in node.outs)
            pos = tuple(i for i, m in enumerate(ins_mc) if m)
            lifts.append(Lift(node, pos, tuple(node.ins[i] for i in pos), node.outs, N, size))
            N += size
        lifted = {id(L.node): L for L in lifts}

        # --- column bounds: x from the box, w from FBBT ------------------
        clo, chi = [vals[0].lo], [vals[0].hi]
        for L in lifts:
            for s in L.out_slots:
                clo.append(jnp.reshape(vals[s].lo, (-1,)))
                chi.append(jnp.reshape(vals[s].hi, (-1,)))
        clo, chi = jnp.concatenate(clo), jnp.concatenate(chi)
        finite = jnp.all(jnp.isfinite(clo) & jnp.isfinite(chi))
        clo = jnp.where(jnp.isfinite(clo), clo, 0.0)
        chi = jnp.where(jnp.isfinite(chi), chi, 1.0)
        chi = jnp.maximum(chi, clo)
        scale = jnp.maximum(chi - clo, 1e-9 * (1 + jnp.abs(clo)))  # v = clo + scale * u

        # --- affine expressions of every MC slot: (E, c), E has a trailing N axis
        aff = {0: (jnp.eye(n, N, dtype=dt), jnp.zeros(n, dt))}

        def value(a, k):  # k = 0: E, 1: c; constants are (0, value)
            if isinstance(a, Const):
                v = jnp.asarray(a.val)
                return jnp.zeros(v.shape + (N,), dt) if k == 0 else v
            if a in aff:
                return aff[a][k]
            return vals[a]  # Unknown / constant computed by propagation

        for node in tape.nodes:
            L = lifted.get(id(node))
            if L is not None:
                k = L.offset
                for s in node.outs:
                    m = int(np.prod(tape.shapes[s][0]))
                    E = jnp.zeros((m, N), dt).at[jnp.arange(m), k + jnp.arange(m)].set(1.0)
                    aff[s] = (jnp.reshape(E, tape.shapes[s][0] + (N,)), jnp.zeros(tape.shapes[s][0], dt))
                    k += m
                continue
            if not any(s in mc_slot for s in node.outs):
                continue
            eqn = node.eqn
            mc_in = [not isinstance(a, Const) and a in aff for a in node.ins]

            def op(*args, eqn=eqn):
                out = eqn.primitive.bind(*args, **eqn.params)
                return out if eqn.primitive.multiple_results else [out]

            cs = [value(a, 1) for a in node.ins]
            c_out = op(*cs)
            zeros = [jnp.zeros_like(c) if m else c for c, m in zip(cs, mc_in)]
            c0 = op(*zeros)

            def col(*cols, zeros=zeros, mc_in=mc_in, c0=c0, op=op):
                args = list(zeros)
                it = iter(cols)
                for i, m in enumerate(mc_in):
                    if m:
                        args[i] = next(it)
                return [o - z for o, z in zip(op(*args), c0)]

            Es = [value(a, 0) for a, m in zip(node.ins, mc_in) if m]
            E_out = jax.vmap(col, in_axes=-1, out_axes=-1)(*Es)
            for s, E, c in zip(node.outs, E_out, c_out):
                if s in mc_slot:
                    aff[s] = (E, c)

        def flat(a):
            E, c = aff[a]
            return jnp.reshape(E, (-1, N)), jnp.reshape(c, (-1,))

        # --- cuts of one lifted node at input points ------------------------
        def node_cuts(L, pts):
            """Rows (A, b) of  A v <= b  for cv/cc cuts at the input points ``pts``."""
            eqn = L.node.eqn
            ins = [vals[a] if not isinstance(a, Const) and a in mc_slot else value(a, 1) for a in L.node.ins]

            def relax_out(*ps):
                args = list(ins)
                for i, p in zip(L.mc_pos, ps):
                    args[i] = MC(ins[i].lo, ins[i].hi, p, p)
                out = RULES[eqn.primitive.name](Ctx(multivariate=multivariate), eqn, *args)
                outs = out if eqn.primitive.multiple_results else [out]
                cv = jnp.concatenate([jnp.reshape(jnp.broadcast_to(o.cv, tape.shapes[s][0]), (-1,))
                                      for o, s in zip(outs, L.out_slots)])  # fmt: skip
                cc = jnp.concatenate([jnp.reshape(jnp.broadcast_to(o.cc, tape.shapes[s][0]), (-1,))
                                      for o, s in zip(outs, L.out_slots)])  # fmt: skip
                return cv, cc

            cv, cc = relax_out(*pts)
            Jcv, Jcc = jax.jacfwd(relax_out, argnums=tuple(range(len(pts))))(*pts)
            W = jnp.zeros((L.size, N), dt).at[jnp.arange(L.size), L.offset + jnp.arange(L.size)].set(1.0)
            A_lo, b_lo = -W, -cv
            A_hi, b_hi = W, cc
            for p, slot, Jl, Jh in zip(pts, L.in_slots, Jcv, Jcc):
                E, c = flat(slot)
                Jl = jnp.reshape(Jl, (L.size, -1))
                Jh = jnp.reshape(Jh, (L.size, -1))
                d = jnp.reshape(p, (-1,)) - c
                # w >= cv + Jl (E v + c - p)  ->  Jl E v - w <= -cv + Jl (p - c)
                A_lo, b_lo = A_lo + Jl @ E, b_lo + Jl @ d
                # w <= cc + Jh (E v + c - p)  ->  w - Jh E v <= cc - Jh (p - c)
                A_hi, b_hi = A_hi - Jh @ E, b_hi - Jh @ d
            A = jnp.concatenate([A_lo, A_hi])
            b = jnp.concatenate([b_lo, b_hi])
            ok = jnp.isfinite(b) & jnp.all(jnp.isfinite(A), axis=1)
            return jnp.where(ok[:, None], A, 0.0), jnp.where(ok, b, jnp.inf)

        def point(L, t):
            """Input points at relative position t (one scalar per MC input)."""
            return [vals[a].lo + ti * (vals[a].hi - vals[a].lo) for a, ti in zip(L.in_slots, t)]

        def safe_point(L, ps):
            out = []
            for a, p in zip(L.in_slots, ps):
                lo_, hi_ = vals[a].lo, vals[a].hi
                mid = jnp.where(jnp.isfinite(lo_) & jnp.isfinite(hi_), 0.5 * (lo_ + hi_), 0.0)
                p = jnp.where(jnp.isfinite(p), p, mid)
                out.append(jnp.clip(p, lo_, hi_))
            return out

        rows = []
        for L in lifts:
            for t in _pattern(len(L.mc_pos)):
                rows.append(node_cuts(L, safe_point(L, point(L, t))))

        # objective and constraint rows
        Ef, cf = flat(tape.outs[0])

        # reduced-space cuts of the whole functions on the x columns:
        #   cv_f(p) + J (x - p) <= f = Ef v + cf,   cv_g(p) + J (x - p) <= tol, ...
        bx_lo, bx_hi = vals[0].lo, vals[0].hi
        pdata = prepare_parts(bx_lo, bx_hi)

        def reduced_cuts(p):
            p = jnp.clip(p, bx_lo, bx_hi)
            val = parts(p, bx_lo, bx_hi, pdata)
            J = jax.jacfwd(parts)(p, bx_lo, bx_hi, pdata)
            A = jnp.zeros((val.size, N), dt).at[:, :n].set(J)
            A = A.at[:n_f].add(-Ef)
            b = J @ p - val
            b = b.at[:n_f].add(cf) + jnp.where(jnp.arange(val.size) < n_f, 0.0, tol)
            ok = jnp.isfinite(b) & jnp.all(jnp.isfinite(A), axis=1)
            return jnp.where(ok[:, None], A, 0.0), jnp.where(ok, b, jnp.inf)

        cx = 0.5 * (bx_lo + bx_hi)
        ax = jnp.concatenate([jnp.eye(n), -jnp.eye(n)]) * ((bx_hi - bx_lo) / 4)
        for pnt in jnp.concatenate([cx[None], cx + ax]):
            rows.append(reduced_cuts(pnt))
        k = 1
        if ineq is not None:
            if not isinstance(tape.outs[k], Const):
                E, c = flat(tape.outs[k])
                rows.append((E, tol - c))
            k += 1
        if eq is not None:
            if not isinstance(tape.outs[k], Const):
                E, c = flat(tape.outs[k])
                rows += [(E, tol - c), (-E, tol + c)]

        # --- LP in unit coordinates u in [0, 1]^N ---------------------------
        def unit(A, b):
            b = b - A @ clo
            A = A * scale
            nrm = jnp.linalg.norm(A, axis=1)
            live = jnp.isfinite(b) & (nrm > 0)
            nrm = jnp.where(live, nrm, 1.0)
            return jnp.where(live[:, None], A / nrm[:, None], 0.0), jnp.where(live, b / nrm, 1.0)

        cu = Ef[0] * scale
        cn = jnp.maximum(jnp.linalg.norm(cu), 1e-300)
        chat = cu / cn
        f0 = cf[0] + Ef[0] @ clo
        box_A = jnp.concatenate([jnp.eye(N), -jnp.eye(N)])
        box_b = jnp.concatenate([jnp.ones(N), jnp.zeros(N)])

        def dual(A, b, y):
            y = jnp.where(jnp.isfinite(y), jnp.maximum(y, 0.0), 0.0)
            coef = chat + A.T @ y
            val = jnp.sum(jnp.minimum(coef, 0.0)) - y @ b
            err = (A.shape[0] + N + 2) * EPS * (jnp.sum(1.0 + jnp.abs(A).T @ y) + jnp.abs(b) @ y)
            return f0 + cn * (val - err), _safe_costs(coef, A, y)

        # interval lower bound of f (with the univariate hull) is a floor for the LP bound
        flo = rel_f(0.5 * (bx_lo + bx_hi), bx_lo, bx_hi, pdata[0]).lo[0]
        best = jnp.where(finite & ~jnp.isnan(flo), flo, -jnp.inf).astype(dt)
        xl, xh = clo[:n], clo[:n] + scale[:n]  # the box the certificate is over
        rlo, rhi = jnp.maximum(lo, vals[0].lo), jnp.minimum(hi, vals[0].hi)
        v = 0.5 * (clo + chi)
        for it in range(n_kelley + 1):
            A, b = unit(jnp.concatenate([r[0] for r in rows]), jnp.concatenate([r[1] for r in rows]))
            nA = A.shape[0]
            z, y, lb = ipm(chat, jnp.concatenate([A, box_A]), jnp.concatenate([b, box_b]), ipm_iters,
                           cert=lambda y, A=A, b=b, nA=nA: dual(A, b, y[:nA])[0])  # fmt: skip
            lb = jnp.where(jnp.isnan(lb) | ~finite, -jnp.inf, lb)
            best = jnp.maximum(best, lb)
            _, r = dual(A, b, y[:nA])
            klo, khi = reduce_ranges(xl, xh, lb, cn * r[:n] / scale[:n], cutoff)
            rlo, rhi = jnp.maximum(rlo, klo), jnp.minimum(rhi, khi)
            v = clo + scale * jnp.clip(z, 0.0, 1.0)
            if it < n_kelley:  # Kelley: cuts at the LP solution's input points
                for L in lifts:
                    ps = []
                    for a in L.in_slots:
                        E, c = aff[a]
                        ps.append(jnp.tensordot(E, v, axes=1) + c)
                    rows.append(node_cuts(L, safe_point(L, ps)))
                rows.append(reduced_cuts(v[:n]))

        x = jnp.clip(v[:n], lo, hi)
        lb = jnp.where(empty, jnp.inf, best)
        rlo = jnp.where(empty | ~finite, lo, rlo)
        rhi = jnp.where(empty | ~finite, hi, rhi)
        return lb, x, empty, rlo, rhi

    return bound

