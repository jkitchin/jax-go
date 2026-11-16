"""Integer-aware lower bounds: a MILP over the linear cuts of the relaxations.

The node bound (``lp.py``) is an LP over affine cuts of the McCormick / alphaBB relaxations
``cv_f``, ``cv_g``, ``cv_h``, ``-cc_h`` of the objective and constraints on a box. Every cut is valid
at every point of the box, integer or not, so the same cuts with the integer variables required
integral define a MILP whose optimal value also bounds the problem from below on that box, and
can be far stronger: on minlp.org's k-means clustering models (convex, binaries in big-M
constraints) the LP bound, SCIP's root bound and jaxglobal's root bound are all 0, while this
MILP reaches 96 % of the optimum at the root (docs/ANALYSIS-root-bounds.md).

``MilpBound.bound`` builds the cuts at the box center, the axis points of the variables some part is
nonlinear in (``lp._nonlinear_vars``) and given points (the incumbent, the relaxation point),
solves the MILP with HiGHS (``scipy.optimize.milp``), adds cuts at the MILP and LP solutions and
solves again (outer approximation), within a time budget. It returns HiGHS's proven dual bound of
the last MILP, lowered by a margin for HiGHS's tolerances.

Not certified like ``lp.py``'s weak-duality bound: the bound is that of a floating-point MILP
solve, whose feasibility and optimality tolerances (1e-6 here) the margin absorbs. ``minimize``
therefore does not use it with ``safe=True``.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from .relax import relax

MAX_COEF = 1e8  # cuts with larger coefficients are dropped (only loosens the bound)
MARGIN_REL, MARGIN_ABS = 1e-6, 1e-7  # subtracted from HiGHS's dual bound
MAX_ROWS = 200_000  # no MILP above this many cut rows
MAX_ROWS0 = 40_000  # initial cut rows at most (fewer points when more)
PROPOSE_GAP = 1e-2  # ``propose``: relative MILP gap at which HiGHS stops (a good assignment is enough)
STALL, PATIENCE = 0.01, 3  # stop after PATIENCE rounds each raising the bound by under STALL of the gap
_DEBUG = False


CHUNK = 8  # points per call of the cut program (``cut_values``)
MAX_EQNS = 100_000  # ``ready``: cut programs with more jaxpr equations are not compiled


def _count_eqns(jaxpr):
    """Equations of a jaxpr, those of its sub-jaxprs (jit, scan, cond bodies) included."""
    c = 0
    for e in jaxpr.eqns:
        c += 1
        for v in e.params.values():
            for j in v if isinstance(v, (tuple, list)) else (v,):
                j = getattr(j, "jaxpr", j)
                if hasattr(j, "eqns"):
                    c += _count_eqns(j)
    return c


def _vec(fn):
    return (lambda x: jnp.atleast_1d(fn(x))) if fn is not None else (lambda x: jnp.zeros(0))


class MilpBound:
    """``bound(lo, hi, points, ...)`` for one problem; the relaxations' cut functions are compiled once."""

    def __init__(self, f, ineq, eq, n, feas_tol=1e-6):
        from .lp import _nonlinear_vars

        self.n, self.feas_tol = n, feas_tol
        g, h = _vec(ineq), _vec(eq)
        rf, rg, rh = relax(f), relax(g), relax(h)

        def parts(x, lo, hi):
            mf, mg, mh = rf(x, lo, hi), rg(x, lo, hi), rh(x, lo, hi)
            return jnp.atleast_1d(mf.cv), mg.cv, mh.cv, mh.cc

        val = jax.vmap(parts, in_axes=(0, None, None))
        jac = jax.vmap(jax.jacfwd(parts), in_axes=(0, None, None))
        self._cuts = jax.jit(lambda X, lo, hi: (val(X, lo, hi), jac(X, lo, hi)))
        self._flo = jax.jit(lambda lo, hi: rf(0.5 * (lo + hi), lo, hi).lo)
        self._ready = None  # ``ready``: None until traced and compiled, then whether it compiled
        nl = _nonlinear_vars(f, ineq, eq, n)
        self.nonlinear = np.ones(n, bool) if nl is None else np.asarray(nl, bool).any(axis=0)
        with jax.ensure_compile_time_eval():
            sd = jax.ShapeDtypeStruct((n,), jnp.float64)
            self.m_ineq = int(np.size(jax.eval_shape(g, sd)))
            self.m_eq = int(np.size(jax.eval_shape(h, sd)))

    def cut_values(self, P, lo, hi):
        """The cut program at the points ``P`` (k, n), evaluated ``CHUNK`` points at a time (the last
        chunk padded), so that it compiles once whatever the number of points."""
        P = np.asarray(P, np.float64)
        lo, hi = jnp.asarray(lo), jnp.asarray(hi)
        outs = []
        for c in range(0, len(P), CHUNK):
            Pc = P[c:c + CHUNK]
            Pc = np.concatenate([Pc, np.repeat(Pc[:1], CHUNK - len(Pc), 0)])
            outs.append(jax.tree_util.tree_map(np.asarray, self._cuts(jnp.asarray(Pc), lo, hi)))
        return jax.tree_util.tree_map(lambda *a: np.concatenate(a)[:len(P)], *outs)

    def ready(self):
        """Trace and compile the cut program (and ``_flo``) once; False when it is too large
        (``MAX_EQNS`` jaxpr equations: smallinvDAXr3b010-011's 122,851 took 52 s and 4.2 GB to
        compile, against 6-10 s and under 0.7 GB for 18,000-91,000 on minlp.org m426 and m474).
        Meant for a background thread; the result is kept."""
        if self._ready is None:
            n = self.n
            args = (jnp.zeros((CHUNK, n)), jnp.zeros(n), jnp.ones(n))
            if _count_eqns(jax.make_jaxpr(self._cuts)(*args).jaxpr) > MAX_EQNS:
                self._ready = False
            else:
                jax.block_until_ready(self._cuts(*args))
                jax.block_until_ready(self._flo(args[1], args[2]))
                self._ready = True
        return self._ready

    def rows(self, P, lo, hi):
        """Cut rows ``(A, row_lo, row_hi)`` over ``[x, t]`` at the points ``P`` (k, n)."""
        (vf, vg, vh, vc), (Jf, Jg, Jh, Jc) = self.cut_values(P, lo, hi)
        P = np.asarray(P, np.float64)
        A, rlo, rhi = [], [], []

        def add(V, J, kind, slack):
            V, J = np.asarray(V, np.float64), np.asarray(J, np.float64)  # (k, m), (k, m, n)
            if V.size == 0:
                return
            rhs = np.einsum("kmn,kn->km", J, P) - V  # J (x - p) + v  <=>  J x  vs  J p - v
            ok = np.isfinite(rhs) & np.all(np.isfinite(J), axis=2) & (np.max(np.abs(J), axis=2) <= MAX_COEF)
            J, rhs = J[ok], rhs[ok]
            if kind == "obj":  # v + J (x - p) - t <= 0
                A.append(np.hstack([J, -np.ones((len(J), 1))]))
                rlo.append(np.full(len(J), -np.inf))
                rhi.append(rhs)
            elif kind == "le":  # v + J (x - p) <= slack
                A.append(np.hstack([J, np.zeros((len(J), 1))]))
                rlo.append(np.full(len(J), -np.inf))
                rhi.append(rhs + slack)
            else:  # "ge": v + J (x - p) >= -slack
                A.append(np.hstack([J, np.zeros((len(J), 1))]))
                rlo.append(rhs - slack)
                rhi.append(np.full(len(J), np.inf))

        add(vf, Jf, "obj", 0.0)
        add(vg, Jg, "le", self.feas_tol)
        add(vh, Jh, "le", self.feas_tol)
        add(vc, Jc, "ge", self.feas_tol)
        if not A:
            return None
        return np.vstack(A), np.concatenate(rlo), np.concatenate(rhi)

    def points(self, lo, hi, extra=(), n_random=None, seed=0):
        """The box center, the axis points (half way to each face) of the nonlinear variables, random points
        (nonlinear coordinates uniform in the box, the others at the center) and ``extra``."""
        c = 0.5 * (lo + hi)
        P = [c]
        nl = np.flatnonzero(self.nonlinear & (hi > lo))
        for j in nl:
            for s in (-0.5, 0.5):
                p = c.copy()
                p[j] += s * (hi[j] - lo[j])
                P.append(p)
        k = 2 * len(nl) if n_random is None else n_random
        if k and len(nl):
            R = np.tile(c, (k, 1))
            R[:, nl] = np.random.default_rng(seed).uniform(lo[nl], hi[nl], (k, len(nl)))
            P += list(R)
        P += [np.clip(np.asarray(x, np.float64), lo, hi) for x in extra if x is not None and np.all(np.isfinite(x))]
        return np.array(P)

    def _initial(self, lo, hi, imask, extra, few=False):
        """The first MILP of ``bound`` / ``propose``: ``(lo, hi, imask, lo_i, hi_i, tlo, (A, rl, rh))``,
        the integer bounds rounded inward, the objective variable's lower bound and the cut rows at
        ``points`` (at most ``MAX_ROWS0`` rows; ``few``: only the center and ``extra``); None when
        there is nothing to solve."""
        lo, hi = np.asarray(lo, np.float64), np.asarray(hi, np.float64)
        if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))) or not np.any(imask):
            return None
        imask = np.asarray(imask, bool)
        lo_i, hi_i = np.where(imask, np.ceil(lo - 1e-9), lo), np.where(imask, np.floor(hi + 1e-9), hi)
        if np.any(lo_i > hi_i):
            return None
        flo = float(self._flo(jnp.asarray(lo), jnp.asarray(hi)))
        tlo = flo if np.isfinite(flo) else -np.inf
        extra = [np.asarray(x, np.float64) for x in extra if x is not None and np.all(np.isfinite(x))]
        P = self.points(lo, hi, extra) if not few else np.array([0.5 * (lo + hi)] + [np.clip(x, lo, hi) for x in extra])
        per = 1 + self.m_ineq + 2 * self.m_eq  # rows per point (at most)
        if len(P) * per > MAX_ROWS0:  # keep the center, the extra points and an even subset of the others
            k = max(1, MAX_ROWS0 // per - 1 - len(extra))
            mid = P[1 : len(P) - len(extra)]
            keep = np.linspace(0, len(mid) - 1, min(k, len(mid))).astype(int) if len(mid) else []
            P = np.vstack([P[:1], mid[keep], P[len(P) - len(extra) :]]) if len(extra) else np.vstack([P[:1], mid[keep]])
        R = self.rows(P, lo, hi)
        if R is None:
            return None
        return lo, hi, imask, lo_i, hi_i, tlo, R

    def bound(self, lo, hi, imask, extra=(), cutoff=np.inf, budget=30.0, rounds=8, round_time=None, tol=0.0):
        """A lower bound on ``f`` over the box ``[lo, hi]`` (feasible points, integers integral), or None.

        Stops after ``rounds`` outer-approximation rounds, ``budget`` seconds, when the bound
        reaches ``cutoff - tol`` (the incumbent: nothing better to prove), or when a round raises
        it by under ``STALL`` of the remaining gap ``PATIENCE`` times in a row (a Kelley bound can sit
        still for a round or two before rising)."""
        from scipy.optimize import Bounds, LinearConstraint, milp

        t0 = time.perf_counter()
        built = self._initial(lo, hi, imask, extra)
        if built is None:
            return None
        lo, hi, imask, lo_i, hi_i, tlo, (A, rl, rh) = built
        n = self.n
        cost = np.r_[np.zeros(n), 1.0]
        box = Bounds(np.r_[lo_i, tlo], np.r_[hi_i, np.inf])
        integrality = np.r_[imask.astype(int), 0]
        best, x_prev, stalled = -np.inf, None, 0
        for _ in range(rounds):
            left = budget - (time.perf_counter() - t0)
            if left <= 0.5 or len(A) > MAX_ROWS:
                break
            tl = left if round_time is None else min(left, round_time)
            try:
                res = milp(cost, constraints=LinearConstraint(A, rl, rh), integrality=integrality, bounds=box,
                           options={"time_limit": float(tl), "mip_rel_gap": 1e-6})  # fmt: skip
            except Exception:  # noqa: BLE001 -- HiGHS failure: keep the bound so far
                break
            if res.status == 2:  # infeasible: no feasible point in the box
                return np.inf
            db = getattr(res, "mip_dual_bound", None)
            if db is None or not np.isfinite(db):
                break
            prev, best = best, max(best, float(db))
            if _DEBUG: print("round", len(A), res.status, float(db), res.fun, time.perf_counter() - t0)
            if best >= cutoff - tol or res.x is None:
                break
            stalled = stalled + 1 if np.isfinite(prev) and best - prev <= STALL * (abs(cutoff - prev) if np.isfinite(cutoff) else 1.0 + abs(prev)) else 0
            if stalled >= PATIENCE:
                break
            x = np.clip(res.x[:n], lo, hi)
            X = [x] if x_prev is None else [x, 0.5 * (x + x_prev)]  # the midpoint stabilizes Kelley's cutting planes
            x_prev = x
            new = self.rows(np.array(X), lo, hi)
            if new is None:
                break
            A, rl, rh = np.vstack([A, new[0]]), np.r_[rl, new[1]], np.r_[rh, new[2]]
        if not np.isfinite(best):
            return None
        return best - MARGIN_REL * abs(best) - MARGIN_ABS


    def propose(self, lo, hi, imask, fixed_solve, extra=(), cutoff=np.inf, budget=10.0, rounds=8, tol=0.0,
                milp_time=None, few=True, state=None):
        """Outer-approximation primal heuristic: integer assignments from MILPs over the cuts, each
        completed by ``fixed_solve``.

        ``fixed_solve(X)`` takes MILP solutions ``X`` (k, n) (integers integral) and returns
        ``(Y, F, ub)``: the points of local solves with those integers fixed, their objective
        values (``inf`` for a point that is no incumbent: it must screen them against the original
        problem) and the incumbent value after them. Each round solves the MILP of ``bound`` plus a
        cutoff row ``t <= ub - tol`` and, when every integer variable is binary, a no-good cut per
        assignment already tried; then adds cuts at the MILP solution and the local solutions (on a
        convex problem these are the cuts of outer approximation, which converges to the optimal
        assignment). Stops after ``rounds`` rounds, ``budget`` seconds, an infeasible MILP (no
        assignment left whose cuts allow beating the incumbent: ``state["done"]``) or a repeated
        assignment.

        Only proposes incumbents: the cutoff and no-good rows make its MILPs no bound of anything,
        and nothing here is returned as one. ``few``: the first MILP has the cuts at the box center
        and ``extra`` only (``bound``'s axis and random points make its MILPs much larger and
        slower; outer approximation adds the cuts it needs). ``state``: the dict a previous call on
        the same box returned, to resume with its cuts and no-goods (``extra`` then adds cuts).
        Returns the state (``rounds``: MILPs solved so far, ``done``: nothing left to propose)."""
        from scipy.optimize import Bounds, LinearConstraint, milp

        t0 = time.perf_counter()
        if state is None:
            built = self._initial(lo, hi, imask, extra, few=few)
            if built is None:
                return dict(done=True, rounds=0)
            lo, hi, imask, lo_i, hi_i, tlo, (A, rl, rh) = built
            ii = np.flatnonzero(imask)
            state = dict(lo=lo, hi=hi, lo_i=lo_i, hi_i=hi_i, tlo=tlo, A=A, rl=rl, rh=rh, ii=ii, NG=[], ng_lo=[],
                         seen=set(), done=False, rounds=0,
                         binary=bool(np.all((lo_i[ii] == 0.0) & (hi_i[ii] == 1.0))))  # fmt: skip
        elif state.get("done"):
            return state
        else:
            ex = [np.asarray(x, np.float64) for x in extra if x is not None and np.all(np.isfinite(x))]
            new = self.rows(np.clip(np.array(ex), state["lo"], state["hi"]), state["lo"], state["hi"]) if ex else None
            if new is not None:
                state["A"], state["rl"], state["rh"] = (np.vstack([state["A"], new[0]]), np.r_[state["rl"], new[1]],
                                                        np.r_[state["rh"], new[2]])  # fmt: skip
        n, st = self.n, state
        lo, hi, ii = st["lo"], st["hi"], st["ii"]
        cost = np.r_[np.zeros(n), 1.0]
        box = Bounds(np.r_[st["lo_i"], st["tlo"]], np.r_[st["hi_i"], np.inf])
        integrality = np.r_[np.isin(np.arange(n), ii).astype(int), 0]
        ub = cutoff
        for _ in range(rounds):
            left = budget - (time.perf_counter() - t0)
            if left <= 0.2 or len(st["A"]) > MAX_ROWS:
                break
            AA, LL, HH = st["A"], st["rl"], st["rh"]
            if np.isfinite(ub):  # t <= ub - tol: only assignments whose cuts allow an improvement
                AA = np.vstack([AA, cost[None]])
                LL, HH = np.r_[LL, -np.inf], np.r_[HH, ub - tol]
            if st["NG"]:  # no-good: sum_{y_j = 0} x_j + sum_{y_j = 1} (1 - x_j) >= 1
                AA = np.vstack([AA, np.array(st["NG"])])
                LL, HH = np.r_[LL, st["ng_lo"]], np.r_[HH, np.full(len(st["NG"]), np.inf)]
            tl = left if milp_time is None else min(left, milp_time)
            try:
                res = milp(cost, constraints=LinearConstraint(AA, LL, HH), integrality=integrality, bounds=box,
                           options={"time_limit": float(tl), "mip_rel_gap": PROPOSE_GAP})  # fmt: skip
                if res.status == 4 and np.isfinite(st["tlo"]):
                    # HiGHS (scipy 1.18) reports a solve error on some small MILPs whose objective
                    # variable has a finite lower bound; without that bound they solve (the cuts bound t)
                    box = Bounds(np.r_[st["lo_i"], -np.inf], np.r_[st["hi_i"], np.inf])
                    st["tlo"] = -np.inf
                    res = milp(cost, constraints=LinearConstraint(AA, LL, HH), integrality=integrality, bounds=box,
                               options={"time_limit": float(tl), "mip_rel_gap": PROPOSE_GAP})  # fmt: skip
            except Exception:  # noqa: BLE001 -- HiGHS failure: no proposal
                break
            st["rounds"] += 1
            if res.status == 2:  # infeasible: no assignment left that the cuts let beat the incumbent
                st["done"] = True
                break
            if _DEBUG: print("propose milp", res.status, res.message)
            if res.x is None:  # no solution within the time
                break
            x = np.clip(np.asarray(res.x[:n], np.float64), lo, hi)
            x[ii] = np.clip(np.round(x[ii]), st["lo_i"][ii], st["hi_i"][ii])
            key = x[ii].tobytes()
            if _DEBUG: print("propose", len(AA), res.status, res.fun, time.perf_counter() - t0)
            if key in st["seen"]:  # (general integers: no no-good cut) the cuts did not move it
                break
            st["seen"].add(key)
            if st["binary"]:
                row = np.zeros(n + 1)
                row[ii] = np.where(x[ii] > 0.5, -1.0, 1.0)
                st["NG"].append(row)
                st["ng_lo"].append(1.0 - float(np.sum(x[ii] > 0.5)))
            Y, F, ub = fixed_solve(x[None])
            P = [x] + [y for y in np.atleast_2d(Y) if np.all(np.isfinite(y))]
            new = self.rows(np.clip(np.array(P), lo, hi), lo, hi)
            if new is not None:
                st["A"], st["rl"], st["rh"] = np.vstack([st["A"], new[0]]), np.r_[st["rl"], new[1]], np.r_[st["rh"], new[2]]
        return st
