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
# ...while there is no incumbent (cutoff +inf): any assignment the cuts allow is worth completing.
# minlp.org m400_CLay0304HRcone, from its root box: the first feasible proposal after 4.3 s and the
# optimum after 6.5 s, against 15-22 s with PROPOSE_GAP (each MILP 0.2-0.5 s instead of 0.5-2.7 s)
PROPOSE_GAP_NOINC = 1.0
STALL, PATIENCE = 0.01, 3  # stop after PATIENCE rounds each raising the bound by under STALL of the gap
_DEBUG = False


CHUNK = 8  # points per call of the cut program (``cut_values``)
MAX_EQNS = 100_000  # ``ready``: cut programs with more jaxpr equations are not compiled
# XLA options of the cut program's compile (``ready``). It runs a few calls per MILP round, so its
# compile is what costs: at LLVM -O0 minlp.org m410_CLay0203HRS's (76k equations) compiles in 9.3 s
# instead of 13.7 s, and a call takes 5.0 ms instead of 6.4 ms (values equal to 2e-10). Empty: defaults.
COMPILE_OPTIONS = {"xla_backend_optimization_level": 0}


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


def form_tangents(Q, n):
    """``MilpBound``'s ``tangents`` from a lifted quadratic form ``Q`` (``quadratic.QuadraticForm``
    of degree 2: rows ``[f, g..., h...] = E v + c`` with ``v = (x, w)``, column ``n + p`` the
    product ``x_a x_b`` of ``pairs[p]``), in numpy: no trace, no compile. Each row's cut is the row
    with every product replaced by the McCormick plane (``a != b``) or the tangent / secant of the
    square (``a == b``) that is active at the point, an underestimator for a positive coefficient
    and an overestimator for a negative one: a valid convex minorant of the row on the box, as
    ``relax``'s. None when some product is of a lifted column (degree above 2).
    Layout: [cv_f, cv_g..., cv_h..., -cc_h...]."""
    E, c, pairs = np.asarray(Q.E, np.float64), np.asarray(Q.c, np.float64), np.asarray(Q.pairs, np.int64)
    if len(pairs) and pairs.max() >= n:
        return None
    n_g, n_h = int(Q.n_g), int(Q.n_h)
    A, B = (pairs[:, 0], pairs[:, 1]) if len(pairs) else (np.zeros(0, np.int64),) * 2
    Ex, Ew = E[:, :n], E[:, n:]
    h = slice(1 + n_g, 1 + n_g + n_h)

    def planes(P, lo, hi, over):
        """(k, Pairs) slopes on x_a, x_b and offsets of the plane for w = x_a x_b at each point:
        the active McCormick under- (``over`` False) or overestimator."""
        la, ua, lb, ub = lo[A], hi[A], lo[B], hi[B]
        xa, xb = P[:, A], P[:, B]
        sq = A == B
        if not over:
            # w >= lb xa + la xb - la lb  and  w >= ub xa + ua xb - ua ub: the larger at the point
            v1, v2 = lb * xa + la * xb - la * lb, ub * xa + ua * xb - ua * ub
            one = v1 >= v2
            ca, cb, c0 = np.where(one, lb, ub), np.where(one, la, ua), np.where(one, -la * lb, -ua * ub)
            # a square: the tangent at the point, 2 p x - p^2 (as slope on x_a; x_b is x_a)
            ca, cb, c0 = np.where(sq, 2 * xa, ca), np.where(sq, 0.0, cb), np.where(sq, -xa * xa, c0)
        else:
            # w <= ub xa + la xb - la ub  and  w <= lb xa + ua xb - ua lb: the smaller at the point
            v1, v2 = ub * xa + la * xb - la * ub, lb * xa + ua * xb - ua * lb
            one = v1 <= v2
            ca, cb, c0 = np.where(one, ub, lb), np.where(one, la, ua), np.where(one, -la * ub, -ua * lb)
            # a square: the secant (la + ua) x - la ua
            ca, cb, c0 = np.where(sq, la + ua, ca), np.where(sq, 0.0, cb), np.where(sq, -la * ua, c0)
        return ca, cb, c0

    def minorant(Er, cr, P, lo, hi):
        """Slopes (k, R, n) and offsets (k, R) of the convex minorants of the rows ``Er v + cr``."""
        Ex_, Ew_ = Er[:, :n], Er[:, n:]
        k = len(P)
        G = np.repeat(Ex_[None], k, 0)
        b0 = np.repeat(cr[None], k, 0)
        if len(A):
            pos, neg = np.maximum(Ew_, 0.0), np.minimum(Ew_, 0.0)
            for over, W in ((False, pos), (True, neg)):
                ca, cb, c0 = planes(P, lo, hi, over)  # (k, Pairs)
                # row r gets W[r, p] * (ca x_a + cb x_b + c0)
                Ga = W[None] * ca[:, None, :]  # (k, R, Pairs)
                Gb = W[None] * cb[:, None, :]
                for q in range(len(A)):
                    G[:, :, A[q]] += Ga[:, :, q]
                    G[:, :, B[q]] += Gb[:, :, q]
                b0 = b0 + np.einsum("rp,kp->kr", W, c0)
        return G, b0

    def tangents(P, lo, hi):
        P = np.atleast_2d(np.asarray(P, np.float64))
        lo, hi = np.asarray(lo, np.float64), np.asarray(hi, np.float64)
        G, b0 = minorant(E, c, P, lo, hi)  # cv of every row
        Gc, bc = minorant(-E[h], -c[h], P, lo, hi)  # cv of -h = -cc of h
        return np.concatenate([G, Gc], axis=1), np.concatenate([b0, bc], axis=1)

    return tangents


class MilpBound:
    """``bound(lo, hi, points, ...)`` for one problem; the relaxations' cut functions are compiled once."""

    def __init__(self, f, ineq, eq, n, feas_tol=1e-6, tangents=None):
        """``tangents``: optional ``tangents(P, lo, hi) -> (a, b)``, numpy ``(k, K, n)`` and ``(k, K)``:
        cuts ``a . x + b`` at the points ``P`` of valid convex minorants on ``[lo, hi]`` of the parts
        [cv_f, cv_g..., cv_h..., -cc_h...] (``lp.make_pool_cuts``' layout; ``b = -inf`` for none).
        ``cut_values`` then takes its cuts from it instead of compiling its own cut program
        (``ready`` / ``lower`` / ``compile`` are not needed)."""
        from .lp import _nonlinear_vars

        self.n, self.feas_tol, self._tangents = n, feas_tol, tangents
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
        self._exe = None  # ``ready``: the cut program's executable for ``cut_values``' argument shapes
        self.eqns = None  # ``lower``: the cut program's jaxpr equations
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
        if self._tangents is not None:
            a, b = (np.asarray(z, np.float64) for z in self._tangents(P, lo, hi))
            with np.errstate(invalid="ignore"):
                v = b + np.einsum("kmn,kn->km", a, P)  # (the value at p of each cut; -inf: no cut)
            a = np.where(np.isfinite(v)[..., None], a, np.nan)  # (dropped by ``rows``)
            e = np.cumsum([1, self.m_ineq, self.m_eq])
            return ((v[:, :1], v[:, e[0]:e[1]], v[:, e[1]:e[2]], -v[:, e[2]:]),
                    (a[:, :1], a[:, e[0]:e[1]], a[:, e[1]:e[2]], -a[:, e[2]:]))  # (-cc_h -> cc_h)
        lo, hi = jnp.asarray(lo, jnp.float64), jnp.asarray(hi, jnp.float64)
        outs = []
        for c in range(0, len(P), CHUNK):
            Pc = P[c:c + CHUNK]
            Pc = np.concatenate([Pc, np.repeat(Pc[:1], CHUNK - len(Pc), 0)])
            cuts = self._exe if self._exe is not None else self._cuts
            outs.append(jax.tree_util.tree_map(np.asarray, cuts(jnp.asarray(Pc), lo, hi)))
        return jax.tree_util.tree_map(lambda *a: np.concatenate(a)[:len(P)], *outs)

    def lower(self):
        """Trace and lower the cut program; None when it is too large (``MAX_EQNS`` jaxpr
        equations: smallinvDAXr3b010-011's 122,851 took 52 s and 4.2 GB to compile, against 6-10 s
        and under 0.7 GB for 18,000-91,000 on minlp.org m426 and m474). The first half of
        ``ready``, for a caller that queues the compile itself."""
        n = self.n
        args = (jnp.zeros((CHUNK, n)), jnp.zeros(n), jnp.ones(n))
        self.eqns = _count_eqns(jax.make_jaxpr(self._cuts)(*args).jaxpr)
        if self.eqns > MAX_EQNS:
            return None
        return self._cuts.trace(*args).lower()

    def compile(self, low):
        """Compile ``lower``'s result (``COMPILE_OPTIONS``) and ``_flo``; the second half of ``ready``.
        Returns whether the cut program is ready (False for ``low`` None)."""
        if self._ready is None:
            if low is None:
                self._ready = False
            else:
                n = self.n
                args = (jnp.zeros((CHUNK, n)), jnp.zeros(n), jnp.ones(n))
                exe = low.compile(compiler_options=dict(COMPILE_OPTIONS)) if COMPILE_OPTIONS else low.compile()
                jax.block_until_ready(exe(*args))
                jax.block_until_ready(self._flo(args[1], args[2]))
                self._exe, self._ready = exe, True
        return self._ready

    def ready(self):
        """Trace and compile the cut program (and ``_flo``) once (``lower``, then ``compile``); False
        when it is too large. Meant for a background thread; the result is kept."""
        if self._ready is None:
            self.compile(self.lower())
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
                           options={"time_limit": float(tl), "mip_rel_gap": PROPOSE_GAP if np.isfinite(ub) else PROPOSE_GAP_NOINC})  # fmt: skip
                if res.status == 4 and np.isfinite(st["tlo"]):
                    # HiGHS (scipy 1.18) reports a solve error on some small MILPs whose objective
                    # variable has a finite lower bound; without that bound they solve (the cuts bound t)
                    box = Bounds(np.r_[st["lo_i"], -np.inf], np.r_[st["hi_i"], np.inf])
                    st["tlo"] = -np.inf
                    res = milp(cost, constraints=LinearConstraint(AA, LL, HH), integrality=integrality, bounds=box,
                               options={"time_limit": float(tl), "mip_rel_gap": PROPOSE_GAP if np.isfinite(ub) else PROPOSE_GAP_NOINC})  # fmt: skip
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
            if _DEBUG: print("complete", np.round(np.asarray(F), 3), ub)
            P = [x] + [y for y in np.atleast_2d(Y) if np.all(np.isfinite(y))]
            new = self.rows(np.clip(np.array(P), lo, hi), lo, hi)
            if new is not None:
                st["A"], st["rl"], st["rh"] = np.vstack([st["A"], new[0]]), np.r_[st["rl"], new[1]], np.r_[st["rh"], new[2]]
        return st
