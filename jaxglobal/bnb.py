"""Batched spatial branch-and-bound using JAX relaxations.

Each iteration takes the ``batch // 2`` open boxes with the lowest lower bound,
bisects them, and bounds all children at once with ``jit(vmap(bound))``.

Lower bound of a box = max of
  * the interval bound of f, and
  * affine cuts of the convex McCormick relaxation cv: for any point x in the
    box with subgradient g = grad(cv)(x),
        min_box f >= min_box cv >= cv(x) + sum_i min(g_i (lo_i - x_i), g_i (hi_i - x_i)).
    The points x are Frank-Wolfe iterates on cv, so the cut is also the
    Frank-Wolfe duality-gap bound.
Upper bounds come from evaluating f at the box center and FW iterate, plus a
batch of local solves (see ``make_local``) whenever a batch improves the incumbent.

Bound tightening (all against the incumbent cutoff ``f <= UB``), every
node unless switched off:
  * FBBT over the jaxpr before bounding (``fbbt.py``);
  * duality-based range reduction from the cuts / LP duals after bounding;
  * OBBT on the root box (and optionally every node) via batched LPs.
Children are created from the tightened box.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from .local import make_local
from .domain import bound_domain
from .fbbt import make_fbbt
from .lifted import make_lifted_bounder
from .lp import make_constrained_bounder, make_obbt, reduce_ranges
from .relax import relax
from .univariate import make_relaxer


def make_bounder(f, n_fw=20, univariate=False, alphabb=False, multivariate=False):
    """Return ``bound(lo, hi, cutoff) -> (lb, x, f(x), lo', hi')`` for one box.

    ``[lo', hi']`` is the box after duality-based range reduction: every
    Frank-Wolfe cut ``f(z) >= cv(x) + g.(z - x)`` with ``f(z) <= cutoff``
    confines each ``z_j`` to an interval.
    """
    rfu = make_relaxer(f, univariate, alphabb, multivariate=multivariate)

    def bound(blo, bhi, cutoff=jnp.inf):
        data = rfu.prepare(blo, bhi)

        def cv(x):
            return rfu(x, blo, bhi, data).cv

        vg = jax.value_and_grad(cv)
        c = 0.5 * (blo + bhi)
        ilb = rfu(c, blo, bhi, data).lo

        def step(carry, k):
            x, best, rlo, rhi = carry
            v, g = vg(x)
            vert = jnp.where(g > 0, blo, bhi)
            lb = v + jnp.dot(g, vert - x)
            ok = jnp.isfinite(lb) & jnp.all(jnp.isfinite(g))
            best = jnp.maximum(best, jnp.where(ok, lb, -jnp.inf))
            klo, khi = reduce_ranges(blo, bhi, jnp.where(ok, lb, -jnp.inf), jnp.where(ok, g, 0.0), cutoff)
            gamma = 2.0 / (k + 2.0)
            xn = jnp.where(ok, x + gamma * (vert - x), x)
            return (xn, best, jnp.maximum(rlo, klo), jnp.minimum(rhi, khi)), None

        init = (c, -jnp.inf, blo, bhi)
        (xfw, cut, rlo, rhi), _ = jax.lax.scan(step, init, jnp.arange(n_fw))
        lb = jnp.maximum(jnp.where(jnp.isnan(ilb), -jnp.inf, ilb), cut)
        fc, ff = f(c), f(xfw)
        use_c = ~(ff < fc)
        return lb, jnp.where(use_c, c, xfw), jnp.where(use_c, fc, ff), rlo, rhi

    return bound


def _hessian(fn):
    return jax.jit(jax.hessian(lambda x: jnp.atleast_1d(fn(x))))


def nonlinear_variables(fns, lo, hi, n_samples=3, seed=0, hessians=None):
    """Variables with a nonzero Hessian row in any function (sampled at random points).

    Branching on a variable that only enters linearly does not tighten
    McCormick relaxations, so B&B only splits the nonlinear ones.
    ``hessians`` are prebuilt ``_hessian(fn)`` for ``fns`` (reused across calls).
    """
    rng = np.random.default_rng(seed)
    nl = np.zeros(lo.size, bool)
    for fn, hess in zip(fns, hessians or [_hessian(fn) for fn in fns]):
        for _ in range(n_samples):
            H = np.asarray(hess(jnp.asarray(rng.uniform(lo, hi))))  # (m, n, n)
            nl |= np.any(H != 0, axis=(0, 2)) | ~np.all(np.isfinite(H), axis=(0, 2))
    return nl if nl.any() else np.ones(lo.size, bool)


# Compiled B&B kernels, reused by later ``minimize`` calls on the same
# functions and options (most of a small solve is tracing and XLA compilation).
# Functions are keyed by identity, as ``jax.jit`` does, so a function whose
# behavior changes through mutable globals is not recompiled; call
# ``clear_cache()`` after such a change.
_CACHE = OrderedDict()
_CACHE_SIZE = 32


def clear_cache():
    """Forget the compiled kernels kept across ``minimize`` calls."""
    _CACHE.clear()


def _cached(key, build):
    """LRU lookup of ``key`` in the kernel cache; unhashable keys are not cached."""
    try:
        hit = key in _CACHE
    except TypeError:
        return build()
    if hit:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    val = _CACHE[key] = build()
    while len(_CACHE) > _CACHE_SIZE:
        _CACHE.popitem(last=False)
    return val


class _Kernels:
    """Everything ``minimize`` compiles for one problem, independent of the box.

    ``bound(lo, hi, cutoff)`` tightens a batch of boxes by FBBT and bounds them;
    ``obbt(lo, hi, cutoff)`` tightens a batch by OBBT then FBBT. They are
    separate kernels that share no code, so neither the bounder nor the OBBT
    LPs are compiled twice (a single kernel that switches OBBT on and off with
    ``lax.cond`` compiles about twice as slowly: batching control flow around
    the OBBT body processes it several times).
    """

    def __init__(self, f, ineq, eq, *, n_fw, n_kelley, fbbt, fbbt_rounds, dbbt, obbt, obbt_rounds,
                 univariate, alphabb, multivariate, lifted, feas_tol):  # fmt: skip
        constrained = ineq is not None or eq is not None
        g, h = _vec(ineq), _vec(eq)

        def violation(x):
            return jnp.maximum(jnp.max(jnp.maximum(g(x), 0.0), initial=0.0), jnp.max(jnp.abs(h(x)), initial=0.0))

        if lifted:
            lb_lifted = make_lifted_bounder(f, ineq, eq, n_kelley=n_kelley, fbbt_rounds=fbbt_rounds, tol=feas_tol,
                                            univariate=univariate, alphabb=alphabb,
                                            multivariate=multivariate)

            def raw_bound(blo, bhi, cutoff):
                lb, x, _, rlo, rhi = lb_lifted(blo, bhi, cutoff)
                fx = jnp.where(violation(x) <= feas_tol, f(x), jnp.inf)
                return lb, x, fx, rlo, rhi
        elif constrained:
            cb = make_constrained_bounder(f, ineq, eq, n_kelley=n_kelley, tol=feas_tol, univariate=univariate,
                                          alphabb=alphabb, multivariate=multivariate)

            def raw_bound(blo, bhi, cutoff):
                lb, x, _, rlo, rhi = cb(blo, bhi, cutoff)
                fx = jnp.where(violation(x) <= feas_tol, f(x), jnp.inf)
                return lb, x, fx, rlo, rhi
        else:
            raw_bound = make_bounder(f, n_fw, univariate=univariate, alphabb=alphabb, multivariate=multivariate)
        tighten = make_fbbt(f, ineq, eq, rounds=fbbt_rounds, feas_tol=feas_tol) if fbbt else None
        lp_tighten = make_obbt(f, ineq, eq, rounds=obbt_rounds, tol=feas_tol, univariate=univariate,
                                alphabb=alphabb, multivariate=multivariate) if obbt else None

        def node(blo, bhi, cutoff):
            """Tighten, then bound one box. Returns (lb, x, f(x), lo', hi')."""
            empty = jnp.array(False)
            if tighten is not None:
                blo, bhi, empty = tighten(blo, bhi, cutoff)
            lb, x, fx, rlo, rhi = raw_bound(blo, bhi, cutoff)
            if not dbbt:
                rlo, rhi = blo, bhi
            # an empty box holds no feasible point with f <= cutoff
            return (jnp.where(empty, cutoff, lb), x, jnp.where(empty, jnp.inf, fx),
                    jnp.where(empty, blo, rlo), jnp.where(empty, bhi, rhi))  # fmt: skip

        def node_obbt(blo, bhi, cutoff):
            """FBBT, then OBBT on the result. Returns (lo', hi', empty).

            ``node`` runs FBBT again on the OBBT box, so the sequence is
            FBBT, OBBT, FBBT, bound."""
            empty = jnp.array(False)
            if tighten is not None:
                blo, bhi, empty = tighten(blo, bhi, cutoff)
            blo, bhi, e = lp_tighten(blo, bhi, cutoff)
            return blo, bhi, empty | e

        self.bound = jax.jit(jax.vmap(node, in_axes=(0, 0, None)))
        self.obbt = jax.jit(jax.vmap(node_obbt, in_axes=(0, 0, None))) if obbt else None
        self.viol = jax.jit(jax.vmap(violation))
        self.fvals = jax.jit(jax.vmap(f))
        self.fns = [fn for fn in (f, ineq, eq) if fn is not None]
        self.hessians = [_hessian(fn) for fn in self.fns]


def _vec(fn):
    return lambda x: jnp.atleast_1d(fn(x)) if fn is not None else jnp.zeros(0)


@dataclass
class Result:
    x: np.ndarray
    fun: float
    lower_bound: float
    converged: bool
    iterations: int
    nodes: int
    time: float
    history: list = field(default_factory=list)
    status: str = ""
    message: str = ""

    @property
    def gap(self):
        return self.fun - self.lower_bound

    def __repr__(self):
        return (
            f"Result(status={self.status!r}, fun={self.fun:.8g}, lower_bound={self.lower_bound:.8g}, "
            f"gap={self.gap:.2e}, x={np.array2string(self.x, precision=6)}, "
            f"nodes={self.nodes}, iterations={self.iterations}, time={self.time:.2f}s)"
        )


def minimize(
    f,
    lo,
    hi,
    *,
    ineq=None,
    eq=None,
    abs_tol=1e-6,
    rel_tol=1e-6,
    feas_tol=1e-6,
    batch=256,
    max_iter=5000,
    n_fw=20,
    n_kelley=4,
    local_solve=True,
    local_solver="auto",
    n_starts=1,
    fbbt=True,
    fbbt_rounds=5,
    dbbt=True,
    obbt="root",
    obbt_rounds=2,
    univariate=True,
    alphabb=True,
    multivariate=False,
    lifted=False,
    time_limit=np.inf,
    verbose=False,
):
    """Globally minimize ``f(x)`` over ``lo <= x <= hi``, ``ineq(x) <= 0``, ``eq(x) == 0``.

    ``f`` is any JAX-traceable Python function of a 1-D array returning a
    scalar; ``ineq`` / ``eq`` return a scalar or 1-D array of constraint values.
    A point is accepted as feasible when ``ineq <= feas_tol`` and ``|eq| <= feas_tol``.
    ``local_solver`` picks the upper-bounding local NLP solver: ``"pounce"``,
    ``"optax"`` (pure JAX), ``"scipy"``, or ``"auto"`` (pounce if installed).
    The search stops with status ``"time_limit"`` after ``time_limit`` seconds
    (checked between batches, so compile time counts).

    Bounds may be infinite: finite ones are derived first from the constraints,
    unused variables, and FBBT against a feasible point found by local solves
    (``domain.py``). If some variable stays unbounded the result has status
    ``"unbounded_domain"`` and ``message`` names the variables.

    Bound tightening against the incumbent: ``fbbt`` (forward-backward interval
    propagation over the jaxpr, ``fbbt_rounds`` sweeps per node), ``dbbt``
    (duality-based range reduction from each node's bound certificate), and
    ``obbt`` (LP-based tightening of every variable: ``"root"``, ``"tree"`` for
    every node, or ``False``).

    Relaxations: ``univariate`` replaces the relaxation of every smooth
    sub-expression of a single variable by a grid-hull envelope
    (``univariate.py``); ``alphabb`` adds whole-function alphaBB with
    scaled-Gershgorin alpha from an interval Hessian for smooth functions
    (``alphabb.py``; ``alphabb="eigen"`` adds the eigenvalue relaxation of
    exact quadratics); ``multivariate`` relaxes products by the multivariate
    McCormick rule of Tsoukalas & Mitsos, written as up to two extra facets
    from its dual (tighter where a factor's interval straddles zero); ``lifted`` bounds nodes with the extended-space LP
    (``lifted.py``) instead of the reduced-space cuts.
    """
    t0 = time.perf_counter()
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    seed_x, seed_f = None, np.inf
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        # derive finite bounds from the constraints and a feasible point (domain.py)
        lo, hi, infeasible0, seed_x, seed_f = bound_domain(f, ineq, eq, lo, hi, feas_tol=feas_tol)
        if infeasible0 or not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
            free = np.flatnonzero(~(np.isfinite(lo) & np.isfinite(hi)))
            status = "infeasible" if infeasible0 else "unbounded_domain"
            return Result(
                x=np.asarray(seed_x if seed_x is not None else np.clip(0.0, lo, hi)),
                fun=float(seed_f) if not infeasible0 else np.inf,
                lower_bound=np.inf if infeasible0 else -np.inf,
                converged=bool(infeasible0), iterations=0, nodes=0, time=time.perf_counter() - t0,
                status=status, message="" if infeasible0 else f"no finite bounds found for variables {free.tolist()}",
            )  # fmt: skip
    width0 = np.where(hi > lo, hi - lo, 1.0)
    constrained = ineq is not None or eq is not None
    if obbt not in (False, None, "root", "tree"):
        raise ValueError(f"obbt must be False, 'root' or 'tree', not {obbt!r}")
    opts = dict(n_fw=n_fw, n_kelley=n_kelley, fbbt=bool(fbbt), fbbt_rounds=fbbt_rounds, dbbt=bool(dbbt),
                obbt=bool(obbt), obbt_rounds=obbt_rounds, univariate=univariate, alphabb=alphabb,
                multivariate=multivariate, lifted=lifted, feas_tol=feas_tol)  # fmt: skip
    K = _cached(("kernels", f, ineq, eq, tuple(sorted(opts.items()))), lambda: _Kernels(f, ineq, eq, **opts))
    branchable = nonlinear_variables(K.fns, lo, hi, hessians=K.hessians)
    width0 = np.where(branchable, width0, np.inf)  # never split linear variables

    solve_local = _cached(("local", f, ineq, eq, local_solver, lo.tobytes(), hi.tobytes()),
                          lambda: make_local(f, ineq, eq, lo, hi, local_solver))

    def local(starts):
        """Local solves from each start; returns (x, f) of the best feasible result."""
        X = np.clip(np.asarray(solve_local(np.asarray(starts, dtype=np.float64))), lo, hi)
        F = np.asarray(K.fvals(jnp.asarray(X)), dtype=np.float64)
        if constrained:
            F = np.where(np.asarray(K.viol(jnp.asarray(X))) <= feas_tol, F, np.inf)
        F = np.where(np.isnan(F), np.inf, F)
        i = int(np.argmin(F))
        return X[i], float(F[i])

    def padded(blo, bhi):
        pad = batch - len(blo)
        return (jnp.asarray(np.concatenate([blo, np.repeat(blo[:1], pad, 0)])),
                jnp.asarray(np.concatenate([bhi, np.repeat(bhi[:1], pad, 0)])))  # fmt: skip

    def evaluate(blo, bhi, with_obbt=False):
        """Tighten and bound a batch of boxes: (lb, x, f(x), lo', hi')."""
        k = len(blo)
        cutoff = jnp.asarray(ub, dtype=jnp.float64)
        empty = np.zeros(k, bool)
        if with_obbt:
            # the root OBBT pass runs on one box: do not pad it to a full batch
            # (padding would compile the OBBT kernel at the batch shape as well)
            args = padded(blo, bhi) if obbt == "tree" else (jnp.asarray(blo), jnp.asarray(bhi))
            olo, ohi, oe = (np.asarray(a)[:k] for a in K.obbt(*args, cutoff))
            blo, bhi, empty = np.where(oe[:, None], blo, olo), np.where(oe[:, None], bhi, ohi), oe
        lb, x, fx, rlo, rhi = (np.asarray(a)[:k] for a in K.bound(*padded(blo, bhi), cutoff))
        # an empty box holds no feasible point with f <= cutoff
        return (np.where(empty, ub, lb), x, np.where(empty, np.inf, fx),
                np.where(empty[:, None], blo, rlo), np.where(empty[:, None], bhi, rhi))  # fmt: skip

    ub, xbest = np.inf, 0.5 * (lo + hi)
    if seed_x is not None:  # feasible point found while bounding the domain
        ub, xbest = float(seed_f), np.asarray(seed_x)

    def update_incumbent(xc, fc, lb):
        """Improve the incumbent from candidate points and local solves."""
        nonlocal ub, xbest
        starts = []
        if np.any(np.isfinite(fc)):
            i = int(np.nanargmin(fc))
            if fc[i] < ub:
                ub, xbest = float(fc[i]), xc[i]
                starts.append(xc[i])
        if constrained:
            # relaxation minimizers of the most promising boxes are good starts
            n_try = n_starts * (4 if not np.isfinite(ub) else 1)
            for i in np.argsort(lb)[:n_try]:
                if lb[i] < ub - tol():
                    starts.append(xc[i])
        if local_solve and starts:
            xl, fl = local(starts)
            if fl < ub:
                ub, xbest = fl, xl

    def tol():
        return max(abs_tol, rel_tol * abs(ub)) if np.isfinite(ub) else abs_tol

    lb, xc, fc, rlo, rhi = evaluate(lo[None], hi[None])
    update_incumbent(xc, fc, lb)
    if obbt and (constrained or np.isfinite(ub)):
        # root OBBT once an incumbent cutoff is known (it is useless without
        # one when there are no constraints)
        lb2, xc, fc, rlo, rhi = evaluate(rlo, rhi, with_obbt=True)
        lb = np.maximum(lb, lb2)
        update_incumbent(xc, fc, lb)
    open_lo, open_hi, open_lb = rlo, rhi, lb
    pruned_lb = np.inf  # smallest lower bound among fathomed boxes
    nodes, history = 1, []
    it = 0
    converged = False
    for it in range(1, max_iter + 1):
        open_lb = np.where(np.isnan(open_lb), -np.inf, open_lb)  # NaN bound = no information
        keep = open_lb < ub - tol()
        if not keep.all():
            pruned_lb = min(pruned_lb, float(open_lb[~keep].min()))
        open_lo, open_hi, open_lb = open_lo[keep], open_hi[keep], open_lb[keep]
        glb = min(float(open_lb.min()) if len(open_lb) else np.inf, pruned_lb, ub)
        history.append((it, nodes, glb, ub, len(open_lb)))
        if verbose and (it == 1 or it % 25 == 0):
            print(f"it {it:5d} nodes {nodes:8d} open {len(open_lb):7d} LB {glb:.8g} UB {ub:.8g}")
        if not len(open_lb) or ub - glb <= tol():
            converged = True
            break
        if time.perf_counter() - t0 > time_limit:
            break

        k = min(batch // 2, len(open_lb))
        sel = np.argpartition(open_lb, k - 1)[:k] if k < len(open_lb) else np.arange(k)
        rest = np.ones(len(open_lb), bool)
        rest[sel] = False
        plo, phi = open_lo[sel], open_hi[sel]

        d = np.argmax(np.where(branchable, (phi - plo) / width0, -1.0), axis=1)
        mid = 0.5 * (plo[np.arange(k), d] + phi[np.arange(k), d])
        left_hi, right_lo = phi.copy(), plo.copy()
        left_hi[np.arange(k), d] = mid
        right_lo[np.arange(k), d] = mid
        clo = np.concatenate([plo, right_lo])
        chi = np.concatenate([left_hi, phi])

        clb, cx, cf, clo, chi = evaluate(clo, chi, with_obbt=obbt == "tree")
        nodes += 2 * k
        # children cannot be worse than their parent's bound
        clb = np.maximum(clb, np.concatenate([open_lb[sel], open_lb[sel]]))
        update_incumbent(cx, cf, clb)

        open_lo = np.concatenate([open_lo[rest], clo])
        open_hi = np.concatenate([open_hi[rest], chi])
        open_lb = np.concatenate([open_lb[rest], clb])

    glb = min(float(open_lb.min()) if len(open_lb) else np.inf, pruned_lb, ub)
    if converged:
        status = "optimal" if np.isfinite(ub) else "infeasible"
    elif time.perf_counter() - t0 > time_limit:
        status = "time_limit"
    else:
        status = "max_iter"
    return Result(
        x=np.asarray(xbest),
        fun=ub,
        lower_bound=glb,
        converged=converged,
        iterations=it,
        nodes=nodes,
        time=time.perf_counter() - t0,
        history=history,
        status=status,
    )
