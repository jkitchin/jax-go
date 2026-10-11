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

import functools
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from .local import make_local
from .crown import crown_auto, crown_supported, make_crown
from .domain import bound_domain
from . import kcache
from .fbbt import make_fbbt
from .lifted import make_lifted_bounder
from .lp import make_constrained_bounder, make_obbt, reduce_ranges
from .relax import SAFE_K
from .univariate import make_relaxer


def make_bounder(f, n_fw=20, univariate=False, alphabb=False, multivariate=False, return_xr=False, safe=False):
    """Return ``bound(lo, hi, cutoff) -> (lb, x, f(x), lo', hi')`` for one box.

    ``[lo', hi']`` is the box after duality-based range reduction: every
    Frank-Wolfe cut ``f(z) >= cv(x) + g.(z - x)`` with ``f(z) <= cutoff``
    confines each ``z_j`` to an interval. With ``return_xr`` a sixth output is
    the final Frank-Wolfe iterate (the relaxation minimizer estimate).
    """
    rfu = make_relaxer(f, univariate, alphabb, multivariate=multivariate, safe=safe)
    eps = float(jnp.finfo(jnp.float64).eps)

    def bound(blo, bhi, cutoff=jnp.inf):
        data = rfu.prepare(blo, bhi)

        def cv(x):
            return rfu(x, blo, bhi, data).cv

        vg = jax.value_and_grad(cv)
        c = 0.5 * (blo + bhi)
        ilb = rfu(c, blo, bhi, data).lo

        def step(carry, k):
            x, best, rlo, rhi = carry
            # one evaluation of (v, g) for the vertex choice and the cut: XLA may otherwise
            # recompute g where it is used and break a facet tie differently (see lifted.node_cuts)
            v, g = jax.lax.optimization_barrier(vg(x))
            vert = jnp.where(g > 0, blo, bhi)
            lb = v + jnp.dot(g, vert - x)
            if safe:  # rounding of the cut value (the subgradient itself is exact for the widened cv)
                lb = lb - SAFE_K * (blo.size + 3) * eps * (jnp.abs(v) + jnp.abs(g) @ jnp.abs(vert - x) + jnp.abs(lb))
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
        out = lb, jnp.where(use_c, c, xfw), jnp.where(use_c, fc, ff), rlo, rhi
        return out + (xfw,) if return_xr else out

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
SAFE_DEFAULT = False  # outward rounding of every bound; set after measuring its overhead (docs/ROADMAP.md)
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


_POOLS = []


def _pools():
    """(tracer, compiler) thread pools for ``_Precompiled``, created on first use."""
    if not _POOLS:
        from concurrent.futures import ThreadPoolExecutor

        _POOLS.extend([ThreadPoolExecutor(1, thread_name_prefix="jaxglobal-trace"),
                       ThreadPoolExecutor(4, thread_name_prefix="jaxglobal-compile")])  # fmt: skip
    return _POOLS


class _Precompiled:
    """A jitted batch kernel plus executables compiled ahead of time in the background.

    On its first call with a new shape ``jax.jit`` traces, lowers and compiles
    on the calling thread, one kernel after another. ``start(b, n)`` instead
    queues trace and lowering on a tracer thread and hands each lowered kernel
    to a compiler pool. XLA compilation releases the GIL, so one kernel
    compiles while the next is traced, and both overlap the rest of the setup
    in ``minimize`` (Hessians, local solver). Calls with a started shape wait
    for that executable; any other call goes through ``jax.jit``.

    With a problem fingerprint (``kcache.py``) the lowered kernel is also
    stored on disk with ``jax.export``, and a later process loads it instead
    of tracing again.
    """

    def __init__(self, jitted, name):
        self.jitted, self.name, self.ready = jitted, name, {}

    @staticmethod
    def _key(args):
        return tuple((a.shape, a.dtype, getattr(a, "weak_type", False)) for a in args)

    def start(self, b, n, fingerprint=None):
        """Compile in the background for ``b`` boxes of dimension ``n`` (single device).

        ``fingerprint()``, called on the tracer thread, names the problem for
        the disk cache (None: do not cache)."""
        args = (jax.ShapeDtypeStruct((b, n), jnp.float64),) * 2 + (jax.ShapeDtypeStruct((), jnp.float64),)
        key = self._key(args)
        if key not in self.ready:
            tracer, compiler = _pools()
            self.ready[key] = tracer.submit(lambda: compiler.submit(self._lower(args, fingerprint).compile))

    def _lower(self, args, fingerprint):
        """The lowered kernel, loaded from the disk cache when possible (and stored there when not)."""
        fp = fingerprint() if fingerprint is not None else None
        if fp is not None:
            fp = f"{fp}-{self.name}-{args[0].shape[0]}"
            exp = kcache.load(fp)
            if exp is None:
                try:
                    exp = jax.export.export(self.jitted)(*args)
                except Exception:  # e.g. a host callback in f: trace without the cache
                    exp = None
                kcache.save(fp, exp)
            if exp:
                return jax.jit(exp.call).trace(*args).lower()
        return self.jitted.trace(*args).lower()

    def __call__(self, *args):
        key = self._key(args)
        fut = self.ready.get(key)
        if fut is not None:
            try:
                return fut.result().result()(*args)
            except Exception:  # let jit raise (or succeed) with its usual traceback
                del self.ready[key]
        return self.jitted(*args)


class _Kernels:
    """Everything ``minimize`` compiles for one problem, independent of the box.

    ``bound(lo, hi, cutoff)`` tightens a batch of boxes by FBBT and bounds them;
    ``obbt(lo, hi, cutoff)`` tightens a batch by OBBT (FBBT first for ``obbt="tree"``). They are
    separate kernels that share no code, so neither the bounder nor the OBBT
    LPs are compiled twice (a single kernel that switches OBBT on and off with
    ``lax.cond`` compiles about twice as slowly: batching control flow around
    the OBBT body processes it several times).
    """

    def __init__(self, f, ineq, eq, *, n_fw, n_kelley, fbbt, fbbt_rounds, dbbt, obbt, obbt_rounds,
                 univariate, alphabb, multivariate, lifted, crown, feas_tol, safe):  # fmt: skip
        opts = {k: v for k, v in locals().items() if k not in ("self", "f", "ineq", "eq")}
        constrained = ineq is not None or eq is not None
        g, h = _vec(ineq), _vec(eq)

        def violation(x):
            return jnp.maximum(jnp.max(jnp.maximum(g(x), 0.0), initial=0.0), jnp.max(jnp.abs(h(x)), initial=0.0))

        if lifted:
            lb_lifted = make_lifted_bounder(f, ineq, eq, n_kelley=n_kelley, fbbt_rounds=fbbt_rounds, tol=feas_tol,
                                            univariate=univariate, alphabb=alphabb,
                                            multivariate=multivariate, safe=safe)

            def raw_bound(blo, bhi, cutoff):
                lb, x, _, rlo, rhi = lb_lifted(blo, bhi, cutoff)
                fx = jnp.where(violation(x) <= feas_tol, f(x), jnp.inf)
                return lb, x, fx, rlo, rhi, x
        elif constrained:
            cb = make_constrained_bounder(f, ineq, eq, n_kelley=n_kelley, tol=feas_tol, univariate=univariate,
                                          alphabb=alphabb, multivariate=multivariate, safe=safe)

            def raw_bound(blo, bhi, cutoff):
                lb, x, _, rlo, rhi = cb(blo, bhi, cutoff)
                fx = jnp.where(violation(x) <= feas_tol, f(x), jnp.inf)
                return lb, x, fx, rlo, rhi, x
        else:
            raw_bound = make_bounder(f, n_fw, univariate=univariate, alphabb=alphabb, multivariate=multivariate,
                                     return_xr=True, safe=safe)
        tighten = make_fbbt(f, ineq, eq, rounds=fbbt_rounds, feas_tol=feas_tol, safe=safe) if fbbt else None
        lp_tighten = make_obbt(f, ineq, eq, rounds=obbt_rounds, tol=feas_tol, univariate=univariate,
                                alphabb=alphabb, multivariate=multivariate, safe=safe) if obbt else None

        crown_lb = make_crown(f, safe=safe) if crown else None

        def node(blo, bhi, cutoff):
            """Tighten, then bound one box. Returns (lb, x, f(x), lo', hi', xr)."""
            empty = jnp.array(False)
            if tighten is not None:
                blo, bhi, empty = tighten(blo, bhi, cutoff)
            lb, x, fx, rlo, rhi, xr = raw_bound(blo, bhi, cutoff)
            if crown_lb is not None:  # both bounds are valid on the box: take the larger
                lb = jnp.maximum(jnp.where(jnp.isnan(lb), -jnp.inf, lb), crown_lb(blo, bhi))
            if not dbbt:
                rlo, rhi = blo, bhi
            # an empty box holds no feasible point with f <= cutoff
            return (jnp.where(empty, cutoff, lb), x, jnp.where(empty, jnp.inf, fx),
                    jnp.where(empty, blo, rlo), jnp.where(empty, bhi, rhi),
                    jnp.where(empty, 0.5 * (blo + bhi), xr))  # fmt: skip

        def node_obbt(blo, bhi, cutoff):
            """OBBT (after FBBT for ``obbt="tree"``). Returns (lo', hi', empty).

            ``node`` runs FBBT again on the OBBT box, so the sequence is
            FBBT, OBBT, FBBT, bound. The root box has already been through
            ``node`` (FBBT included), so for ``obbt="root"`` the leading FBBT
            is left out: it would only add its trace and compile time."""
            empty = jnp.array(False)
            if tighten is not None and obbt == "tree":
                blo, bhi, empty = tighten(blo, bhi, cutoff)
            blo, bhi, e = lp_tighten(blo, bhi, cutoff)
            return blo, bhi, empty | e

        self.bound = _Precompiled(jax.jit(jax.vmap(node, in_axes=(0, 0, None))), "bound")
        self.obbt = _Precompiled(jax.jit(jax.vmap(node_obbt, in_axes=(0, 0, None))), "obbt") if obbt else None
        self._problem, self._opts, self._fingerprints = (f, ineq, eq), opts, {}
        self.viol = jax.jit(jax.vmap(violation))
        self.fvals = jax.jit(jax.vmap(f))
        self.fns = [fn for fn in (f, ineq, eq) if fn is not None]
        self.hessians = [_hessian(fn) for fn in self.fns]

    def fingerprint(self, n):
        """Disk-cache key of this problem on ``n`` variables (``kcache.fingerprint``), or None."""
        if n not in self._fingerprints:
            self._fingerprints[n] = kcache.fingerprint(self._problem, n, sorted(self._opts.items()))
        return self._fingerprints[n]


def _vec(fn):
    return lambda x: jnp.atleast_1d(fn(x)) if fn is not None else jnp.zeros(0)


def _verify(f, ineq, eq, x, fun, lo, hi, feas_tol):
    """Re-evaluate ``x`` outside jit in float64: ``(max_violation, ok, reason)``.

    Checks the box, ``ineq <= feas_tol``, ``|eq| <= feas_tol`` and
    ``f(x) == fun`` (to 1e-8 relative). With no incumbent (``fun`` infinite)
    there is nothing to verify: ``(inf, False, "")``.
    """
    if not np.isfinite(fun):
        return float("inf"), False, ""
    xj = jnp.asarray(x, dtype=jnp.float64)
    with jax.disable_jit():
        fx = float(np.sum(np.asarray(f(xj))))
        g = np.atleast_1d(np.asarray(ineq(xj), dtype=np.float64)) if ineq is not None else np.zeros(0)
        h = np.atleast_1d(np.asarray(eq(xj), dtype=np.float64)) if eq is not None else np.zeros(0)
    box = max(float(np.max(np.maximum(lo - x, 0.0), initial=0.0)), float(np.max(np.maximum(x - hi, 0.0), initial=0.0)))
    viol = max(box, float(np.max(np.maximum(g, 0.0), initial=0.0)), float(np.max(np.abs(h), initial=0.0)))
    if np.isnan(viol) or np.isnan(fx):
        return float("nan"), False, "NaN in x, f or a constraint"
    why = []
    if box > 1e-9 * (1 + np.max(np.abs(np.concatenate([lo, hi])))):
        why.append(f"x outside the box by {box:.3g}")
    if viol - box > feas_tol:
        why.append(f"constraint violation {viol:.3g} > feas_tol {feas_tol:.3g}")
    if abs(fx - fun) > 1e-8 * (1 + abs(fun)):
        why.append(f"f(x) = {fx:.12g} != fun = {fun:.12g}")
    return viol, not why, "; ".join(why)


class _Padder:
    """Cut batches of boxes into calls of a few fixed sizes; spread big calls over devices.

    Every distinct batch shape is traced and compiled separately, and on CPU a
    call costs about the same per box from a few dozen boxes up, so ``sizes``
    is kept small (``batch_sizes``, capped by ``batch``; the full ``batch``
    when empty). A batch up to the largest size is padded to the smallest size
    that holds it; a bigger one is cut into calls of the largest size. With
    several JAX devices (e.g. CPU cores exposed by
    ``XLA_FLAGS=--xla_force_host_platform_device_count=N``) the largest size is
    rounded up to a multiple of the device count and its calls are split across
    them.
    """

    def __init__(self, batch, sizes, devices):
        self.mesh = None
        sizes = {min(int(s), batch) for s in sizes} or {batch}
        if len(devices) > 1:
            from jax.sharding import Mesh, NamedSharding, PartitionSpec

            self.mesh = NamedSharding(Mesh(np.array(devices), ("boxes",)), PartitionSpec("boxes"))
            top = max(sizes)
            sizes = (sizes - {top}) | {-(-top // len(devices)) * len(devices)}
        self.sizes = sorted(sizes)

    def __call__(self, *arrays):
        """List of calls, each a list of the arrays padded to one of ``sizes``."""
        k, top = len(arrays[0]), self.sizes[-1]
        calls = []
        for s in range(0, k, top):
            part = [a[s : s + top] for a in arrays]
            m = len(part[0])
            size = next(z for z in self.sizes if z >= m)
            out = [np.concatenate([a, np.repeat(a[:1], size - m, 0)]) for a in part]
            if self.mesh is not None and size == top:
                calls.append([jax.device_put(a, self.mesh) for a in out])
            else:
                calls.append([jnp.asarray(a) for a in out])
        return calls


_LAM, _BETA = 0.75, 0.2  # branch_point="relax": pull toward the relaxation minimizer, margin from faces
_FLOOR = 0.05  # pseudocosts are floored at this fraction of the mean over observed variables
_ETA, _M = 2, 4  # branch_var="reliability": observations to trust a pseudocost, strong-branching candidates


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
    evaluations: int = 0  # boxes bounded, including strong-branching trials
    max_violation: float = float("nan")  # of x, re-evaluated outside jit: box, ineq > 0, |eq|
    verified: bool = False  # x is feasible to feas_tol and f(x) == fun, checked in float64 after the search

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
    batch_sizes=(16,),
    devices=None,
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
    branch_point="mid",
    branch_var="width",
    univariate=True,
    alphabb=True,
    multivariate=False,
    lifted=False,
    crown=False,
    safe=SAFE_DEFAULT,
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
    (``lifted.py``) instead of the reduced-space cuts. ``crown`` adds the CROWN
    backward linear bound of ``f`` (``crown.py``) to every node bound (the
    larger of the two is used): ``False``, ``True``, or ``"auto"`` (on when
    every primitive of ``f`` has a CROWN rule and ``f`` has a smooth
    saturating activation or ReLU; otherwise off). With ``True`` and an
    unsupported ``f`` the CROWN bound is ``-inf``, i.e. no effect.

    ``safe`` rounds every quantity feeding a lower bound outward (interval and
    McCormick values, envelopes, alphaBB, CROWN lines, LP rows and the dual
    certificate) at the primitive-dispatch level; see docs/ROADMAP.md for what
    that does and does not guarantee.

    After the search the incumbent is re-evaluated outside jit in float64
    (box, ``ineq``, ``eq``, ``f(x) == fun``): ``Result.max_violation`` and
    ``Result.verified`` report it, and a failed check changes status
    ``"optimal"`` to ``"unverified"`` (with ``message``) instead of passing silently.

    Branching: ``branch_point`` is ``"mid"`` (bisect) or ``"relax"`` (split at
    the relaxation minimizer, pulled toward the midpoint and kept away from the
    faces, see ``_LAM`` and ``_BETA``); ``branch_var`` is ``"width"`` (largest
    relative width among nonlinear variables), ``"pseudocost"`` (product score
    of per-variable bound gains observed so far) or ``"reliability"`` (like
    pseudocost, but variables with fewer than ``_ETA`` observations are
    strong-branched, up to ``_M`` of them per box, within the ``batch`` budget).
    ``Result.evaluations`` counts every box bounded.

    Up to ``batch`` boxes are bounded per iteration, in kernel calls of the
    sizes in ``batch_sizes`` (capped by ``batch``; just ``batch`` when empty):
    a batch is padded to the smallest size that holds it or cut into calls of
    the largest. Each size is traced and compiled once, so the default is a
    single small size. The kernels compile in background threads and are
    cached on disk across processes (``jaxglobal/kcache.py``; disable with
    ``JAXGLOBAL_CACHE=0``). With several JAX devices calls of the largest size
    are split across them; on CPU, expose cores by setting
    ``XLA_FLAGS=--xla_force_host_platform_device_count=N`` before JAX is
    imported. ``devices`` overrides the device list (e.g.
    ``devices=jax.devices()[:1]`` to disable splitting).
    """
    t0 = time.perf_counter()
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    lo_user, hi_user = lo, hi
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
    if branch_point not in ("mid", "relax"):
        raise ValueError(f"branch_point must be 'mid' or 'relax', not {branch_point!r}")
    if branch_var not in ("width", "pseudocost", "reliability"):
        raise ValueError(f"branch_var must be 'width', 'pseudocost' or 'reliability', not {branch_var!r}")
    if obbt not in (False, None, "root", "tree"):
        raise ValueError(f"obbt must be False, 'root' or 'tree', not {obbt!r}")
    if crown not in (False, None, True, "auto"):
        raise ValueError(f"crown must be False, True or 'auto', not {crown!r}")
    if crown == "auto":
        crown = crown_auto(f, lo)
    elif crown:
        crown = crown_supported(f, lo)  # an unsupported f gets no CROWN kernel
    opts = dict(n_fw=n_fw, n_kelley=n_kelley, fbbt=bool(fbbt), fbbt_rounds=fbbt_rounds, dbbt=bool(dbbt),
                obbt=obbt or False, obbt_rounds=obbt_rounds, univariate=univariate, alphabb=alphabb,
                multivariate=multivariate, lifted=lifted, crown=bool(crown), feas_tol=feas_tol, safe=bool(safe))  # fmt: skip
    K = _cached(("kernels", f, ineq, eq, tuple(sorted(opts.items()))), lambda: _Kernels(f, ineq, eq, **opts))
    padded = _Padder(batch, batch_sizes, jax.devices() if devices is None else list(devices))
    if padded.mesh is None:  # compile the kernels in the background while the setup below runs
        fp = functools.partial(K.fingerprint, lo.size) if kcache.cache_root() else None
        for b in padded.sizes:
            K.bound.start(b, lo.size, fp)
        if obbt:
            for b in padded.sizes if obbt == "tree" else [1]:
                K.obbt.start(b, lo.size, fp)
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

    def run(kernel, calls, cutoff, k):
        """Apply ``kernel`` to each padded call (dispatched before any result is read); first ``k`` rows."""
        fn = kernel.jitted if padded.mesh is not None else kernel  # sharded calls compile through jit
        outs = [fn(*c, cutoff) for c in calls]
        return [np.concatenate([np.asarray(o[i]) for o in outs])[:k] for i in range(len(outs[0]))]

    def evaluate(blo, bhi, with_obbt=False):
        """Tighten and bound a batch of boxes: (lb, x, f(x), lo', hi', xr)."""
        nonlocal evaluations
        k = len(blo)
        evaluations += k
        cutoff = jnp.asarray(ub, dtype=jnp.float64)
        empty = np.zeros(k, bool)
        if with_obbt:
            # the root OBBT pass runs on one box: do not pad it to a full batch
            # (padding would compile the OBBT kernel at the batch shape as well)
            calls = padded(blo, bhi) if obbt == "tree" else [[jnp.asarray(blo), jnp.asarray(bhi)]]
            olo, ohi, oe = run(K.obbt, calls, cutoff, k)
            blo, bhi, empty = np.where(oe[:, None], blo, olo), np.where(oe[:, None], bhi, ohi), oe
        lb, x, fx, rlo, rhi, xr = run(K.bound, padded(blo, bhi), cutoff, k)
        # an empty box holds no feasible point with f <= cutoff
        return (np.where(empty, ub, lb), x, np.where(empty, np.inf, fx),
                np.where(empty[:, None], blo, rlo), np.where(empty[:, None], bhi, rhi),
                np.where(empty[:, None] | ~np.isfinite(xr), 0.5 * (blo + bhi), xr))  # fmt: skip

    evaluations = 0
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

    lb, xc, fc, rlo, rhi, rxr = evaluate(lo[None], hi[None])
    update_incumbent(xc, fc, lb)
    if obbt and (constrained or np.isfinite(ub)) and not (ub - float(lb.min()) <= tol()):
        # root OBBT once an incumbent cutoff is known (it is useless without
        # one when there are no constraints), and not when the root bound has
        # already closed the gap (the search ends at once; OBBT would only
        # cost its own trace and compilation)
        lb2, xc, fc, rlo, rhi, rxr = evaluate(rlo, rhi, with_obbt=True)
        lb = np.maximum(lb, lb2)
        update_incumbent(xc, fc, lb)
    open_lo, open_hi, open_lb, open_xr = rlo, rhi, lb, rxr
    sumL, sumR, cnt = np.zeros(lo.size), np.zeros(lo.size), np.zeros(lo.size)  # pseudocost sums, observations

    def pseudo_scores(relw, free):
        """Product-rule scores per (box, variable), -1 for non-candidates; and whether any pseudocost is known."""
        cand = branchable & free
        cand = np.where(cand.any(axis=1, keepdims=True), cand, branchable)
        seen = cnt > 0
        if not seen.any():
            return np.where(cand, relw, -1.0), False
        mL, mR = (sumL[seen] / cnt[seen]).mean(), (sumR[seen] / cnt[seen]).mean()
        # a variable whose early splits gained nothing would otherwise never be tried again
        psiL = np.where(seen, np.maximum(sumL / np.maximum(cnt, 1), _FLOOR * mL), mL)
        psiR = np.where(seen, np.maximum(sumR / np.maximum(cnt, 1), _FLOOR * mR), mR)
        eps = 1e-6 * (1 + abs(ub)) if np.isfinite(ub) else 1e-6
        r2 = relw**2
        sc = np.maximum(psiL * r2, eps) * np.maximum(psiR * r2, eps) * (1 + 1e-3 * relw)  # width breaks ties
        return np.where(cand & np.isfinite(sc), sc, np.where(cand, 0.0, -1.0)), True

    pruned_lb = np.inf  # smallest lower bound among fathomed boxes
    nodes, history = 1, []
    it = 0
    converged = False
    for it in range(1, max_iter + 1):
        open_lb = np.where(np.isnan(open_lb), -np.inf, open_lb)  # NaN bound = no information
        keep = open_lb < ub - tol()
        if not keep.all():
            pruned_lb = min(pruned_lb, float(open_lb[~keep].min()))
        open_lo, open_hi, open_lb, open_xr = open_lo[keep], open_hi[keep], open_lb[keep], open_xr[keep]
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
        if branch_var == "reliability":
            sel = sel[np.argsort(open_lb[sel], kind="stable")]
        plo, phi, pxr, plb = open_lo[sel], open_hi[sel], open_xr[sel], open_lb[sel]
        relw = (phi - plo) / width0  # relative width (0 for variables never split)

        if branch_var == "width":
            pp = np.arange(k)
            pv = np.argmax(np.where(branchable, relw, -1.0), axis=1)
            groups = [[i] for i in range(k)]
        else:
            score, _ = pseudo_scores(relw, phi > plo)
            if branch_var == "pseudocost":
                pv = np.argmax(score, axis=1)
                pp = np.arange(k)
                groups = [[i] for i in range(k)]
            else:
                pp_l, pv_l, groups, used = [], [], [], 0
                for i in range(k):
                    cand = score[i] >= 0
                    unrel = np.flatnonzero(cand & (cnt < _ETA))
                    unrel = unrel[np.argsort(-relw[i, unrel], kind="stable")][:_M]
                    best = int(np.argmax(score[i]))
                    vs = [best] + [int(v) for v in unrel if v != best] if cnt[best] >= _ETA else [int(v) for v in unrel]
                    if not vs:
                        vs = [best]
                    if used and used + 2 * len(vs) > batch:
                        break
                    vs = vs[: max(batch // 2, 1)]
                    groups.append(list(range(len(pp_l), len(pp_l) + len(vs))))
                    pp_l += [i] * len(vs)
                    pv_l += vs
                    used += 2 * len(vs)
                pp, pv = np.array(pp_l), np.array(pv_l)
            if len(groups) < k:  # batch budget: take fewer parents this iteration
                k = len(groups)
                sel, plo, phi, pxr, plb, relw = sel[:k], plo[:k], phi[:k], pxr[:k], plb[:k], relw[:k]
        rest = np.ones(len(open_lb), bool)
        rest[sel] = False

        M = len(pp)
        ar = np.arange(M)
        lov, hiv = plo[pp, pv], phi[pp, pv]
        pt = 0.5 * (lov + hiv)
        if branch_point == "relax":
            w = hiv - lov
            pt = np.clip(_LAM * pxr[pp, pv] + (1 - _LAM) * pt, lov + _BETA * w, hiv - _BETA * w)
        left_hi, right_lo = phi[pp].copy(), plo[pp].copy()
        left_hi[ar, pv] = pt
        right_lo[ar, pv] = pt
        clo = np.concatenate([plo[pp], right_lo])
        chi = np.concatenate([left_hi, phi[pp]])

        clb, cx, cf, clo, chi, cxr = evaluate(clo, chi, with_obbt=obbt == "tree")
        update_incumbent(cx, cf, clb)
        if branch_var != "width":
            # pseudocost observations: bound gain per squared relative width
            gl, gr = np.maximum(clb[:M] - plb[pp], 0.0), np.maximum(clb[M:] - plb[pp], 0.0)
            if np.isfinite(ub):
                cap = np.maximum(ub - plb[pp], 0.0)
                gl, gr = np.minimum(gl, cap), np.minimum(gr, cap)
            else:  # no scale for a huge gain (an infeasible child) without an incumbent: skip it
                big = 1e12 * (1 + np.abs(plb[pp]))
                gl, gr = np.where(gl <= big, gl, np.inf), np.where(gr <= big, gr, np.inf)
            r2 = relw[pp, pv] ** 2
            good = np.isfinite(gl) & np.isfinite(gr) & (r2 > 1e-24)
            np.add.at(sumL, pv[good], gl[good] / r2[good])
            np.add.at(sumR, pv[good], gr[good] / r2[good])
            np.add.at(cnt, pv[good], 1)
        lbl, lbr = clb[:M], clb[M:]
        if M > k:  # strong branching: keep the best candidate of each parent
            eps = 1e-6 * (1 + abs(ub)) if np.isfinite(ub) else 1e-6
            gl, gr = np.maximum(lbl - plb[pp], 0.0), np.maximum(lbr - plb[pp], 0.0)
            gl, gr = np.where(np.isfinite(gl), gl, 1e150), np.where(np.isfinite(gr), gr, 1e150)
            strong = np.maximum(gl, eps) * np.maximum(gr, eps)
            chosen = np.array([g[np.lexsort((relw[pp[g], pv[g]], strong[g]))[-1]] for g in groups])
            pmax = np.array([np.max(np.minimum(lbl[g], lbr[g])) for g in groups])
        else:
            chosen = np.arange(M)
            pmax = np.minimum(lbl, lbr)
        rows = np.concatenate([chosen, chosen + M])
        nodes += 2 * k
        # children cannot be worse than their parent's bound, nor than the
        # weaker side of any tried split of it
        raised = np.maximum(plb, pmax)
        clb = np.maximum(clb[rows], np.concatenate([raised, raised]))

        open_lo = np.concatenate([open_lo[rest], clo[rows]])
        open_hi = np.concatenate([open_hi[rest], chi[rows]])
        open_lb = np.concatenate([open_lb[rest], clb])
        open_xr = np.concatenate([open_xr[rest], cxr[rows]])

    glb = min(float(open_lb.min()) if len(open_lb) else np.inf, pruned_lb, ub)
    if converged:
        status = "optimal" if np.isfinite(ub) else "infeasible"
    elif time.perf_counter() - t0 > time_limit:
        status = "time_limit"
    else:
        status = "max_iter"
    xbest = np.asarray(xbest)
    maxv, verified, why = _verify(f, ineq, eq, xbest, ub, lo_user, hi_user, feas_tol)
    message = ""
    if np.isfinite(ub) and not verified:
        message = f"incumbent failed float64 verification: {why}"
        if status == "optimal":
            status, converged = "unverified", False
    return Result(
        x=xbest,
        fun=ub,
        lower_bound=glb,
        converged=converged,
        iterations=it,
        nodes=nodes,
        evaluations=evaluations,
        time=time.perf_counter() - t0,
        history=history,
        status=status,
        message=message,
        max_violation=maxv,
        verified=verified,
    )
