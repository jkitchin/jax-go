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
import itertools
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from .local import make_local, near_zero_options
from .crown import crown_auto, crown_supported, make_crown
from .domain import DomainTimeout, bound_domain
from . import kcache
from .fbbt import make_fbbt
from .lifted import make_lifted_bounder
from .lp import make_constrained_bounder, make_obbt, make_pool_cuts, pool_parts, reduce_ranges
from .quadratic import binary_only, make_quadratic_bounder, make_quadratic_obbt, polynomial_form, quadratic_form
from .structure import structure
from .relax import SAFE_K
from .univariate import make_relaxer


def _gap_scores(Q, lo, hi, x):
    """Per (box, variable): the sum, over the products it is a factor of, of the gap between
    the McCormick over- and underestimators at the relaxation point ``x`` (how far the
    product's LP column can be from the product there)."""
    I, J = Q.pairs[:, 0], Q.pairs[:, 1]
    li, ui, lj, uj = lo[:, I], hi[:, I], lo[:, J], hi[:, J]
    xi, xj = np.clip(x[:, I], li, ui), np.clip(x[:, J], lj, uj)
    sq = I == J
    cv = np.where(sq, xi * xi, np.maximum(lj * xi + li * xj - li * lj, uj * xi + ui * xj - ui * uj))
    cc = np.where(sq, (li + ui) * xi - li * ui, np.minimum(lj * xi + ui * xj - ui * lj, uj * xi + li * xj - li * uj))
    gap = np.where(np.isfinite(cc - cv), np.maximum(cc - cv, 0.0), 0.0)
    out = np.zeros_like(lo)
    np.add.at(out.T, I, gap.T)
    np.add.at(out.T, J, np.where(sq, 0.0, gap).T)
    return out


_SHRINK = (8.0, 100.0)  # _branch_widths: rescale only variables root tightening narrowed at least [0]-fold, by at most [1]-fold


def _branch_widths(width0, rlo, rhi, shrink=_SHRINK):
    """The reference widths that branching scores measure relative widths against: ``width0``,
    except for a variable that root tightening (FBBT, OBBT; the widest over the root boxes)
    narrowed at least ``shrink[0]``-fold, which is measured against its root width instead,
    but at most ``shrink[1]``-fold narrower than ``width0``.

    Measured against the user's box, a variable that root OBBT narrowed a lot starts the search
    looking almost split already, and pseudocost (gains per squared relative width; untried
    variables scored at the mean pseudocost) then never tries it: ex8_4_1's slope, in all ten
    bilinear terms, is cut from [-2, 2] to width 0.24 and was never branched on (23811 nodes;
    1523 with its root width). Modest narrowing is left alone: rescaling nous1's variables
    narrowed 1.2-7x doubled its nodes. The ``shrink[1]`` floor keeps a variable that tightening
    all but fixed (width 1e-6 of 10 on pooling_adhya1stp) from looking as wide as an untouched
    one. Only branching scores use these widths: bounds are unaffected."""
    rw = np.max(np.asarray(rhi) - np.asarray(rlo), axis=0)
    ok = np.isfinite(width0) & np.isfinite(rw) & (rw > 0) & (rw * shrink[0] <= width0)
    return np.where(ok, np.maximum(np.where(ok, rw, 1.0), width0 / shrink[1]), width0)


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


def structural_nonlinear(f, ineq, eq, n):
    """Variables that enter some function nonlinearly, read off its tape: those an input of a
    non-affine equation depends on (``univariate._dep_masks``). A node is affine in its variable
    inputs when it is structural, a negation or difference, a product or ``dot_general`` with one
    variable input, or a division by a constant (the rule of ``univariate``). Exact, and needs no
    compilation; None if the functions close over a value being transformed."""
    from .fbbt import Const, _has_tracer, joint_tape
    from .fbbt_interp import scalarize, use_interpreter
    from .univariate import AFFINE, _dep_masks

    with jax.ensure_compile_time_eval():
        tape = joint_tape(f, ineq, eq, np.zeros(n))
    if _has_tracer(tape):
        return None
    prog = scalarize(tape, n) if use_interpreter(tape, n, min_rows=0) else None
    if prog is not None:  # scalar tapes (.nl models): dependency bitmasks per scalar slot
        const = ~np.isnan(prog.const_val)
        deps = [1 << j for j in range(n)] + [0] * (len(prog.const_val) - n)
        nl_bits = 0
        for o, a, b, out in zip(prog.op.tolist(), prog.a.tolist(), prog.b.tolist(), prog.out.tolist()):
            name, d = prog.opcodes[o][0], deps[a] | deps[b]
            affine = (name in ("add", "add_any", "sub", "neg") or (name == "mul" and (const[a] or const[b]))
                      or (name == "div" and const[b]))  # fmt: skip
            if not affine:
                nl_bits |= d
            deps[out] = d
        return np.array([(nl_bits >> j) & 1 for j in range(n)], bool)
    with jax.ensure_compile_time_eval():
        masks = _dep_masks(tape, n)
    nl = np.zeros(n, bool)
    for node in tape.nodes:
        name = node.eqn.primitive.name
        vin = [a for a in node.ins if not isinstance(a, Const) and a in masks]
        if not vin:
            continue
        affine = (name in AFFINE or (name in ("mul", "dot_general") and len(vin) == 1)
                  or (name == "div" and node.ins[1] not in vin))  # fmt: skip
        if not affine:
            for a in vin:
                nl |= (np.asarray(masks[a]).reshape(-1, n) > 0).any(axis=0)
    return nl


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


class _CompileTimeout(Exception):
    """A kernel was still being traced or compiled when ``minimize``'s time limit ran out."""


class _Deadline(_CompileTimeout):
    """``minimize``'s time limit ran out before a kernel call was dispatched.

    A subclass of ``_CompileTimeout`` so that every handler of a late kernel also
    handles a late call: both leave the caller's state as it was before the call."""


def _past(deadline):
    """Whether ``deadline`` (a ``time.perf_counter()`` value, or None) has passed."""
    return deadline is not None and time.perf_counter() > deadline


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
        self.dispatched = 0.0  # perf_counter() of the last dispatch, after any wait for the executable

    @staticmethod
    def _key(args):
        return tuple((a.shape, a.dtype, getattr(a, "weak_type", False)) for a in args)

    def start(self, b, n, fingerprint=None, extra=(), args=None):
        """Compile in the background for ``b`` boxes of dimension ``n`` (single device).

        ``fingerprint()``, called on the tracer thread, names the problem for
        the disk cache (None: do not cache). ``extra``: shapes of further
        (unbatched) float64 arguments after the cutoff. ``args``: all the argument
        shapes instead, when they are not ``(lo, hi, cutoff, *extra)``."""
        args = self._args(b, n, extra) if args is None else args
        key = self._key(args)
        if key not in self.ready:
            tracer, compiler = _pools()
            self.ready[key] = tracer.submit(lambda: compiler.submit(self._lower(args, fingerprint).compile))

    def wait(self, *args, deadline=None):
        """Wait until ``deadline`` for the background executable of ``args``' shapes, if one was
        started; ``_CompileTimeout`` if it is not ready by then. (A failed compile is left to
        ``__call__``.)"""
        fut = self.ready.get(self._key(args))
        if fut is None:
            return
        left = lambda: None if deadline is None or not np.isfinite(deadline) else max(deadline - time.perf_counter(), 0.0)  # noqa: E731
        try:
            fut.result(timeout=left()).result(timeout=left())
        except TimeoutError:
            raise _CompileTimeout(self.name) from None
        except Exception:
            pass

    @staticmethod
    def _args(b, n, extra=()):
        """``start``'s default argument shapes: ``(lo, hi, cutoff, *extra)``."""
        return (jax.ShapeDtypeStruct((b, n), jnp.float64),) * 2 + (jax.ShapeDtypeStruct((), jnp.float64),) + \
            tuple(jax.ShapeDtypeStruct(tuple(sh), jnp.float64) for sh in extra)  # fmt: skip

    def pending(self, b, n, extra=(), args=None):
        """Whether the background build that ``start(b, n, extra=extra, args=args)`` queues was
        started and has not finished."""
        args = self._args(b, n, extra) if args is None else args
        fut = self.ready.get(self._key(args))
        if fut is None:
            return False
        if not fut.done():
            return True
        return fut.exception() is None and not fut.result().done()

    def _lower(self, args, fingerprint):
        """The lowered kernel, loaded from the disk cache when possible (and stored there when not)."""
        fp = fingerprint() if fingerprint is not None else None
        if fp is not None:
            fp = f"{fp}-{self.name}-{args[0].shape[0]}" + "".join(f"-{'x'.join(map(str, a.shape))}" for a in args[3:])
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

    def __call__(self, *args, deadline=None):
        """Run on ``args``; with a started shape, wait for its executable until ``deadline``
        (a ``time.perf_counter()`` value), else raise ``_CompileTimeout``. Past the deadline
        nothing is dispatched: ``_Deadline`` if the executable is ready, else ``_CompileTimeout``
        (also instead of the synchronous ``jax.jit`` fallback, which has no deadline)."""
        key = self._key(args)
        fut = self.ready.get(key)
        if fut is not None:
            wait = None if deadline is None or not np.isfinite(deadline) else max(deadline - time.perf_counter(), 0.0)
            try:
                compiled = fut.result(timeout=wait)
                compiled = compiled.result(timeout=None if wait is None else max(deadline - time.perf_counter(), 0.0))
            except TimeoutError:
                raise _CompileTimeout(self.name) from None
            except Exception:  # let jit raise (or succeed) with its usual traceback
                del self.ready[key]
            else:
                if _past(deadline):  # ready, but out of time: no new call
                    raise _Deadline(self.name)
                self.dispatched = time.perf_counter()
                return compiled(*args)
        if _past(deadline):  # jit would trace and compile synchronously, with no deadline
            raise _CompileTimeout(self.name)
        self.dispatched = time.perf_counter()
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
                 univariate, alphabb, multivariate, lifted, crown, feas_tol, safe, integer=(), quadratic=True,
                 cut_pool=0, exact=False):  # fmt: skip
        opts = {k: v for k, v in locals().items() if k not in ("self", "f", "ineq", "eq")}
        constrained = ineq is not None or eq is not None
        g, h = _vec(ineq), _vec(eq)

        def violation(x):
            return jnp.maximum(jnp.max(jnp.maximum(g(x), 0.0), initial=0.0), jnp.max(jnp.abs(h(x)), initial=0.0))

        poly_forms, poly_lock = {}, threading.Lock()

        def lifted_form(n):
            """The lifted LP's form on ``n`` variables: the quadratic one (``self.quadratic``), else
            the polynomial one (``quadratic.polynomial_form``: every monomial a column that is the
            product of two earlier columns), else None (reduced-space cuts)."""
            Q = self.quadratic(n)
            if Q is None and quadratic and not lifted:
                with poly_lock:
                    if n not in poly_forms:
                        try:  # an optional faster bounder: any failure falls back to the reduced-space path
                            poly_forms[n] = polynomial_form(f, ineq, eq, n, integer=integer)
                        except Exception:
                            poly_forms[n] = None
                    Q = poly_forms[n]
            return Q

        self.lifted_form = lifted_form
        if lifted:
            lb_lifted = make_lifted_bounder(f, ineq, eq, n_kelley=n_kelley, fbbt_rounds=fbbt_rounds, tol=feas_tol,
                                            univariate=univariate, alphabb=alphabb,
                                            multivariate=multivariate, safe=safe)

            def raw_bound(blo, bhi, cutoff, pool=None, root=None):
                lb, x, _, rlo, rhi = lb_lifted(blo, bhi, cutoff)
                fx = jnp.where(violation(x) <= feas_tol, f(x), jnp.inf)
                return lb, x, fx, rlo, rhi, x
        elif constrained:
            cb = make_constrained_bounder(f, ineq, eq, n_kelley=n_kelley, tol=feas_tol, univariate=univariate,
                                          alphabb=alphabb, multivariate=multivariate, safe=safe, exact=exact)

            def raw_bound(blo, bhi, cutoff, pool=None, root=None):
                Q = lifted_form(blo.size)
                if Q is None:
                    lb, x, _, rlo, rhi = cb(blo, bhi, cutoff, pool, root)
                else:  # (the lifted LP takes no pool)
                    lb, x, _, rlo, rhi = make_quadratic_bounder(Q, n_kelley=n_kelley, tol=feas_tol, safe=safe)(blo, bhi, cutoff)
                fx = jnp.where(violation(x) <= feas_tol, f(x), jnp.inf)
                return lb, x, fx, rlo, rhi, x
        else:
            unc_bound = make_bounder(f, n_fw, univariate=univariate, alphabb=alphabb, multivariate=multivariate,
                                     return_xr=True, safe=safe)

            def raw_bound(blo, bhi, cutoff, pool=None, root=None):
                return unc_bound(blo, bhi, cutoff)
        tighten = make_fbbt(f, ineq, eq, rounds=fbbt_rounds, feas_tol=feas_tol, safe=safe, integer=integer) if fbbt else None
        cut_obbt = make_obbt(f, ineq, eq, rounds=obbt_rounds, tol=feas_tol, univariate=univariate,
                             alphabb=alphabb, multivariate=multivariate, safe=safe, exact=exact) if obbt else None

        def lp_tighten(blo, bhi, cutoff):
            Q = lifted_form(blo.size) if constrained else None
            if Q is None:
                return cut_obbt(blo, bhi, cutoff)
            return make_quadratic_obbt(Q, rounds=obbt_rounds, tol=feas_tol, safe=safe)(blo, bhi, cutoff)

        crown_lb = make_crown(f, safe=safe) if crown else None
        idx = np.asarray(integer, dtype=np.int64)

        def round_box(lo, hi):
            """Integer bounds rounded inward (``_round_bounds``); True if that empties the box."""
            if not idx.size:
                return lo, hi, jnp.array(False)
            il, ih = jnp.ceil(lo[idx] - INT_TOL), jnp.floor(hi[idx] + INT_TOL)
            return lo.at[idx].set(il), hi.at[idx].set(jnp.maximum(il, ih)), jnp.any(il > ih)

        def node(blo, bhi, cutoff, pool=None, root=None):
            """Tighten, then bound one box (with the cut pool ``pool``, valid on ``root``; see
            ``make_constrained_bounder``). Returns (lb, x, f(x), lo', hi', xr)."""
            empty = jnp.array(False)
            if tighten is not None and not (constrained and not lifted and binary_only(lifted_form(blo.size), blo.size)):
                blo, bhi, empty = tighten(blo, bhi, cutoff)
            blo, bhi, e = round_box(blo, bhi)
            empty |= e
            lb, x, fx, rlo, rhi, xr = raw_bound(blo, bhi, cutoff, pool, root)
            if crown_lb is not None:  # both bounds are valid on the box: take the larger
                lb = jnp.maximum(jnp.where(jnp.isnan(lb), -jnp.inf, lb), crown_lb(blo, bhi))
            if not dbbt:
                rlo, rhi = blo, bhi
            if idx.size:
                # an incumbent candidate must be integral (the relaxation point xr keeps the
                # fractional values for branching): the LP point rounded at several thresholds
                # (nearest, up, down, 1/4, 3/4); the best feasible one, else the least violated
                # (the start of the fix-and-solve in update_incumbent)
                lo_i, hi_i = blo[idx], bhi[idx]
                fl = jnp.floor(x[idx])
                X = jnp.stack([x.at[idx].set(jnp.clip(fl + (x[idx] - fl > t), lo_i, hi_i)) for t in ROUNDINGS])
                v = jax.vmap(violation)(X)
                fv = jnp.where(v <= feas_tol, jax.vmap(f)(X), jnp.inf)
                best = jnp.where(jnp.isfinite(fv).any(), jnp.argmin(fv), jnp.argmin(v))
                x, fx = X[best], fv[best]
                rlo, rhi, e = round_box(rlo, rhi)  # no integer point of the box has f <= cutoff
                empty |= e
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
            blo, bhi, e2 = round_box(blo, bhi)
            return blo, bhi, empty | e | e2

        self.bound = _Precompiled(jax.jit(jax.vmap(node, in_axes=(0, 0, None))), "bound")
        # the same kernel with the cut pool (``minimize`` picks one): reduced-space cuts only
        self.bound_pool = None
        self.pool_cuts = None
        if constrained and not lifted and cut_pool:
            self.bound_pool = _Precompiled(jax.jit(jax.vmap(node, in_axes=(0, 0, None, None, None))), "bound_pool")
            self.pool_cuts = jax.jit(jax.vmap(make_pool_cuts(f, ineq, eq, univariate=univariate, alphabb=alphabb,
                                                             multivariate=multivariate, safe=safe, exact=exact),
                                              in_axes=(0, None, None)))  # fmt: skip
        self.obbt = _Precompiled(jax.jit(jax.vmap(node_obbt, in_axes=(0, 0, None))), "obbt") if obbt else None
        self._problem, self._opts, self._fingerprints, self._nonlinear, self._structure = (f, ineq, eq), opts, {}, {}, {}
        self._quadratic, self._use_quadratic, self._qlock = {}, bool(quadratic) and not lifted, threading.Lock()
        self.viol = jax.jit(jax.vmap(violation))
        self.dive = None
        if idx.size and fbbt:
            # its own FBBT (own tape and interpreter caches): this kernel is traced on the
            # calling thread while the bound kernel may be traced on the tracer thread
            dive_fbbt = make_fbbt(f, ineq, eq, rounds=fbbt_rounds, feas_tol=feas_tol, safe=safe, integer=integer)

            def dive(blo, bhi, x, cutoff, failed, p0, p1):
                """Propagation dive from the point ``x`` in a box: fix the integers one at a time
                (most fractional first) to their rounded value, or the other rounding when FBBT
                then empties the box. Runs steps ``p0`` to ``p1`` of the dive (``minimize`` calls
                it in chunks, checking the time between them; the order depends only on ``x``,
                so the chunks continue one dive). Returns the box and whether a variable had no
                value left (the dive failed), given the box and flag after step ``p0``."""
                frac = jnp.abs(x[idx] - jnp.round(x[idx]))
                order = jnp.asarray(idx)[jnp.argsort(-frac)]

                def fix(lo, hi, j, val):
                    lo, hi, e = dive_fbbt(lo.at[j].set(val), hi.at[j].set(val), cutoff)
                    lo, hi, e2 = round_box(lo, hi)
                    return lo, hi, e | e2

                def step(p, carry):
                    lo, hi, failed = carry
                    j = order[p]
                    xj = jnp.clip(x[j], lo[j], hi[j])
                    v = jnp.round(xj)
                    alt = jnp.clip(jnp.where(xj >= v, v + 1.0, v - 1.0), lo[j], hi[j])
                    L, H, E = jax.vmap(lambda val: fix(lo, hi, j, val))(jnp.stack([v, alt]))
                    k = jnp.where(E[0], 1, 0)
                    skip = failed | (hi[j] <= lo[j])
                    lo, hi = jnp.where(skip, lo, L[k]), jnp.where(skip, hi, H[k])
                    return lo, hi, failed | (~skip & E[0] & E[1])

                return jax.lax.fori_loop(p0, p1, step, (blo, bhi, failed))

            self.dive = _Precompiled(jax.jit(jax.vmap(dive, in_axes=(0, 0, 0, None, 0, None, None))), "dive")
        self.fvals = jax.jit(jax.vmap(f))
        self.fns = [fn for fn in (f, ineq, eq) if fn is not None]
        self.hessians = [_hessian(fn) for fn in self.fns]

    def quadratic(self, n):
        """``quadratic.quadratic_form`` of the problem on ``n`` variables (cached; kernels are
        traced on several threads), or None when the lifted quadratic LP is not used."""
        with self._qlock:
            if n not in self._quadratic:
                self._quadratic[n] = quadratic_form(*self._problem, n) if self._use_quadratic else None
            return self._quadratic[n]

    def structure(self, n):
        """``structure.structure`` of the problem on ``n`` variables (cached), or None."""
        if n not in self._structure:
            try:
                self._structure[n] = structure(*self._problem, n)
            except Exception:  # an equation the analysis cannot follow
                self._structure[n] = None
        return self._structure[n]

    def nonlinear(self, n):
        """Variables entering nonlinearly (cached): from ``structure``, else ``structural_nonlinear``, else None."""
        if n not in self._nonlinear:
            st = self.structure(n)
            if st is not None:
                self._nonlinear[n] = st.nonlinear
            else:
                try:
                    self._nonlinear[n] = structural_nonlinear(*self._problem, n)
                except Exception:  # an equation the dependency masks cannot follow: sample Hessians
                    self._nonlinear[n] = None
        return self._nonlinear[n]

    def fingerprint(self, n):
        """Disk-cache key of this problem on ``n`` variables (``kcache.fingerprint``), or None."""
        if n not in self._fingerprints:
            self._fingerprints[n] = kcache.fingerprint(self._problem, n, sorted(self._opts.items()))
        return self._fingerprints[n]


def _pool_parts(f, ineq, eq, n, safe):
    """``lp.pool_parts``, or None when the analysis fails."""
    try:
        return pool_parts(f, ineq, eq, n, safe=safe)
    except Exception:
        return None


def _vec(fn):
    return lambda x: jnp.atleast_1d(fn(x)) if fn is not None else jnp.zeros(0)


def _integer_mask(integer, n):
    """``integer`` (None, a boolean mask or indices) as a boolean mask of length ``n``."""
    mask = np.zeros(n, bool)
    if integer is None:
        return mask
    a = np.asarray(integer)
    if a.dtype == bool:
        if a.shape != (n,):
            raise ValueError(f"integer mask has shape {a.shape}, expected ({n},)")
        return a.copy()
    mask[a.astype(np.int64)] = True
    return mask


def _round_bounds(lo, hi, mask):
    """Integer bounds rounded inward, ``ceil(lo - INT_TOL)`` and ``floor(hi + INT_TOL)`` (infinite ones stay)."""
    lo, hi = lo.copy(), hi.copy()
    lo[mask], hi[mask] = np.ceil(lo[mask] - INT_TOL), np.floor(hi[mask] + INT_TOL)
    return lo, hi


_REPAIR_SIZE = 10_000  # integer repair: largest rows x variables Jacobian it extracts (else it is off; the eager read takes seconds beyond)
_REPAIR_STEPS = 64  # integer repair: most +-1 moves per rounded point
_REPAIR_SWAPS = 200  # integer repair: swap moves (one +1, another -1) only with at most this many integers
_REPAIR_PTS = 4  # integer repair: most rounded relaxation points repaired per batch of local solves


def _integer_rows(ineq, eq, lo, hi, imask, jac=None, seed=0):
    """The linear constraint rows on integer variables only: ``(A, c, is_eq)`` with each row's
    value ``A @ x[imask] + c`` (an inequality ``<= 0``, an equality ``== 0``), or None.

    Rounding the integers of a relaxation point can violate such a row (a budget), and no
    solve of the continuous variables can repair that. Read numerically: the constraint
    Jacobian at two random points of the box; a row qualifies when its Jacobian is the same at
    both, zero on every continuous variable, and its values agree with that linear form. The
    rows only guide ``_repair_rounding``; every point it proposes is still screened against
    all the constraints. None when there are no such rows or the Jacobian would exceed
    ``_REPAIR_SIZE`` entries. ``jac``: the Jacobian's sparsity ``(rows, cols)``
    (``Structure.jac``), when known: without a row on integers only, nothing is computed."""
    g, h = _vec(ineq), _vec(eq)
    n = lo.size
    if not imask.any() or (ineq is None and eq is None):
        return None
    if jac is not None:
        r, cl = (np.asarray(a) for a in jac)
        touched = np.zeros(int(r.max(initial=-1)) + 1, bool)
        touched[r] = True
        cont_rows = np.zeros_like(touched)
        cont_rows[r[~imask[cl]]] = True
        if not (touched & ~cont_rows).any():
            return None
    try:
        rows = lambda x: jnp.concatenate([g(x), h(x)])  # noqa: E731
        with jax.ensure_compile_time_eval():
            m_ineq = int(np.size(jax.eval_shape(g, jax.ShapeDtypeStruct((n,), jnp.float64))))
            m = m_ineq + int(np.size(jax.eval_shape(h, jax.ShapeDtypeStruct((n,), jnp.float64))))
        if m == 0 or m * n > _REPAIR_SIZE:
            return None
        rng = np.random.default_rng(seed)
        a = np.where(np.isfinite(lo), lo, np.minimum(hi, 0.0) - 10.0)  # a finite part of the box
        b = np.where(np.isfinite(hi), hi, a + 20.0)
        P = a + (b - a) * rng.uniform(0.1, 0.9, size=(2, n))
        jacobian = jax.jacfwd(rows)  # (eager: tracing and compiling it under jit costs more)
        V = [np.asarray(rows(jnp.asarray(p)), np.float64) for p in P]
        J = [np.asarray(jacobian(jnp.asarray(p)), np.float64).reshape(m, n) for p in P]
    except Exception:  # an optional heuristic: anything the Jacobian cannot follow turns it off
        return None
    scale = 1.0 + np.abs(J[0]).max(axis=1)
    pred = V[0] + J[0] @ (P[1] - P[0])
    ok = (np.all(np.isfinite(J[0]) & np.isfinite(J[1]), axis=1) & np.isfinite(V[0]) & np.isfinite(V[1])
          & np.all(J[0][:, ~imask] == 0.0, axis=1) & np.all(J[1][:, ~imask] == 0.0, axis=1)
          & np.all(np.abs(J[0] - J[1]) <= 1e-9 * scale[:, None], axis=1)
          & (np.abs(pred - V[1]) <= 1e-8 * (scale + np.abs(V[1])))
          & np.any(J[0][:, imask] != 0.0, axis=1))  # fmt: skip
    if not ok.any():
        return None
    A = J[0][ok][:, imask]
    c = V[0][ok] - A @ P[0][imask]
    return A, c, (np.arange(m) >= m_ineq)[ok]


def _repair_rounding(xi, ref, lo, hi, rows, tol, max_steps=_REPAIR_STEPS):
    """Greedy repair of rounded integer values ``xi`` against linear rows ``rows`` (``_integer_rows``).

    Each step makes the move (one integer +-1, or when none helps and there are few integers
    a swap: one +1, another -1) that most reduces the rows' scaled violation per unit of
    added distance from the relaxation values ``ref``; it stops when every row holds to
    ``tol`` or no move reduces the violation. ``lo``, ``hi`` are the integers' bounds.
    Returns the repaired integers, or None if the rows are not met. It only proposes a
    candidate incumbent: no bound depends on it."""
    A0, c0, is_eq = rows
    s = 1.0 + np.abs(A0).max(axis=1)
    A, c = A0 / s[:, None], c0 / s  # rows scaled: the violation sums comparable units
    x = np.clip(np.round(xi), lo, hi)

    def viol(V):  # (m, k) row values -> (k,) total violation
        return np.sum(np.where(is_eq[:, None], np.abs(V), np.maximum(V, 0.0)), axis=0)

    def holds(x):
        r = A0 @ x + c0
        return bool(np.all(np.where(is_eq, np.abs(r), r) <= tol))

    k = x.size
    for _ in range(max_steps):
        if holds(x):
            return x
        v = A @ x + c
        cur = float(viol(v[:, None])[0])
        best = None
        jj, dd = np.tile(np.arange(k), 2), np.repeat([1.0, -1.0], k)  # single moves x_j += d
        okm = (x[jj] + dd >= lo[jj]) & (x[jj] + dd <= hi[jj])
        red = cur - viol(v[:, None] + A[:, jj] * dd)
        dist = np.abs(x[jj] + dd - ref[jj]) - np.abs(x[jj] - ref[jj])  # in [-1, 1]
        score = np.where(okm & (red > 1e-12), red / (1.0 + dist + 1e-3), -np.inf)
        i = int(np.argmax(score))
        if np.isfinite(score[i]):
            best = ((jj[i],), (dd[i],))
        if best is None and k <= _REPAIR_SWAPS:
            up, dn = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
            up, dn = up.ravel(), dn.ravel()
            sel = (up != dn) & (x[up] + 1 <= hi[up]) & (x[dn] - 1 >= lo[dn])
            up, dn = up[sel], dn[sel]
            if len(up):
                red = cur - viol(v[:, None] + A[:, up] - A[:, dn])
                dist = (np.abs(x[up] + 1 - ref[up]) - np.abs(x[up] - ref[up])
                        + np.abs(x[dn] - 1 - ref[dn]) - np.abs(x[dn] - ref[dn]))
                score = np.where(red > 1e-12, red / (2.0 + dist + 1e-3), -np.inf)
                i = int(np.argmax(score))
                if np.isfinite(score[i]):
                    best = ((up[i], dn[i]), (1.0, -1.0))
        if best is None:
            return None
        for jj, dd in zip(*best):
            x[jj] += dd
    return x if holds(x) else None


def _verify(f, ineq, eq, x, fun, lo, hi, feas_tol, integer=None):
    """Re-evaluate ``x`` outside jit in float64: ``(max_violation, ok, reason)``.

    Checks the box, ``ineq <= feas_tol``, ``|eq| <= feas_tol``, integrality of
    the ``integer`` variables (a mask) to ``INT_TOL`` and ``f(x) == fun`` (to
    1e-8 relative). With no incumbent (``fun`` infinite) there is nothing to
    verify: ``(inf, False, "")``.
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
    if integer is not None and np.any(integer):
        off = float(np.max(np.abs(x[integer] - np.round(x[integer]))))
        if off > INT_TOL:
            why.append(f"integer variables off integral values by {off:.3g}")
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
INT_TOL = 1e-6  # integer variables: a value within INT_TOL of an integer counts as integral
ROUNDINGS = (0.5, INT_TOL, 1.0 - INT_TOL, 0.25, 0.75)  # incumbent candidates: round up fractions above each
_DIVES = 16  # propagation dives per call (integer variables)
_PROBE = True  # integer variables: root probing on binaries (``probe`` in ``minimize``)
_ENUM = 32  # integer variables: at the root, fix-and-solve every assignment when there are at most this many
_FLOOR = 0.05  # pseudocosts are floored at this fraction of the mean over observed variables
_DIVE_BACKOFF = 80  # most iterations between periodic dives that do not improve the incumbent
# once there is an incumbent, a dive is skipped when the dives' total wall time (with its estimated
# cost) would pass this fraction of the elapsed time: on smallinvDAX (1583-row tape, 30 integers)
# one dive call costs ~9 s, and four of them took 36 s of a 60 s run whose incumbent was already
# optimal
_DIVE_FRAC = 0.2
_POOL_NLP_FRAC = 0.3  # at most this share of the wall time goes to the cut pool's local solves
_POOL_CHUNK = 16  # points per call of the cut pool's tangent kernel
_POOL_ADD = 2  # LP points of the lowest-bound boxes of each batch whose tangents join the cut pool
# the per-child local solves of the cut pool are probed (half the children get them) on
# _POOL_PROBE_N boxes of each kind and kept when they add _POOL_GAIN (of the gap) to the
# children's mean bound gain; flay04m (whose node LP is already tight) spent 30% of its time
# on them for nothing. The verdict is revisited after _POOL_REPROBE iterations.
_POOL_PROBE_N = 24
_POOL_GAIN = 0.02
_POOL_REPROBE = 40
_LOCAL_BACKOFF = 16  # most iterations skipped between local searches that do not improve the incumbent
_POLISH_ROUNDS = 5  # incumbent polish (``polish`` in ``minimize``): rounds while it keeps improving
_POLISH_CANDS = 512  # integer neighbours (one step, swaps) tried per polish round
_POLISH_CHUNK = 64  # most neighbours per batch of local solves (the time budget is checked between batches)
_NCPU = os.cpu_count() or 1  # pounce solves batches in parallel on this many threads (polish time caps)
_POLISH_FIRST = 8  # neighbours in the first batch, before the cost of a solve is known
_POLISH_MIN, _POLISH_FRAC = 2.0, 0.1  # polish time budget: _POLISH_MIN s plus this fraction of the elapsed time
_NEAR_ZERO_STARTS = 4  # extra random starts for the scaled local solves of a near-zero objective
_STRICT_FRAC = 1e-3  # repair_incumbent: violation (x feas_tol) above which a reported incumbent is repaired
_REPAIR_SLACK = 100.0  # repair_incumbent: largest accepted rise of f, in units of feas_tol * (1 + |f|)
_ETA, _M = 2, 4  # branch_var="reliability": observations to trust a pseudocost, strong-branching candidates
_BAL = 10.0  # branch_var="strong": candidates are at least 1/_BAL as wide as the widest variable
_Q, _ETA_MS = 2, 0.5  # branch_var="strong": split on up to _Q variables at once, those scoring >= _ETA_MS * best
_PROFILE = None  # benchmarks/profile_root.py: a callable(stage) told when each setup / root stage of ``minimize`` begins


def _mark(stage):
    """Tell ``_PROFILE`` (off by default) that ``minimize`` enters ``stage``."""
    if _PROFILE is not None:
        _PROFILE(stage)


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
    integer=None,
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
    branch_var="auto",
    univariate=True,
    alphabb=True,
    multivariate=False,
    lifted=False,
    quadratic=True,
    crown=False,
    cut_pool=128,
    safe=SAFE_DEFAULT,
    time_limit=np.inf,
    verbose=False,
):
    """Globally minimize ``f(x)`` over ``lo <= x <= hi``, ``ineq(x) <= 0``, ``eq(x) == 0``.

    ``f`` is any JAX-traceable Python function of a 1-D array returning a
    scalar; ``ineq`` / ``eq`` return a scalar or 1-D array of constraint values.
    A point is accepted as feasible when ``ineq <= feas_tol`` and ``|eq| <= feas_tol``.
    ``integer`` marks integer variables (a boolean mask or an array of indices):
    their bounds are rounded inward, they are branched on at integer values, and
    a returned ``x`` has them integral (``Result.verified`` checks it to
    ``INT_TOL``). The relaxations are those of the continuous problem.
    ``local_solver`` picks the upper-bounding local NLP solver: ``"pounce"``,
    ``"optax"`` (pure JAX), ``"scipy"``, or ``"auto"`` (pounce if installed).
    Once there is an incumbent, local searches and integer dives that fail to
    improve it are run less often (``_LOCAL_BACKOFF``, ``_DIVE_BACKOFF``).
    The search stops with status ``"time_limit"`` after ``time_limit`` seconds,
    counted from the call: tracing and compiling the bounding kernels count, and
    if they are not ready in time the result has no bound (``lower_bound=-inf``)
    and ``message`` says so.

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
    (``lifted.py``) instead of the reduced-space cuts. ``quadratic``: a constrained problem whose
    functions are all quadratic (and whose tape scalarizes) is bounded, and OBBT-tightened, over
    the lifted LP with one McCormick-linked column per product (``quadratic.py``) instead of the
    reduced-space cuts; others are unaffected. ``crown`` adds the CROWN
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
    faces, see ``_LAM`` and ``_BETA``); ``branch_var`` is ``"auto"`` (the default:
    ``"pseudocost"`` without integer variables, ``"width"`` with them; on start50 at 60 s
    pseudocost solves 38, width and strong 36, and pseudocost loses none; on int50 each of
    pseudocost and strong solves one more than width but loses flay04m or nous1), ``"width"`` (largest
    relative width among nonlinear variables; on the lifted quadratic LP, relative width
    times the McCormick gap of the variable's products at the relaxation point), ``"pseudocost"`` (product score
    of per-variable bound gains observed so far) or ``"reliability"`` (like
    pseudocost, but variables with fewer than ``_ETA`` observations are
    strong-branched, up to ``_M`` of them per box, within the ``batch`` budget)
    or ``"strong"`` (every parent is split on its best-scored variable, and the
    padding slots of that kernel call strong-branch further candidates, ranked
    by the same scores, so it splits as many parents in as many kernel calls
    per iteration as ``"width"``; each parent is then split on its ``_Q`` best
    candidates at once into up to ``2**_Q`` orthants, see ``_BAL`` and ``_ETA_MS``).
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
    imask = _integer_mask(integer, lo.size)
    _mark("bound_domain")
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        # derive finite bounds from the constraints and a feasible point (domain.py)
        try:
            lo, hi, infeasible0, seed_x, seed_f = bound_domain(f, ineq, eq, lo, hi, feas_tol=feas_tol,
                                                               deadline=t0 + time_limit, integer=imask)
        except DomainTimeout:
            x = np.clip(np.zeros(lo.size), lo, hi)
            return Result(x=x, fun=np.inf, lower_bound=-np.inf, converged=False, iterations=0, nodes=0,
                          time=time.perf_counter() - t0, status="time_limit",
                          message="time_limit reached while deriving bounds for unbounded variables")
        if not infeasible0 and not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))) and \
                time.perf_counter() - t0 > time_limit:  # fmt: skip
            return Result(x=np.clip(np.zeros(lo.size), lo, hi), fun=np.inf, lower_bound=-np.inf, converged=False,
                          iterations=0, nodes=0, time=time.perf_counter() - t0, status="time_limit",
                          message="time_limit reached while deriving bounds for unbounded variables")  # fmt: skip
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
    _mark("setup")
    if imask.any():
        lo, hi = _round_bounds(lo, hi, imask)
        if np.any(lo > hi):
            bad = np.flatnonzero(lo > hi).tolist()
            return Result(x=np.clip(np.round(0.5 * (lo_user + hi_user)), lo_user, hi_user), fun=np.inf,
                          lower_bound=np.inf, converged=True, iterations=0, nodes=0,
                          time=time.perf_counter() - t0, status="infeasible",
                          message=f"no integer value within the bounds of variables {bad}")  # fmt: skip
        if seed_x is not None and np.any(np.abs(seed_x[imask] - np.round(seed_x[imask])) > INT_TOL):
            seed_x, seed_f = None, np.inf  # a feasible point of the relaxation only
    width0 = np.where(hi > lo, hi - lo, 1.0)
    constrained = ineq is not None or eq is not None
    if branch_point not in ("mid", "relax"):
        raise ValueError(f"branch_point must be 'mid' or 'relax', not {branch_point!r}")
    if branch_var not in ("auto", "width", "pseudocost", "reliability", "strong"):
        raise ValueError(
            f"branch_var must be 'auto', 'width', 'pseudocost', 'reliability' or 'strong', not {branch_var!r}"
        )
    if branch_var == "auto":  # measured choice, see the docstring
        branch_var = "width" if imask.any() else "pseudocost"
    if obbt not in (False, None, "root", "tree"):
        raise ValueError(f"obbt must be False, 'root' or 'tree', not {obbt!r}")
    if crown not in (False, None, True, "auto"):
        raise ValueError(f"crown must be False, True or 'auto', not {crown!r}")
    if crown == "auto":
        crown = crown_auto(f, lo)
    elif crown:
        crown = crown_supported(f, lo)  # an unsupported f gets no CROWN kernel
    _mark("cut_pool")
    # convexity certificates on the root box: which parts get a cut pool (reduced-space LP only),
    # and whether every certified verdict holds on the whole box (then on every node box too,
    # so a certified function needs no base relaxation: ``exact``)
    pmask, exact, convex = None, False, False
    if constrained and not lifted and cut_pool:
        parts_ok = _cached(("pool_parts", f, ineq, eq, lo.size, bool(safe)),
                           lambda: _pool_parts(f, ineq, eq, lo.size, bool(safe)))  # fmt: skip
        if parts_ok is not None:
            pmask, exact, convex = parts_ok(lo, hi, convex=True)
    _mark("kernels")
    opts = dict(n_fw=n_fw, n_kelley=n_kelley, fbbt=bool(fbbt), fbbt_rounds=fbbt_rounds, dbbt=bool(dbbt),
                obbt=obbt or False, obbt_rounds=obbt_rounds, univariate=univariate, alphabb=alphabb,
                multivariate=multivariate, lifted=lifted, crown=bool(crown), feas_tol=feas_tol, safe=bool(safe),
                integer=tuple(int(i) for i in np.flatnonzero(imask)), quadratic=bool(quadratic),
                cut_pool=int(cut_pool) if pmask is not None and pmask.any() else 0, exact=bool(exact))  # fmt: skip
    K = _cached(("kernels", f, ineq, eq, tuple(sorted(opts.items()))), lambda: _Kernels(f, ineq, eq, **opts))
    # the cut pool: tangents of the root box's relaxation at good points, added to every node LP
    pool = None
    if K.bound_pool is not None and K.lifted_form(lo.size) is None:
        rows0 = np.zeros((int(cut_pool), lo.size + 2))
        rows0[:, lo.size] = -np.inf
        pool = dict(rows=rows0, root=np.stack([lo, hi]), mask=pmask, next=0, args=None)
    bound_kernel = K.bound if pool is None else K.bound_pool

    def pool_args():
        """The pool's kernel arguments (none without a pool)."""
        if pool is None:
            return ()
        if pool["args"] is None:
            pool["args"] = (jnp.asarray(pool["rows"]), jnp.asarray(pool["root"]))
        return pool["args"]

    def pool_add(X):
        """Add the root-box tangents of the pooled parts at the points ``X`` (k, n) to the pool
        (a ring: the oldest rows go first; rows already in it are skipped)."""
        if pool is None or pool.get("plain"):
            return
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        X = X[np.all(np.isfinite(X), axis=1)]
        if not len(X):
            return
        root = pool["root"]
        X = np.clip(X, root[0], root[1])
        # the tangent kernel (compiled for _POOL_CHUNK points at a time) compiles in the
        # background after the bound kernels: do not wait for it (flay04m stalled for seconds
        # at the root); keep the latest points until it is ready
        if not (pool["exe"].done() and pool["exe"].result().done()):
            pool["pending"] = np.concatenate([pool.get("pending", X[:0]), X])[-_POOL_CHUNK:]
            return
        exe = pool["exe"].result().result()
        if len(pool.get("pending", ())):
            X, pool["pending"] = np.concatenate([pool["pending"], X]), X[:0]
        a, b = [], []
        for c in range(0, len(X), _POOL_CHUNK):
            Xc = X[c:c + _POOL_CHUNK]
            Xc = np.concatenate([Xc, np.repeat(Xc[:1], _POOL_CHUNK - len(Xc), 0)])
            ac, bc = exe(jnp.asarray(Xc), jnp.asarray(root[0]), jnp.asarray(root[1]))
            a.append(np.asarray(ac))
            b.append(np.asarray(bc))
        a, b = np.concatenate(a)[:len(X)], np.concatenate(b)[:len(X)]
        rows, n_ = pool["rows"], lo.size
        for i in range(len(X)):
            for k in np.flatnonzero(pool["mask"]):
                if not np.isfinite(b[i, k]):
                    continue
                r = np.concatenate([a[i, k], [b[i, k], 1.0 if k == 0 else 0.0]])
                scale = 1.0 + np.abs(r[:n_ + 1]).max()
                live = np.isfinite(rows[:, n_])
                if np.any(live & (rows[:, n_ + 1] == r[n_ + 1]) &
                          np.all(np.abs(rows[:, :n_ + 1] - r[:n_ + 1]) <= 1e-9 * scale, axis=1)):  # fmt: skip
                    continue
                rows[pool["next"]] = r
                pool["next"] = (pool["next"] + 1) % len(rows)
                pool["args"] = None

    _mark("start_compile")
    extra = () if pool is None else (pool["rows"].shape, pool["root"].shape)  # the bound kernel's pool arguments
    padded = _Padder(batch, batch_sizes, jax.devices() if devices is None else list(devices))
    if padded.mesh is None:  # compile the kernels in the background while the setup below runs
        fp = functools.partial(K.fingerprint, lo.size) if kcache.cache_root() else None
        for b in padded.sizes:
            bound_kernel.start(b, lo.size, fp, extra)
        if obbt == "tree":  # obbt="root": started once the root shows OBBT will run (``need_obbt``)
            for b in padded.sizes:
                K.obbt.start(b, lo.size, fp)
    if pool is not None:  # the pool's tangent kernel, traced after the bound kernels
        tracer, compiler = _pools()
        pts, vec = jax.ShapeDtypeStruct((_POOL_CHUNK, lo.size), jnp.float64), jax.ShapeDtypeStruct((lo.size,), jnp.float64)
        pool["exe"] = tracer.submit(lambda: compiler.submit(K.pool_cuts.trace(pts, vec, vec).lower().compile))
    _mark("start_dive_compile")
    if K.dive is not None and padded.mesh is None:  # the dive kernel too, traced after the others
        box, cut = jax.ShapeDtypeStruct((_DIVES, lo.size), jnp.float64), jax.ShapeDtypeStruct((), jnp.float64)
        flag, step = jax.ShapeDtypeStruct((_DIVES,), jnp.bool_), jax.ShapeDtypeStruct((), jnp.int32)
        K.dive.start(_DIVES, lo.size, fp, args=(box, box, box, cut, flag, step, step))
    if pool is not None and padded.mesh is None:  # for the pool kernel without rows (pool_judge)
        pool["fp"] = fp
    # integer variables are split even where they enter linearly: that is what makes them integral
    _mark("nonlinear")
    nl = K.nonlinear(lo.size)
    if nl is None:
        nl = nonlinear_variables(K.fns, lo, hi, hessians=K.hessians)
    elif not nl.any():
        nl = np.ones(lo.size, bool)  # as nonlinear_variables: nothing nonlinear, split anything
    branchable = nl | imask
    _mark("quadratic")
    quad = K.quadratic(lo.size) if constrained else None  # the lifted quadratic LP's form, if used
    lf = K.lifted_form(lo.size) if constrained and not lifted else None
    if lf is not None and lf.flip >= 0 and np.all(lo[imask] == 0.0) and np.all(hi[imask] == 1.0):
        lo = lo.copy()  # complement symmetry (quadratic._flip): x_flip = 1 keeps an optimal point
        lo[lf.flip] = 1.0
    width0 = np.where(branchable, width0, np.inf)  # never split linear variables

    _mark("make_local")
    solve_local = _cached(("local", f, ineq, eq, local_solver, lo.tobytes(), hi.tobytes()),
                          lambda: make_local(f, ineq, eq, lo, hi, local_solver, patterns=K.structure(lo.size)))

    def local(starts, options=None):
        """Local solves from each start; returns (x, f) of the best feasible result.

        ``options``: extra local-solver options for these solves (``near_zero_options``).

        With integer variables a start (an LP point, which can violate the nonlinear
        constraints far more than any rounding can repair) is first solved as the
        continuous relaxation; that point is rounded at the ``ROUNDINGS`` thresholds and
        each rounding is fix-and-solved: integers fixed, continuous variables free. A nearest
        rounding that gets no feasible point is also repaired (``repaired``) and solved again."""
        S = np.clip(np.asarray(starts, dtype=np.float64), lo, hi)
        if imask.any():
            R = np.clip(np.asarray(solve_local(S, options=options)), lo, hi)  # NLP relaxation
            pool_add(R)
            fl = np.floor(R)
            fixed = np.concatenate([np.clip(fl + (R - fl > t), lo, hi) for t in ROUNDINGS])
            fixed = np.where(imask, fixed, np.concatenate([R] * len(ROUNDINGS)))
            fixed, inv = np.unique(fixed, axis=0, return_inverse=True)
            L, H = np.where(imask, fixed, lo), np.where(imask, fixed, hi)
            X = np.clip(np.asarray(solve_local(fixed, L, H, options=options)), L, H)
            F = screen(X)
            # a nearest rounding (ROUNDINGS[0]) with no feasible completion: repair its integers
            # against the linear rows on integers only, and fix-and-solve the repaired points too
            near = np.asarray(inv).reshape(-1)[:len(R)]
            bad = np.unique(near[~np.isfinite(F[near])])
            Y = repaired(fixed[bad], R[[int(np.flatnonzero(near == b)[0]) for b in bad]]) if len(bad) else []
            if len(Y):
                L, H = np.where(imask, Y, lo), np.where(imask, Y, hi)
                Y = np.clip(np.asarray(solve_local(Y, L, H, options=options)), L, H)
                X, F = np.concatenate([X, Y]), np.concatenate([F, screen(Y)])
        else:
            X = np.clip(np.asarray(solve_local(S, options=options)), lo, hi)
            F = screen(X)
        i = int(np.argmin(F))
        if np.isfinite(F[i]):
            pool_add(X[i])
        return X[i], float(F[i])

    int_rows = []  # [rows] once read (``_integer_rows``): the linear rows on integers only, or [None]

    def repaired(X, R):
        """The rounded points ``X`` (k, n) whose integers violate a linear row on integers only,
        repaired (``_repair_rounding``, towards the relaxation points ``R``): (k', n), the
        continuous values kept. At most ``_REPAIR_PTS`` repairs per call; they only propose
        candidate incumbents (fix-and-solved, then screened)."""
        out = np.zeros((0, lo.size))
        if _past(deadline):
            return out
        if not int_rows:
            st = K.structure(lo.size)
            int_rows.append(_integer_rows(ineq, eq, lo, hi, imask, jac=None if st is None else st.jac))
        rows = int_rows[0]
        if rows is None:
            return out
        A, c, is_eq = rows
        res = A @ X[:, imask].T + c[:, None]
        bad = np.flatnonzero(np.any(np.where(is_eq[:, None], np.abs(res), res) > _STRICT_FRAC * feas_tol, axis=0))
        Y = []
        for i in bad[:_REPAIR_PTS]:
            if not (np.all(np.isfinite(X[i])) and np.all(np.isfinite(R[i]))):
                continue
            xi = _repair_rounding(X[i, imask], R[i, imask], lo[imask], hi[imask], rows, _STRICT_FRAC * feas_tol)
            if xi is not None:
                y = X[i].copy()
                y[imask] = xi
                Y.append(y)
        return np.array(Y) if Y else out

    def screen(X):
        """Objective values of candidate incumbents ``X`` (k, n), ``inf`` for any that is not one.

        A candidate must lie in the box, have its integer variables integral (to ``INT_TOL``)
        and satisfy every constraint to ``feas_tol`` (``K.viol``, float64), and ``f`` must be
        a number. Every point a local solve or the polish proposes passes through here before
        it can become the incumbent, i.e. before it is used as a cutoff to prune."""
        X = np.asarray(X, dtype=np.float64)
        F = np.asarray(K.fvals(jnp.asarray(X)), dtype=np.float64)
        bad = np.isnan(F) | np.any((X < lo) | (X > hi) | np.isnan(X), axis=1)
        if constrained:
            bad |= ~(np.asarray(K.viol(jnp.asarray(X))) <= feas_tol)
        if imask.any():
            bad |= np.any(np.abs(X[:, imask] - np.round(X[:, imask])) > INT_TOL, axis=1)
        return np.where(bad, np.inf, F)

    def repair_incumbent(x, fx):
        """The incumbent to report: ``x`` itself, or a strictly feasible point near it.

        The search accepts points with constraint violation up to ``feas_tol``, and relaxation
        minimizers sit exactly there; a violation of 1e-6 in a sensitive constraint can make
        ``f`` smaller than the problem's true optimum (st_z, ex8_5_3, ex9_2_3). When the
        violation exceeds ``_STRICT_FRAC * feas_tol`` one local solve from ``x`` (integers
        fixed, tight constraint tolerance) looks for a point within that; if it finds one it is
        reported instead, with its own objective value (larger when the violation was helping,
        smaller when the solve also improves the point). Pruning, the lower bound and the status
        are unchanged: they are those of the problem relaxed by ``feas_tol``.
        Returns (x, f(x), note)."""
        strict = _STRICT_FRAC * feas_tol
        v = float(np.asarray(K.viol(jnp.asarray(x[None], dtype=jnp.float64)))[0])
        if not v > strict:
            return x, fx, ""
        L, H = np.where(imask, x, lo), np.where(imask, x, hi)
        opts = {"constr_viol_tol": 0.1 * strict, **(near_zero_options(fx, abs_tol) or {})}
        try:
            X = np.clip(np.asarray(solve_local(x[None], L[None], H[None], options=opts)), L, H)
            F, vr = screen(X), np.asarray(K.viol(jnp.asarray(X)))
        except Exception:  # an optional improvement: any failure keeps the incumbent
            return x, fx, ""
        # The status and lower bound stay those of the feas_tol-relaxed problem, so a repaired point
        # may only cost what a feas_tol perturbation plausibly does (a multiplier up to _REPAIR_SLACK)
        # or the gap tolerance: a local solve that walks far away to strict feasibility (f rising by
        # much more) would be reported "optimal" with a gap far beyond the tolerance, possibly with
        # a much better strictly feasible point elsewhere. Then keep the relaxed-feasible incumbent.
        rise = max(abs_tol, rel_tol * abs(fx), _REPAIR_SLACK * feas_tol * (1 + abs(fx)))
        if np.isfinite(F[0]) and vr[0] <= strict and F[0] <= fx + rise:
            return X[0], float(F[0]), (f"incumbent repaired from violation {v:.2g} to {vr[0]:.2g} "
                                       f"(f {fx:.10g} -> {F[0]:.10g})")  # fmt: skip
        return x, fx, ""

    deadline = t0 + time_limit if np.isfinite(time_limit) else None

    def pool_nlp(blo, bhi, starts, pair):
        """Pool the tangents at local solutions of the continuous problem in each box (from
        ``starts``): on a convex part those are the cuts that close the box's LP bound to its
        NLP bound. Within ``_POOL_NLP_FRAC`` of the wall time, and only while it pays off: in a
        probe phase only the children of every other parent (``pair``: the parent of each
        child) get the solves, and ``pool_judge`` compares their bound gains with the others'.
        Returns the mask of the boxes that got them, or None."""
        if pool is None or not local_solve or _past(deadline):
            return None
        t = time.perf_counter()
        if pool.setdefault("nlp_time", 0.0) > _POOL_NLP_FRAC * (t - t0):
            return None
        st = pool.setdefault("nlp", dict(mode="probe", since=it, gt=[], gu=[]))
        if st["mode"] == "on" and it - st["since"] >= _POOL_REPROBE:
            st.update(mode="probe", since=it, gt=[], gu=[])
        if st["mode"] == "off":
            return None
        do = (pair % 2 == 0) if st["mode"] == "probe" else np.ones(len(blo), bool)
        if do.any():
            X = np.clip(np.asarray(solve_local(np.clip(starts[do], blo[do], bhi[do]), blo[do], bhi[do])),
                        blo[do], bhi[do])  # fmt: skip
            pool_add(X)
            pool["nlp_x"] = X
        pool["nlp_time"] += time.perf_counter() - t
        return do

    def pool_judge(do, clb, plb_c):
        """Probe phase of ``pool_nlp``: the gains ``clb - plb`` (relative to the gap) of the
        boxes with and without the local solves; once there are enough of both, keep the solves
        on when they at least add ``_POOL_GAIN`` to the mean gain (else switch them off) until
        the next probe ``_POOL_REPROBE`` iterations later."""
        st = pool.get("nlp") if pool is not None else None
        if do is None or st is None or st["mode"] != "probe" or not np.isfinite(ub):
            return
        cap = np.maximum(ub - plb_c, 1e-12)
        g = np.clip(np.where(np.isfinite(clb), clb - plb_c, cap), 0.0, cap) / cap
        st["gt"] += g[do].tolist()
        st["gu"] += g[~do].tolist()
        if min(len(st["gt"]), len(st["gu"])) >= _POOL_PROBE_N:
            on = np.mean(st["gt"]) >= np.mean(st["gu"]) + _POOL_GAIN
            st.update(mode="on" if on else "off", since=it)
            if not on and padded.mesh is None:
                # the pool's rows cost the node LP up to 2x (process): the pool kernel without
                # rows for the rest of the search, once it has compiled in the background (not
                # at the start: that compile competed with the root phase of every solve)
                pool["plain"] = True
                for b in padded.sizes:
                    bound_kernel.start(b, lo.size, pool["fp"], ((0, lo.size + 2), pool["root"].shape))

    def plain_kernel():
        """Whether the pool kernel runs without rows (``pool_judge``): once that is compiled."""
        if pool is None or not pool.get("plain"):
            return False
        futs = [f for k, f in bound_kernel.ready.items() if len(k) > 3 and k[3][0][0] == 0]
        return bool(futs) and all(f.done() and f.result().done() for f in futs)

    bound_call = [0.0]  # seconds of the slowest bound-kernel call so far (first-dive cost estimate)

    def run(kernel, calls, cutoff, k, extra=()):
        """Apply ``kernel`` to each padded call; first ``k`` rows.

        Once the time limit has passed no further call is dispatched: ``_Deadline`` (or
        ``_CompileTimeout`` while the kernel compiles, see ``_Precompiled.__call__``) is
        raised instead (a call already running cannot be interrupted, so a solve overruns
        its limit by at most one call). Each call is waited for before the next is
        dispatched; on CPU the calls run one after another anyway, so this costs nothing
        and makes the check before each call meaningful."""
        outs = []
        for c in calls:
            if padded.mesh is not None:  # sharded calls compile through jit (synchronously)
                if _past(deadline):
                    raise _Deadline(kernel.name)
                outs.append(jax.block_until_ready(kernel.jitted(*c, cutoff, *extra)))
                continue  # (a call that compiled is no estimate of a call's cost)
            o = kernel(*c, cutoff, *extra, deadline=deadline)
            outs.append(jax.block_until_ready(o))
            if kernel is bound_kernel:  # not counting the wait for the executable
                bound_call[0] = max(bound_call[0], time.perf_counter() - kernel.dispatched)
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
        # a pool switched off by the probe: the zero-row variant skips the rows' LP cost
        args = (jnp.zeros((0, lo.size + 2)), pool_args()[1]) if plain_kernel() else pool_args()
        lb, x, fx, rlo, rhi, xr = run(bound_kernel, padded(blo, bhi), cutoff, k, args)
        # a NaN bound carries no information: -inf, never a bound that could prune the box
        # or claim optimality (and np.maximum / min would otherwise spread it to other bounds)
        lb = np.where(np.isnan(lb), -np.inf, lb)
        x = np.where(np.isnan(x).any(axis=1, keepdims=True), 0.5 * (blo + bhi), x)  # (a NaN point is no local start)
        if pool is not None:  # the LP points of the lowest bounds: where the pool's cuts are needed next
            live = np.isfinite(lb) & (lb < ub)
            if live.any():
                pool_add(xr[np.flatnonzero(live)[np.argsort(lb[live])[:_POOL_ADD]]])
        # an empty box holds no feasible point with f <= cutoff
        return (np.where(empty, ub, lb), x, np.where(empty, np.inf, fx),
                np.where(empty[:, None], blo, rlo), np.where(empty[:, None], bhi, rhi),
                np.where(empty[:, None] | ~np.isfinite(xr), 0.5 * (blo + bhi), xr))  # fmt: skip

    evaluations = 0
    ub, xbest = np.inf, 0.5 * (lo + hi)
    if seed_x is not None:  # feasible point found while bounding the domain
        ub, xbest = float(seed_f), np.asarray(seed_x)

    backoff = [0, 0]  # local-search back-off: current wait length, iterations left to skip

    def update_incumbent(xc, fc, lb):
        """Improve the incumbent from candidate points and local solves."""
        nonlocal ub, xbest
        starts = []
        if np.any(np.isfinite(fc)):
            i = int(np.nanargmin(fc))
            if fc[i] < ub:
                fs = float(screen(np.asarray(xc[i])[None])[0])  # the kernel checked feasibility; screen re-checks
                if fs < ub:                                     # it in float64 with the box and integrality
                    ub, xbest = fs, xc[i]
                starts.append(xc[i])
        if constrained:
            # relaxation minimizers of the most promising boxes are good starts
            n_try = n_starts * (4 if not np.isfinite(ub) else 1)
            for i in np.argsort(lb)[:n_try]:
                if lb[i] < ub - tol():
                    starts.append(xc[i])
        if local_solve and starts and not _past(deadline):  # (a batch already running is not interrupted)
            # with an incumbent, back off after local searches that did not improve it (skip
            # 1, 3, 7, ... iterations, up to _LOCAL_BACKOFF); an improvement resets the wait
            if np.isfinite(ub) and backoff[1] > 0:
                backoff[1] -= 1
                return
            xl, fl = local(starts)
            nz = near_zero_options(min(ub, fl), abs_tol)
            if nz is not None and not _past(deadline):
                # near-zero objective: solve again with the objective scaled up (local.near_zero_options),
                # also from the incumbent and a few random points of the box
                extra = [xl if fl < ub else xbest] + list(prng.uniform(lo, hi, size=(_NEAR_ZERO_STARTS, lo.size)))
                x2, f2 = local(starts + extra, options=nz)
                if f2 < fl:
                    xl, fl = x2, f2
            if fl < ub:
                ub, xbest = fl, xl
                backoff[:] = [0, 0]
            elif np.isfinite(ub):
                backoff[0] = min(2 * backoff[0] + 1, _LOCAL_BACKOFF)
                backoff[1] = backoff[0]
        polish()

    prng = np.random.default_rng(1)  # random starts of the incumbent heuristics (``rng`` drives the dives)
    polished = dict(ub=np.inf, time=0.0, per=None)  # incumbent value last polished, total polish time,
    # wall seconds per local solve in the last polish batch

    def polish():
        """Improve a new incumbent locally, within a time budget (``_POLISH_*``).

        Each round: for a near-zero objective, a local solve of the continuous variables from
        the incumbent with ``near_zero_options`` (integers fixed); then, with integer variables, the
        integer neighbours of the incumbent (one variable +-1; one +1 and another -1), each
        with its continuous variables solved again (or just evaluated when there are none).
        The best candidate that passes ``screen`` and beats the incumbent replaces it; rounds
        stop when one does not improve. Only an upper bound can change."""
        nonlocal ub, xbest
        if not local_solve or not np.isfinite(ub) or ub >= polished["ub"]:
            return
        start = time.perf_counter()
        stop = min(start + _POLISH_MIN + _POLISH_FRAC * (start - t0) - polished["time"], t0 + time_limit)
        cont = ~imask
        try:
            for _ in range(_POLISH_ROUNDS):
                if time.perf_counter() >= stop:
                    break
                before = ub
                x = np.where(imask, np.round(xbest), xbest)
                L, H = np.where(imask, x, lo), np.where(imask, x, hi)
                nz = near_zero_options(ub, abs_tol)
                if cont.any() and nz is not None:  # (the incumbent is already a local solution otherwise)
                    X = np.clip(np.asarray(solve_local(x[None], L[None], H[None], options=nz)), L, H)
                    F = screen(X)
                    if F[0] < ub:
                        ub, xbest = float(F[0]), X[0]
                if imask.any():
                    x = np.where(imask, np.round(xbest), xbest)
                    C = neighbours(x)
                    c = 0
                    while c < len(C):
                        left = stop - time.perf_counter()
                        if left <= 0:
                            break
                        # a batch of local solves cannot be interrupted: size it to the time left
                        # at the measured cost per solve, and cap each solve's wall time at its
                        # share of the time left with the solves run ``os.cpu_count()`` at a time
                        # (pounce's ``max_wall_time``). On batch 64 neighbours, most fixing an
                        # infeasible assignment, took 40 s against a 2 s budget.
                        per = polished["per"]
                        m = _POLISH_FIRST if per is None else int(np.clip(left / per, 1, _POLISH_CHUNK))
                        Y = C[c:c + m]
                        c += len(Y)
                        if cont.any():
                            L, H = np.where(imask, Y, lo), np.where(imask, Y, hi)
                            t1 = time.perf_counter()
                            Y = np.clip(np.asarray(solve_local(Y, L, H, options={"max_wall_time": left * min(1.0, _NCPU / len(Y))})), L, H)
                            polished["per"] = (time.perf_counter() - t1) / len(Y)
                        F = screen(Y)
                        i = int(np.argmin(F))
                        if F[i] < ub:
                            ub, xbest = float(F[i]), Y[i]
                if not ub < before:
                    break
        finally:
            polished["ub"] = ub
            polished["time"] += time.perf_counter() - start

    def neighbours(x):
        """Integer neighbours of ``x`` in the box: one integer variable +-1, then swaps (one +1,
        another -1); at most ``_POLISH_CANDS``, the swaps sampled when there are more."""
        ii = np.flatnonzero(imask)
        one = [(j, d) for j in ii for d in (-1.0, 1.0) if lo[j] <= x[j] + d <= hi[j]]
        up = [j for j in ii if x[j] + 1 <= hi[j]]
        dn = [j for j in ii if x[j] - 1 >= lo[j]]
        room = max(_POLISH_CANDS - len(one), 0)
        pairs = [(a, b) for a in up for b in dn if a != b]
        if len(pairs) > room:
            pairs = [pairs[i] for i in prng.choice(len(pairs), room, replace=False)]
        Y = np.repeat(x[None], len(one) + len(pairs), 0)
        for r, (j, d) in enumerate(one):
            Y[r, j] += d
        for r, (a, b) in enumerate(pairs, len(one)):
            Y[r, a] += 1.0
            Y[r, b] -= 1.0
        return Y[:_POLISH_CANDS]

    def tol():
        return max(abs_tol, rel_tol * abs(ub)) if np.isfinite(ub) else abs_tol

    rng = np.random.default_rng(0)
    dive_cost = [None]  # wall seconds of the last complete dive (kernel runs and fixed solves)
    step_cost = [None]  # wall seconds per step (integer fixed) of the last dive kernel run
    dive_total = [0.0]  # wall seconds of all dives so far

    def time_for(estimate):
        """Whether ``1.5 * estimate`` seconds of incumbent-only work (dives, root enumeration)
        still fit before the time limit. With no estimate yet (``None``): whether a quarter of
        the time limit is left. Skipping such work never changes a bound (see ``dive``)."""
        if deadline is None:
            return True
        left = deadline - time.perf_counter()
        return left >= (0.25 * time_limit if estimate is None else 1.5 * estimate)

    def dive(blo, bhi, points):
        """Propagation dives (``_Kernels.dive``) from up to ``_DIVES`` points in their boxes,
        then a local solve of the continuous variables in each box that ends with every
        integer fixed; improves the incumbent.

        Dives only produce incumbents, so skipping or cutting one short is sound. The dive
        kernel runs in chunks of steps (1, 2, 4, ... integers fixed), and under a time limit
        the next chunk is dispatched only while 1.5 times its estimated cost fits
        (``time_for``): the measured cost per step so far, or before the first chunk, the
        slowest bound-kernel call (a step runs FBBT on both roundings of one integer, the bound
        kernel FBBT once plus the LP). A dive cut short gives no fixed solve. The per-step cost
        varies by more than an order of magnitude between problems (smallinvDAX: a step costs
        about one bound call; batch: 1/24 of one), so no estimate made before running is
        reliable; chunking bounds the overrun by one chunk instead. Once there is an incumbent,
        a dive is skipped when the dives' total wall time with its estimated cost (the last
        dive's, else the steps times the per-step estimate) would pass ``_DIVE_FRAC`` of the
        elapsed time."""
        nonlocal ub, xbest
        if K.dive is None or not len(points):
            return
        k = min(len(points), _DIVES)
        pad = lambda A: np.concatenate([A[:k], np.repeat(A[:1], _DIVES - k, 0)])  # noqa: E731
        steps = int(imask.sum())
        DL, DH, X = jnp.asarray(pad(blo)), jnp.asarray(pad(bhi)), jnp.asarray(pad(points))
        F, cut = jnp.zeros(_DIVES, jnp.bool_), jnp.asarray(ub, jnp.float64)
        step = lambda p: jnp.asarray(p, jnp.int32)  # noqa: E731
        try:  # compiled in the background; skipped if not ready by the deadline
            K.dive.wait(DL, DH, X, cut, F, step(0), step(0), deadline=deadline)
        except _CompileTimeout:
            return
        per_step = step_cost[0] or bound_call[0] or None
        est = dive_cost[0] if dive_cost[0] is not None else (None if per_step is None else per_step * steps)
        t = time.perf_counter()
        if np.isfinite(ub) and est is not None and dive_total[0] + est > _DIVE_FRAC * (t - t0):
            return  # with an incumbent, dives get at most _DIVE_FRAC of the wall time
        p, chunk = 0, 1
        while p < steps:
            c = min(chunk, steps - p)
            if not time_for(None if per_step is None else per_step * c):
                break
            try:
                DL, DH, F = jax.block_until_ready(K.dive(DL, DH, X, cut, F, step(p), step(p + c), deadline=deadline))
            except (_CompileTimeout, _Deadline):
                break
            p, chunk = p + c, 2 * chunk
            per_step = step_cost[0] = (time.perf_counter() - t) / p
        if p == steps:  # a dive cut short gives no fixed solve
            DL, DH, F = np.asarray(DL), np.asarray(DH), np.asarray(F)
            ok = ~F[:k] & np.all(DL[:k, imask] == DH[:k, imask], axis=1)
            if ok.any() and not _past(deadline):
                fixed_solve(DL[:k][ok], DH[:k][ok], pad(points)[:k][ok])
            dive_cost[0] = time.perf_counter() - t
        dive_total[0] += time.perf_counter() - t

    def probe(rlo, rhi, lb):
        """Root probing on the binary variables: bound the box with each one at 0 and at 1 (one
        batch). A side that is empty or cannot beat the incumbent fixes the variable to the other
        side; every variable shrinks to the hull of the two sides' tightened boxes (any feasible
        point lies in one of them); and the root bound rises to the larger side minimum."""
        j = np.flatnonzero(imask & (rhi[0] - rlo[0] == 1.0))
        if not len(j):
            return rlo, rhi, lb
        P_lo, P_hi = np.repeat(rlo, 2 * len(j), 0), np.repeat(rhi, 2 * len(j), 0)
        r = np.arange(len(j))
        P_hi[2 * r, j] = rlo[0, j]  # side 0: x_j at its lower bound
        P_lo[2 * r + 1, j] = rhi[0, j]  # side 1: x_j at its upper bound
        plb, px, pf, plo, phi, _ = evaluate(P_lo, P_hi)
        update_incumbent(px, pf, plb)
        plb = np.where(np.isnan(plb), -np.inf, plb).reshape(-1, 2)
        dead = plb >= ub - tol()
        plo, phi = plo.reshape(-1, 2, lo.size), phi.reshape(-1, 2, lo.size)
        # a side's tightened box; one with non-finite bounds tightens nothing (its probe box)
        bad = ~(np.isfinite(plo) & np.isfinite(phi))
        plo = np.where(bad, P_lo.reshape(-1, 2, lo.size), plo)
        phi = np.where(bad, P_hi.reshape(-1, 2, lo.size), phi)
        # a dead side does not widen the hull (masked, not inf arithmetic: inf - inf is NaN)
        hull_lo = np.min(np.where(dead[..., None], np.inf, plo), axis=1)
        hull_hi = np.max(np.where(dead[..., None], -np.inf, phi), axis=1)
        new_lo = np.maximum(rlo[0], np.max(np.where(dead.all(axis=1)[:, None], rlo[0], hull_lo), axis=0))
        new_hi = np.minimum(rhi[0], np.min(np.where(dead.all(axis=1)[:, None], rhi[0], hull_hi), axis=0))
        new_lb = np.maximum(lb, np.max(np.min(plb, axis=1)))
        if np.any(new_lo > new_hi) or dead.all(axis=1).any():  # every side of some binary is dead
            return rlo, rhi, np.maximum(new_lb, ub)
        return new_lo[None], new_hi[None], new_lb

    def fixed_solve(L, H, S):
        """Local solves of the continuous variables in boxes whose integers are fixed (``L``,
        ``H``) from the points ``S``; improves the incumbent."""
        nonlocal ub, xbest
        if _past(deadline):
            return
        X = np.clip(np.asarray(solve_local(np.clip(S, L, H), L, H)), L, H)
        F = screen(X)
        i = int(np.argmin(F))
        if F[i] < ub:
            ub, xbest = float(F[i]), X[i]
        polish()

    def need_obbt():
        """Whether the root OBBT pass runs: with an incumbent cutoff known (it is useless without
        one when there are no constraints), and not when the root bound already closes the gap."""
        return bool(obbt) and (constrained or np.isfinite(ub)) and not (ub - float(lb.min()) <= tol())

    def start_root_obbt():
        """Compile the root OBBT kernel in the background (once; ``start`` skips a started shape)."""
        if obbt == "root" and padded.mesh is None:
            K.obbt.start(1, lo.size, functools.partial(K.fingerprint, lo.size) if kcache.cache_root() else None)

    _mark("root_bound")
    if obbt == "root" and (constrained or np.isfinite(ub)) and any(bound_kernel.pending(b, lo.size, extra) for b in padded.sizes):
        # the main thread is about to wait for the bound kernel's build: trace the OBBT kernel in that
        # idle time (queued behind the bound and dive kernels) rather than after the root bound,
        # when ``need_obbt`` can first tell. A warm cache has the bound kernel ready: no guess then.
        start_root_obbt()
    try:
        lb, xc, fc, rlo, rhi, rxr = evaluate(lo[None], hi[None])
    except _CompileTimeout as e:  # no bound at all yet: report what is known
        why = "before the root bound" if isinstance(e, _Deadline) else f"while compiling the '{e}' kernel"
        xbest = np.asarray(xbest)
        maxv, verified, _ = _verify(f, ineq, eq, xbest, ub, lo_user, hi_user, feas_tol, integer=imask)
        return Result(x=xbest, fun=ub, lower_bound=-np.inf, converged=False, iterations=0, nodes=0,
                      evaluations=evaluations, time=time.perf_counter() - t0, status="time_limit",
                      message=f"time_limit reached {why}; no bound computed",
                      max_violation=maxv, verified=verified)  # fmt: skip
    _mark("root_incumbent")
    update_incumbent(xc, fc, lb)

    if need_obbt():
        # compile the root OBBT kernel only now, while the root dives run: a problem whose root
        # bound and first incumbent close the gap (most small ones) never traces or compiles it,
        # and the bound kernel's compile does not compete with it for the CPU
        start_root_obbt()
    _mark("root_dives")
    if imask.any() and np.all(np.isfinite(lb)) and np.all(np.isfinite(rxr)):  # root dives: relaxation point, perturbed
        P = np.clip(rxr + rng.normal(size=(_DIVES, lo.size)) * 0.1 * (rhi - rlo) * (np.arange(_DIVES) > 0)[:, None],
                    rlo, rhi)  # fmt: skip
        dive(np.repeat(rlo, _DIVES, 0), np.repeat(rhi, _DIVES, 0), P)
        # few integer assignments: fix-and-solve every one (rounding and dives try one or two)
        ii = np.flatnonzero(imask)
        sizes = rhi[0, ii] - rlo[0, ii] + 1
        _mark("root_enum")
        if np.prod(sizes) <= _ENUM and time_for(None):  # incumbent-only, like the dives
            A = np.array(list(itertools.product(*[np.arange(rlo[0, j], rhi[0, j] + 1) for j in ii])))
            A = np.repeat(A, 4, 0)  # each from the relaxation point and 3 random points of the box
            L, H = np.repeat(rlo, len(A), 0), np.repeat(rhi, len(A), 0)
            L[:, ii] = H[:, ii] = A
            S = np.where((np.arange(len(A)) % 4 == 0)[:, None], rxr, rng.uniform(L, H))
            fixed_solve(L, H, S)
    _mark("root_obbt")
    if need_obbt():  # (the search would end at once on a closed gap: OBBT would only cost its compile)
        # the dives may have found the first incumbent (no constraints): start the kernel now if
        # not yet, so that the call below waits for it until the deadline instead of compiling
        # synchronously through ``jax.jit`` (which has no deadline)
        start_root_obbt()
        try:
            lb2, xc, fc, rlo2, rhi2, rxr2 = evaluate(rlo, rhi, with_obbt=True)
        except _CompileTimeout:  # the root bound stands; the search loop then stops on the time limit
            pass
        else:
            rlo, rhi, rxr = rlo2, rhi2, rxr2
            lb = np.maximum(lb, lb2)
            _mark("root_obbt_incumbent")
            update_incumbent(xc, fc, lb)
    _mark("probe")
    if _PROBE and imask.any() and np.all(np.isfinite(lb)) and not (ub - float(lb.min()) <= tol()):
        try:
            rlo, rhi, lb = probe(rlo, rhi, lb)
        except _CompileTimeout:
            pass
    _mark("search")
    open_lo, open_hi, open_lb, open_xr = rlo, rhi, lb, rxr
    width0 = _branch_widths(width0, rlo, rhi)
    sumL, sumR, cnt = np.zeros(lo.size), np.zeros(lo.size), np.zeros(lo.size)  # pseudocost sums, observations

    def pseudo_scores(relw, free, xr):
        """Product-rule scores per (box, variable), -1 for non-candidates; and whether any pseudocost is known.

        A continuous variable's gains are per squared relative width; an integer variable's
        (as in MIP) per unit of the distance its split moves it, ``f`` down and ``1 - f`` up
        for the fractional part ``f`` of the relaxation value (0.5 each when integral)."""
        cand = branchable & free
        cand = np.where(cand.any(axis=1, keepdims=True), cand, free)
        seen = cnt > 0
        if not seen.any():
            return np.where(cand, relw, -1.0), False
        mL, mR = (sumL[seen] / cnt[seen]).mean(), (sumR[seen] / cnt[seen]).mean()
        # a variable whose early splits gained nothing would otherwise never be tried again
        psiL = np.where(seen, np.maximum(sumL / np.maximum(cnt, 1), _FLOOR * mL), mL)
        psiR = np.where(seen, np.maximum(sumR / np.maximum(cnt, 1), _FLOOR * mR), mR)
        eps = 1e-6 * (1 + abs(ub)) if np.isfinite(ub) else 1e-6
        r2 = relw**2
        fr = xr - np.floor(xr)
        fr = np.where(fr > INT_TOL, fr, 0.5)
        dn, up = np.where(imask, fr, r2), np.where(imask, 1.0 - fr, r2)
        sc = np.maximum(psiL * dn, eps) * np.maximum(psiR * up, eps) * (1 + 1e-3 * relw)  # width breaks ties
        return np.where(cand & np.isfinite(sc), sc, np.where(cand, 0.0, -1.0)), True

    pruned_lb = np.inf  # smallest lower bound among fathomed boxes
    dive_every = 2 if not np.isfinite(ub) else 10
    next_dive = dive_every
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
        if branch_var in ("reliability", "strong"):
            sel = sel[np.argsort(open_lb[sel], kind="stable")]
        plo, phi, pxr, plb = open_lo[sel], open_hi[sel], open_xr[sel], open_lb[sel]
        relw = (phi - plo) / width0  # relative width (0 for variables never split)

        free = phi > plo
        if branch_var == "width":
            pp = np.arange(k)
            score = np.where(branchable & free, relw, -1.0)
            if quad is not None:  # lifted quadratic LP: relative width times the McCormick gap
                gsc = _gap_scores(quad, plo, phi, pxr)
                gmax = gsc.max(axis=1, keepdims=True)
                score = np.where(branchable & free & (gmax > 0), relw * gsc / np.where(gmax > 0, gmax, 1.0), score)
            if imask.any():  # integers fractional at the relaxation point first, most fractional first
                frac = np.abs(pxr - np.round(pxr))
                score = np.where(imask & free & (frac > INT_TOL), 2.0 + frac, score)
            none = (score < 0).all(axis=1)
            if none.any():  # no branchable variable left to split: the widest of any kind
                score[none] = np.where(free[none], (phi - plo)[none] / np.where(hi > lo, hi - lo, 1.0), -1.0)
            # a box with no free variable at all (range reduction can shrink one to a point)
            # is "split" into two copies of itself; bounding a point is exact, so they are pruned
            pv = np.argmax(score, axis=1)
            groups = [[i] for i in range(k)]
        else:
            score, _ = pseudo_scores(relw, phi > plo, pxr)
            if branch_var == "pseudocost":
                pv = np.argmax(score, axis=1)
                pp = np.arange(k)
                groups = [[i] for i in range(k)]
            elif branch_var == "strong":
                # balance: only variables at least 1/_BAL as wide as the widest are
                # candidates, so no variable is starved (gain-based scores otherwise
                # keep splitting one variable of nonsmooth f, e.g. holder_table)
                wmax = np.max(np.where(branchable, relw, 0.0), axis=1, keepdims=True)
                score = np.where(relw * _BAL >= wmax, score, -1.0)
                # every parent gets its best-scored variable, so as many parents are
                # split as with the other rules; spare slots then go round-robin,
                # lowest bound first, to each parent's next-best variables
                order = np.argsort(-score, axis=1, kind="stable")
                ncand = (score >= 0).sum(axis=1)
                lists = [[int(order[i, 0])] for i in range(k)]
                # fill only the padding of the call the 2k children need anyway: those
                # slots are computed regardless, so the trials cost no extra time
                # (filling to ``batch`` made each iteration a full-size call, 2-7x slower)
                top = padded.sizes[-1]
                fill = next((z for z in padded.sizes if z >= 2 * k), -(-2 * k // top) * top)
                spare, r = (fill - 2 * k) // 2, 1
                while spare > 0 and (ncand > r).any():
                    for i in np.flatnonzero(ncand > r)[:spare]:
                        lists[i].append(int(order[i, r]))
                    spare -= int(min((ncand > r).sum(), spare))
                    r += 1
                pp_l, pv_l, groups = [], [], []
                for i, vs in enumerate(lists):
                    groups.append(list(range(len(pp_l), len(pp_l) + len(vs))))
                    pp_l += [i] * len(vs)
                    pv_l += vs
                pp, pv = np.array(pp_l), np.array(pv_l)
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
        rpt = pt
        isint = np.zeros(M, bool)
        if imask.any():
            # integer variables: children [lo, p] and [p + 1, hi], p = floor of the relaxation
            # value when it is fractional and inside the box, else of the midpoint
            v = pxr[pp, pv]
            v = np.where((np.abs(v - np.round(v)) > INT_TOL) & (v >= lov) & (v <= hiv), v, 0.5 * (lov + hiv))
            ip = np.clip(np.floor(v), lov, hiv - 1.0)
            isint = imask[pv] & (hiv > lov)
            v_split = v
            pt, rpt = np.where(isint, ip, pt), np.where(isint, ip + 1.0, pt)
        left_hi, right_lo = phi[pp].copy(), plo[pp].copy()
        left_hi[ar, pv] = pt
        right_lo[ar, pv] = rpt
        clo = np.concatenate([plo[pp], right_lo])
        chi = np.concatenate([left_hi, phi[pp]])

        nlp_do = pool_nlp(clo, chi, np.concatenate([pxr[pp], pxr[pp]]), np.concatenate([ar, ar]))
        try:
            clb, cx, cf, clo, chi, cxr = evaluate(clo, chi, with_obbt=obbt == "tree")
        except _CompileTimeout:  # out of time while a kernel compiles; the open boxes are unchanged
            break
        pool_judge(nlp_do, clb, np.concatenate([plb[pp], plb[pp]]))
        if convex and nlp_do is not None and nlp_do.any():
            # a convex continuous relaxation: the local solutions are its minimizers in each box,
            # better places to branch than the LP points (an LP over tangents of the convex
            # parts is often degenerate at its optimum, with a point far from the minimizer:
            # on cvxnonsep_normcon40 the nodes fell from 547 to 291). Only branching uses it.
            X, rows = pool["nlp_x"], np.flatnonzero(nlp_do)
            ok = np.isfinite(X).all(axis=1)  # a failed solve keeps its LP point
            cxr = cxr.copy()
            cxr[rows[ok]] = np.clip(X[ok], clo[rows[ok]], chi[rows[ok]])
        update_incumbent(cx, cf, clb)
        if imask.any() and it >= next_dive:  # dives from the most promising children
            o = np.argsort(clb)[:_DIVES]
            before = ub
            dive(clo[o], chi[o], cxr[o])
            # every 2 iterations without an incumbent, then every 10, doubling (up to
            # _DIVE_BACKOFF) after dives that do not improve it
            dive_every = 2 if not np.isfinite(ub) else (10 if ub < before else min(2 * dive_every, _DIVE_BACKOFF))
            next_dive = it + dive_every
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
            # integer splits: per unit moved, v - p down and p + 1 - v up (p = the split point)
            dl = np.where(isint, np.maximum(v_split - pt, 1e-6), r2) if imask.any() else r2
            dr = np.where(isint, np.maximum(pt + 1.0 - v_split, 1e-6), r2) if imask.any() else r2
            good = np.isfinite(gl) & np.isfinite(gr) & (dl > 1e-24) & (dr > 1e-24)
            np.add.at(sumL, pv[good], gl[good] / dl[good])
            np.add.at(sumR, pv[good], gr[good] / dr[good])
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
        # children cannot be worse than their parent's bound, nor than the
        # weaker side of any tried split of it
        raised = np.maximum(plb, pmax)
        if branch_var == "strong" and _Q > 1 and M > k:
            # orthant multisection: split each parent on its q best strong-branched
            # variables at once. An orthant is the intersection of one (tightened)
            # half per variable, so it lies in each of them and its bound is the
            # largest of theirs; no box is bounded twice.
            ol, oh, ob, ox = [], [], [], []
            for i, g in enumerate(groups):
                g = np.asarray(g)
                g = g[np.argsort(-strong[g], kind="stable")]  # NaN scores sort last; all NaN: q = g[:1]
                q = g[strong[g] >= _ETA_MS * strong[g[0]]][:_Q]
                q = q[np.unique(pv[q], return_index=True)[1]] if len(q) else g[:1]  # one split per variable
                for bits in itertools.product((0, M), repeat=len(q)):
                    h = q + np.array(bits)
                    lo_o, hi_o = clo[h].max(axis=0), chi[h].min(axis=0)
                    if np.any(lo_o > hi_o):
                        continue
                    ol.append(lo_o)
                    oh.append(hi_o)
                    ob.append(max(raised[i], float(clb[h].max())))
                    ox.append(np.clip(cxr[h[0]], lo_o, hi_o))
            nodes += len(ob)
            new_lo, new_hi = np.array(ol).reshape(-1, lo.size), np.array(oh).reshape(-1, lo.size)
            new_lb, new_xr = np.array(ob), np.array(ox).reshape(-1, lo.size)
        else:
            rows = np.concatenate([chosen, chosen + M])
            nodes += 2 * k
            new_lo, new_hi, new_xr = clo[rows], chi[rows], cxr[rows]
            new_lb = np.maximum(clb[rows], np.concatenate([raised, raised]))

        open_lo = np.concatenate([open_lo[rest], new_lo])
        open_hi = np.concatenate([open_hi[rest], new_hi])
        open_lb = np.concatenate([open_lb[rest], new_lb])
        open_xr = np.concatenate([open_xr[rest], new_xr])

    open_lb = np.where(np.isnan(open_lb), -np.inf, open_lb)  # (children added by the last iteration)
    glb = min(float(open_lb.min()) if len(open_lb) else np.inf, pruned_lb, ub)
    if converged:
        status = "optimal" if np.isfinite(ub) else "infeasible"
    elif time.perf_counter() - t0 > time_limit:
        status = "time_limit"
    else:
        status = "max_iter"
    xbest = np.asarray(xbest)
    xbest = np.asarray(xbest)
    if imask.any() and np.isfinite(ub):
        xbest = np.where(imask, np.round(xbest), xbest)
    message = ""
    if constrained and local_solve and np.isfinite(ub):
        xbest, ub, message = repair_incumbent(xbest, ub)
        glb = min(glb, ub)  # (the repair solve may also lower f)
    maxv, verified, why = _verify(f, ineq, eq, xbest, ub, lo_user, hi_user, feas_tol, integer=imask)
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
