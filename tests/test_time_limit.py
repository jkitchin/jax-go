"""minimize's time_limit also covers tracing and compiling the bounding kernels."""

import concurrent.futures
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxglobal import bnb, minimize


def test_time_limit_covers_compilation():
    # a fresh function: its kernels are not in the in-memory cache, so the root bound has to wait for them
    f = lambda x: jnp.sum(jnp.sin(3 * x) * jnp.exp(-x**2)) + 0.123456  # noqa: E731
    t = time.perf_counter()
    r = minimize(f, -np.ones(3), np.ones(3), time_limit=1e-6)
    assert time.perf_counter() - t < 30  # returns promptly instead of waiting for the compile
    assert r.status == "time_limit"
    assert r.lower_bound == -np.inf and r.nodes == 0
    # still compiling, or (kernels cached on disk) compiled but past the deadline before the call
    assert "compiling" in r.message or "before the root bound" in r.message


def test_time_limit_does_not_affect_normal_solves():
    r = minimize(lambda x: (x[0] - 0.3) ** 2 + x[1] ** 2, -np.ones(2), np.ones(2), time_limit=120)
    assert r.status == "optimal" and abs(r.fun) < 1e-6


def test_time_limit_covers_deriving_bounds():
    """Unbounded variables: bound_domain's FBBT kernel compiles before any bounding kernel."""
    f = lambda x: (x[0] - 0.5) ** 2 + jnp.cos(x[1]) + 0.654321  # noqa: E731
    g = lambda x: jnp.array([x[0] + x[1] - 1.0, -x[0] - 2.0, x[1] ** 2 - 4.0])  # noqa: E731
    t = time.perf_counter()
    r = minimize(f, -np.inf * np.ones(2), np.inf * np.ones(2), ineq=g, time_limit=1e-6)
    assert time.perf_counter() - t < 30
    assert r.status == "time_limit" and r.lower_bound == -np.inf
    assert "bounds" in r.message


# ---------------------------------------------------------------- deadline in the search (A1)


def _hard(x):  # many local minima; with zero tolerances the search never closes the gap
    return jnp.sum(jnp.sin(5 * x) * x + 0.1 * x**2) + 0.0271828


def _mixed(x):  # four continuous and three integer variables
    return _hard(x[:4]) + jnp.sum((x[4:] - 1.37) ** 2 * jnp.cos(x[:1] + x[4:])) + 0.0314159


MIXED = dict(integer=[4, 5, 6], abs_tol=0.0, rel_tol=0.0, max_iter=10**6)
MIXED_BOX = np.array([-3.0] * 4 + [0] * 3), np.array([3.0] * 4 + [4] * 3)


def _slow_bound(monkeypatch, seconds):
    """Make every bound-kernel call take ``seconds`` longer (the sleep follows the dispatch, so
    it counts as kernel run time) and count the calls refused by the deadline."""
    orig, n = bnb._Precompiled.__call__, {"calls": 0, "deadline": 0}

    def call(self, *args, **kw):
        try:
            out = orig(self, *args, **kw)
        except bnb._Deadline:
            n["deadline"] += 1
            raise
        if self.name == "bound":
            n["calls"] += 1
            jax.block_until_ready(out)
            time.sleep(seconds)
        return out

    monkeypatch.setattr(bnb._Precompiled, "__call__", call)
    return n


def test_deadline_mid_search_keeps_pre_iteration_bound(monkeypatch):
    """A kernel call past the deadline is not dispatched; the iteration it belongs to is dropped
    and the result carries the lower bound from before that iteration.

    Every bound call is slowed by 0.2 s, and the deadline is made to pass during the first call
    of an iteration that has more than one call (the first such iteration after half the limit), so that the check before the second
    call, not the per-iteration check at the top of the loop, stops the search. (Slowing every
    call instead left it to timing which check came first: when the limit passed during an
    iteration's last call, the loop stopped at its top with no refused call.)"""
    lo, hi = -3 * np.ones(4), 3 * np.ones(4)
    opts = dict(abs_tol=0.0, rel_tol=0.0, batch=64, batch_sizes=(16,))
    minimize(_hard, lo, hi, max_iter=3, **opts)  # compile
    limit, n = 3.0, {"deadline": 0, "slept": 0}
    run = {"calls": 0, "pos": 0}  # calls in the current padded batch, and the next call's index
    pad_orig, call_orig = bnb._Padder.__call__, bnb._Precompiled.__call__

    def pad(self, *arrays):
        calls = pad_orig(self, *arrays)
        run.update(calls=len(calls), pos=0)
        return calls

    def call(self, *args, **kw):
        try:
            out = call_orig(self, *args, **kw)
        except bnb._Deadline:
            n["deadline"] += 1
            raise
        if self.name == "bound":
            jax.block_until_ready(out)
            first, run["pos"] = run["pos"] == 0, run["pos"] + 1
            if first and run["calls"] > 1 and not n["slept"] and time.perf_counter() - t > limit / 2:
                n["slept"] = 1  # this call runs past the deadline; the next one is refused
                time.sleep(max(t + limit + 0.2 - time.perf_counter(), 0.0))
            else:
                time.sleep(0.2)  # slow enough that the search does not finish first
        return out

    monkeypatch.setattr(bnb._Padder, "__call__", pad)
    monkeypatch.setattr(bnb._Precompiled, "__call__", call)
    t = time.perf_counter()
    r = minimize(_hard, lo, hi, time_limit=limit, max_iter=10**6, **opts)
    wall = time.perf_counter() - t
    assert n["slept"] == 1  # the search reached a multi-call iteration after half the limit
    assert r.status == "time_limit" and np.isfinite(r.lower_bound)
    assert wall < limit + 1.5  # one call (0.2 s here) past the limit, plus the float64 check
    assert n["deadline"] == 1  # stopped by the check before a call, inside an iteration
    assert r.lower_bound == r.history[-1][2]  # the bound before the aborted iteration
    assert r.lower_bound <= r.fun


def test_deadline_does_not_change_the_search_before_the_limit():
    lo, hi = -3 * np.ones(3), 3 * np.ones(3)
    a = minimize(_hard, lo, hi, max_iter=30)
    b = minimize(_hard, lo, hi, max_iter=30, time_limit=600)
    assert a.history == b.history and a.lower_bound == b.lower_bound and a.fun == b.fun


class _CountingDive:
    """Stands in for ``_Kernels.dive`` (a ``_Precompiled``): counts its runs, each made
    ``seconds`` longer."""

    def __init__(self, jitted, seconds=0.0):
        self.jitted, self.seconds, self.runs = jitted, seconds, 0

    def start(self, *a, **k):
        pass

    def wait(self, *args, deadline=None):
        pass

    def __call__(self, *args, deadline=None):
        out = jax.block_until_ready(self.jitted(*args))
        self.runs += 1
        time.sleep(self.seconds)
        return out


def _kernels(f):
    return next(v for k, v in bnb._CACHE.items() if k[0] == "kernels" and k[1] is f)


def test_dive_budget_skips_when_time_is_short(monkeypatch):
    lo, hi = MIXED_BOX
    minimize(_mixed, lo, hi, **{**MIXED, "max_iter": 3})  # compile and cache the kernels
    K = _kernels(_mixed)
    # lift the dives' share of the wall time (_DIVE_FRAC): with cached kernels this short search
    # takes so little time that the share alone skips every dive; this test is about time_limit
    monkeypatch.setattr(bnb, "_DIVE_FRAC", np.inf)
    monkeypatch.setattr(K, "dive", _CountingDive(K.dive))
    minimize(_mixed, lo, hi, **{**MIXED, "max_iter": 12})  # no time limit: root and periodic dives
    assert K.dive.runs >= 2
    # a 2 s dive under a 4 s limit: the root dive runs, then no later dive fits in 1.5 x 2 s
    # (bound calls slowed to 0.1 s so that the search does not finish first)
    monkeypatch.setattr(K, "dive", _CountingDive(K.dive.jitted, seconds=2.0))
    _slow_bound(monkeypatch, 0.1)
    t = time.perf_counter()
    r = minimize(_mixed, lo, hi, time_limit=4.0, **MIXED)
    assert r.status == "time_limit" and K.dive.runs == 1
    assert time.perf_counter() - t < 4.0 + 1.5


def test_dive_share_of_wall_time(monkeypatch):
    """With an incumbent, dives get at most _DIVE_FRAC of the elapsed wall time: at 0 none runs,
    without the cap the same search dives at the root and periodically."""
    lo, hi = MIXED_BOX
    minimize(_mixed, lo, hi, **{**MIXED, "max_iter": 3})
    K = _kernels(_mixed)
    monkeypatch.setattr(K, "dive", _CountingDive(K.dive))
    monkeypatch.setattr(bnb, "_DIVE_FRAC", 0.0)
    r = minimize(_mixed, lo, hi, **{**MIXED, "max_iter": 12})
    assert np.isfinite(r.fun) and K.dive.runs == 0
    monkeypatch.setattr(bnb, "_DIVE_FRAC", np.inf)
    minimize(_mixed, lo, hi, **{**MIXED, "max_iter": 12})
    assert K.dive.runs >= 2


def test_first_dive_estimate_from_bound_kernel(monkeypatch):
    """Before any dive its cost is estimated as (integer variables) x (slowest bound call):
    3 x 0.5 s here, so with less than 2.25 s left the root dive is skipped."""
    lo, hi = MIXED_BOX
    minimize(_mixed, lo, hi, **{**MIXED, "max_iter": 3})
    K = _kernels(_mixed)
    monkeypatch.setattr(bnb, "_DIVE_FRAC", np.inf)  # skipped by the time limit, not by the share
    monkeypatch.setattr(K, "dive", _CountingDive(K.dive))
    _slow_bound(monkeypatch, 0.5)
    r = minimize(_mixed, lo, hi, time_limit=2.0, **MIXED)
    assert r.status == "time_limit" and K.dive.runs == 0
    monkeypatch.undo()


def _done(value=None, error=None):
    fut = concurrent.futures.Future()
    fut.set_exception(error) if error is not None else fut.set_result(value)
    return fut


ARGS = (jnp.zeros((2, 3)), jnp.ones((2, 3)), jnp.asarray(0.0))


@pytest.mark.parametrize("started", [True, False])
def test_compile_fallback_respects_deadline(started):
    """No synchronous jit compile past the deadline, whether the background compile failed or
    the shape was never started."""
    calls = []
    k = bnb._Precompiled(lambda *a: calls.append(a), "bound")
    if started:
        k.ready[k._key(ARGS)] = _done(error=RuntimeError("background compile failed"))
    with pytest.raises(bnb._CompileTimeout) as e:
        k(*ARGS, deadline=time.perf_counter() - 1.0)
    assert not isinstance(e.value, bnb._Deadline) and not calls
    k(*ARGS, deadline=time.perf_counter() + 60.0)  # in time: jit as before
    assert len(calls) == 1


def test_ready_kernel_not_dispatched_past_deadline():
    ran = []
    k = bnb._Precompiled(None, "bound")
    k.ready[k._key(ARGS)] = _done(_done(lambda *a: ran.append(a)))
    with pytest.raises(bnb._Deadline):
        k(*ARGS, deadline=time.perf_counter() - 1.0)
    assert not ran
    k(*ARGS, deadline=None)
    assert len(ran) == 1


def _late_incumbent(x):
    return jnp.sum((x - 0.4) ** 2) + jnp.sin(3 * x[0]) * x[1] + 0.3 * jnp.cos(5 * x[2])


def test_root_obbt_kernel_started_when_dives_find_first_incumbent(monkeypatch):
    """No constraints and no incumbent after the root bound: root OBBT is not needed yet, so its
    kernel is not started. When the root dives then find the first incumbent, OBBT runs: its
    kernel must be started in the background (and waited for until the deadline), not compiled
    synchronously by the ``jax.jit`` fallback, which has no deadline."""
    calls, first = [], [True]
    call = bnb._Precompiled.__call__

    def counting(self, *args, **kw):
        if self.name == "obbt":
            calls.append(self._key(args) in self.ready)
        out = call(self, *args, **kw)
        if self.name == "bound" and first[0]:  # the root bound finds no feasible candidate
            first[0] = False
            out = (*out[:2], jnp.full_like(out[2], jnp.inf), *out[3:])
        return out

    monkeypatch.setattr(bnb._Precompiled, "__call__", counting)
    r = minimize(_late_incumbent, -3 * np.ones(4), 3 * np.ones(4), integer=[0, 1, 2, 3],
                 local_solve=False, time_limit=120)  # fmt: skip
    assert np.isfinite(r.fun)
    assert calls == [True]
