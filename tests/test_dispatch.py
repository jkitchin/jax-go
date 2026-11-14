"""Concurrent dispatch of the batch-kernel calls in ``minimize`` (``bnb._dispatch_threads``).

The calls of one batch of boxes run from a thread pool; every box must get exactly the bound,
point and tightened box it gets when the calls run one after another, in the same order.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from jaxglobal import bnb, minimize


def _f(x):
    return jnp.sum((x - jnp.array([0.3, -1.7, 2.2, 0.9])) ** 2) + 0.5 * x[0] * x[1] - 0.25 * x[2] * x[3]


def _g(x):
    return jnp.array([x[0] + x[1] + x[2] - 2.5, x[3] ** 2 - x[0] - 3.0])


LO, HI = np.full(4, -3.0), np.full(4, 3.0)
INT = np.array([True, True, True, False])
_CALL = bnb._Precompiled.__call__


def _solve(monkeypatch, threads):
    """The result and every bound-kernel call: {(input bytes): outputs}, in call order."""
    monkeypatch.setattr(bnb, "DISPATCH_THREADS", threads)
    monkeypatch.delenv("JAXGLOBAL_DISPATCH_THREADS", raising=False)
    calls, orig = [], _CALL

    def spy(self, *args, deadline=None):
        out = orig(self, *args, deadline=deadline)
        if self.name.startswith("bound"):
            calls.append((b"".join(np.asarray(a).tobytes() for a in args), [np.asarray(o) for o in out]))
        return out

    monkeypatch.setattr(bnb._Precompiled, "__call__", spy)
    r = minimize(_f, LO, HI, ineq=_g, integer=INT, batch=32, batch_sizes=(4,), max_iter=6, local_solve=False)
    return r, calls


def test_threaded_dispatch_matches_sequential(monkeypatch):
    # a first solve compiles: its search depends on which background kernels are ready when
    # (cut pool variants, dive cost estimates), so it is only a warm-up
    _solve(monkeypatch, 1)
    r1, c1 = _solve(monkeypatch, 1)
    r4, c4 = _solve(monkeypatch, 4)
    assert len(c1) == len(c4) and len(c1) > 4  # several calls per batch
    seq, thr = dict(c1), dict(c4)
    assert seq.keys() == thr.keys()  # the same boxes, in calls with the same padding and cutoff
    for key in seq:
        for a, b in zip(seq[key], thr[key]):
            np.testing.assert_array_equal(a, b)
    assert r1.nodes == r4.nodes
    assert r1.lower_bound == r4.lower_bound
    assert r1.fun == r4.fun
    np.testing.assert_array_equal(r1.x, r4.x)


@pytest.mark.parametrize("env, want", [("1", 1), ("3", 3), ("", None)])
def test_dispatch_threads_env(monkeypatch, env, want):
    monkeypatch.setattr(bnb, "DISPATCH_THREADS", None)
    monkeypatch.setenv("JAXGLOBAL_DISPATCH_THREADS", env)
    monkeypatch.setenv("RAYON_NUM_THREADS", "2")  # benchmarks/run.py --threads-per-job 2
    assert bnb._dispatch_threads() == (2 if want is None else want)
