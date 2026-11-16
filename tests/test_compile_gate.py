"""Compile scheduling and the time limit around kernel calls (``bnb._Gate``, ``bnb._await``,
``bnb._batched``)."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from jaxglobal import bnb  # noqa: E402


def test_gate_holds_until_opened():
    g = bnb._Gate()
    ran = threading.Event()
    fut = g.submit(lambda: ran.set() or 7)
    time.sleep(0.2)
    assert not ran.is_set() and not fut.done()
    g.open()
    assert fut.result(timeout=10) == 7
    assert g.submit(lambda: 8).result(timeout=10) == 8  # open: runs at once


def test_gate_release_on_wait_and_failures():
    g = bnb._Gate()
    fut = g.submit(lambda: 1 / 0)
    bnb._Gate.release(fut)  # a caller needs it
    with pytest.raises(ZeroDivisionError):
        fut.result(timeout=10)
    bnb._Gate.release(object())  # not a held future: nothing to do


def test_gate_after_inner_future():
    """``after`` opens once the build future's inner (compile) future is done, or at once when
    ``small()`` holds."""
    ex = ThreadPoolExecutor(2)
    go = threading.Event()
    inner = ex.submit(go.wait)
    outer = ex.submit(lambda: inner)
    g = bnb._Gate()
    g.after(outer)
    held = g.submit(lambda: 3)
    time.sleep(0.2)
    assert not held.done()
    go.set()
    assert held.result(timeout=10) == 3
    g2 = bnb._Gate()
    g2.after(ex.submit(lambda: ex.submit(threading.Event().wait, 30)), small=lambda: True)
    assert g2.submit(lambda: 4).result(timeout=10) == 4
    ex.shutdown(wait=False, cancel_futures=True)


def _slow(n=1500):
    A = jnp.ones((n, n)) / n

    @jax.jit
    def f(a):
        return jax.lax.fori_loop(0, 60, lambda i, x: x @ a, a)

    jax.block_until_ready(f(A))  # compile (and one run) outside the timing
    return f, A


def test_await_deadline():
    f, A = _slow()
    t = time.perf_counter()
    jax.block_until_ready(f(A))
    dt = time.perf_counter() - t
    if dt < 0.2:
        pytest.skip(f"the slow call takes only {dt:.3f} s here")
    t = time.perf_counter()
    with pytest.raises(bnb._Deadline):
        bnb._await(f(A), time.perf_counter() + 0.05 * dt, "slow")
    assert time.perf_counter() - t < 0.5 * dt + 0.05
    out = bnb._await(f(A), time.perf_counter() + 100 * dt + 10, "slow")
    assert np.all(np.isfinite(np.asarray(out)))
    assert np.asarray(bnb._await(jnp.ones(3), None, "x")).sum() == 3


def test_batched_one_shape():
    """``_batched`` gives every row's value and compiles a single batch shape."""
    fn = jax.jit(jax.vmap(lambda x: jnp.sum(x**2)))
    rng = np.random.default_rng(0)
    for k in (1, 7, 8, 9, 30, 64):
        X = rng.normal(size=(k, 3))
        assert np.allclose(bnb._batched(fn, X), np.sum(X**2, axis=1))
    assert fn._cache_size() == 1
