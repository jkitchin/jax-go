"""Single-threaded OpenBLAS while ``minimize`` runs (``jaxglobal/blas.py``)."""

import threading

import jax.numpy as jnp
import numpy as np
import pytest

from jaxglobal import blas, bnb, minimize

if not blas.thread_counts():  # (no OpenBLAS loaded, e.g. another BLAS build: nothing to test)
    pytest.skip("no OpenBLAS library loaded", allow_module_level=True)


def _f(x):
    return (x[0] - 0.3) ** 2 + x[0] * x[1]


def test_limited_sets_and_restores():
    before = blas.thread_counts()
    with blas.limited(1):
        assert blas.thread_counts() == [1] * len(before)
        with blas.limited(1):  # nested: the inner exit does not restore
            pass
        assert blas.thread_counts() == [1] * len(before)
    assert blas.thread_counts() == before


def test_limited_zero_leaves_libraries_alone():
    before = blas.thread_counts()
    with blas.limited(0):
        assert blas.thread_counts() == before


def test_limited_restores_after_exception():
    before = blas.thread_counts()
    with pytest.raises(RuntimeError):
        with blas.limited(1):
            raise RuntimeError("boom")
    assert blas.thread_counts() == before


def test_concurrent_holders_restore_once_all_left():
    before = blas.thread_counts()
    inside, release = threading.Event(), threading.Event()

    def hold():
        with blas.limited(1):
            inside.set()
            release.wait(10)

    t = threading.Thread(target=hold)
    t.start()
    inside.wait(10)
    with blas.limited(1):
        pass
    assert blas.thread_counts() == [1] * len(before)  # the other holder is still inside
    release.set()
    t.join()
    assert blas.thread_counts() == before


def test_minimize_runs_single_threaded_blas_and_restores(monkeypatch):
    before = blas.thread_counts()
    seen = []
    orig = bnb._minimize

    def spy(*a, **k):
        seen.append(blas.thread_counts())
        return orig(*a, **k)

    monkeypatch.setattr(bnb, "_minimize", spy)
    monkeypatch.setattr(blas, "THREADS", 1)
    r = minimize(_f, np.array([-1.0, -1.0]), np.array([1.0, 1.0]), ineq=lambda x: jnp.array([x[0] + x[1] - 1.5]))
    assert r.status == "optimal"
    assert seen == [[1] * len(before)]
    assert blas.thread_counts() == before
