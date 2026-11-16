"""The cut-generation memory budget (jaxglobal/memory.py) and the reduced cut points (lp.axis_vars)."""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import jaxglobal.lp as LP  # noqa: E402
from jaxglobal import memory  # noqa: E402


def test_point_chunk():
    assert memory.point_chunk(None, 100) is None
    assert memory.point_chunk(1.0, 100, boxes=16, limit=1 << 30) is None  # fits: one vmap
    # 100 points x 16 boxes x 1 MB > 512 MB: 32 points fit, as 4 equal chunks of 25
    assert memory.point_chunk(2**20, 100, boxes=16, limit=512 * 2**20) == 25
    assert memory.point_chunk(2**30, 100, boxes=16, limit=1) == 1


def test_parse_budget(monkeypatch):
    monkeypatch.setenv("JAXGLOBAL_MEMORY_BUDGET", "2G")
    assert memory.budget() == 2 << 30
    monkeypatch.setenv("JAXGLOBAL_MEMORY_BUDGET", "300M")
    assert memory.budget() == 300 << 20
    monkeypatch.delenv("JAXGLOBAL_MEMORY_BUDGET")
    assert 0 < memory.budget() <= memory.DEFAULT_BUDGET


def test_map_points_matches_vmap():
    f = lambda p: (jnp.sin(p).sum(), jnp.outer(p, p))  # noqa: E731
    pts = jnp.asarray(np.random.default_rng(0).normal(size=(13, 4)))
    a = jax.vmap(f)(pts)
    for chunk in (None, 1, 4, 5, 13):
        b = memory.map_points(f, pts, chunk)
        for x, y in zip(a, b):
            np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=0, atol=1e-14)


def _model():
    """6 variables, two of them (0, 1) in a bilinear constraint; the rest linear."""

    def f(x):
        return x[0] + 2 * x[2] - x[3] + 0.5 * x[4] + x[5]

    def g(x):
        return jnp.array([x[0] * x[1] - 1.0, x[2] + x[3] + x[4] - 2.0, -x[5] + x[0] - 1.5])

    lo, hi = np.zeros(6), np.full(6, 3.0)
    return f, g, lo, hi


def test_axis_vars():
    f, g, lo, hi = _model()
    nl = LP._nonlinear_vars(f, g, None, 6)
    assert list(LP.axis_vars(nl, 6)) == [0, 1]
    assert list(LP.axis_vars(None, 6)) == list(range(6))
    pts = LP.cut_points(jnp.full(6, 1.5), jnp.full(6, 3.0), LP.axis_vars(nl, 6))
    assert pts.shape == (5, 6)


def test_bound_same_with_and_without_chunks(monkeypatch):
    """The node bound and OBBT give the same results when the budget forces one point at a time."""
    f, g, lo, hi = _model()
    boxes = [(lo, hi), (np.array([0.5, 0.5, 0, 0, 0, 0]), np.array([2.0, 2.5, 3, 3, 3, 3]))]

    def run(limit):
        monkeypatch.setattr(memory, "BUDGET", limit)
        monkeypatch.setattr(memory, "cut_point_bytes", lambda fns, n: 1000)  # every tape "interpreted"
        cb = LP.make_constrained_bounder(f, g, None, tol=1e-6, univariate=True, alphabb=True)
        ob = LP.make_obbt(f, g, None, tol=1e-6, univariate=True, alphabb=True)
        out = []
        for l, h in boxes:
            out.append([np.asarray(v) for v in jax.jit(cb)(jnp.asarray(l), jnp.asarray(h), jnp.inf)])
            out.append([np.asarray(v) for v in jax.jit(ob)(jnp.asarray(l), jnp.asarray(h), jnp.inf)])
        return out

    a, b = run(1 << 40), run(1)
    for x, y in zip(a, b):
        for u, v in zip(x, y):
            np.testing.assert_allclose(u, v, rtol=1e-9, atol=1e-9)


def test_memory_threads(monkeypatch):
    """Concurrent kernel calls are capped so that their executables' working memory fits the budget."""
    from jaxglobal import bnb

    class K(bnb._Precompiled):
        def __init__(self, tb):
            super().__init__(None, "k")
            self.tb = tb

        def temp_bytes(self, *args):
            return self.tb

    monkeypatch.setattr(memory, "BUDGET", 1000)
    assert bnb._memory_threads(K(None), 4, ()) == 4  # not compiled yet: unchanged
    assert bnb._memory_threads(K(200), 4, ()) == 4  # 800 <= 1000
    assert bnb._memory_threads(K(300), 4, ()) == 3
    assert bnb._memory_threads(K(5000), 4, ()) == 1
    assert bnb._memory_threads(object(), 4, ()) == 4  # a plain jitted function
