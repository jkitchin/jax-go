"""minimize's time_limit also covers tracing and compiling the bounding kernels."""

import time

import jax.numpy as jnp
import numpy as np

from jaxglobal import minimize


def test_time_limit_covers_compilation():
    # a fresh function: its kernels are not in the in-memory cache, so the root bound has to wait for them
    f = lambda x: jnp.sum(jnp.sin(3 * x) * jnp.exp(-x**2)) + 0.123456  # noqa: E731
    t = time.perf_counter()
    r = minimize(f, -np.ones(3), np.ones(3), time_limit=1e-6)
    assert time.perf_counter() - t < 30  # returns promptly instead of waiting for the compile
    assert r.status == "time_limit"
    assert r.lower_bound == -np.inf and r.nodes == 0
    assert "compiling" in r.message


def test_time_limit_does_not_affect_normal_solves():
    r = minimize(lambda x: (x[0] - 0.3) ** 2 + x[1] ** 2, -np.ones(2), np.ones(2), time_limit=120)
    assert r.status == "optimal" and abs(r.fun) < 1e-6
