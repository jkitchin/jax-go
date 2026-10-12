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


def test_time_limit_covers_deriving_bounds():
    """Unbounded variables: bound_domain's FBBT kernel compiles before any bounding kernel."""
    f = lambda x: (x[0] - 0.5) ** 2 + jnp.cos(x[1]) + 0.654321  # noqa: E731
    g = lambda x: jnp.array([x[0] + x[1] - 1.0, -x[0] - 2.0, x[1] ** 2 - 4.0])  # noqa: E731
    t = time.perf_counter()
    r = minimize(f, -np.inf * np.ones(2), np.inf * np.ones(2), ineq=g, time_limit=1e-6)
    assert time.perf_counter() - t < 30
    assert r.status == "time_limit" and r.lower_bound == -np.inf
    assert "bounds" in r.message
