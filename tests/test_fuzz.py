"""Fuzz: random expression trees must never produce a wrong bound or an infeasible incumbent.

Each case builds a random smooth/nonsmooth expression on 1-4 variables (domain
guarded), optionally 1-2 random inequality constraints, solves it under a node
cap, and checks against a dense random sample of the box:

  (a) lower_bound <= min f over the feasible samples (a valid bound cannot
      exceed any feasible value);
  (b) the returned x is feasible, f(x) == fun, and the result says so
      (``verified``);
  (c) status "optimal" means fun is no worse than the best sample by more than
      the tolerance; status "infeasible" means no sample is feasible.

Seeded and reproducible: case ``i`` uses ``np.random.default_rng(i)`` and the
option set ``VARIANTS[i % len(VARIANTS)]``. ``JAXGLOBAL_FUZZ=N`` runs N cases
(default 8); ``JAXGLOBAL_FUZZ_START=k`` starts at case k.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxglobal import minimize

N_CASES = int(os.environ.get("JAXGLOBAL_FUZZ", "8"))
START = int(os.environ.get("JAXGLOBAL_FUZZ_START", "0"))
FEAS_TOL = 1e-6
N_SAMPLE = 20000

VARIANTS = [
    dict(safe=True),
    dict(safe=False),
    dict(safe=True, crown=True),
    dict(safe=True, univariate=False),
    dict(safe=True, alphabb="eigen"),
    dict(safe=False, crown=True, univariate=False),
    dict(safe=True, lifted=True),
    dict(safe=True, multivariate=True),
]

LEAF = ("var", "const")
UN = ("exp", "log", "sqrt", "sin", "cos", "tanh", "pow2", "pow3", "neg")
BIN = ("add", "sub", "mul", "div", "max", "min", "where")


def gen(rng, n, depth):
    """Random tree as nested tuples; every leaf is a variable index or a constant."""
    if depth == 0 or rng.random() < 0.2:
        if rng.random() < 0.8:
            return ("var", int(rng.integers(n)))
        return ("const", float(np.round(rng.uniform(-2, 2), 2)))
    if rng.random() < 0.45:
        return (str(rng.choice(UN)), gen(rng, n, depth - 1))
    return (str(rng.choice(BIN)), gen(rng, n, depth - 1), gen(rng, n, depth - 1), gen(rng, n, depth - 1))


def ev(t, x):
    """Evaluate a tree with jnp. log / sqrt / div arguments are shifted to stay in their domain."""
    k = t[0]
    if k == "var":
        return x[t[1]]
    if k == "const":
        return jnp.asarray(t[1], x.dtype)
    a = ev(t[1], x)
    if k == "exp":
        return jnp.exp(a if t[1][0] in LEAF else 2.0 * jnp.tanh(a))  # no overflow from deep arguments
    if k == "log":
        return jnp.log(0.5 + a * a)
    if k == "sqrt":
        return jnp.sqrt(0.1 + a * a)
    if k == "sin":
        return jnp.sin(a)
    if k == "cos":
        return jnp.cos(a)
    if k == "tanh":
        return jnp.tanh(a)
    if k == "pow2":
        return a**2
    if k == "pow3":
        return a**3
    if k == "neg":
        return -a
    b = ev(t[2], x)
    if k == "add":
        return a + b
    if k == "sub":
        return a - b
    if k == "mul":
        return a * b
    if k == "div":
        return a / (1.5 + jnp.cos(b))  # denominator in [0.5, 2.5]
    if k == "max":
        return jnp.maximum(a, b)
    if k == "min":
        return jnp.minimum(a, b)
    c = ev(t[3], x)
    return jnp.where(a > 0.3, b, c)


def build(seed):
    """(f, ineq or None, lo, hi, samples, f values, feasible mask, options) for case ``seed``."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 5))
    lo = rng.uniform(-2.0, 0.5, n)
    hi = lo + rng.uniform(0.5, 3.0, n)
    tf = gen(rng, n, 3)
    f = lambda x: ev(tf, x)  # noqa: E731
    X = rng.uniform(lo, hi, (N_SAMPLE, n))
    X[: 2**n] = np.array(np.meshgrid(*[[a, b] for a, b in zip(lo, hi)])).reshape(n, -1).T  # box vertices
    F = np.asarray(jax.jit(jax.vmap(f))(jnp.asarray(X)))
    ineq, feas = None, np.ones(N_SAMPLE, bool)
    m = int(rng.integers(0, 3))
    if m:
        trees = [gen(rng, n, 2) for _ in range(m)]
        G = np.stack([np.asarray(jax.jit(jax.vmap(lambda x, t=t: ev(t, x)))(jnp.asarray(X))) for t in trees], axis=1)
        thr = np.array([np.nanquantile(G[:, j], 0.4) for j in range(m)])  # ~40% of the box satisfies each
        ineq = lambda x: jnp.stack([ev(t, x) - c for t, c in zip(trees, thr)])  # noqa: E731
        feas = np.all(G <= thr, axis=1)
    return f, ineq, lo, hi, X, F, feas, VARIANTS[seed % len(VARIANTS)]


@pytest.mark.parametrize("seed", range(START, START + N_CASES))
def test_fuzz(seed):
    f, ineq, lo, hi, X, F, feas, opts = build(seed)
    ok = feas & np.isfinite(F)
    best = float(F[ok].min()) if ok.any() else np.inf
    r = minimize(f, lo, hi, ineq=ineq, feas_tol=FEAS_TOL, batch=32, batch_sizes=(8, 16), max_iter=40,
                 time_limit=40.0, **opts)  # fmt: skip
    where = f"seed {seed} opts {opts}: {r}"
    tol = 1e-6 * (1 + abs(best)) if np.isfinite(best) else 0.0
    assert not (r.lower_bound > best + tol), f"lower bound above a feasible sample value ({best}); {where}"
    if np.isfinite(r.fun):
        x = np.asarray(r.x)
        assert np.all(x >= lo - 1e-9) and np.all(x <= hi + 1e-9), where
        if ineq is not None:
            assert np.max(np.asarray(ineq(jnp.asarray(x)))) <= FEAS_TOL + 1e-12, f"infeasible x; {where}"
        assert abs(float(f(jnp.asarray(x))) - r.fun) <= 1e-8 * (1 + abs(r.fun)), f"f(x) != fun; {where}"
        assert r.verified and r.max_violation <= FEAS_TOL + 1e-12, where
    if r.status == "optimal":
        assert np.isfinite(r.fun), where
        assert r.fun - best <= max(1e-5, 1e-5 * abs(best)) + r.gap, f"optimal but a sample is better; {where}"
    if r.status == "infeasible":
        assert not ok.any(), f"infeasible but a sample is feasible; {where}"


def test_safe_sqrt_of_sum_of_squares_touching_zero():
    """Regression: outward widening put lo of ``x**2 + y**2`` below 0, so sqrt gave -inf / NaN (holder_table)."""
    from jaxglobal.relax import relax

    f = lambda x: jnp.sqrt(x[0] ** 2 + x[1] ** 2) / 3.0 + jnp.log(0.5 + x[0] * x[1])  # noqa: E731
    lo, hi = jnp.zeros(2), jnp.full(2, 1e-3)
    m = relax(f, safe=True)(0.5 * (lo + hi), lo, hi)
    assert np.isfinite(float(m.lo)) and np.isfinite(float(m.cv)), m
    assert float(m.lo) <= float(f(lo)) and float(m.hi) >= float(f(hi))
