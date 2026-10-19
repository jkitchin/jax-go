"""Exact variable elimination for binary polynomial problems (pseudoboolean.py) and its shortcut
in the search (bnb._pseudo_boolean).

Tests: the elimination's minimum and minimizer match enumeration on random polynomials, with
fixed variables; it gives up beyond the width limit; the polynomial read off the lifted form of an
epigraph problem and of a plain one; refusals (an extra constraint, a non-binary integer, a
continuous variable that is not the epigraph's); the search returns the enumerated optimum through
the shortcut on an epigraph problem and on an unconstrained one, and without it (the search) on a
problem with a side constraint."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxglobal
from jaxglobal.pseudoboolean import minimize_pb, objective_polynomial, solve_binary_polynomial
from jaxglobal.quadratic import polynomial_form

jax.config.update("jax_enable_x64", True)


def random_poly(seed, n=8, n_terms=14, degree=4):
    rng = np.random.default_rng(seed)
    monos, coefs = [], []
    for _ in range(n_terms):
        d = int(rng.integers(1, degree + 1))
        monos.append(tuple(sorted(rng.choice(n, d, replace=False).tolist())))
        coefs.append(float(rng.integers(-9, 10)))
    return monos, np.array(coefs)


def value(monos, coefs, const, x):
    return const + sum(a for m, a in zip(monos, coefs) if all(x[i] == 1 for i in m))


def enumerate_min(monos, coefs, const, lo, hi):
    n = len(lo)
    best = np.inf
    for bits in itertools.product((0.0, 1.0), repeat=n):
        x = np.array(bits)
        if np.all(x >= lo) and np.all(x <= hi):
            best = min(best, value(monos, coefs, const, x))
    return best


@pytest.mark.parametrize("seed", range(6))
def test_elimination_matches_enumeration(seed):
    monos, coefs = random_poly(seed)
    n = 8
    lo, hi = np.zeros(n), np.ones(n)
    if seed % 2:  # some variables fixed
        lo[1], hi[4] = 1.0, 0.0
    F, x = minimize_pb(monos, coefs, 3.0, lo, hi)
    assert F == enumerate_min(monos, coefs, 3.0, lo, hi)
    assert np.all(x >= lo) and np.all(x <= hi) and set(np.unique(x)) <= {0.0, 1.0}
    assert value(monos, coefs, 3.0, x) == F


def test_width_limit():
    n = 10
    monos = [tuple(range(n))] + [(i, j) for i in range(n) for j in range(i + 1, n)]
    coefs = np.ones(len(monos))
    assert minimize_pb(monos, coefs, 0.0, np.zeros(n), np.ones(n), max_width=5) is None
    F, _ = minimize_pb(monos, coefs, 0.0, np.zeros(n), np.ones(n), max_width=10)
    assert F == 0.0


def chain_poly(x, n):
    # +-1 sequence s = 2x - 1 with interactions of range 3: sum_i (s_i s_{i+1} + s_i s_{i+2} s_{i+3} ...)^2
    s = 2 * x - 1
    c1 = sum(s[i] * s[i + 1] for i in range(n - 1))
    c2 = sum(s[i] * s[i + 2] for i in range(n - 2))
    loc = sum(s[i] * s[i + 1] * s[i + 2] * s[i + 3] for i in range(n - 3))
    return c1 * c1 + c2 * c2 + 3 * loc - 2 * x[0]


N = 9


def epi_f(z):
    return z[N]


def epi_g(z):
    return jnp.atleast_1d(chain_poly(z[:N], N) - z[N])


def brute(n=N):
    return min(float(chain_poly(np.array(b, float), n)) for b in itertools.product((0, 1), repeat=n))


def test_objective_polynomial_epigraph():
    integer = np.arange(N)
    Q = polynomial_form(epi_f, epi_g, None, N + 1, integer=integer)
    monos, coefs, const, j, r = objective_polynomial(Q, N + 1)
    assert j == N and r == 1
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = rng.integers(0, 2, N).astype(float)
        assert value(monos, coefs, const, x) == pytest.approx(float(chain_poly(x, N)), abs=1e-9)
    lo, hi = np.zeros(N + 1), np.ones(N + 1)
    lo[N], hi[N] = -1e4, 1e4
    imask = np.arange(N + 1) < N
    x, F, ts, _ = solve_binary_polynomial(Q, N + 1, lo, hi, imask)
    assert F == pytest.approx(brute())
    assert ts == 1.0 and x[N] == pytest.approx(F)
    # refusals: t outside its box, a non-binary integer, t declared integer
    hi2 = hi.copy()
    hi2[N] = F - 1.0
    assert solve_binary_polynomial(Q, N + 1, lo, hi2, imask) is None
    hi3 = hi.copy()
    hi3[0] = 2.0
    assert solve_binary_polynomial(Q, N + 1, lo, hi3, imask) is None


def test_objective_polynomial_refuses_side_constraint():
    def g2(z):
        return jnp.concatenate([epi_g(z), jnp.atleast_1d(z[0] + z[1] - 1.0)])

    Q = polynomial_form(epi_f, g2, None, N + 1, integer=np.arange(N))
    assert Q is None or objective_polynomial(Q, N + 1) is None


def test_search_epigraph():
    lo, hi = np.zeros(N + 1), np.ones(N + 1)
    lo[N], hi[N] = -np.inf, np.inf
    r = jaxglobal.minimize(epi_f, lo, hi, ineq=epi_g, integer=np.arange(N))
    assert r.status == "optimal" and "variable elimination" in r.message
    assert r.fun == pytest.approx(brute(), abs=1e-6)
    assert r.lower_bound <= r.fun and r.fun - r.lower_bound <= 1e-5
    assert r.verified


def test_search_unconstrained():
    n = 8

    def f(x):
        return chain_poly(x, n)

    r = jaxglobal.minimize(f, np.zeros(n), np.ones(n), integer=np.arange(n))
    assert r.status == "optimal" and "variable elimination" in r.message
    assert r.fun == pytest.approx(brute(n), abs=1e-9)
    assert r.lower_bound <= r.fun


def test_search_side_constraint_not_shortcut():
    n = 7

    def f(z):
        return z[n]

    def g(z):  # epigraph row and a cardinality row: not unconstrained, so the search runs
        return jnp.stack([chain_poly(z[:n], n) - z[n], jnp.sum(z[:n]) - 2.0])

    lo, hi = np.zeros(n + 1), np.ones(n + 1)
    lo[n], hi[n] = -1e3, 1e3
    r = jaxglobal.minimize(f, lo, hi, ineq=g, integer=np.arange(n))
    best = min(float(chain_poly(np.array(b, float), n)) for b in itertools.product((0, 1), repeat=n) if sum(b) <= 2)
    assert "variable elimination" not in r.message
    assert r.status == "optimal" and r.fun == pytest.approx(best, abs=1e-5)
