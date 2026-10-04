"""Scalable synthetic problems (size n, seed) for scaling studies.

Each generator is deterministic in its arguments and returns a ``Problem`` or
``ConstrainedProblem`` from ``problems.py``. For the constrained families
``xstar`` is a known *feasible* point (not a minimizer) and ``fstar`` is None.
"""

import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize_scalar

from .problems import ConstrainedProblem, Problem


def univariate_sum(n, seed=0):
    """sum_i g_i(x_i), each g_i either a sin + quadratic or a double-well quartic; fstar is exact."""
    r = np.random.default_rng(seed)
    kind = r.random(n) < 0.5
    A, B = np.where(kind, r.uniform(0.5, 2, n), 0), r.uniform(1, 4, n)
    C = np.where(kind, r.uniform(0.05, 0.3, n), 0)
    K, D = np.where(~kind, r.uniform(1, 4, n), 0), np.where(~kind, r.uniform(-1, 1, n), 0)
    P = (~kind).astype(float)
    lo, hi = np.full(n, -3.0), np.full(n, 3.0)

    def f(x):
        return jnp.sum(A * jnp.sin(B * x) + C * x**2 + P * (x**4 / 4 - K * x**2 + D * x))

    def g(i, t):
        return A[i] * np.sin(B[i] * t) + C[i] * t**2 + P[i] * (t**4 / 4 - K[i] * t**2 + D[i] * t)

    grid, fstar = np.linspace(-3, 3, 20001), 0.0
    for i in range(n):
        t0 = grid[np.argmin(g(i, grid))]
        res = minimize_scalar(lambda t: g(i, t), bounds=(max(-3, t0 - 1e-3), min(3, t0 + 1e-3)), method="bounded",
                              options=dict(xatol=1e-12))
        fstar += min(res.fun, g(i, t0))
    return Problem(f"univariate_sum_{n}", f, lo, hi, float(fstar), f"separable sum of {n} 1-D nonconvex terms")


def _sparse_sym(r, n, per_var, scale):
    """Symmetric Q with ~per_var bilinear partners per variable and a few squares (indefinite)."""
    Q = np.zeros((n, n))
    for i in range(n):
        for j in r.choice(n, size=min(per_var, n), replace=False):
            Q[i, j] += scale * r.normal() / 2
            Q[j, i] += scale * r.normal() / 2
    Q[np.diag_indices(n)] = 0.3 * scale * r.normal(size=n)
    return (Q + Q.T) / 2


def qcqp(n, m, seed=0, per_var=2):
    """Nonconvex x'Qx + c'x over [-1,1]^n with m quadratic inequalities, all strictly satisfied at x0."""
    r = np.random.default_rng(seed)
    Q, c = _sparse_sym(r, n, per_var, 1.0), r.normal(size=n)
    Qs = np.stack([_sparse_sym(r, n, per_var, 1.0) for _ in range(m)]) if m else np.zeros((0, n, n))
    cs = r.normal(size=(m, n))
    x0 = r.uniform(-0.5, 0.5, n)
    b = np.einsum("kij,i,j->k", Qs, x0, x0) + cs @ x0 + r.uniform(0.2, 1.0, m)  # g(x0) = -margin < 0

    def f(x):
        return jnp.sum(x * (Q @ x)) + c @ x  # x @ Q @ x (var @ var) is not supported by the relaxation

    def ineq(x):
        return jnp.sum(x * (Qs @ x), axis=1) + cs @ x - b

    return ConstrainedProblem(f"qcqp_{n}_{m}", f, ineq if m else None, None, np.full(n, -1.0), np.full(n, 1.0),
                              None, x0, f"sparse indefinite QCQP, {n} vars, {m} quadratic constraints")


def pooling(n_pools, n_inputs, n_products, seed=0):
    """p-formulation pooling: inputs -> pools -> products, one quality, bilinear quality balances.

    Variables are y[i,p] (input to pool), q[p] (pool quality), z[p,j] (pool to product).
    Equalities: flow balance and quality balance per pool; inequalities: supply, demand, product quality spec.
    Zero flow is feasible.
    """
    r = np.random.default_rng(seed)
    P, I, J = n_pools, n_inputs, n_products
    u = r.uniform(1.0, 4.0, I)  # input quality (e.g. sulfur)
    cost = 12 - 2 * u + r.uniform(-0.5, 0.5, I)  # cleaner inputs cost more
    price = r.uniform(10, 16, J)
    spec = r.uniform(1.8, 2.6, J)
    supply, demand = r.uniform(30, 60, I), r.uniform(30, 60, J)
    cap = r.uniform(40, 80, P)
    ny, nz = I * P, P * J

    def split(w):
        return w[:ny].reshape(I, P), w[ny : ny + P], w[ny + P :].reshape(P, J)

    def f(w):
        y, q, z = split(w)
        return jnp.sum(cost @ y) - jnp.sum(z @ price)

    def ineq(w):
        y, q, z = split(w)
        return jnp.concatenate([y.sum(1) - supply, z.sum(0) - demand, y.sum(0) - cap, (q[:, None] * z).sum(0) - spec * z.sum(0)])

    def eq(w):
        y, q, z = split(w)
        return jnp.concatenate([y.sum(0) - z.sum(1), u @ y - q * z.sum(1)])

    lo = np.concatenate([np.zeros(ny), np.full(P, u.min()), np.zeros(nz)])
    hi = np.concatenate([np.minimum(supply[:, None], cap[None, :]).ravel(), np.full(P, u.max()),
                         np.minimum(cap[:, None], demand[None, :]).ravel()])
    x0 = np.concatenate([np.zeros(ny), np.full(P, 0.5 * (u.min() + u.max())), np.zeros(nz)])
    return ConstrainedProblem(f"pooling_{P}_{I}_{J}", f, ineq, eq, lo, hi, None, x0,
                              f"pooling, {P} pools, {I} inputs, {J} products")


def nn_surrogate(n_in, hidden=8, layers=2, seed=0, act="tanh", half_width=2.0):
    """Minimize the output of a random MLP (n_in -> hidden x layers -> 1) over [-half_width, half_width]^n_in."""
    r = np.random.default_rng(seed)
    dims = [n_in] + [hidden] * layers + [1]
    Ws = [r.normal(size=(a, b)) * 1.5 / np.sqrt(a) for a, b in zip(dims[:-1], dims[1:])]
    bs = [r.normal(size=b) * 0.5 for b in dims[1:]]
    nl = jnp.tanh if act == "tanh" else (lambda t: jnp.maximum(t, 0.0))

    def f(x):
        for W, b in zip(Ws[:-1], bs[:-1]):
            x = nl(x @ W + b)
        return (x @ Ws[-1] + bs[-1])[0]

    return Problem(f"nn_{act}_{n_in}x{hidden}x{layers}", f, np.full(n_in, -half_width), np.full(n_in, half_width),
                   None, f"random {act} MLP, {n_in} inputs")


SIZES = {
    "univariate_sum": [dict(n=n) for n in (5, 10, 20, 50, 100)],
    "qcqp": [dict(n=n, m=m) for n, m in ((5, 2), (10, 5), (20, 10), (50, 25), (100, 50))],
    "pooling": [dict(n_pools=p, n_inputs=i, n_products=j) for p, i, j in ((2, 4, 2), (3, 4, 3), (4, 6, 4), (5, 8, 5), (6, 9, 6))],
    "nn_surrogate": [dict(n_in=n, hidden=8, layers=2) for n in (2, 5, 10, 20)],
}
_GEN = dict(univariate_sum=univariate_sum, qcqp=qcqp, pooling=pooling, nn_surrogate=nn_surrogate)


def suite(families=None, seed=0):
    """List of (family, instance) over ``SIZES`` for the given families (default all)."""
    return [(fam, _GEN[fam](**kw, seed=seed)) for fam in (families or SIZES) for kw in SIZES[fam]]
