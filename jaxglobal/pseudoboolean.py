"""Exact minimization of binary polynomial problems of small induced width.

A problem whose objective, at every point with its integer variables binary, is a polynomial
``F(x) = c + sum_T a_T prod_{i in T} x_i`` of binary variables (``quadratic._lattice``: ``f``
itself, or an epigraph ``f = s t + c_0`` over a single inequality ``alpha t + P(x) <= 0``) and
that has no other constraint is an unconstrained pseudo-Boolean optimization problem. Variable
elimination (the "basic algorithm" of Hammer, Rosenberg & Rudeanu 1963; Crama, Hansen & Jaumard
1990, "The basic algorithm for pseudo-Boolean programming revisited") minimizes it exactly: for
each variable in turn, the terms that contain it are summed into a table over the variables they
share with it (its bucket), and the variable is minimized out of that table, which becomes a new
term. The cost is exponential only in the largest bucket (the induced width of the elimination
order), not in the number of variables: on a polynomial whose monomials span at most ``w``
consecutive variables (a sequence with interactions of bounded range) every bucket has at most
``w`` variables. ``minimize_pb`` gives up (returns None) when the greedy order's widest bucket
exceeds ``max_width``; ``bnb.minimize`` then searches as before.
"""

from __future__ import annotations

import numpy as np

EPS = np.finfo(np.float64).eps
MAX_WIDTH = 16  # largest bucket (variables of one table, 2**MAX_WIDTH entries) eliminated exactly
MAX_WORK = 5e7  # total table entries over the elimination: beyond this give up


def objective_polynomial(Q, n):
    """``(monos, coefs, const, j, r)`` when, at every point whose polynomial variables are binary,
    the minimum of ``f`` over the remaining variables is ``const + sum_k coefs[k] prod_{i in
    monos[k]} x_i`` and the problem has no other constraint, else None.

    ``Q`` is a ``quadratic.QuadraticForm`` with a ``Lattice``. ``j = -1``: ``f`` is the polynomial
    (no constraint rows at all). ``j >= 0``: ``f = s t + c_0`` with ``t = x_j`` and the single
    inequality row ``r`` (``alpha t + P(x) + c_r <= 0``, ``s alpha < 0``): the polynomial is
    ``(-s / alpha) (P + c_r) + c_0``, attained at ``t = -(P + c_r) / alpha`` (within ``t``'s box,
    which the caller checks)."""
    lat = Q.lattice
    if lat is None or Q.n_h or len(Q.den_row):
        return None
    E, c = np.asarray(Q.E), np.asarray(Q.c)
    if lat.j < 0:
        if Q.n_g:
            return None
        row, const, r = E[0].copy(), float(c[0]), -1
    else:
        if Q.n_g != 1 or E[1, lat.j] == 0.0:
            return None
        r = 1
        scale = -lat.s / lat.alpha
        row = E[r].copy()
        row[lat.j] = 0.0
        row, const = row * scale, float(c[r] * scale + c[0])
    mono = [(i,) for i in range(n)]
    for a, b in np.asarray(Q.pairs).tolist():
        mono.append(tuple(sorted(set(mono[a]) | set(mono[b]))))  # binary: x^2 = x
    nz = np.flatnonzero(row)
    return [mono[k] for k in nz.tolist()], row[nz], const, int(lat.j), r


def _order(scopes, free):
    """Greedy elimination order (fewest variables in the bucket first) and its widest bucket."""
    adj = {v: set() for v in free}
    for s in scopes:
        for v in s:
            adj[v].update(s)
    for v in adj:
        adj[v].discard(v)
    order, width, work, left = [], 0, 0.0, set(free)
    while left:
        v = min(left, key=lambda u: (len(adj[u]), u))
        nb = adj[v]
        width = max(width, len(nb) + 1)
        work += 2.0 ** (len(nb) + 1)
        for u in nb:
            adj[u].update(nb)
            adj[u].discard(u)
            adj[u].discard(v)
        order.append(v)
        left.discard(v)
        del adj[v]
    return order, width, work


def minimize_pb(monos, coefs, const, lo, hi, max_width=MAX_WIDTH, max_work=MAX_WORK):
    """Exact minimum of ``const + sum_k coefs[k] prod_{i in monos[k]} x_i`` over the binary points
    of the box ``[lo, hi]`` (every variable of the monomials with bounds in {0, 1}), by variable
    elimination: ``(value, x)`` with ``x`` a minimizer on those variables (others: ``lo``), or None
    when the elimination order has a bucket of more than ``max_width`` variables or more than
    ``max_work`` table entries in all. ``value`` is the float sum of the minimizer's terms (each
    an exact product of a coefficient and 0 or 1)."""
    lo, hi = np.asarray(lo, np.float64), np.asarray(hi, np.float64)
    x = lo.copy()
    # substitute the fixed variables; merge monomials with the same remaining variables
    terms = {}
    for m, a in zip(monos, np.asarray(coefs, np.float64).tolist()):
        if any(hi[i] == 0.0 for i in m):
            continue
        k = tuple(i for i in m if lo[i] != 1.0)
        terms[k] = terms.get(k, 0.0) + a
    base = const + terms.pop((), 0.0)
    free = sorted({i for m in terms for i in m})
    order, width, work = _order(list(terms), free)
    if width > max_width or work > max_work:
        return None
    factors = []  # (scope: sorted tuple, table: ndarray of shape (2,) * len(scope))
    for m, a in terms.items():
        t = np.zeros((2,) * len(m))
        t[(1,) * len(m)] = a
        factors.append((m, t))
    trace = []  # (v, rest scope, argmin table over rest)
    for v in order:
        bucket = [f for f in factors if v in f[0]]
        factors = [f for f in factors if v not in f[0]]
        U = tuple(sorted({u for s, _ in bucket for u in s}))
        tab = np.zeros((2,) * len(U))
        for s, t in bucket:
            tab = tab + t.reshape([2 if u in s else 1 for u in U])
        ax = U.index(v)
        rest = U[:ax] + U[ax + 1:]
        trace.append((v, rest, np.argmin(tab, axis=ax)))
        factors.append((rest, np.min(tab, axis=ax)))
    for v, rest, arg in reversed(trace):
        x[v] = float(arg[tuple(int(x[u]) for u in rest)])
    value = base
    for m, a in terms.items():
        if all(x[i] == 1.0 for i in m):
            value += a
    return value, x


def solve_binary_polynomial(Q, n, lo, hi, integer, max_width=MAX_WIDTH):
    """The exact minimum of the problem of ``Q`` (``objective_polynomial``) over the box, or None.

    Returns ``(x, F, tol_scale, mag)``: a minimizer ``x`` (with ``t = x_j`` at its value for an
    epigraph), the polynomial's minimum ``F``, ``|s / alpha|`` (how far below ``F`` the objective
    may go when the epigraph row only holds to a tolerance; 0 without one), and the sum of the
    magnitudes of the polynomial's terms (a bound on the float error of ``F`` is
    ``(len(monos) + 2) * EPS * mag``). None when it does not apply: some variable other than
    ``t`` is not an integer with bounds in {0, 1}, the polynomial is too wide, or the minimizer's
    ``t`` is outside its box."""
    form = objective_polynomial(Q, n)
    if form is None:
        return None
    monos, coefs, const, j, r = form
    lo, hi = np.asarray(lo, np.float64), np.asarray(hi, np.float64)
    integer = np.asarray(integer, bool)
    others = np.ones(n, bool)
    if j >= 0:
        others[j] = False
    if not (np.all(integer[others]) and np.all(np.isin(lo[others], (0.0, 1.0)))
            and np.all(np.isin(hi[others], (0.0, 1.0))) and np.all(lo <= hi)):  # fmt: skip
        return None
    out = minimize_pb(monos, coefs, const, np.where(others, lo, 0.0), np.where(others, hi, 0.0), max_width)
    if out is None:
        return None
    F, x = out
    mag = abs(const) + float(np.sum(np.abs(coefs)))
    if j < 0:
        return x, F, 0.0, mag
    lat = Q.lattice
    E, c = np.asarray(Q.E), np.asarray(Q.c)
    mono = [(i,) for i in range(n)]
    for a, b in np.asarray(Q.pairs).tolist():
        mono.append(tuple(sorted(set(mono[a]) | set(mono[b]))))
    v = np.array([float(all(x[i] == 1.0 for i in m)) for m in mono])
    v[j] = 0.0
    t = -(E[r] @ v + c[r]) / lat.alpha
    if not lo[j] <= t <= hi[j]:
        return None
    x[j] = t
    return x, F, abs(lat.s / lat.alpha), mag
