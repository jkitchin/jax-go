import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxglobal import minimize, relax
from jaxglobal.lp import make_constrained_bounder
from jaxglobal.problems import CONSTRAINED_PROBLEMS

IDS = [p.name for p in CONSTRAINED_PROBLEMS]


def boxes_around(p, x, k=6, seed=0):
    """Random sub-boxes of the domain that contain x."""
    rng = np.random.default_rng(seed)
    w = p.hi - p.lo
    for _ in range(k):
        a = np.maximum(p.lo, x - rng.uniform(0, 0.5, x.size) * w)
        b = np.minimum(p.hi, x + rng.uniform(0, 0.5, x.size) * w)
        yield jnp.asarray(a), jnp.asarray(b)


@pytest.mark.parametrize("p", [p for p in CONSTRAINED_PROBLEMS if p.eq is not None], ids=lambda p: p.name)
def test_concave_cuts_valid(p):
    """Affine overestimators of cc (used for -cc_h <= 0) lie above h on the box."""
    rh = relax(lambda x: jnp.atleast_1d(p.eq(x)))
    rng = np.random.default_rng(1)
    for _ in range(5):
        a, b = np.sort(rng.uniform(p.lo, p.hi, (2, p.lo.size)), axis=0)
        a, b = jnp.asarray(a), jnp.asarray(b)
        pt = jnp.asarray(rng.uniform(a, b))
        cc = rh(pt, a, b).cc
        J = jax.jacfwd(lambda x: rh(x, a, b).cc)(pt)
        xs = jnp.asarray(rng.uniform(a, b, (200, p.lo.size)))
        hv = jax.vmap(lambda x: jnp.atleast_1d(p.eq(x)))(xs)
        cut = cc + (xs - pt) @ J.T
        assert np.all(np.asarray(cut) >= np.asarray(hv) - 1e-8 * (1 + np.abs(np.asarray(hv))))


@pytest.mark.parametrize("p", [p for p in CONSTRAINED_PROBLEMS if p.xstar is not None], ids=lambda p: p.name)
def test_node_bound_below_optimum(p):
    """Any box containing a global minimizer must bound below f*."""
    bound = jax.jit(make_constrained_bounder(p.f, p.ineq, p.eq))
    for a, b in boxes_around(p, p.xstar):
        lb = float(bound(a, b)[0])
        assert lb <= p.fstar + 1e-5 * (1 + abs(p.fstar)), (lb, p.fstar)


@pytest.mark.parametrize("p", CONSTRAINED_PROBLEMS, ids=IDS)
def test_constrained_global_minimum(p):
    r = minimize(p.f, p.lo, p.hi, ineq=p.ineq, eq=p.eq, max_iter=3000)
    assert r.status == "optimal"
    tol = 1e-4 * (1 + abs(p.fstar))
    assert abs(r.fun - p.fstar) <= tol
    assert r.lower_bound <= p.fstar + tol


def test_infeasible_problem():
    r = minimize(lambda x: x[0] + x[1], [-1, -1], [1, 1], ineq=lambda x: 1.5 - x[0] ** 2 - x[1] ** 2 * 0.1)
    assert r.status == "infeasible"
    assert r.fun == np.inf


# kall_circles_c6a (MINLPLib): circle packing with two equalities. The IPM duals
# diverge to ~1e170 on its degenerate LPs, and the certificate used to turn the
# rounding noise into lower bounds of 1e47..1e276 (the solver then "proved" a
# wrong optimum). x* is SCIP's solution, feasible to ~4e-8.
C6A_X = [0.49999999114651, 0.499999990679916, 0.499999991021435, 1.49999998666014, 2.68174240423438,
         0.499999990656766, 2.68174240477416, 1.49999998663835, 3.68174240029873, 0.499999990760315,
         3.68174240067707, 1.49999998652489, 1.59087119780145, 0.999999989105133, 4.18174239168908,
         1.99999997728567, 8.36348468839271, 2.11171530774902]  # fmt: skip


def test_degenerate_lp_bound_valid():
    import pathlib

    from jaxglobal.nl import read_nl

    M = read_nl(pathlib.Path(__file__).parent / "data" / "kall_circles_c6a.nl")
    x = np.clip(np.array(C6A_X), M.lo, M.hi)
    fstar = float(M.f(jnp.asarray(x)))
    cb = jax.jit(make_constrained_bounder(M.f, M.ineq, M.eq, tol=1e-6))
    lbs = []
    for w in [None, 1e-2, 1e-3, 1e-5]:
        lo, hi = (M.lo, M.hi) if w is None else (np.maximum(M.lo, x - w), np.minimum(M.hi, x + w))
        lbs.append(float(cb(jnp.asarray(lo), jnp.asarray(hi))[0]))
    assert all(lb <= fstar + 1e-6 for lb in lbs), lbs
    assert lbs[-1] > fstar - 1e-3  # and still tight on a tiny box


def mixed_model():
    """Nonlinear objective; linear and nonlinear inequalities and equalities."""

    def f(x):
        return x[0] * x[1] + jnp.exp(0.5 * x[2]) - x[0] ** 2

    def g(x):
        return jnp.stack([x[0] + 2.0 * x[1] - 1.0, x[0] * x[1] - 0.5, 3.0 * (x[2] - x[0]) / 2.0 - 0.25])

    def h(x):
        return jnp.stack([x[0] - x[1] + x[2] - 0.1, x[0] ** 2 - x[2]])

    return f, g, h


def test_affine_parts():
    """Parts [cv_f, cv_g..., cv_h..., -cc_h...]: linear rows are found from the scalarized tape;
    vectorized code is not scalarized (None: every part counts as nonlinear)."""
    from jaxglobal.lp import _affine_parts

    f, g, h = mixed_model()
    assert _affine_parts(f, g, h, 3).tolist() == [False, True, False, True, True, False, True, False]
    W = jnp.asarray(np.random.default_rng(0).normal(size=(50, 3)))
    assert _affine_parts(lambda x: jnp.sum(jnp.tanh(W @ x) ** 2), lambda x: W @ x, None, 3) is None


@pytest.mark.parametrize("dedup", [True, False])
def test_affine_rows_once(dedup, monkeypatch):
    """With one row per linear part (instead of one per cut point) the node bound and OBBT stay
    valid on sampled feasible points, and the bound matches the one from all rows."""
    import jaxglobal.lp as LP

    f, g, h = mixed_model()
    lo0, hi0 = -np.ones(3), 2.0 * np.ones(3)
    # the equalities are the curve x2 = x0^2, x1 = x0 + x0^2 - 0.1
    t = np.linspace(-1, 2, 200001)
    X = np.stack([t, t + t**2 - 0.1, t**2], axis=1)
    X = X[np.all((X >= lo0) & (X <= hi0), axis=1) & np.all(np.asarray(jax.vmap(g)(jnp.asarray(X))) <= 0, axis=1)]
    F = np.asarray(jax.vmap(f)(jnp.asarray(X)))
    tol = 1e-6

    def bounds(on):
        """Compiled bounder and OBBT; the affine parts are read while tracing, so compile here."""
        if not on:
            monkeypatch.setattr(LP, "_nonlinear_vars", lambda *a: None)
        box = (jnp.asarray(lo0), jnp.asarray(hi0))
        cb = jax.jit(LP.make_constrained_bounder(f, g, h, tol=tol, univariate=True, alphabb=True)).lower(*box).compile()
        ob = jax.jit(LP.make_obbt(f, g, h, tol=tol, univariate=True, alphabb=True)).lower(*box).compile()
        monkeypatch.undo()
        return cb, ob

    cb, ob = bounds(dedup)
    cb_all, _ = bounds(False)
    rng = np.random.default_rng(3)
    centers = X[rng.choice(len(X), 8)]
    boxes = [(lo0, hi0)] + [(np.maximum(lo0, c - rng.uniform(0, 1, 3)), np.minimum(hi0, c + rng.uniform(0, 1, 3)))
                            for c in centers]  # fmt: skip
    for a, b in boxes:
        inside = np.all((X >= a) & (X <= b), axis=1)
        assert inside.any()
        lb = float(cb(jnp.asarray(a), jnp.asarray(b))[0])
        olo, ohi, oempty = ob(jnp.asarray(a), jnp.asarray(b))
        assert lb <= F[inside].min() + 1e-9
        assert not bool(oempty)
        assert np.all(X[inside] >= np.asarray(olo) - 1e-9) and np.all(X[inside] <= np.asarray(ohi) + 1e-9)
        ref = float(cb_all(jnp.asarray(a), jnp.asarray(b))[0])
        assert abs(lb - ref) <= 1e-4 * (1 + abs(ref)), (a, b, lb, ref)


def haverly1():
    """Haverly's pooling problem 1 (p-formulation, minimizing): f* = -400 at ``HAVERLY_X``.
    x = (A, B, C, Px, Py, Cx, Cy, p): source flows, pool-to-product flows, direct flows of C,
    pool quality."""

    def f(x):
        A, B, C, Px, Py, Cx, Cy, p = (x[i] for i in range(8))
        return -(9.0 * (Px + Cx) + 15.0 * (Py + Cy) - 6.0 * A - 16.0 * B - 10.0 * C)

    def g(x):
        A, B, C, Px, Py, Cx, Cy, p = (x[i] for i in range(8))
        return jnp.stack([p * Px + 2.0 * Cx - 2.5 * (Px + Cx), p * Py + 2.0 * Cy - 1.5 * (Py + Cy),
                          Px + Cx - 100.0, Py + Cy - 200.0])  # fmt: skip

    def h(x):
        A, B, C, Px, Py, Cx, Cy, p = (x[i] for i in range(8))
        return jnp.stack([A + B - Px - Py, C - Cx - Cy, p * (Px + Py) - 3.0 * A - B])

    lo, hi = np.r_[np.zeros(7), 1.0], np.r_[np.full(7, 300.0), 3.0]
    return f, g, h, lo, hi


HAVERLY_X = np.array([0.0, 100.0, 100.0, 0.0, 100.0, 0.0, 100.0, 1.0])


def test_nonlinear_vars_pooling():
    """Each part keeps its cuts at the axis points of the variables it is nonlinear in only."""
    from jaxglobal.lp import _cut_rows, _nonlinear_vars

    f, g, h, lo, hi = haverly1()
    nl = _nonlinear_vars(f, g, h, 8)
    # parts: f, g0..g3, h0..h2, -cc of h0..h2; the bilinear terms are p*Px, p*Py, p*(Px + Py)
    want = np.zeros((11, 8), bool)
    want[[1, 7, 10], 3] = want[[2, 7, 10], 4] = want[[1, 2, 7, 10], 7] = True
    assert (nl == want).all()
    P = 2 * 8 + 1 + 4  # center, axis points, Kelley points
    keep = _cut_rows((P, 11), nl)
    # every part at the center, 4 (g0, g1) or 6 (h2, -cc h2) axis points, 4 Kelley points
    assert len(keep) == 11 + 2 * (4 + 4) + 2 * (6 + 4)


def test_pooling_rows_valid():
    """On a pooling model the node bound with the reduced cut rows is valid on boxes holding the
    optimum and matches the bound from all rows."""
    import jaxglobal.lp as LP

    f, g, h, lo0, hi0 = haverly1()
    tol = 1e-6

    def bounder(on, monkeypatch=None):
        cb = LP.make_constrained_bounder(f, g, h, tol=tol, univariate=True, alphabb=True)
        if on:
            return jax.jit(cb)
        orig = LP._nonlinear_vars
        LP._nonlinear_vars = lambda *a: None
        try:
            return jax.jit(cb).lower(jnp.asarray(lo0), jnp.asarray(hi0)).compile()
        finally:
            LP._nonlinear_vars = orig

    cb, cb_all = bounder(True), bounder(False)
    rng = np.random.default_rng(5)
    w = hi0 - lo0
    boxes = [(lo0, hi0)] + [(np.maximum(lo0, HAVERLY_X - rng.uniform(0, 0.5, 8) * w),
                             np.minimum(hi0, HAVERLY_X + rng.uniform(0, 0.5, 8) * w)) for _ in range(6)]  # fmt: skip
    for a, b in boxes:
        lb = float(cb(jnp.asarray(a), jnp.asarray(b))[0])
        ref = float(cb_all(jnp.asarray(a), jnp.asarray(b))[0])
        assert lb <= -400.0 + 1e-6, (a, b, lb)
        assert abs(lb - ref) <= 1e-4 * (1 + abs(ref)), (a, b, lb, ref)
