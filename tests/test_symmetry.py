"""Permutation symmetry (``symmetry.py``): detected groups are symmetries, lex-leader pruning
keeps the lex-largest point of every orbit, and the search with it proves the right optimum."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxglobal
from jaxglobal.quadratic import polynomial_form
from jaxglobal.symmetry import automorphisms, is_automorphism, lex_prune

jax.config.update("jax_enable_x64", True)


def _sign(p):
    return (-1) ** sum(p[i] > p[j] for i in range(len(p)) for j in range(i + 1, len(p)))


def det_poly(m):
    """det of the m x m matrix of variables x_{i m + j} as a monomial dict."""
    return {tuple(sorted(i * m + p[i] for i in range(m))): float(_sign(p)) for p in itertools.permutations(range(m))}


def det_problem(m):
    """max det(B) over 0/1 matrices B as min -t, t - det(B) <= 0 (MINLPLib hadamard_m)."""
    n = m * m

    def f(x):
        return -x[n]

    def g(x):
        det = 0.0
        for p in itertools.permutations(range(m)):
            t = float(_sign(p))
            for i in range(m):
                t = t * x[i * m + p[i]]
            det = det + t
        return jnp.stack([x[n] - det])

    return f, g, n


def test_det_group():
    """The symmetries of det(B) >= t moving only B: row and column permutations of equal sign
    and the transpose, 2 (m!)^2 / 2 elements, the identity left out."""
    m = 3
    P = automorphisms(m * m + 1, {(m * m,): -1.0}, [{**{k: -v for k, v in det_poly(m).items()}, (m * m,): 1.0}], [],
                      range(m * m))  # fmt: skip
    assert len(P) == 36 - 1
    assert len({p.tobytes() for p in P}) == len(P)
    for p in P:
        assert p[m * m] == m * m  # the continuous t stays
        assert is_automorphism(p, {(m * m,): -1.0}, [{**{k: -v for k, v in det_poly(m).items()}, (m * m,): 1.0}], [])


def test_none_found_without_symmetry():
    f = {(0,): 1.0, (1,): 2.0, (2,): 3.0, (0, 1): 1.0}
    assert automorphisms(3, f, [], [], range(3)).shape == (0, 3)
    # x0 and x1 swap; x2 is continuous and may not move even where it would fit
    P = automorphisms(4, {(0,): 1.0, (1,): 1.0, (2,): 1.0, (3,): 1.0}, [], [], [0, 1])
    assert [p.tolist() for p in P] == [[1, 0, 2, 3]]


def test_rows_permuted_and_equalities():
    """Inequality rows may be permuted among themselves; an asymmetric equality breaks the swap."""
    f = {(0, 1): 1.0}
    gs = [{(0,): 1.0, (): -1.0}, {(1,): 1.0, (): -1.0}]
    assert len(automorphisms(2, f, gs, [], [0, 1])) == 1
    assert len(automorphisms(2, f, gs, [{(0,): 1.0, (): -1.0}], [0, 1])) == 0
    assert len(automorphisms(2, f, [{(0,): 1.0, (): -1.0}, {(1,): 2.0, (): -1.0}], [], [0, 1])) == 0


def test_polynomial_form_finds_det_group():
    f, g, n = det_problem(3)
    Q = polynomial_form(f, g, None, n + 1, integer=np.arange(n))
    assert Q.flip == -1 and Q.sym.shape == (35, n + 1)
    assert polynomial_form(f, g, None, n + 1, integer=None).sym.size == 0  # no integers: no claim


def _lex_key(x):
    return tuple(x.tolist())


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_lex_prune_keeps_orbit_leaders(seed):
    """No box holding the lex-largest point of its orbit is pruned, and a pruned box holds only
    points with a lex-larger image (checked by enumeration over 0/1 points of the box)."""
    m = 3
    n = m * m
    f = {(n,): -1.0}
    g = [{**{k: -v for k, v in det_poly(m).items()}, (n,): 1.0}]
    P = automorphisms(n + 1, f, g, [], range(n))
    group = np.vstack([np.arange(n + 1), P])
    pts = np.array(list(itertools.product([0.0, 1.0], repeat=n)))
    pts = np.c_[pts, np.zeros(len(pts))]
    leader = np.array([all(_lex_key(x) >= _lex_key(x[p]) for p in group) for x in pts])
    rng = np.random.default_rng(seed)
    lo = np.zeros((300, n + 1))
    hi = np.ones((300, n + 1))
    hi[:, n] = 0.0
    for b in range(300):  # random partial assignments
        fix = rng.random(n) < rng.random()
        val = (rng.random(n) < 0.5).astype(float)
        lo[b, :n] = np.where(fix, val, 0.0)
        hi[b, :n] = np.where(fix, val, 1.0)
    dead = lex_prune(lo, hi, P)
    assert dead.any() and not dead.all()
    for b in np.flatnonzero(dead):
        inside = np.all((pts >= lo[b]) & (pts <= hi[b]), axis=1)
        assert not (inside & leader).any()
        for x in pts[inside]:
            assert any(_lex_key(x[p]) > _lex_key(x) for p in P)


def test_minimize_det_with_symmetry():
    """max det over 0/1 4 x 4 matrices is 3: proved with the pruning on, from the full group."""
    f, g, n = det_problem(4)
    Q = polynomial_form(f, g, None, n + 1, integer=np.arange(n))
    assert len(Q.sym) == 2 * 24 * 24 // 2 - 1
    lo, hi = np.r_[np.zeros(n), -30.0], np.r_[np.ones(n), 30.0]
    r = jaxglobal.minimize(f, lo, hi, ineq=g, integer=np.arange(n), abs_tol=1e-6, rel_tol=1e-6)
    assert r.status == "optimal"
    assert r.fun == pytest.approx(-3.0, abs=1e-6)
    assert r.lower_bound <= -3.0 + 1e-6
    B = np.round(r.x[:n]).reshape(4, 4)
    assert np.linalg.det(B) == pytest.approx(3.0, abs=1e-9)


def _sym_sum_problem(n):
    """min sum x_i^3 over integers with sum x_i >= 3: every permutation is a symmetry (S_n)."""
    f = {(i, i, i): 1.0 for i in range(n)}
    g = [{**{(i,): -1.0 for i in range(n)}, (): 3.0}]
    return f, g


def test_large_group_listing_is_capped_and_small():
    """S_120: the transpositions are found directly, the listing stops at the cap without forming
    the full products, and the generators (transversal elements) come first."""
    n = 120
    f, g = _sym_sum_problem(n)
    P = automorphisms(n, f, g, [], range(n), max_elements=300, deadline=np.inf)  # (no wall clock in tests)
    assert P.shape == (300, n)
    assert len({p.tobytes() for p in P}) == len(P)
    first = P[0]
    assert (first != np.arange(n)).sum() == 2  # a transposition
    for p in P[::37]:
        assert is_automorphism(p, f, g, [])


def test_search_deadline_gives_up_cleanly():
    """Out of time (an injected clock), the search returns no elements; the call budget still
    keeps what it found."""
    f, g, n = det_problem(3)
    fp = {(n,): -1.0}
    gp = [{**{k: -v for k, v in det_poly(3).items()}, (n,): 1.0}]
    ticks = iter(range(10**6))
    P = automorphisms(n + 1, fp, gp, [], range(n), deadline=5.0, clock=lambda: next(ticks))
    assert P.shape == (0, n + 1)
    assert len(automorphisms(n + 1, fp, gp, [], range(n), deadline=np.inf)) == 35


def test_lex_prune_uses_a_bounded_prefix():
    """lex_prune looks at only the first max_elements elements: a box only a later element prunes
    survives with a small cap."""
    P = np.array([[0, 1, 2, 3], [0, 1, 3, 2]])
    lo, hi = np.array([[0.0, 0.0, 0.0, 1.0]]), np.array([[0.0, 0.0, 0.0, 1.0]])
    assert lex_prune(lo, hi, P).tolist() == [True]
    assert lex_prune(lo, hi, P, max_elements=1).tolist() == [False]


def test_polynomial_form_survives_a_failing_search(monkeypatch):
    """An exception in the symmetry search drops only the symmetry, not the lifted form."""
    import jaxglobal.quadratic as quadratic

    def boom(*a, **k):
        raise RuntimeError

    monkeypatch.setattr(quadratic, "automorphisms", boom)
    f, g, n = det_problem(3)
    Q = polynomial_form(f, g, None, n + 1, integer=np.arange(n))
    assert Q is not None and Q.sym.size == 0
