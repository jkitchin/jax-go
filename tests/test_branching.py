import pytest

from jaxglobal import minimize
from jaxglobal.problems import CONSTRAINED_PROBLEMS, PROBLEMS

CAMEL = next(p for p in PROBLEMS if p.name == "six_hump_camel")
G24 = next(p for p in CONSTRAINED_PROBLEMS if p.name == "G24")

OPTIONS = [
    {"branch_var": "width"},
    {"branch_point": "relax"},
    {"branch_var": "pseudocost"},
    {"branch_var": "reliability"},
    {"branch_point": "relax", "branch_var": "reliability"},
    {"branch_var": "strong"},
]
IDS = [",".join(f"{k}={v}" for k, v in o.items()) for o in OPTIONS]


def solve(p, **kw):
    return minimize(
        p.f, p.lo, p.hi, ineq=getattr(p, "ineq", None), eq=getattr(p, "eq", None), **kw
    )


@pytest.mark.parametrize("opts", OPTIONS, ids=IDS)
@pytest.mark.parametrize("p", [CAMEL, G24], ids=lambda p: p.name)
def test_branching_options_reach_optimum(p, opts):
    r = solve(p, **opts)
    assert r.status == "optimal"
    assert abs(r.fun - p.fstar) < 1e-4 * (1 + abs(p.fstar))
    assert r.evaluations >= r.nodes - 1


@pytest.mark.parametrize("opts", [{}] + OPTIONS, ids=["default"] + IDS)
def test_small_batch_keeps_boxes_in_sync(opts):
    """batch=8 forces a lot of filtering and, for strong branching, parent trimming."""
    r = solve(CAMEL, batch=8, **opts)
    assert r.status == "optimal"
    assert abs(r.fun - CAMEL.fstar) < 1e-4


def test_default_evaluations_match_nodes():
    """Neither rule the default picks (pseudocost, or width with integers) bounds trial boxes."""
    for kw in ({}, {"integer": [0]}):
        r = solve(CAMEL, **kw)
        assert r.status == "optimal"
        assert r.evaluations in (
            r.nodes,
            r.nodes + 1,
        )  # +1: the root is re-bounded after root OBBT


def test_auto_branching_rule(monkeypatch):
    """branch_var="auto" (the default) is pseudocost without integer variables, width with them.

    The decision is checked directly: node counts are not compared across solves, since the
    first solve with new kernels can take a different path while they compile."""
    import numpy as np

    from jaxglobal import bnb

    assert bnb._auto_branch_var(np.zeros(2, bool)) == "pseudocost"
    assert bnb._auto_branch_var(np.array([True, False])) == "width"
    picked = []
    rule = bnb._auto_branch_var
    monkeypatch.setattr(
        bnb, "_auto_branch_var", lambda m: picked.append(rule(m)) or picked[-1]
    )
    sh = next(p for p in PROBLEMS if p.name == "shubert")
    assert solve(sh).status == solve(sh, integer=[0]).status == "optimal"
    assert (
        solve(sh, branch_var="width").status == "optimal"
    )  # an explicit rule is not replaced
    assert picked == ["pseudocost", "width"]


def test_pseudocost_with_integers():
    """Pseudocost on an integer problem splits fractional integers first and records integer
    pseudocosts per unit of the split's move (0.5 each side for an integral value)."""
    sh = next(p for p in PROBLEMS if p.name == "shubert")
    for integer in ([0], [0, 1]):
        r = solve(sh, integer=integer, branch_var="pseudocost")
        assert r.status == "optimal" and r.verified


def test_invalid_options():
    with pytest.raises(ValueError, match="branch_point"):
        solve(CAMEL, branch_point="left")
    with pytest.raises(ValueError, match="branch_var"):
        solve(CAMEL, branch_var="random")


@pytest.mark.parametrize("q", [2, 3])
@pytest.mark.parametrize("p", [CAMEL, G24], ids=lambda p: p.name)
def test_orthant_multisection_reaches_optimum(p, q, monkeypatch):
    """branch_var="strong" with _Q > 1 splits a parent into up to 2**q orthants."""
    import jaxglobal.bnb as bnb

    monkeypatch.setattr(bnb, "_Q", q)
    r = solve(p, branch_var="strong")
    assert r.status == "optimal"
    assert abs(r.fun - p.fstar) < 1e-4 * (1 + abs(p.fstar))
    assert r.lower_bound <= p.fstar + 1e-6 * (1 + abs(p.fstar))


def test_branch_widths_root_box_with_floor():
    """A variable root tightening narrowed at least ``_SHRINK[0]``-fold is measured against its
    root width, floored at ``width0 / _SHRINK[1]``; modestly narrowed, never-split (inf) and
    root-fixed variables keep the original width."""
    import numpy as np

    from jaxglobal.bnb import _SHRINK, _branch_widths

    w0 = np.array([4.0, 10.0, 1.0, np.inf, 2.0, 2.0])
    rlo = np.array([[-0.7, 0.0, 0.0, 0.0, 1.0, 0.0], [-0.6, 0.0, 0.2, 0.0, 1.0, 0.0]])
    rhi = np.array([[-0.4, 1e-6, 1.0, 5.0, 1.0, 0.5], [-0.5, 1e-6, 0.5, 5.0, 1.0, 0.5]])
    w = _branch_widths(w0, rlo, rhi)
    assert np.allclose(w[:3], [0.3, 10.0 / _SHRINK[1], 1.0])  # widest root box; floor; unchanged
    assert w[3] == np.inf and w[4] == 2.0 and w[5] == 2.0  # never split; root-fixed; only 4x


def test_failed_local_solves_on_a_convex_problem_keep_finite_branch_points(monkeypatch):
    """On convex problems children branch at the local solutions; a solve that returns NaN keeps
    the LP point (NaN branch points made NaN boxes that were never pruned: max_iter at 13823 nodes)."""
    import jax.numpy as jnp
    import numpy as np

    from jaxglobal import bnb

    make_local = bnb.make_local

    def nan_local(*a, **k):
        solve = make_local(*a, **k)

        def failing(S, lo=None, hi=None, **kw):
            X = np.array(
                solve(S, lo, hi, **kw) if lo is not None else solve(S, **kw),
                dtype=float,
            )
            if X.ndim == 2 and lo is not None:
                X[:] = np.nan  # the per-child solves fail
            return X

        return failing

    monkeypatch.setattr(bnb, "make_local", nan_local)
    monkeypatch.setattr(bnb, "_cached", lambda key, build: build())
    n = 4

    def f(z):
        return (
            sum(jnp.exp(1.5 * z[j]) + (z[j] - 0.3 * j) ** 4 for j in range(n))
            + (z[0] + z[1] - z[2] + 0.5 * z[3]) ** 2
        )

    def g(z):  # written out: a dot product hides the convexity
        return jnp.stack(
            [
                z[0] ** 2 + 2 * z[1] ** 2 + z[2] ** 2 + z[3] ** 2 - 3.0,
                jnp.exp(z[0]) + jnp.exp(-z[3]) - 3.0,
                0.4 - z[0] - z[1] - z[2],
            ]
        )

    r = minimize(
        f,
        -3 * np.ones(n),
        3 * np.ones(n),
        ineq=g,
        branch_point="relax",
        n_kelley=0,
        max_iter=60,
        abs_tol=1e-10,
        rel_tol=1e-10,
    )
    assert r.status == "optimal" and r.lower_bound <= r.fun + 1e-9


def test_gap_weights():
    """The width score on the lifted quadratic LP (which also weighs pseudocost scores there):
    relative width times the variable's McCormick gap over the box's largest; the relative
    width alone in a box whose products are exact at the relaxation point; -1 where not ok."""
    from types import SimpleNamespace

    import numpy as np

    from jaxglobal.bnb import _gap_weights

    Q = SimpleNamespace(pairs=np.array([[0, 0], [0, 1]]))  # x0^2 and x0*x1; x2 in no product
    lo = np.zeros((2, 3))
    hi = np.array([[2.0, 2.0, 1.0]] * 2)
    x = np.array([[1.0, 1.0, 0.5], [0.0, 0.0, 0.0]])  # box 1 at a vertex: every gap is 0
    relw = np.array([[1.0, 0.5, 0.25]] * 2)
    ok = np.array([[True, True, True], [True, True, False]])
    w = _gap_weights(Q, lo, hi, x, relw, ok)
    # box 0: gaps x0^2 -> 1, x0*x1 -> 2; per variable 3, 2, 0
    assert np.allclose(w[0], [1.0, 0.5 * 2 / 3, 0.0])
    assert np.allclose(w[1], [1.0, 0.5, -1.0])
