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
    return minimize(p.f, p.lo, p.hi, ineq=getattr(p, "ineq", None), eq=getattr(p, "eq", None), **kw)


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
        assert r.evaluations in (r.nodes, r.nodes + 1)  # +1: the root is re-bounded after root OBBT


def test_auto_branching_rule():
    """branch_var="auto" (the default) is pseudocost without integer variables, width with them.

    Both problems are ones where the two rules take different numbers of nodes
    (hartmann3 157 width / 93 pseudocost; shubert with x0 integer 53 / 77)."""
    h3 = next(p for p in PROBLEMS if p.name == "hartmann3")
    sh = next(p for p in PROBLEMS if p.name == "shubert")
    n = lambda r: (r.status, r.nodes, r.evaluations)  # noqa: E731
    assert n(solve(h3, branch_var="width")) != n(solve(h3, branch_var="pseudocost")) == n(solve(h3))
    w, pc = solve(sh, integer=[0], branch_var="width"), solve(sh, integer=[0], branch_var="pseudocost")
    assert n(pc) != n(w) == n(solve(sh, integer=[0]))


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
