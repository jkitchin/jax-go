import pytest

from jaxglobal import minimize
from jaxglobal.problems import CONSTRAINED_PROBLEMS, PROBLEMS

CAMEL = next(p for p in PROBLEMS if p.name == "six_hump_camel")
G24 = next(p for p in CONSTRAINED_PROBLEMS if p.name == "G24")

OPTIONS = [
    {"branch_point": "relax"},
    {"branch_var": "pseudocost"},
    {"branch_var": "reliability"},
    {"branch_point": "relax", "branch_var": "reliability"},
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
    r = solve(CAMEL)
    assert r.evaluations in (r.nodes, r.nodes + 1)  # +1: the root is re-bounded after root OBBT


def test_invalid_options():
    with pytest.raises(ValueError, match="branch_point"):
        solve(CAMEL, branch_point="left")
    with pytest.raises(ValueError, match="branch_var"):
        solve(CAMEL, branch_var="random")
