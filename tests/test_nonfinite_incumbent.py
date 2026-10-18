"""A point whose objective or constraint values are not finite never becomes the incumbent.

``f = -inf`` (log at 0) is no attained minimum: accepted as the incumbent it would be a cutoff
that prunes every box, and the search would end with ``fun = -inf`` and status "infeasible"
(``ub`` not finite). ``screen`` rejects such points, so every incumbent entry point (point
boxes, local solves, dives, integer repair, the polish, the kernel's candidates) does."""

import jax.numpy as jnp
import numpy as np

import jaxglobal  # noqa: F401  (enables x64)
from jaxglobal import minimize


def test_integer_neg_inf_objective_point_is_skipped():
    """log(x0) over the integers 0..3: x0 = 0 gives -inf; the minimum is f(1) = 0."""
    r = minimize(lambda x: jnp.log(x[0]), lo=[0.0], hi=[3.0], integer=[0])
    assert r.status == "optimal" and r.converged
    assert r.fun == 0.0 and np.allclose(r.x, [1.0])
    assert np.isfinite(r.lower_bound)


def test_integer_neg_inf_constraint_point_is_skipped():
    """min x0 s.t. log(x0) <= 0 over the integers 0..3: g(0) = -inf is no satisfied constraint
    (the problem is undefined there), so the minimum is at x0 = 1."""
    r = minimize(lambda x: x[0], lo=[0.0], hi=[3.0], ineq=lambda x: jnp.log(x[0]), integer=[0])
    assert r.status == "optimal"
    assert r.fun == 1.0 and np.allclose(r.x, [1.0])


def test_continuous_neg_inf_objective_is_not_an_incumbent():
    """The continuous analogue: log(x0) on [0, 3] is unbounded below. The local solve walks to
    x0 = 0 (f = -inf); that is no incumbent, so the result keeps a finite one and an honest
    status (not "infeasible", not "optimal") with lower_bound -inf."""
    r = minimize(lambda x: jnp.log(x[0]), lo=[0.0], hi=[3.0], max_iter=10)
    assert np.isfinite(r.fun) and r.x[0] > 0.0
    assert r.status not in ("infeasible", "optimal") and not r.converged
    assert r.lower_bound == -np.inf
