"""A point whose objective is not finite never becomes the incumbent (nor one whose constraint
violation is NaN or inf; g = -inf is a satisfied inequality).

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


def test_neg_inf_constraint_value_is_satisfied():
    """g = -inf is a satisfied inequality (a ``where`` that switches the constraint off): such points
    stay incumbents, else boxes the bounder sees as feasible would never close."""
    g = lambda x: jnp.where(x[0] < 1, -jnp.inf, x[0] - 1.5)  # noqa: E731
    r = minimize(lambda x: (x[0] - 0.2) ** 2, lo=[-2.0], hi=[2.0], ineq=g, max_iter=200)
    assert r.status == "optimal" and abs(r.fun) < 1e-8 and abs(r.x[0] - 0.2) < 1e-4


def test_continuous_neg_inf_objective_is_not_an_incumbent():
    """The continuous analogue: log(x0) on [0, 3] is unbounded below. The local solve walks to
    x0 = 0 (f = -inf); that is no incumbent, so the result keeps a finite one and an honest
    status (not "infeasible", not "optimal") with lower_bound -inf."""
    r = minimize(lambda x: jnp.log(x[0]), lo=[0.0], hi=[3.0], max_iter=10)
    assert np.isfinite(r.fun) and r.x[0] > 0.0
    assert r.status not in ("infeasible", "optimal") and not r.converged
    assert r.lower_bound == -np.inf
