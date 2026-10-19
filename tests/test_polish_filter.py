"""``bnb._moves_keep``: the polish skips integer moves that break a row on integers only."""

import jax
import jax.numpy as jnp
import numpy as np

from jaxglobal.bnb import _integer_rows, _moves_keep

jax.config.update("jax_enable_x64", True)


def test_assignment_row_moves():
    # x0..x3 binary with x0 + x1 + x2 + x3 == 1; x4 continuous in a row with x0 (not integer-only)
    eq = lambda x: jnp.array([x[0] + x[1] + x[2] + x[3] - 1.0])  # noqa: E731
    ineq = lambda x: jnp.array([x[4] - 10.0 * x[0], x[1] + x[2] - 1.0])  # noqa: E731
    imask = np.array([True] * 4 + [False])
    lo, hi = np.zeros(5), np.array([1.0, 1, 1, 1, 10])
    rows = _integer_rows(ineq, eq, lo, hi, imask)
    assert rows is not None and len(rows[0]) == 2  # the assignment row and x1 + x2 <= 1, not the mixed row
    A = rows[0]
    xi = np.array([1.0, 0, 0, 0])
    one = [(j, d) for j in range(4) for d in (-1.0, 1.0) if 0 <= xi[j] + d <= 1]
    D = np.array([A[:, j] * d for j, d in one])
    assert not _moves_keep(rows, xi, D, 1e-6).any()  # a single flip always breaks the assignment
    pairs = [(a, b) for a in range(4) for b in range(4) if a != b and xi[a] + 1 <= 1 and xi[b] - 1 >= 0]
    D = np.array([A[:, a] - A[:, b] for a, b in pairs])
    assert _moves_keep(rows, xi, D, 1e-6).all()  # swaps that move the 1 keep it
    xi = np.array([0.0, 1, 0, 0])  # moving the 1 from x1 to x2 is fine, but x1 + x2 <= 1 is checked too
    D = np.array([A[:, 2] - A[:, 1]])
    assert _moves_keep(rows, xi, D, 1e-6).all()


def test_tolerance():
    rows = (np.array([[1.0, 1.0]]), np.array([-1.0]), np.array([True]))
    xi = np.array([1.0, 0.0])
    D = np.array([[5e-6], [1e-3]])
    assert _moves_keep(rows, xi, D, 1e-6).tolist() == [True, False]
