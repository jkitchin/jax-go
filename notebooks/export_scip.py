"""Export a JAX problem to SCIP and to an AMPL .nl file, and check it against jaxglobal.

The model below uses ``jnp.where``, ``jnp.maximum``, ``jnp.tanh``, ``jax.nn.relu``
and a ``fori_loop`` (a fixed number of iterations), plus an integer variable.
The ``jnp``-to-Pyomo shim in ``compare.py`` cannot translate any of these, so
every one had to be reformulated by hand. ``jaxglobal.export`` handles them
directly and reports how it did it.

Read the report before trusting a solve. ``approximated`` lists what is not
exact: here the comparisons inside ``jnp.where`` are relaxed at ties
(``x[0] == 0.3``), which is harmless because both branches agree there (the
switch is continuous). For a piecewise-constant model it is not, so the script
re-evaluates ``f`` in JAX at the solver's ``x``, as you should.

Two ``.nl`` files are written. The default one keeps ``where`` / ``max`` /
``tanh`` as .nl operators (exact, read back by ``jaxglobal.nl.read_nl``), but
the ``scip`` and ``couenne`` binaries reject those operators; the
``portable=True`` one uses only operators both read. When the binaries are on
PATH they are run on it.

Run:  PYTHONPATH=. python notebooks/export_scip.py [time_limit_seconds]
"""

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jaxglobal import export, minimize
from jaxglobal.nl import read_nl

TIME_LIMIT = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0

W = jnp.array([[1.5, -0.7], [0.4, 1.2], [-1.1, 0.9]])
B = jnp.array([0.1, -0.3, 0.2])
V = jnp.array([0.8, -1.0, 0.6])


def f(x):
    """A small tanh network plus switching costs; x[2] is a whole number of units."""
    net = V @ jnp.tanh(W @ x[:2] + B)
    switch = jnp.where(x[0] > 0.3, (x[0] - 0.3) ** 2, 0.5 * (0.3 - x[0]))  # continuous at 0.3
    hinge = jnp.maximum(x[1], 0.2 * x[0]) + jax.nn.relu(x[0] - x[1] - 0.5)
    damped = jax.lax.fori_loop(0, 3, lambda i, c: 0.5 * c + 0.1 * x[i % 2], 0.0)  # unrolled by the export
    units = 0.15 * (x[2] - 1.6) ** 2
    return net + switch + hinge + damped + units


def g(x):  # g(x) <= 0
    return jnp.array([x[0] ** 2 + x[1] ** 2 - 2.0, x[1] - x[0] - 0.5 * x[2]])


lo = np.array([-1.5, -1.5, 0.0])
hi = np.array([1.5, 1.5, 4.0])
integer = [2]


def main():
    print("== export report (SCIP) ==")
    model = export.to_scip(f, lo, hi, ineq=g, integer=integer)
    print(model.data.report)

    model.hideOutput()
    model.setParam("limits/time", TIME_LIMIT)
    model.optimize()
    xs = export.solution(model)
    print(
        f"\nSCIP: status {model.getStatus()}, objective {model.getObjVal():.6f}, dual bound {model.getDualbound():.6f}"
    )
    print(f"      x = {np.round(xs, 6)}, f(x) re-evaluated in JAX = {float(f(jnp.asarray(xs))):.6f}")
    print(f"      max g(x) = {float(jnp.max(g(jnp.asarray(xs)))):.2e}")

    r = minimize(f, lo, hi, ineq=g, integer=integer, time_limit=TIME_LIMIT * 6)
    print(f"\njaxglobal: status {r.status}, f = {r.fun:.6f}, lower bound {r.lower_bound:.6f}, x = {np.round(r.x, 6)}")
    print(f"difference SCIP - jaxglobal: {model.getObjVal() - r.fun:+.2e}")

    path = Path(tempfile.mkdtemp()) / "export_example.nl"
    rep = export.to_nl(f, lo, hi, path, ineq=g, integer=integer)
    m = read_nl(str(path), integers="mask")
    x_nl = jnp.asarray(xs[rep.var_order])  # .nl writes integer variables last
    print(f"\n.nl file: {path} ({path.stat().st_size} bytes)")
    print(
        f"      read back with jaxglobal.nl.read_nl: f = {float(m.f(x_nl)):.6f}, integer mask {np.asarray(m.integer)}"
    )

    portable = path.with_name("export_example_portable.nl")
    prep = export.to_nl(f, lo, hi, portable, ineq=g, integer=integer, portable=True)
    print(f"portable .nl file: {portable} ({prep.n_aux} auxiliary variables, {prep.n_binary} binary)")
    for cmd, pat in (
        (["scip", "-c", f"set limits time {TIME_LIMIT} read {portable} opt quit"], r"Primal Bound\s*:\s*(\S+)"),
        (["couenne", str(portable), "-AMPL"], r"Upper bound:\s*(\S+)"),
    ):
        if shutil.which(cmd[0]) is None:
            print(f"      {cmd[0]}: not on PATH (run: {' '.join(cmd)})")
            continue
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=10 * TIME_LIMIT)
            hit = re.search(pat, out.stdout + out.stderr)
            print(f"      {cmd[0]} binary: objective {hit.group(1) if hit else 'not found (see its output)'}")
        except subprocess.TimeoutExpired:
            print(f"      {cmd[0]} binary: timed out")


if __name__ == "__main__":
    main()
