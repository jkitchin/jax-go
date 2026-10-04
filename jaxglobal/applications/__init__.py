"""Code-defined application models for the flagship benchmark.

Each class module (``kinetics``, ``logic``, ``nn``) exports ``PROBLEMS``, a list
of ``AppProblem``. ``fstar`` is the best known global minimum of *that exact
formulation* (the discretized or tabulated model that ``f`` evaluates), and
``pyomo()`` builds the same formulation for the equation-based solvers.
"""

from __future__ import annotations

import importlib
from typing import NamedTuple

import numpy as np


class AppProblem(NamedTuple):
    name: str
    f: callable  # JAX objective
    lo: np.ndarray
    hi: np.ndarray
    ineq: callable | None  # g(x) <= 0, vector
    eq: callable | None
    fstar: float | None  # best known global minimum of THIS formulation
    source: str  # citation / how fstar was obtained
    pyomo: callable | None  # () -> pyomo ConcreteModel of the same formulation (objective `obj`, variable `x` indexed 0..n-1)
    note: str


CLASSES = ("kinetics", "logic", "nn")


def suite(classes=None):
    """All problems of the given classes (default all); a missing class module is skipped."""
    out = []
    for c in classes or CLASSES:
        try:
            mod = importlib.import_module(f"{__name__}.{c}")
        except ModuleNotFoundError as e:
            if e.name != f"{__name__}.{c}":
                raise
            continue
        out.extend(mod.PROBLEMS)
    return out
