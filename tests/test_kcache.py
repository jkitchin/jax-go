"""Disk cache of traced kernels (jaxglobal/kcache.py)."""

import os
import subprocess
import sys
from pathlib import Path

import jax.numpy as jnp

from jaxglobal import kcache

ROOT = Path(__file__).resolve().parent.parent


def test_fingerprint_tracks_function_and_constants():
    a = kcache.fingerprint((lambda x: jnp.sum(x**2), None, None), 2, ())
    assert a == kcache.fingerprint((lambda x: jnp.sum(x**2), None, None), 2, ())
    assert a != kcache.fingerprint((lambda x: jnp.sum(x**3), None, None), 2, ())
    assert a != kcache.fingerprint((lambda x: jnp.sum(x**2), None, None), 3, ())
    assert a != kcache.fingerprint((lambda x: jnp.sum(x**2), None, None), 2, (("n_fw", 3),))
    assert a != kcache.fingerprint((lambda x: jnp.sum(x**2), lambda x: x[0], None), 2, ())
    c1, c2 = jnp.arange(50.0), jnp.arange(50.0).at[49].set(0.0)  # differ in one entry of a closed-over array
    assert kcache.fingerprint((lambda x: jnp.sum(x**2) + c1 @ c1, None, None), 2, ()) != kcache.fingerprint(
        (lambda x: jnp.sum(x**2) + c2 @ c2, None, None), 2, ()
    )


_SCRIPT = """
import os
from jaxglobal import minimize
from jaxglobal.problems import PROBLEMS
p = next(p for p in PROBLEMS if p.name == "six_hump_camel")
r = minimize(p.f, p.lo, p.hi)
print(r.status, repr(r.fun), r.nodes, sorted(os.listdir(os.path.join(os.environ["JAXGLOBAL_CACHE_DIR"], "kernels"))))
"""


def test_second_process_loads_kernels(tmp_path):
    env = dict(os.environ, JAXGLOBAL_CACHE_DIR=str(tmp_path), PYTHONPATH=str(ROOT), JAX_PLATFORMS="cpu")
    env.pop("JAXGLOBAL_CACHE", None)
    runs = [subprocess.run([sys.executable, "-c", _SCRIPT], env=env, capture_output=True, text=True, timeout=600)
            for _ in range(2)]  # fmt: skip
    for r in runs:
        assert r.returncode == 0, r.stdout + r.stderr
    first, second = (r.stdout.strip().splitlines()[-1] for r in runs)
    assert first.startswith("optimal") and first == second  # same result, same files
    assert "-bound-16.bin" in first and (tmp_path / "xla").is_dir()


_SCRIPT_LAPACK = """
import os, jax, re
from jaxglobal import minimize
from jaxglobal.problems import CONSTRAINED_PROBLEMS
p = next(p for p in CONSTRAINED_PROBLEMS if p.name == "G24")
r = minimize(p.f, p.lo, p.hi, ineq=p.ineq, eq=p.eq)
d = os.path.join(os.environ["JAXGLOBAL_CACHE_DIR"], "kernels")
calls = set()
for name in os.listdir(d):
    if name.endswith(".bin"):
        mlir = jax.export.deserialize(bytearray(open(os.path.join(d, name), "rb").read())).mlir_module()
        calls |= set(re.findall(r"custom_call @(lapack_\\w+)", mlir))
print(r.status, repr(r.fun), sorted(calls))
"""


def test_second_process_runs_cached_lapack_kernels(tmp_path):
    """A kernel whose node LP calls LAPACK, loaded from disk in a fresh process.

    Before jax 0.11.2 (hence the requirement in pyproject.toml) jaxlib loaded
    its LAPACK function pointers only while lowering a linear algebra op, and a
    kernel restored with ``jax.export.deserialize`` is not lowered from Python,
    so the second process segfaulted on the first bound.
    """
    env = dict(os.environ, JAXGLOBAL_CACHE_DIR=str(tmp_path), PYTHONPATH=str(ROOT), JAX_PLATFORMS="cpu")
    env.pop("JAXGLOBAL_CACHE", None)
    runs = [subprocess.run([sys.executable, "-c", _SCRIPT_LAPACK], env=env, capture_output=True, text=True, timeout=600)
            for _ in range(2)]  # fmt: skip
    for r in runs:
        assert r.returncode == 0, f"exit {r.returncode}\n" + r.stdout + r.stderr[-2000:]
    first, second = (r.stdout.strip().splitlines()[-1] for r in runs)
    assert first.startswith("optimal") and first == second
    assert "lapack_" in first  # the cached kernels really do call LAPACK
