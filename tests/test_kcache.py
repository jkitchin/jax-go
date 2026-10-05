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
