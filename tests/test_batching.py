import os
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

from jaxglobal import minimize
from jaxglobal.bnb import _Padder
from jaxglobal.problems import CONSTRAINED_PROBLEMS, PROBLEMS

ROOT = Path(__file__).resolve().parent.parent
CAMEL = next(p for p in PROBLEMS if p.name == "six_hump_camel")
G24 = next(p for p in CONSTRAINED_PROBLEMS if p.name == "G24")


@pytest.mark.parametrize(
    "k, size", [(1, 16), (16, 16), (17, 64), (64, 64), (65, 256), (256, 256)]
)
def test_padder_sizes_one_device(k, size):
    pad = _Padder(256, (16, 64), jax.devices()[:1])
    a = np.arange(k * 3, dtype=float).reshape(k, 3)
    b = np.arange(k, dtype=float) + 100
    pa, pb = pad(a, b)
    assert pa.shape == (size, 3) and pb.shape == (size,)
    np.testing.assert_array_equal(pa[:k], a)
    np.testing.assert_array_equal(pb[:k], b)
    np.testing.assert_array_equal(pa[k:], np.broadcast_to(a[0], (size - k, 3)))
    np.testing.assert_array_equal(pb[k:], np.full(size - k, b[0]))


def test_padder_sizes_capped_by_batch():
    pad = _Padder(10, (16, 64), jax.devices()[:1])
    assert pad.sizes == [10]
    assert pad(np.zeros((3, 2)))[0].shape == (10, 2)


def _solve(p, **kw):
    ineq = getattr(p, "ineq", None)
    eq = getattr(p, "eq", None)
    return minimize(p.f, p.lo, p.hi, ineq=ineq, eq=eq, **kw)


@pytest.mark.parametrize("p", [CAMEL, G24], ids=lambda p: p.name)
def test_batch_sizes_same_optimum(p):
    r0 = _solve(p, batch_sizes=())
    r1 = _solve(p)
    assert r0.status == r1.status == "optimal"
    assert abs(r0.fun - p.fstar) < 1e-4 and abs(r1.fun - p.fstar) < 1e-4
    assert abs(r0.fun - r1.fun) < 1e-5


_SCRIPT = """
import jax
from jaxglobal import minimize
from jaxglobal.problems import CONSTRAINED_PROBLEMS, PROBLEMS
assert len(jax.devices()) == 4, jax.devices()
camel = next(p for p in PROBLEMS if p.name == "six_hump_camel")
g24 = next(p for p in CONSTRAINED_PROBLEMS if p.name == "G24")
for batch in (10, 32):
    r = minimize(camel.f, camel.lo, camel.hi, batch=batch)
    assert r.status == "optimal", (batch, r)
    assert abs(r.fun - camel.fstar) < 1e-4, (batch, r)
    r = minimize(g24.f, g24.lo, g24.hi, ineq=g24.ineq, batch=batch)
    assert r.status == "optimal", (batch, r)
    assert abs(r.fun - g24.fstar) < 1e-4, (batch, r)
print("ok")
"""


def test_multi_device_subprocess():
    env = dict(os.environ)
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["JAX_PLATFORMS"] = "cpu"
    out = subprocess.run(
        [sys.executable, "-c", _SCRIPT], env=env, capture_output=True, text=True, timeout=600
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert out.stdout.strip().endswith("ok")
