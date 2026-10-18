"""Disk cache of traced kernels (jaxglobal/kcache.py)."""

import os
import subprocess
import sys
import time
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


def _file(path, size, age):
    """A file of ``size`` bytes last used ``age`` seconds ago."""
    path.write_bytes(b"x" * size)
    t = time.time() - age
    os.utime(path, (t, t))
    return path


def test_prune_xla_removes_least_recently_used(tmp_path):
    files = [_file(tmp_path / f"k{i}-cache", 100, age=1000 - i) for i in range(10)]  # k0 oldest
    kcache.prune_xla(str(tmp_path), max_bytes=2000)  # under the cap: nothing removed
    assert all(f.exists() for f in files)
    kcache.prune_xla(str(tmp_path), max_bytes=800)  # down to 0.75 * 800 = 600 bytes
    assert [f.exists() for f in files] == [False] * 4 + [True] * 6


def test_prune_kernels_drops_stale_sources_and_old_files(tmp_path, monkeypatch):
    monkeypatch.setenv("JAXGLOBAL_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("JAXGLOBAL_CACHE", raising=False)
    monkeypatch.setattr(kcache, "_MAX_KERNELS", 3)
    d = tmp_path / "kernels"
    d.mkdir()
    mine = kcache._source_hash()[:12]
    old_mine = [_file(d / f"{mine}-{i:064x}-bound-16.bin", 10, age=3 * 86400 + i) for i in range(4)]
    stale = _file(d / f"{'0' * 12}-{1:064x}-bound-16.bin", 10, age=2 * 86400)  # another source, unused
    shared = _file(d / f"{'1' * 12}-{2:064x}-bound-16.bin", 10, age=60)  # another checkout, in use
    legacy = _file(d / f"{3:064x}-bound-16.bin", 10, age=2 * 86400)  # unprefixed (older layout)
    kcache._prune_kernels(str(d), force=True)
    assert not stale.exists() and not legacy.exists() and shared.exists()
    # the 3 most recently used remain: the shared one and the 2 newest of this source
    assert [f.exists() for f in old_mine] == [True, True, False, False]
    assert kcache._path("ab").startswith(str(d / f"{mine}-ab"))


def test_clear_cache_removes_only_jaxglobal_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("JAXGLOBAL_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("JAXGLOBAL_CACHE", raising=False)
    for sub in ("xla", "kernels", "other"):
        (tmp_path / sub).mkdir()
        _file(tmp_path / sub / "f", 100, age=0)
    assert kcache.clear_cache() == 200
    assert not (tmp_path / "xla").exists() and not (tmp_path / "kernels").exists()
    assert (tmp_path / "other" / "f").exists()
    monkeypatch.setenv("JAXGLOBAL_CACHE", "0")
    assert kcache.clear_cache() == 0
