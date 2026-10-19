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


def _xla(d, i, size=100, age=0):
    """A JAX compilation cache entry (named as JAX names them)."""
    return _file(d / f"jit_f-{i:064x}-cache", size, age)


def test_prune_xla_removes_least_recently_used(tmp_path):
    files = [_xla(tmp_path, i, age=1000 - i) for i in range(10)]  # entry 0 oldest
    kcache.prune_xla(str(tmp_path), max_bytes=2000)  # under the cap: nothing removed
    assert all(f.exists() for f in files)
    kcache.prune_xla(str(tmp_path), max_bytes=800)  # down to 0.75 * 800 = 600 bytes
    assert [f.exists() for f in files] == [False] * 4 + [True] * 6


def test_concurrent_xla_prunes_do_not_overshoot(tmp_path, monkeypatch):
    """Another process prunes between this one's scan and its first removal: the files it removed
    count as removed here too."""
    for i in range(100):
        _xla(tmp_path, i, age=1000 - i)
    remove, fired = os.remove, []

    def racing_remove(p):
        if not fired:
            fired.append(1)
            monkeypatch.setattr(kcache.os, "remove", remove)
            kcache.prune_xla(str(tmp_path), max_bytes=8000)  # the other process, start to end
            monkeypatch.setattr(kcache.os, "remove", racing_remove)
        remove(p)

    monkeypatch.setattr(kcache.os, "remove", racing_remove)
    kcache.prune_xla(str(tmp_path), max_bytes=8000)
    assert len(os.listdir(tmp_path)) * 100 == 6000


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
    legacy = _file(d / f"{3:064x}-obbt-1.noexport", 10, age=2 * 86400)  # unprefixed (older layout)
    tmp_old = _file(d / "jaxglobal-tmp-abc", 10, age=2 * 86400)  # left by a crashed write
    tmp_new = _file(d / "jaxglobal-tmp-def", 10, age=0)  # another process writing
    kcache._prune_kernels(str(d), force=True)
    assert not stale.exists() and not legacy.exists() and not tmp_old.exists()
    assert shared.exists() and tmp_new.exists()
    # the 3 most recently used remain: the shared one and the 2 newest of this source
    assert [f.exists() for f in old_mine] == [True, True, False, False]
    assert kcache._path("ab").startswith(str(d / f"{mine}-ab"))


def test_foreign_files_survive_pruning_and_clearing(tmp_path, monkeypatch):
    monkeypatch.setenv("JAXGLOBAL_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("JAXGLOBAL_CACHE", raising=False)
    monkeypatch.setattr(kcache, "_MAX_KERNELS", 1)
    (tmp_path / "xla").mkdir()
    (tmp_path / "kernels").mkdir()
    foreign = [_file(tmp_path / sub / name, 10**4, age=10 * 86400)
               for sub in ("xla", "kernels") for name in ("my_cuda_kernel.cu", "notes-cache", "a.bin")]  # fmt: skip
    ours = [_xla(tmp_path / "xla", i) for i in range(3)]
    # +60 s: the marker save() writes below takes its mtime from the filesystem clock, which can trail time.time()
    ours += [_file(tmp_path / "kernels" / f"{'0' * 12}-{i:064x}-bound-16.bin", 100, age=86400 * (2 - i) + 60)
             for i in range(3)]  # fmt: skip
    kcache.prune_xla(str(tmp_path / "xla"), max_bytes=1)
    kcache._prune_kernels(str(tmp_path / "kernels"), force=True)
    kcache.save("0" * 64 + "-bound-16", None)  # its own LRU step too
    assert all(f.exists() for f in foreign) and not any(f.exists() for f in ours)
    assert len(os.listdir(tmp_path / "kernels")) == 4  # the foreign files and the new marker
    kcache.clear_cache()
    assert all(f.exists() for f in foreign)
    assert sorted(os.listdir(tmp_path / "kernels")) == sorted(f.name for f in foreign if f.parent.name == "kernels")


def test_clear_cache_removes_only_jaxglobal_entries(tmp_path, monkeypatch):
    monkeypatch.setenv("JAXGLOBAL_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("JAXGLOBAL_CACHE", raising=False)
    (tmp_path / "xla").mkdir()
    (tmp_path / "kernels").mkdir()
    _xla(tmp_path / "xla", 1)
    _file(tmp_path / "kernels" / f"{1:064x}-bound-16.bin", 100, age=0)
    other = tmp_path / "other"
    other.mkdir()
    _xla(other, 2)
    assert kcache.clear_cache() == 200
    assert os.listdir(tmp_path / "xla") == [] and os.listdir(tmp_path / "kernels") == []
    assert len(os.listdir(other)) == 1
    monkeypatch.setenv("JAXGLOBAL_CACHE", "0")
    assert kcache.clear_cache() == 0


def test_clear_cache_skips_a_linked_cache_directory(tmp_path, monkeypatch):
    target = tmp_path / "elsewhere"
    target.mkdir()
    entry = _xla(target, 1)
    root = tmp_path / "root"
    root.mkdir()
    (root / "xla").symlink_to(target)
    monkeypatch.setenv("JAXGLOBAL_CACHE_DIR", str(root))
    monkeypatch.delenv("JAXGLOBAL_CACHE", raising=False)
    assert kcache.clear_cache() == 0 and entry.exists()


def test_load_refreshes_noexport_marker(tmp_path, monkeypatch):
    monkeypatch.setenv("JAXGLOBAL_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("JAXGLOBAL_CACHE", raising=False)
    key = f"{4:064x}-bound-16"
    kcache.save(key, None)
    marker = kcache._path(key) + ".noexport"
    t = time.time() - 5 * 86400
    os.utime(marker, (t, t))
    assert kcache.load(key) is False
    assert time.time() - os.path.getmtime(marker) < 60


def test_wait_for_builds_follows_nested_futures():
    import threading
    from concurrent.futures import Future

    gate = threading.Event()
    inner, outer, failed = Future(), Future(), Future()
    failed.set_exception(RuntimeError("build failed"))
    outer.set_result(inner)  # a trace that handed its lowered kernel to a compiler pool
    kcache.track(failed)
    kcache.track(outer)
    assert not kcache.wait_for_builds(0.0)  # the compile is still running
    threading.Thread(target=lambda: (gate.wait(), inner.set_result("executable"))).start()
    gate.set()
    assert kcache.wait_for_builds(30.0)
    assert outer not in kcache._BUILDS or kcache._settled(outer)
    kcache.track(Future()).set_result(None)  # finished builds are dropped from the list
    assert all(not kcache._settled(f) for f in kcache._BUILDS[:-1])


def test_precompiled_builds_are_tracked():
    import jax

    from jaxglobal.bnb import _Precompiled

    k = _Precompiled(jax.jit(lambda lo, hi, c: jnp.sum(lo * hi, axis=1) + c), "test")
    k.start(2, 3)
    fut = k.ready[k._key(k._args(2, 3))]
    assert fut in kcache._BUILDS
    assert kcache.wait_for_builds(60.0)
    assert kcache._settled(fut)
