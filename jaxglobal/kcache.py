"""On-disk caches that let a new process skip most of the JIT compile tax.

Most of a small solve is spent building the bounding kernels: tracing the
relaxation interpreters over the problem (Python) and compiling the result
(XLA). Two caches under one directory remove both on later runs:

* ``xla/``: JAX's persistent compilation cache (compiled executables);
* ``kernels/``: traced and lowered kernels serialized with ``jax.export``,
  keyed by a fingerprint of the problem, so tracing is skipped as well.

The directory is ``$JAXGLOBAL_CACHE_DIR`` (default ``~/.cache/jaxglobal``);
``JAXGLOBAL_CACHE=0`` disables both. An already configured JAX compilation
cache is left alone.

Both are pruned once per process, oldest first: ``xla/`` beyond
``_MAX_XLA_BYTES`` (by last access where the filesystem records reads, else by
write: APFS does not), ``kernels/`` beyond ``_MAX_KERNELS`` files (a load marks
its file used), and kernels of other jaxglobal sources (which can never load
again) once unused for ``_STALE_SECS``. Only files named like cache entries are
ever removed. ``clear_cache()`` (``python -c "import jaxglobal.kcache as k; k.clear_cache()"``)
removes both.

A problem's fingerprint hashes the jaxprs of ``f``, ``ineq`` and ``eq`` and of
their Hessians (which bring in custom derivative rules), with every constant
and literal, together with the kernel options and shapes, the jaxglobal
source and the JAX version and platform. A function whose jaxpr cannot be
fingerprinted is not cached.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
import time

import jax
import jax.numpy as jnp
import numpy as np

_MAX_KERNELS = 512  # serialized kernels kept on disk (oldest removed first)
# size cap of ``xla/`` (bytes): every compile is cached, hundreds per solve. Pruned once per process,
# down to _XLA_PRUNE_TO of the cap, least recently used first. (JAX's own cap,
# ``jax_compilation_cache_max_size``, rescans the directory on every write: 0.3 s at 9000 files.)
_MAX_XLA_BYTES = 2 * 1024**3
_XLA_PRUNE_TO = 0.75
_STALE_SECS = 86400.0  # kernels of another jaxglobal source are removed once unused this long
_SOURCE = []
# names of the files the caches own (nothing else in their directories is ever removed): JAX's
# entries ``<module>-<sha256>-cache`` (and ``-atime``), jaxglobal's kernels ``[<source>-]<sha256>-...``
# and their temporary files
_XLA_NAME = re.compile(r"^.+-[0-9a-f]{64}-(cache|atime)$")
_KERNEL_NAME = re.compile(r"^(?:[0-9a-f]{12}-)?[0-9a-f]{64}-.*\.(bin|noexport)$")
_TMP_PREFIX = "jaxglobal-tmp-"
_PRUNED = []  # kernels/ was pruned in this process


def cache_root():
    """The cache directory, or None when caching is disabled."""
    if os.environ.get("JAXGLOBAL_CACHE", "1").lower() in ("0", "false", "no", "off"):
        return None
    return os.environ.get("JAXGLOBAL_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "jaxglobal")


def enable_compilation_cache():
    """Point JAX's persistent compilation cache at ``<root>/xla`` unless one is configured."""
    root = cache_root()
    if root is None or jax.config.jax_compilation_cache_dir or not jax.config.jax_enable_compilation_cache:
        return
    path = os.path.join(root, "xla")
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        return
    jax.config.update("jax_compilation_cache_dir", path)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    if jax.config.jax_compilation_cache_max_size == -1:  # not capped by the user
        prune_xla(path)


def prune_xla(path, max_bytes=None):
    """Remove the oldest JAX cache entries of the directory ``path`` (by last access, where the
    filesystem records reads, else by last write) when they take more than ``max_bytes`` (default
    ``_MAX_XLA_BYTES``), down to ``_XLA_PRUNE_TO`` of that. Other files are left alone."""
    max_bytes = _MAX_XLA_BYTES if max_bytes is None else max_bytes
    files = [(max(st.st_atime, st.st_mtime), st.st_size, f) for f, st in _owned(path, _XLA_NAME.match)]
    total = sum(f[1] for f in files)
    if total <= max_bytes:
        return
    for _, size, f in sorted(files):
        if total <= _XLA_PRUNE_TO * max_bytes:
            break
        try:
            os.remove(f)
        except FileNotFoundError:  # another process pruned it: gone all the same
            pass
        except OSError:
            continue
        total -= size


def _owned(d, match):
    """``(path, lstat)`` of the regular files in directory ``d`` whose names ``match`` (none if ``d``
    is missing or a symbolic link)."""
    out = []
    try:
        if os.path.islink(d):
            return out
        with os.scandir(d) as it:
            for e in it:
                if match(e.name) and e.is_file(follow_symlinks=False):
                    try:
                        out.append((e.path, e.stat(follow_symlinks=False)))
                    except OSError:
                        pass
    except OSError:
        pass
    return out


def _is_kernel(name):
    return bool(_KERNEL_NAME.match(name)) or name.startswith(_TMP_PREFIX)


def _source_hash():
    if not _SOURCE:
        h = hashlib.sha256()
        here = os.path.dirname(os.path.abspath(__file__))
        for name in sorted(os.listdir(here)):
            if name.endswith(".py"):
                with open(os.path.join(here, name), "rb") as fh:
                    h.update(name.encode() + fh.read())
        _SOURCE.append(h.hexdigest())
    return _SOURCE[0]


def _hash_value(h, v):
    a = np.asarray(v)
    if a.dtype == object:
        raise TypeError("object constant")
    h.update(f"{a.dtype}{a.shape}".encode())
    h.update(np.ascontiguousarray(a).tobytes())


def _hash_jaxpr(h, jaxpr, consts):
    from jax.extend.core import ClosedJaxpr, Jaxpr, Literal

    h.update(str(jaxpr).encode())
    for c in consts:
        _hash_value(h, c)
    for eqn in jaxpr.eqns:
        for v in eqn.invars:
            if isinstance(v, Literal):
                _hash_value(h, v.val)
        for prm in eqn.params.values():
            for s in prm if isinstance(prm, (tuple, list)) else (prm,):
                if isinstance(s, ClosedJaxpr):
                    _hash_jaxpr(h, s.jaxpr, s.consts)
                elif isinstance(s, Jaxpr):
                    _hash_jaxpr(h, s, ())
                elif isinstance(s, (np.ndarray, jax.Array)):
                    _hash_value(h, s)


def fingerprint(fns, n, extra):
    """Hex digest identifying the kernels of ``fns = (f, ineq, eq)`` on ``n`` variables, or None."""
    try:
        h = hashlib.sha256()
        import jaxlib

        dev = jax.devices()[0]
        h.update(repr((_source_hash(), jax.__version__, jaxlib.__version__, dev.platform, dev.device_kind,
                       bool(jax.config.jax_enable_x64), n, extra)).encode())  # fmt: skip
        x = jax.ShapeDtypeStruct((n,), jnp.float64)
        for fn in fns:
            if fn is None:
                h.update(b"none")
                continue
            for g in (fn, jax.hessian(fn)):
                cj = jax.make_jaxpr(g)(x)
                _hash_jaxpr(h, cj.jaxpr, cj.consts)
        return h.hexdigest()
    except Exception:  # anything unusual in the functions: no disk cache for them
        return None


def _path(key):
    """The file (without extension) of kernel ``key``: named after the jaxglobal source, so that
    kernels of other sources can be told apart (and removed: ``_prune_kernels``)."""
    root = cache_root()
    if root is None or key is None:
        return None
    _prune_kernels(os.path.join(root, "kernels"))
    return os.path.join(root, "kernels", f"{_source_hash()[:12]}-{key}")


def _prune_kernels(d, force=False):
    """Once per process (unless ``force``): remove from the kernel directory ``d`` the files of
    other jaxglobal sources unused for ``_STALE_SECS`` (several checkouts may share one cache), then
    the least recently used beyond ``_MAX_KERNELS``."""
    if _PRUNED and not force:
        return
    _PRUNED.append(True)
    mine, now = _source_hash()[:12] + "-", time.time()
    keep = []
    for path, st in _owned(d, _is_kernel):
        name = os.path.basename(path)
        if name.startswith(mine) or now - st.st_mtime <= _STALE_SECS:
            if not name.startswith(_TMP_PREFIX):  # a recent temporary file: another process's write
                keep.append((st.st_mtime, path))
            continue
        try:
            os.remove(path)
        except OSError:
            pass
    _drop_oldest(keep)


def _drop_oldest(files):
    """Remove the oldest of ``files`` (``(mtime, path)``) beyond ``_MAX_KERNELS``."""
    for _, path in sorted(files)[:-_MAX_KERNELS]:
        try:
            os.remove(path)
        except OSError:
            pass


def clear_cache():
    """Remove jaxglobal's cache entries (in ``xla/`` and ``kernels/`` under ``cache_root()``);
    returns the bytes freed. Other files there, a directory that is a symbolic link, and a JAX
    compilation cache configured elsewhere are left alone."""
    root, freed = cache_root(), 0
    if root is None:
        return 0
    for sub, match in (("xla", _XLA_NAME.match), ("kernels", _is_kernel)):
        for path, st in _owned(os.path.join(root, sub), match):
            try:
                os.remove(path)
                freed += st.st_size
            except OSError:
                pass
    return freed


def load(key):
    """The serialized kernel stored under ``key``: an ``Exported``, ``False`` if it cannot be exported, or None."""
    path = _path(key)
    if path is None:
        return None
    if os.path.exists(path + ".noexport"):
        try:
            os.utime(path + ".noexport")  # recently used: kept by the pruning
        except OSError:
            pass
        return False
    try:
        with open(path + ".bin", "rb") as fh:
            exported = jax.export.deserialize(bytearray(fh.read()))
    except Exception:
        return None
    try:
        os.utime(path + ".bin")  # recently used: kept by the pruning
    except OSError:
        pass
    return exported


def save(key, exported):
    """Store ``exported`` under ``key`` (``None``: remember that this kernel cannot be exported)."""
    path = _path(key)
    if path is None:
        return
    d = os.path.dirname(path)
    try:
        os.makedirs(d, exist_ok=True)
        data, ext = (b"", ".noexport") if exported is None else (exported.serialize(), ".bin")
        fd, tmp = tempfile.mkstemp(dir=d, prefix=_TMP_PREFIX)
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path + ext)
        _drop_oldest([(st.st_mtime, f) for f, st in _owned(d, _KERNEL_NAME.match)])
    except OSError:
        pass

