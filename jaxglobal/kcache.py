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

A problem's fingerprint hashes the jaxprs of ``f``, ``ineq`` and ``eq`` and of
their Hessians (which bring in custom derivative rules), with every constant
and literal, together with the kernel options and shapes, the jaxglobal
source and the JAX version and platform. A function whose jaxpr cannot be
fingerprinted is not cached.
"""

from __future__ import annotations

import hashlib
import os
import tempfile

import jax
import jax.numpy as jnp
import numpy as np

_MAX_KERNELS = 512  # serialized kernels kept on disk (oldest removed first)
_SOURCE = []


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
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.2)


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
    root = cache_root()
    return None if root is None or key is None else os.path.join(root, "kernels", key)


def load(key):
    """The serialized kernel stored under ``key``: an ``Exported``, ``False`` if it cannot be exported, or None."""
    path = _path(key)
    if path is None:
        return None
    if os.path.exists(path + ".noexport"):
        return False
    try:
        with open(path + ".bin", "rb") as fh:
            return jax.export.deserialize(bytearray(fh.read()))
    except Exception:
        return None


def save(key, exported):
    """Store ``exported`` under ``key`` (``None``: remember that this kernel cannot be exported)."""
    path = _path(key)
    if path is None:
        return
    d = os.path.dirname(path)
    try:
        os.makedirs(d, exist_ok=True)
        data, ext = (b"", ".noexport") if exported is None else (exported.serialize(), ".bin")
        fd, tmp = tempfile.mkstemp(dir=d)
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path + ext)
        files = sorted((os.path.join(d, f) for f in os.listdir(d)), key=os.path.getmtime)
        for old in files[:-_MAX_KERNELS]:
            os.remove(old)
    except OSError:
        pass
