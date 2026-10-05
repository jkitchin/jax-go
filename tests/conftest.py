"""Persistent caches for the test suite under .jax_cache/ (git-ignored).

XLA executables go to .jax_cache/ and jaxglobal's traced kernels to
.jax_cache/jaxglobal/kernels (see jaxglobal/kcache.py). Opt out of both with
JAXGLOBAL_NO_CACHE=1.
"""

import os
import pathlib

import jax

CACHE = pathlib.Path(__file__).resolve().parents[1] / ".jax_cache"

if os.environ.get("JAXGLOBAL_NO_CACHE"):
    os.environ["JAXGLOBAL_CACHE"] = "0"
else:
    os.environ.setdefault("JAXGLOBAL_CACHE_DIR", str(CACHE / "jaxglobal"))
    jax.config.update("jax_compilation_cache_dir", str(CACHE))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
