"""Persistent XLA compilation cache for the test suite (.jax_cache/, git-ignored).

Opt out with JAXGLOBAL_NO_CACHE=1.
"""

import os
import pathlib

import jax

if not os.environ.get("JAXGLOBAL_NO_CACHE"):
    jax.config.update("jax_compilation_cache_dir", str(pathlib.Path(__file__).resolve().parents[1] / ".jax_cache"))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
