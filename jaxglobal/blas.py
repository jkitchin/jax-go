"""Single-threaded OpenBLAS while ``minimize`` runs.

jaxlib's CPU LAPACK kernels (Cholesky, triangular solves: the IPMs of ``lp.py`` and
``quadratic.py``) call SciPy's OpenBLAS, which by default runs every call on as many threads
as there are cores. The node LPs are small (a few hundred columns) and ``minimize`` already
runs several kernel calls, XLA compiles and local solves at once, so those threads only
oversubscribe the CPU. OpenBLAS threads busy-wait for work, and with more runnable threads
than cores a call waits for its spinning helpers to be scheduled: on 4 cores at load ~6, a
316 x 316 Cholesky took 270 ms (median) instead of 0.7 ms single-threaded, and the root bound of
minlp.org m441_kClus_3_10_2_HRc_1 (lifted LP, 16 boxes) 47 s instead of 0.45 s
(docs/ANALYSIS-root-cost.md).

``limited(n)`` sets every loaded OpenBLAS (SciPy's, NumPy's) to ``n`` threads and restores the
previous counts when the last concurrent holder leaves. ``THREADS`` (environment
``JAXGLOBAL_BLAS_THREADS``, default 1) is what ``minimize`` uses; 0 leaves the libraries alone.
Only OpenBLAS is handled (the BLAS of the NumPy / SciPy wheels); other libraries are left as
they are. The thread count of OpenBLAS is process-wide, so other threads of the process see
it too while a solve runs.
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import sys
import threading

THREADS = int(os.environ.get("JAXGLOBAL_BLAS_THREADS", "1") or 0)

# (get, set) symbol pairs of the OpenBLAS builds in circulation (plain, scipy-openblas 32 / 64-bit ints)
_SYMBOLS = (
    ("openblas_get_num_threads", "openblas_set_num_threads"),
    ("scipy_openblas_get_num_threads", "scipy_openblas_set_num_threads"),
    ("scipy_openblas_get_num_threads64_", "scipy_openblas_set_num_threads64_"),
    ("openblas_get_num_threads64_", "openblas_set_num_threads64_"),
)
_LOCK = threading.Lock()
_STATE = {"depth": 0, "saved": []}


def _loaded_paths():
    """Paths of the shared libraries loaded in this process whose name mentions OpenBLAS."""
    paths = set()
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/self/maps") as fh:
                for line in fh:
                    p = line.split()[-1] if line.strip() else ""
                    if "openblas" in os.path.basename(p).lower():
                        paths.add(p)
        except OSError:
            pass
    elif sys.platform == "darwin":
        try:
            dyld = ctypes.CDLL(None)
            dyld._dyld_get_image_name.restype = ctypes.c_char_p
            for i in range(dyld._dyld_image_count()):
                p = (dyld._dyld_get_image_name(i) or b"").decode(errors="replace")
                if "openblas" in os.path.basename(p).lower():
                    paths.add(p)
        except (OSError, AttributeError):
            pass
    return sorted(paths)


def _controls():
    """(get, set) functions of every loaded OpenBLAS."""
    try:  # jaxlib's LAPACK kernels use SciPy's: make sure it is loaded before looking
        import scipy.linalg.cython_lapack  # noqa: F401
    except ImportError:
        pass
    out = []
    for p in _loaded_paths():
        try:
            lib = ctypes.CDLL(p)
        except OSError:
            continue
        for g, s in _SYMBOLS:
            if hasattr(lib, g) and hasattr(lib, s):
                get, set_ = getattr(lib, g), getattr(lib, s)
                get.restype, set_.restype, set_.argtypes = ctypes.c_int, None, [ctypes.c_int]
                out.append((get, set_))
                break
    return out


def thread_counts():
    """Current thread count of each loaded OpenBLAS (for tests and diagnostics)."""
    return [get() for get, _ in _controls()]


@contextlib.contextmanager
def limited(n=None):
    """Run the block with every loaded OpenBLAS at ``n`` threads (default ``THREADS``; 0: no change).

    Nested and concurrent uses share one setting: the first entry saves the counts and sets
    them, the last exit restores them."""
    n = THREADS if n is None else int(n)
    if n <= 0:
        yield
        return
    with _LOCK:
        if _STATE["depth"] == 0:
            saved = []
            for get, set_ in _controls():
                try:
                    saved.append((set_, int(get())))
                    set_(n)
                except Exception:  # noqa: BLE001 -- a library we cannot drive is left alone
                    continue
            _STATE["saved"] = saved
        _STATE["depth"] += 1
    try:
        yield
    finally:
        with _LOCK:
            _STATE["depth"] -= 1
            if _STATE["depth"] == 0:
                for set_, k in _STATE["saved"]:
                    try:
                        set_(k)
                    except Exception:  # noqa: BLE001
                        pass
                _STATE["saved"] = []
