"""jaxglobal: proof-of-concept deterministic global optimization of plain JAX functions."""

import jax

# Global optimization needs double precision for meaningful tolerances.
jax.config.update("jax_enable_x64", True)


# Modules that solver code imports lazily, some of them on background threads (the kernel tracer,
# the domain and local-solver builders): imported here, once, on the importing thread. Two threads
# importing scipy.linalg for the first time at once can raise
# ``_DeadlockError: deadlock detected by _ModuleLock('scipy.linalg.cython_blas')``: three runs of
# the 2026-10-20 MINLPLib re-run (flay03m, ex14_2_1, wastewater05m1) crashed with it.
import scipy.linalg  # noqa: E402,F401
import scipy.optimize  # noqa: E402,F401
import scipy.sparse  # noqa: E402,F401
import jax.scipy.linalg  # noqa: E402,F401

from . import kcache  # noqa: E402

kcache.enable_compilation_cache()  # see kcache.py: compiled kernels are kept on disk across processes

from . import kernels  # noqa: E402  (registers the kernel primitives' rules)
from . import gp  # noqa: E402  (registers gp_std's rule)
from . import export  # noqa: E402  (SCIP / .nl export; pyscipopt is imported on use)
from .bnb import Result, make_bounder, minimize  # noqa: E402
from .relax import MC, interval, relax  # noqa: E402

__all__ = ["MC", "Result", "export", "interval", "make_bounder", "minimize", "relax"]
