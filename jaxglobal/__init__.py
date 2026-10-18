"""jaxglobal: proof-of-concept deterministic global optimization of plain JAX functions."""

import jax

# Global optimization needs double precision for meaningful tolerances.
jax.config.update("jax_enable_x64", True)


from . import kcache  # noqa: E402

kcache.enable_compilation_cache()  # see kcache.py: compiled kernels are kept on disk across processes

from . import kernels  # noqa: E402  (registers the kernel primitives' rules)
from . import gp  # noqa: E402  (registers gp_std's rule)
from . import export  # noqa: E402  (SCIP / .nl export; pyscipopt is imported on use)
from .bnb import Result, make_bounder, minimize  # noqa: E402
from .relax import MC, interval, relax  # noqa: E402

__all__ = ["MC", "Result", "export", "interval", "make_bounder", "minimize", "relax"]
