"""jaxglobal: proof-of-concept deterministic global optimization of plain JAX functions."""

import jax

# Global optimization needs double precision for meaningful tolerances.
jax.config.update("jax_enable_x64", True)

from .bnb import Result, make_bounder, minimize  # noqa: E402
from .relax import MC, interval, relax  # noqa: E402

__all__ = ["MC", "Result", "interval", "make_bounder", "minimize", "relax"]
