"""Working-memory estimates of the bounding kernels, and the budget they are held to.

Some kernel paths need memory that grows faster than the model: the node bound and
OBBT linearize every convex part at ``P`` cut points per box (``lp.cut_points``: the
center and two axis points per nonlinear variable, ``P <= 2n + 1``) by forward mode,
``jax.jacfwd`` with ``n`` tangents. On a tape the relaxation interpreter runs
(``relax_interp.make_mc``) the carry of that pass is ``(S, 4 + 2n)`` float64 per point
(``lo, hi, cv, cc`` of every slot and the gradients of ``cv`` and ``cc``), and all
``P`` points of all boxes of a call are one ``vmap``: ``b * P * S * (4 + 2n) * 8`` bytes,
which XLA keeps twice (the loop's input and output buffers). On m34_convex2
(n = 100, S = 1154) and m410_CLay0203HRS (n = 90, S = 1838) that was 14 GB per call
of the 16-box node kernel: the process was OOM-killed (docs/ANALYSIS-memory.md).

``cut_point_bytes`` estimates the per-point, per-box working memory of the cut
generation from the tapes alone (no tracing), and ``point_chunk`` turns the budget
into the number of points evaluated at once: the kernels then map over the points in
chunks (``lax.map(..., batch_size=chunk)``, the same values in the same order) instead
of one ``vmap`` over all of them. Under the budget nothing changes.

The budget, ``BUDGET`` bytes of working memory for the cut generation of one kernel
call, is set from the environment variable ``JAXGLOBAL_MEMORY_BUDGET`` (bytes, or with
a ``K``/``M``/``G`` suffix) when given, else ``min(DEFAULT_BUDGET, MEM_SHARE`` of the
machine's physical memory``)``. The rest of a solve's memory (Python, JAX, XLA's
compilation of the kernels, 1-2.5 GB on these models) comes on top.

The other super-linear path, alphaBB's interpreted interval Hessian (carry
``S * (2 + 2n + 2n^2)`` float64 per box), has its own bytes estimate and budget
(``alphabb.taylor2_bytes``, ``alphabb.TAYLOR_BYTES``): above it the traced Hessian is
used instead. Both decisions are logged on the ``jaxglobal.memory`` logger.
"""

from __future__ import annotations

import logging
import os

import numpy as np

log = logging.getLogger("jaxglobal.memory")

DEFAULT_BUDGET = 1 << 30  # bytes: cut-generation working memory of one kernel call
MEM_SHARE = 0.125  # at most this share of physical memory
BATCH = 16  # boxes per node-kernel call the estimate assumes when not told (minimize's batch_sizes)
COPIES = 2  # XLA keeps the loop carry twice (as ``alphabb.taylor2_bytes``)


def _parse(v):
    v = str(v).strip().upper()
    mult = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}.get(v[-1:], 1)
    return int(float(v[:-1] if mult > 1 else v) * mult)


def _physical():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def budget():
    """The cut-generation budget in bytes (see the module docstring)."""
    env = os.environ.get("JAXGLOBAL_MEMORY_BUDGET")
    if env:
        try:
            return _parse(env)
        except ValueError:
            log.warning("ignoring JAXGLOBAL_MEMORY_BUDGET=%r (not a byte count)", env)
    phys = _physical()
    return DEFAULT_BUDGET if phys is None else int(min(DEFAULT_BUDGET, MEM_SHARE * phys))


BUDGET = budget()


def tape_slots(fn, n):
    """Carry slots of the interpreted relaxation of ``fn`` on ``n`` variables: the interpreter's
    laid-out slot count (``relax_interp.make_mc``) when ``relax_interp`` interprets the tape, 0
    when it does not (straight-line code has no carry: XLA fuses its element-wise rules and
    reuses their buffers; measured below 0.5 GB per 16-box call on every MINLPLib int50 /
    start50 instance whose interpreted estimate would exceed the budget, see
    docs/ANALYSIS-memory.md), None when the tape cannot be read."""
    import jax
    import jax.numpy as jnp

    from .fbbt import _has_tracer, make_tape
    from .fbbt_interp import layout, scalarize
    from .relax_interp import fuse_xlogx, use_relax_interpreter

    try:
        with jax.ensure_compile_time_eval():
            tape = make_tape(lambda x: jnp.atleast_1d(fn(x)), np.zeros(n))
        if _has_tracer(tape) or not use_relax_interpreter(tape, n):
            return 0
        prog = scalarize(tape, n)
        if prog is None or not len(prog.op) or len(tape.outs) != 1:
            return 0
        return int(layout(fuse_xlogx(prog)).prog.const_val.size)
    except Exception:
        return None


def cut_point_bytes(fns, n):
    """Estimated bytes per cut point and box of the forward-mode cut pass over the relaxations of
    ``fns`` (``f``, ``ineq``, ``eq``; None skipped): ``COPIES * S * (4 + 2n) * 8`` summed over the
    interpreted tapes (``tape_slots``), or None when a tape cannot be read."""
    total = 0
    for fn in fns:
        if fn is None:
            continue
        s = tape_slots(fn, n)
        if s is None:
            return None
        total += COPIES * s * (4 + 2 * n) * 8
    return total


def point_chunk(per_point, points, boxes=BATCH, what="cuts", limit=None):
    """Points per chunk so that ``boxes * chunk * per_point <= limit`` (``BUDGET``), or None when
    all ``points`` fit (one ``vmap``, as without a budget). Logs the decision when it chunks."""
    limit = BUDGET if limit is None else limit
    if per_point is None or points <= 1:
        return None
    need = boxes * points * per_point
    if need <= limit:
        return None
    chunk = max(1, int(limit // max(boxes * per_point, 1)))
    if chunk >= points:
        return None
    chunk = -(-points // -(-points // chunk))  # equal chunks: as many as needed, none larger
    log.info("%s: %d points x %d boxes x %.1f MB > budget %.0f MB: %d points at a time",
             what, points, boxes, per_point / 2**20, limit / 2**20, chunk)
    return chunk


def map_points(fn, pts, chunk):
    """``jax.vmap(fn)(pts)``, or the same in chunks of ``chunk`` points (``point_chunk``)."""
    import jax

    if chunk is None or chunk >= pts.shape[0]:
        return jax.vmap(fn)(pts)
    return jax.lax.map(fn, pts, batch_size=int(chunk))
