"""Which interval Hessian whole-function alphaBB uses (alphabb.py, ``interpret="auto"``).

The traced Hessian is chosen unless its size (``hessian_size``) says XLA would take
long to compile it, and then the interpreter only if its memory (``taylor2_bytes``)
fits ``TAYLOR_BYTES``. Both are interval enclosures of the same Hessian, so the
choice changes cost, never validity: the alpha, the relaxations and the bounds agree.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxglobal  # noqa: F401  (enables x64)
from jaxglobal import alphabb
from jaxglobal.alphabb import INTERVAL_CTX, make_alphabb
from jaxglobal.fbbt import make_tape
from jaxglobal.problems import CONSTRAINED_PROBLEMS
from jaxglobal.relax_interp import make_taylor2


def dense(x):
    """Every pair of variables coupled: a dense Hessian, written element by element."""
    s = 0.0
    n = x.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            s = s + jnp.sin(x[i] * x[j]) + 0.1 * x[i] * x[j] ** 2
    return jnp.atleast_1d(s)


def boxes(n, k, seed, scale=2.0):
    rng = np.random.default_rng(seed)
    A = np.sort(rng.uniform(-scale, scale, (2, k, n)), 0)
    return jnp.asarray(A[0]), jnp.asarray(A[1])


@pytest.mark.parametrize("n", [10, 16])
def test_taylor2_bytes_matches_xla(n):
    """The estimate is XLA's working memory of the interpreter over a batch of boxes, within 25%."""
    lo = -jnp.ones(n)
    run = make_taylor2(make_tape(dense, lo), n, INTERVAL_CTX, n)
    slots = int(run.program.const_val.size)
    batch = 16
    eye = jnp.eye(n)
    B = jnp.zeros((batch, n))
    c = jax.jit(jax.vmap(lambda lo, hi: run(lo, hi, eye))).lower(B - 1, B + 1).compile()
    temp = c.memory_analysis().temp_size_in_bytes
    est = alphabb.taylor2_bytes(slots, n, batch)
    assert 0.95 * temp <= est <= 1.25 * temp


def test_taylor2_bytes_scaling():
    assert alphabb.taylor2_bytes(1, 0, 1) == 2 * 2 * 8
    assert alphabb.taylor2_bytes(10, 3, 4) == 10 * alphabb.taylor2_bytes(1, 3, 4)
    assert alphabb.taylor2_bytes(10, 3, 4) == 4 * alphabb.taylor2_bytes(10, 3, 1)
    assert alphabb.taylor2_bytes(5, 7) == alphabb.taylor2_bytes(5, 7, alphabb.TAYLOR_BATCH)
    # n = 26 (hadamard_5) with 1,000 slots fits the budget; 5,000 (smallinvDAX's size) does not
    assert alphabb.taylor2_bytes(1000, 26) < alphabb.TAYLOR_BYTES < alphabb.taylor2_bytes(5000, 26)


def test_hessian_size_counts_live_ops():
    n = 6
    lo = jnp.zeros(n)
    closed = jax.make_jaxpr(lambda lo, hi: jax.jacfwd(jax.jacfwd(dense))(lo))(lo, lo)
    ops, volume = alphabb.hessian_size(closed, n)
    assert 0 < ops and 0 < volume <= ops * n * n and volume % (n * n) == 0
    # dead code is not counted
    dead = jax.make_jaxpr(lambda lo, hi: (jnp.sin(hi) * 3, lo + 1.0)[1])(lo, lo)
    assert alphabb.hessian_size(dead, n)[0] == 1


def test_auto_path_small_is_traced():
    prep, _ = make_alphabb(dense)
    assert prep.path(jnp.zeros(4)) == "traced"
    assert make_alphabb(lambda x: jnp.atleast_1d(jnp.abs(x[0])))[0].path(jnp.zeros(2)) is None


def test_auto_path_safe_counts_rounding():
    """Outward rounding multiplies the traced Hessian's operations (n = 5 dense: 16k, which
    XLA compiles in tens of seconds); the interpreter's compile time does not grow."""
    assert make_alphabb(dense, safe=True)[0].path(jnp.zeros(5)) == "interpret"


def test_auto_path_large_is_interpreted(monkeypatch):
    monkeypatch.setattr(alphabb, "HESSIAN_OPS", 0)
    assert make_alphabb(dense)[0].path(jnp.zeros(4)) == "interpret"


def test_auto_path_volume(monkeypatch):
    monkeypatch.setattr(alphabb, "HESSIAN_VOLUME", 0)
    assert make_alphabb(dense)[0].path(jnp.zeros(4)) == "interpret"


def test_auto_path_memory_guard(monkeypatch):
    """Over the memory budget the traced Hessian is kept, however large."""
    monkeypatch.setattr(alphabb, "HESSIAN_OPS", 0)
    monkeypatch.setattr(alphabb, "TAYLOR_BYTES", 1)
    assert make_alphabb(dense)[0].path(jnp.zeros(4)) == "traced"


def test_auto_path_independent_of_min_rows(monkeypatch):
    """The decision is alphaBB's own, not the McCormick interpreter's row threshold."""
    import jaxglobal.relax_interp as RI

    monkeypatch.setattr(RI, "MIN_ROWS", 10**9)
    monkeypatch.setattr(alphabb, "HESSIAN_OPS", 0)
    assert make_alphabb(dense)[0].path(jnp.zeros(4)) == "interpret"
    monkeypatch.setattr(alphabb, "HESSIAN_OPS", 10**9)
    monkeypatch.setattr(RI, "MIN_ROWS", 0)
    assert make_alphabb(dense)[0].path(jnp.zeros(4)) == "traced"


def test_explicit_interpret_is_honored(monkeypatch):
    monkeypatch.setattr(alphabb, "TAYLOR_BYTES", 1)  # the budget is for "auto" only
    assert make_alphabb(dense, interpret=True)[0].path(jnp.zeros(4)) == "interpret"
    monkeypatch.setattr(alphabb, "HESSIAN_OPS", 0)
    assert make_alphabb(dense, interpret=False)[0].path(jnp.zeros(4)) == "traced"


@pytest.mark.parametrize("safe", [False, True])
def test_paths_agree_and_underestimate(monkeypatch, safe):
    """Interpreted and traced Hessians give the same alpha, and the relaxation is below f."""
    n = 5
    L, H = boxes(n, 8, 3)
    trace_prep, apply = make_alphabb(dense, safe=safe, interpret=False)
    assert trace_prep.path(L[0]) == "traced"
    a0, b0, _ = jax.jit(jax.vmap(trace_prep))(L, H)
    monkeypatch.setattr(alphabb, "HESSIAN_OPS", 0)
    interp_prep, apply = make_alphabb(dense, safe=safe)
    assert interp_prep.path(L[0]) == "interpret"
    data = jax.jit(jax.vmap(interp_prep))(L, H)
    a1, b1 = data[0], data[1]
    np.testing.assert_allclose(a1, a0, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(b1, b0, rtol=1e-9, atol=1e-12)
    rng = np.random.default_rng(0)
    X = jnp.asarray(rng.uniform(np.asarray(L), np.asarray(H), (64,) + L.shape))
    f = jax.vmap(jax.vmap(dense))(X)
    cv, cc = jax.vmap(lambda x: jax.vmap(apply)(x, L, H, data))(X)
    assert np.all(cv <= f + 1e-9 * (1 + np.abs(f)))
    assert np.all(cc >= f - 1e-9 * (1 + np.abs(f)))


def _fresh(fn):
    """A new function object, so no kernel compiled for ``fn`` under other thresholds is reused."""
    return None if fn is None else (lambda x: fn(x))


@pytest.mark.parametrize("name", ["G06", "G08", "G24", "G04"])
def test_bound_valid_and_not_weaker(monkeypatch, name):
    """Forcing the interpreted Hessian: the bound stays below the known optimum on boxes
    around it, and is no weaker than the traced Hessian's."""
    import jaxglobal.lp as LP

    p = next(p for p in CONSTRAINED_PROBLEMS if p.name == name)
    lo0, hi0 = jnp.asarray(p.lo), jnp.asarray(p.hi)
    xs = jnp.asarray(p.xstar)
    w = hi0 - lo0
    bxs = [(lo0, hi0)] + [(jnp.maximum(lo0, xs - s * w), jnp.minimum(hi0, xs + s * w)) for s in (0.2, 0.05, 0.01)]

    def bounds():
        cb = jax.jit(LP.make_constrained_bounder(_fresh(p.f), _fresh(p.ineq), _fresh(p.eq),
                                                 univariate=True, alphabb=True))  # fmt: skip
        return np.array([float(cb(lo, hi)[0]) for lo, hi in bxs])

    old = bounds()
    monkeypatch.setattr(alphabb, "HESSIAN_OPS", 0)
    new = bounds()
    tol = 1e-6 * (1 + abs(p.fstar))
    assert np.all(new <= p.fstar + tol)
    assert np.all(new >= old - tol)
