"""The relaxation tape interpreter (jaxglobal/relax_interp.py) against the traced relaxations."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxglobal  # noqa: F401  (enables x64)
from jaxglobal.alphabb import INTERVAL_CTX, make_alphabb
from jaxglobal.fbbt import make_tape
from jaxglobal.relax import MC, eval_jaxpr_mc, relax
from jaxglobal.relax_interp import make_taylor2
from jaxglobal.univariate import find_blocks, make_relaxer, make_univariate
from test_fbbt_interp import _has_switch, scalar_model


def coupled(x):
    """Univariate pieces (hull blocks) plus coupling terms, written element by element."""
    s = 0.0
    for i in range(3):
        s = s + 0.5 * (x[i] ** 4 - 16 * x[i] ** 2 + 5 * x[i])
    return s + jnp.sin(x[0]) * x[1] + jnp.exp(-x[2] ** 2) * jnp.cos(3 * x[1]) + x[0] / (2.5 + x[2])


def boxes(n, k, seed, scale=2.0):
    rng = np.random.default_rng(seed)
    A = np.sort(rng.uniform(-scale, scale, (2, k, n)), 0)
    return A[0], A[1], rng.uniform(A[0], A[1])


def _close(a, b, rtol=1e-13):
    a, b = np.asarray(a), np.asarray(b)
    assert np.array_equal(np.isfinite(a), np.isfinite(b))
    fin = np.isfinite(a)
    assert np.all(np.abs(a - b)[fin] <= rtol * (1 + np.abs(a)[fin]))


@pytest.mark.parametrize("safe", [False, True])
@pytest.mark.parametrize("which", ["f", "coupled"])
def test_same_values_and_subgradients_as_traced(which, safe):
    """Same rules on the same tape: values and the gradients of cv and cc (forward and reverse
    mode) agree with the traced relax_with up to the rounding of re-associated sums, hull
    envelopes included (``coupled`` has univariate blocks). ``g`` is left out: its ``x[0] * x[0]``
    reads one tape slot twice, which the interpreter relaxes as a square, while the traced
    rule sees two jaxpr variables (two slices) and uses the bilinear product (see below)."""
    f, g, _ = scalar_model()
    fn = {"f": f, "g": g, "coupled": coupled}[which]
    if which == "coupled":
        assert len(find_blocks(make_tape(fn, jnp.zeros(3)), 3)) > 0
    L, H, X = boxes(3, 64, 0)
    out = []
    for interpret in (False, True):
        prep, rw = make_univariate(fn, safe=safe, interpret=interpret)

        def one(x, lo, hi, prep=prep, rw=rw):
            d = prep(lo, hi)
            r = lambda x: rw(x, lo, hi, d)  # noqa: E731
            return r(x), jax.jacfwd(lambda x: r(x).cv)(x), jax.jacrev(lambda x: r(x).cc)(x)

        out.append(jax.jit(jax.vmap(one))(X, L, H))
    (a, ga, gca), (b, gb, gcb) = out
    for u, v in zip(list(a) + [ga, gca], list(b) + [gb, gcb]):
        _close(u, v, 1e-12)


def test_interpreted_mc_is_never_looser():
    """Against ``relax`` and the tape-based relaxer the interpreter is the same or tighter: an
    operation reading one slot twice (``x[0] * x[0]`` after the tape merged the two slices) is
    relaxed as a square."""
    f, g, _ = scalar_model()
    for fn in (f, g, coupled):
        pi, ri = make_univariate(fn, interpret=True)
        L, H, X = boxes(3, 64, 7)
        b = jax.jit(jax.vmap(lambda x, lo, hi: ri(x, lo, hi, pi(lo, hi))))(X, L, H)
        prep, rw = make_univariate(fn, interpret=False)
        for rf in (relax(fn), lambda x, lo, hi: rw(x, lo, hi, prep(lo, hi))):
            a = jax.jit(jax.vmap(rf))(X, L, H)
            tol = 1e-12 * (1 + np.abs(np.asarray(a.cv)))
            assert np.all(np.asarray(b.cv) >= np.asarray(a.cv) - tol)
            assert np.all(np.asarray(b.cc) <= np.asarray(a.cc) + 1e-12 * (1 + np.abs(np.asarray(a.cc))))


def test_relaxation_is_valid_on_sampled_points():
    """cv <= f <= cc at points of the box, and the cut cv(x) + g.(z - x) stays below f."""
    R = make_relaxer(coupled, univariate=True, alphabb=True, interpret=True)
    k = jax.jit(jax.vmap(lambda x, lo, hi: (lambda d: (R(x, lo, hi, d), jax.grad(lambda x: R(x, lo, hi, d).cv)(x)))(
        R.prepare(lo, hi))))  # fmt: skip
    L, H, X = boxes(3, 32, 2)
    m, g = k(X, L, H)
    fx = jax.vmap(coupled)(jnp.asarray(X))
    assert np.all(np.asarray(m.cv) <= np.asarray(fx) + 1e-9) and np.all(np.asarray(fx) <= np.asarray(m.cc) + 1e-9)
    Z = np.random.default_rng(3).uniform(L[:, None], H[:, None], (32, 200, 3))
    fz = np.asarray(jax.vmap(jax.vmap(coupled))(jnp.asarray(Z)))
    cut = np.asarray(m.cv)[:, None] + np.einsum("bj,bkj->bk", np.asarray(g), Z - X[:, None])
    ok = np.isfinite(cut)
    assert np.all((cut <= fz + 1e-8 * (1 + np.abs(fz)))[ok])


@pytest.mark.parametrize("which", ["f", "g", "coupled"])
def test_interval_hessian_matches_traced_and_is_sound(which):
    f, g, _ = scalar_model()
    fn = {"f": f, "g": g, "coupled": coupled}[which]
    flat = lambda x: jnp.reshape(fn(x), (-1,))  # noqa: E731
    tay = make_taylor2(make_tape(flat, jnp.zeros(3)), 3, INTERVAL_CTX, 3)
    ids = jnp.asarray(tay.program.outs[0])

    def new(lo, hi):
        r = tay(lo, hi, jnp.eye(3))
        return r[4][ids], r[5][ids]

    def ref(lo, hi):
        closed = jax.make_jaxpr(jax.jacfwd(jax.jacfwd(flat)))(lo)
        (h,) = eval_jaxpr_mc(closed.jaxpr, closed.consts, [MC(lo, hi, lo, hi)], INTERVAL_CTX)
        h = h if isinstance(h, MC) else MC(h, h, h, h)  # a quadratic's Hessian is a constant
        shp = (ids.size, 3, 3)
        return jnp.broadcast_to(h.lo, shp), jnp.broadcast_to(h.hi, shp)

    L, H, _ = boxes(3, 16, 4)
    a, b = jax.jit(jax.vmap(new))(L, H), jax.jit(jax.vmap(ref))(L, H)
    _close(a[0], b[0], 1e-12)
    _close(a[1], b[1], 1e-12)
    Z = np.random.default_rng(5).uniform(L[:, None], H[:, None], (16, 50, 3))
    hz = np.asarray(jax.vmap(jax.vmap(jax.hessian(flat)))(jnp.asarray(Z)))  # (16, 50, m, 3, 3)
    lo, hi = np.asarray(a[0])[:, None], np.asarray(a[1])[:, None]
    tol = 1e-9 * (1 + np.abs(hz))
    assert np.all((lo <= hz + tol) & (hz <= hi + tol))


def test_alphabb_uses_the_interpreted_hessian():
    abb_on = make_alphabb(coupled, eigen=False, interpret=True)[0]
    abb_off = make_alphabb(coupled, eigen=False, interpret=False)[0]
    L, H, _ = boxes(3, 8, 6)
    a, b = jax.jit(jax.vmap(abb_on))(L, H), jax.jit(jax.vmap(abb_off))(L, H)
    _close(a[0], b[0], 1e-12)
    _close(a[1], b[1], 1e-12)


def test_interpreter_is_used_inside_the_bound_kernel():
    """Like fbbt's: inside jit/vmap the relaxer must not fall back to the traced path."""
    from jaxglobal.bnb import make_bounder

    B = jnp.zeros((4, 3))

    def jaxpr(interpret):
        R = make_relaxer(coupled, univariate=True, alphabb=True, interpret=interpret)
        return jax.make_jaxpr(jax.jit(jax.vmap(lambda lo, hi: jax.value_and_grad(
            lambda x: R(x, lo, hi, R.prepare(lo, hi)).cv)(0.5 * (lo + hi)))))(B, B + 1).jaxpr  # fmt: skip

    assert _has_switch(jaxpr(True)) and not _has_switch(jaxpr(False))
    assert make_bounder(coupled)  # the default bounder builds with the interpreter available
