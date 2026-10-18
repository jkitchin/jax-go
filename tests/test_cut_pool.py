"""Global cut pool of convex parts (``lp.make_pool_cuts``, ``lp.pool_parts``), the exact
convexity mode (``convexity.exact_certificate``) and the early-stopping FBBT rounds."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxglobal import convexity, fbbt_interp, minimize
from jaxglobal.convexity import certificate_of, exact_certificate
from jaxglobal.lp import make_constrained_bounder, make_pool_cuts, pool_parts
from jaxglobal.univariate import make_relaxer

jax.config.update("jax_enable_x64", True)


def norm_obj(x):  # convex unconditionally
    return jnp.sqrt(0.01 + x[0] ** 2 + (x[1] - x[2]) ** 2 + 3 * x[2] ** 2)


def nsig(x):  # convex where x >= 0
    return jnp.atleast_1d(1.0 - 0.2 * x[0] ** 0.3 * x[1] ** 0.4 * x[2] ** 0.2)


def ball(x):  # convex constraint: |x - c|^2 <= 1
    return jnp.atleast_1d((x[0] - 0.3) ** 2 + (x[1] + 0.2) ** 2 + x[2] ** 2 - 1.0)


def test_exact_certificate_unconditional_and_conditional():
    x = jnp.ones(3)
    c_norm = certificate_of(lambda z: jnp.atleast_1d(norm_obj(z)), x)
    c_nsig = certificate_of(nsig, x)
    assert exact_certificate(c_norm, ("cv",)) is not None
    assert exact_certificate(c_nsig, ("cv",)) is None  # needs x >= 0
    assert exact_certificate(c_nsig, ("cv",), conditional=True) is not None
    assert exact_certificate(c_norm, ("cv", "cc")) is None  # not concave
    mixed = certificate_of(lambda z: jnp.stack([z[0] + z[1], z[0] * z[1]]), x)
    assert exact_certificate(mixed, ("cv",), conditional=True) is None  # a bilinear output
    aff = certificate_of(lambda z: jnp.stack([z[0] + z[1]]), x)
    assert aff is None or exact_certificate(aff, ("cv",)) is None  # nothing to gain


@pytest.mark.parametrize("safe", [False, True])
def test_conditional_exact_relaxer(safe):
    """In conditional exact mode the relaxation is f itself where the conditions hold and -inf
    where they fail (so it stays a valid underestimator on every box)."""
    r = make_relaxer(nsig, univariate=True, alphabb=True, safe=safe, sides=("cv",), conditional=True)
    x = jnp.array([2.0, 3.0, 4.0])
    lo, hi = jnp.ones(3), 10 * jnp.ones(3)
    cv = float(r(x, lo, hi, r.prepare(lo, hi)).cv[0])
    assert cv <= float(nsig(x)[0]) and cv == pytest.approx(float(nsig(x)[0]), rel=1e-12)
    lo2 = jnp.array([-1.0, 1.0, 1.0])
    cv2 = float(r(jnp.array([0.5, 3.0, 4.0]), lo2, hi, r.prepare(lo2, hi)).cv[0])
    assert cv2 == -np.inf


def test_pool_parts_mask_and_exact():
    po = pool_parts(lambda z: norm_obj(z), nsig, ball, 3)
    assert po is not None
    mask, exact = po(jnp.ones(3) * 0.1, jnp.ones(3))
    # parts: cv_f, cv_g (nsig), cv_h (ball), -cc_h (ball is not concave)
    assert mask.tolist() == [True, True, True, False]
    assert exact
    mask, exact = po(-jnp.ones(3), jnp.ones(3))  # nsig's condition fails
    assert mask.tolist() == [True, False, True, False]
    assert not exact
    assert pool_parts(lambda z: z[0] * z[1], lambda z: jnp.atleast_1d(z[0] + z[2]), None, 3) is None


@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize("safe", [False, True])
def test_pool_cuts_minorize_on_sub_boxes(safe, exact):
    """Tangents taken on the root box minorize each certified part everywhere in it."""
    lo, hi = jnp.full(3, 0.5), jnp.full(3, 3.0)
    cuts = make_pool_cuts(lambda z: norm_obj(z), nsig, ball, univariate=True, alphabb=True, safe=safe, exact=exact)
    mask, _ = pool_parts(lambda z: norm_obj(z), nsig, ball, 3)(lo, hi)
    rng = np.random.default_rng(0)
    xs = rng.uniform(lo, hi, (300, 3))
    vals = np.stack([[float(norm_obj(x)), float(nsig(x)[0]), float(ball(x)[0])] for x in xs])
    for _ in range(6):
        p = jnp.asarray(rng.uniform(lo, hi))
        J, b = cuts(p, lo, hi)
        J, b = np.asarray(J), np.asarray(b)
        assert np.all(np.isfinite(b[:3]))
        cut = xs @ J[:3].T + b[:3]
        for k in range(3):
            assert mask[k]
            assert np.all(cut[:, k] <= vals[:, k] + 1e-12 * (1 + np.abs(vals[:, k])))
        # tight at p
        fp = np.array([float(norm_obj(p)), float(nsig(p)[0]), float(ball(p)[0])])
        assert np.allclose(np.asarray(p) @ J[:3].T + b[:3], fp, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("safe", [False, True])
def test_bounder_with_pool(safe):
    """Pool rows tighten the bound to the convex optimum on boxes inside the root box, never
    past it, and are ignored on a box that leaves the root box."""
    f, g = (lambda z: norm_obj(z)), ball
    root = jnp.stack([jnp.full(3, -2.0), jnp.full(3, 2.0)])
    cuts = jax.jit(make_pool_cuts(f, g, None, univariate=True, alphabb=True, safe=safe))
    rng = np.random.default_rng(1)
    rows = []
    for p in rng.uniform(-1.0, 1.0, (24, 3)):
        J, b = cuts(jnp.asarray(p), root[0], root[1])
        rows.append(np.concatenate([np.asarray(J[0]), [float(b[0])], [1.0]]))
        rows.append(np.concatenate([np.asarray(J[1]), [float(b[1])], [0.0]]))
    pool = jnp.asarray(np.stack(rows + [np.r_[np.zeros(3), -np.inf, 0.0]]))
    bound = jax.jit(make_constrained_bounder(f, g, None, univariate=True, alphabb=True, safe=safe))
    # the optimum: min over the ball, found by sampling (an upper bound on f*)
    xs = rng.normal(size=(20000, 3))
    xs = xs / np.linalg.norm(xs, axis=1, keepdims=True) * rng.uniform(0, 1, (20000, 1)) ** (1 / 3)
    xs = xs + np.array([0.3, -0.2, 0.0])
    fstar = min(float(norm_obj(x)) for x in xs[:4000])
    lo, hi = root[0], root[1]
    lb0 = float(bound(lo, hi)[0])
    lb1 = float(bound(lo, hi, jnp.inf, pool, root)[0])
    assert lb1 <= fstar + 1e-9
    assert lb1 >= lb0 - 1e-9
    out = jnp.stack([root[0] - 1.0, root[1]])
    lb2 = float(bound(out[0], out[1], jnp.inf, pool, root)[0])
    lb3 = float(bound(out[0], out[1])[0])
    assert lb2 == pytest.approx(lb3, rel=1e-6, abs=1e-9)  # dead rows only (IPM round-off)


def test_minimize_convex_with_pool():
    """A convex problem with a norm objective and ball constraints solves with and without the pool."""
    f = norm_obj

    def g(x):
        return jnp.stack([(x[0] - 1.0) ** 2 + (x[1] - 1.0) ** 2 + x[2] ** 2 - 1.0, -x[0] - x[1] + 1.2])

    lo, hi = -3 * np.ones(3), 3 * np.ones(3)
    r0 = minimize(f, lo, hi, ineq=g, cut_pool=0, max_iter=2000)
    r1 = minimize(f, lo, hi, ineq=g, max_iter=2000)
    assert r0.status == r1.status == "optimal"
    assert r1.fun == pytest.approx(r0.fun, rel=1e-5, abs=1e-6)
    assert r1.lower_bound <= r0.fun + 1e-5



@pytest.mark.parametrize("verdict", ["on", "off"])
def test_pool_probe_verdicts(monkeypatch, verdict):
    """The probe of the per-child local solves (``bnb.pool_judge``): whichever way it decides
    the search stays exact, and an "off" verdict moves the bound kernel to its zero-row variant
    (no pool rows: a weaker but still valid relaxation)."""
    import jaxglobal.bnb as B

    monkeypatch.setattr(B, "_POOL_PROBE_N", 1)
    monkeypatch.setattr(B, "_POOL_GAIN", -np.inf if verdict == "on" else np.inf)
    rows = []
    call = B._Precompiled.__call__

    def spy(self, *args, **kw):
        if len(args) > 3:
            rows.append(args[3].shape[0])
        return call(self, *args, **kw)

    monkeypatch.setattr(B._Precompiled, "__call__", spy)

    def g(x):
        return jnp.stack([(x[0] - 1.0) ** 2 + (x[1] - 1.0) ** 2 + x[2] ** 2 - 1.0, -x[0] - x[1] + 1.2])

    def f(x):  # a convex norm plus a nonconvex wave: the search needs a few dozen nodes
        return norm_obj(x) + 0.3 * jnp.sin(3 * x[0]) * jnp.cos(3 * x[1])

    lo, hi = -3 * np.ones(3), 3 * np.ones(3)
    r0 = minimize(f, lo, hi, ineq=g, cut_pool=0, max_iter=2000)
    for _ in range(2):  # the zero-row kernel compiles in the background: certainly ready the second time
        rows.clear()
        r = minimize(f, lo, hi, ineq=g, max_iter=2000)
        assert r.status == "optimal"
        assert r.fun == pytest.approx(r0.fun, rel=1e-5, abs=1e-6)
        assert r.lower_bound <= r0.fun + 1e-6
    assert rows[0] > 0  # the probe runs with the pool's rows
    assert (0 in rows) == (verdict == "off")

def test_fbbt_rounds_stop_early():
    """``_rounds`` stops when a round moves nothing, and runs all rounds while bounds move."""
    P0 = jnp.array([[0.0, 1.0], [0.0, 8.0]])
    count = []

    def halve(k, c):
        P, e = c
        return P.at[:, 1].set(P[:, 1] * 0.5), e

    def still(k, c):
        return c

    P, e = jax.jit(lambda P: fbbt_interp._rounds(halve, 5, P))(P0)
    assert float(P[1, 1]) == pytest.approx(8.0 / 32)
    calls = []

    def counted(k, c):
        jax.debug.callback(lambda: calls.append(1))
        return c

    jax.block_until_ready(fbbt_interp._rounds(counted, 5, P0))
    assert len(calls) == 1
    assert not bool(fbbt_interp._progress(P0, P0))
    assert bool(fbbt_interp._progress(P0, P0.at[0, 0].set(0.5)))
    assert bool(fbbt_interp._progress(jnp.array([[-jnp.inf, 1.0]]), jnp.array([[0.0, 1.0]])))
    del count, still


def test_exact_switch_off_keeps_base():
    """``convexity.EXACT = False`` keeps the base relaxation (max(base, f))."""
    fn = lambda z: jnp.atleast_1d(norm_obj(z))  # noqa: E731
    lo, hi = -jnp.ones(3), jnp.ones(3)
    x = jnp.array([0.1, 0.2, -0.3])
    on = make_relaxer(fn, univariate=True, alphabb=True, sides=("cv",))
    m_on = on(x, lo, hi, on.prepare(lo, hi))
    assert not np.isfinite(float(m_on.lo[0]))  # exact: interval bounds not computed
    convexity.EXACT = False
    try:
        off = make_relaxer(fn, univariate=True, alphabb=True, sides=("cv",))
        m_off = off(x, lo, hi, off.prepare(lo, hi))
    finally:
        convexity.EXACT = True
    assert np.isfinite(float(m_off.lo[0]))
    assert float(m_on.cv[0]) == pytest.approx(float(m_off.cv[0]), rel=1e-12)
