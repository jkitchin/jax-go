"""Structural convexity certificates (``convexity.py``): per-rule validity on random boxes.

For every rule: random boxes, some satisfying the rule's sign condition and some
violating it. Wherever the certificate is accepted on a box the combined
relaxation must still be a convex underestimator there (``cv(y) >= cv(x) +
cv'(x)(y - x)`` and ``cv(y) <= f(y)`` at random points, likewise for ``cc``),
and where the condition fails the certificate must be refused. Non-convex
compositions must get no verdict at all.
"""

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxglobal import convexity
from jaxglobal.convexity import certificate_of
from jaxglobal.convexity import prepare as cert_prepare
from jaxglobal.univariate import make_relaxer

jax.config.update("jax_enable_x64", True)

N = 3


def vec(*terms):
    return lambda x: jnp.stack([t(x) for t in terms])


# name -> (function of x (3,), expected verdict per output: "cv", "cc", None, box sampler kind)
RULES = {
    "nsig": (lambda x: 1.0 - 0.2 * x[0] ** 0.3 * x[1] ** 0.4 * x[2] ** 0.2, "cv"),
    "posy_neg": (lambda x: 2.0 * x[0] ** -1.5 * x[1] ** -0.5, "cv"),
    "one_pos": (lambda x: x[0] ** 2.5 / x[1], "cv"),
    "one_pos_int": (lambda x: x[0] ** 3 / (x[1] * x[2]), "cv"),
    "concave_mono": (lambda x: x[0] ** 0.3 * x[1] ** 0.5, "cc"),
    "affine_atoms": (lambda x: (x[0] + 2 * x[1] + 0.5) ** 1.5 * (x[2] + 1.5) ** -0.5, "cv"),
    "norm": (lambda x: jnp.sqrt(0.01 + x[0] ** 2 + (x[1] - x[2]) ** 2 + 3 * x[2] ** 2), "cv"),
    "norm_of_cvx": (lambda x: jnp.sqrt(jnp.exp(x[0]) ** 2 + (x[1] - x[2]) ** 2), "cv"),
    "pcon": (lambda x: (2.0 ** (x[0] + x[1]) + 2.0 ** (x[1] + x[2])) ** 2 - 3.0, "cv"),
    "lse": (lambda x: jnp.log(jnp.exp(x[0]) + 2 * jnp.exp(x[1] - x[2]) + jnp.exp(-x[0])), "cv"),
    "sq_cvx_nn": (lambda x: (jnp.exp(x[0]) + x[1] ** 2) ** 2, "cv"),
    "odd_pow": (lambda x: (x[0] + x[1]) ** 3, "cv"),
    "inv_affine": (lambda x: 1.0 / (x[0] + x[1]), "cv"),
    "inv_concave": (lambda x: 2.0 / jnp.sqrt(x[0]), "cv"),
    "neg_inv": (lambda x: -1.0 / (x[0] + 0.5 * x[2]), "cc"),
    "log_affine": (lambda x: jnp.log(x[0] + x[1]), "cc"),
    "log_concave": (lambda x: jnp.log(jnp.sqrt(x[0]) + x[1]), "cc"),
    "sqrt": (lambda x: jnp.sqrt(x[0]), "cc"),
    "max": (lambda x: jnp.maximum(x[0] ** 2, jnp.exp(x[1])), "cv"),
    "abs": (lambda x: jnp.abs(x[0] - x[1]), "cv"),
    "exp_cvx": (lambda x: jnp.exp(x[0] ** 2 + x[1]), "cv"),
    "exp_dec": (lambda x: 0.5 ** jnp.sqrt(x[0]), "cv"),
    "pow_neg_concave": (lambda x: jnp.sqrt(x[0]) ** -1.5, "cv"),
    "pow_gt1_cvx": (lambda x: (x[0] ** 2 + x[1] ** 2) ** 1.5, "cv"),
    "pow_lt1_ccv": (lambda x: (jnp.sqrt(x[0]) + x[1]) ** 0.5, "cc"),
    "quad_psd": (lambda x: x[0] ** 2 + x[0] * x[1] + x[1] ** 2 + 2 * x[2] ** 2 - x[1] * x[2] + x[0] - 1, "cv"),
    "quad_nsd": (lambda x: -(x[0] - x[1]) * (x[0] - x[1]) - 0.5 * x[2] * x[2], "cc"),
    "sum_mixed": (lambda x: jnp.exp(x[0]) - jnp.log(x[1] + 3) + 4 * x[2] ** 2 - jnp.sqrt(x[2] + 2), "cv"),
    "neg_scaled": (lambda x: -3.0 * jnp.exp(x[0] - x[1]) + jnp.sqrt(x[2] + 2), "cc"),
}

# non-convex compositions: no verdict on any box
NONCONVEX = {
    "sqrt_not_convex": (lambda x: jnp.sqrt(x[0] + 4.0), "cc"),  # concave only
    "bilinear": (lambda x: x[0] * x[1], None),
    "gauss": (lambda x: jnp.exp(-x[0] ** 2), None),
    "sq_minus": (lambda x: (x[0] ** 2 - 1.0) ** 2, None),
    "quad_indef": (lambda x: x[0] ** 2 - x[1] ** 2 + x[0] * x[2], None),
    "sin": (lambda x: jnp.sin(x[0]) + x[1] ** 2, None),
    "mono_mixed": (lambda x: x[0] ** 0.5 * x[1] ** -0.5, None),  # exponents of both signs, sum 0
    "mono_sum_gt1": (lambda x: x[0] ** 0.7 * x[1] ** 0.6, None),  # all >= 0, sum > 1
    "mono_one_pos_lt1": (lambda x: x[0] ** 1.5 * x[1] ** -0.6, None),  # one positive, sum < 1
    "exp_of_concave": (lambda x: jnp.exp(jnp.sqrt(x[0] + 4.0)), None),
    "log_of_convex": (lambda x: jnp.log(x[0] ** 2 + 1.0), None),
}


def boxes(rng, k=24):
    """Random boxes: a third strictly positive, a third crossing zero, a third negative-ish."""
    out = []
    for i in range(k):
        kind = i % 3
        if kind == 0:
            lo = rng.uniform(0.05, 2.0, N)
        elif kind == 1:
            lo = rng.uniform(-2.0, 0.5, N)
        else:
            lo = rng.uniform(-3.0, -0.5, N)
        hi = lo + rng.uniform(0.05, 3.0, N)
        out.append((lo, hi))
    out.append((np.zeros(N), np.ones(N)))  # lo = 0 exactly: strict positivity fails
    return out


def _flags(fn, lo, hi):
    cert = certificate_of(lambda x: jnp.reshape(fn(x), (-1,)), jnp.asarray(lo))
    if cert is None:
        return None, None, None
    ok_cv, ok_cc = cert_prepare(cert, jnp.asarray(lo), jnp.asarray(hi))
    return cert, bool(ok_cv[0]), bool(ok_cc[0])


def _check_relaxation(fn, lo, hi, rng, n_pts=200, safe=False, univariate=False, alphabb=False):
    """cv convex & <= f, cc concave & >= f at random point pairs in the box (where f is finite)."""
    r = make_relaxer(lambda x: jnp.atleast_1d(fn(x)), univariate=univariate, alphabb=alphabb, safe=safe)
    lo, hi = jnp.asarray(lo), jnp.asarray(hi)
    data = r.prepare(lo, hi)

    def cvcc(x):
        m = r(x, lo, hi, data)
        return jnp.reshape(m.cv, ()), jnp.reshape(m.cc, ())

    def tangent_check(x, y):
        (cvx, ccx), (dcv, dcc) = jax.jvp(cvcc, (x,), (y - x,))
        cvy, ccy = cvcc(y)
        fy = jnp.reshape(fn(y), ())
        return cvx, ccx, dcv, dcc, cvy, ccy, fy

    X = lo + rng.random((n_pts, N)) * (hi - lo)
    Y = lo + rng.random((n_pts, N)) * (hi - lo)
    cvx, ccx, dcv, dcc, cvy, ccy, fy = jax.jit(jax.vmap(tangent_check))(X, Y)
    fin = np.isfinite(np.asarray(fy))
    tol = 1e-9 * (1 + np.abs(np.asarray(fy)))
    lin_cv = np.asarray(cvx + dcv)
    lin_cc = np.asarray(ccx + dcc)
    good_cv = np.isfinite(lin_cv) & fin
    good_cc = np.isfinite(lin_cc) & fin
    # NaN means "no information" to every consumer (the base relaxation gives it on e.g. x / [-1, 1])
    fcv = fin & ~np.isnan(np.asarray(cvy))
    fcc = fin & ~np.isnan(np.asarray(ccy))
    assert np.all(np.asarray(cvy)[fcv] <= np.asarray(fy)[fcv] + tol[fcv]), "cv above f"
    assert np.all(np.asarray(ccy)[fcc] >= np.asarray(fy)[fcc] - tol[fcc]), "cc below f"
    # the tangent of cv at x lies below f at y (what the LP cuts rely on), and below cv itself
    assert np.all(lin_cv[good_cv] <= np.asarray(fy)[good_cv] + tol[good_cv]), "tangent of cv above f"
    assert np.all(lin_cc[good_cc] >= np.asarray(fy)[good_cc] - tol[good_cc]), "tangent of cc below f"


@pytest.mark.parametrize("name", sorted(RULES))
def test_rule_verdict_and_validity(name):
    fn, want = RULES[name]
    rng = np.random.default_rng(abs(hash(name)) % 2**32)
    accepted = 0
    for lo, hi in boxes(rng):
        cert, ok_cv, ok_cc = _flags(fn, lo, hi)
        assert cert is not None, f"{name}: no certificate"
        side = ok_cv if want == "cv" else ok_cc
        accepted += side
        # the check that matters: whatever was accepted is a valid relaxation on this box
        _check_relaxation(fn, lo, hi, rng)
    assert accepted > 0, f"{name}: never certified on any box"


def _true_convex_on(fn, lo, hi, rng, side, n=400):
    """Sampled check (test oracle only) that ``fn`` is convex (concave) on the box: midpoint inequality."""
    X = lo + rng.random((n, N)) * (hi - lo)
    Y = lo + rng.random((n, N)) * (hi - lo)
    f = jax.jit(jax.vmap(fn))
    fx, fy, fm = np.asarray(f(X)), np.asarray(f(Y)), np.asarray(f(0.5 * (X + Y)))
    ok = np.isfinite(fx) & np.isfinite(fy) & np.isfinite(fm)
    d = 0.5 * (fx + fy) - fm if side == "cv" else fm - 0.5 * (fx + fy)
    return np.all(d[ok] >= -1e-9 * (1 + np.abs(fm[ok]))) and np.all(ok)


@pytest.mark.parametrize("name", sorted(RULES))
def test_rule_refused_where_condition_fails(name):
    """On every box where the certificate is accepted the function really is convex (concave) there
    (oracle: midpoint inequality at random pairs); on boxes where it is not, nothing is claimed."""
    fn, want = RULES[name]
    rng = np.random.default_rng(1 + abs(hash(name)) % 2**31)
    for lo, hi in boxes(rng, 30):
        _, ok_cv, ok_cc = _flags(fn, lo, hi)
        if ok_cv:
            assert _true_convex_on(fn, lo, hi, rng, "cv"), f"{name}: convex claimed on a box where it is not"
        if ok_cc:
            assert _true_convex_on(fn, lo, hi, rng, "cc"), f"{name}: concave claimed on a box where it is not"


def test_conditions_fail_on_nonpositive_boxes():
    lo, hi = np.array([-1.0, 1.0, 1.0]), np.array([2.0, 2.0, 2.0])
    for name in ("nsig", "posy_neg", "one_pos", "concave_mono", "inv_affine", "log_affine"):
        _, ok_cv, ok_cc = _flags(RULES[name][0], lo, hi)
        assert not ok_cv and not ok_cc, name
    # lo = 0 exactly: strict positivity (monomials) fails, nonnegativity (sqrt) holds
    lo, hi = np.array([0.0, 1.0, 1.0]), np.array([2.0, 2.0, 2.0])
    assert _flags(RULES["nsig"][0], lo, hi)[1] is False
    assert _flags(RULES["sqrt"][0], lo, hi)[2] is True
    # odd power: convex only where x0 + x1 >= 0
    f = RULES["odd_pow"][0]
    # (lo exactly 0 is refused: the interval of a two-term atom carries a rounding margin)
    assert _flags(f, np.array([0.0, 0.0, 0.0]), np.ones(3))[1] is False
    assert _flags(f, np.array([0.5, 0.0, 0.0]), np.ones(3))[1] is True
    assert _flags(f, np.array([-1.0, 0.0, 0.0]), np.ones(3))[1] is False


@pytest.mark.parametrize("name", sorted(NONCONVEX))
def test_nonconvex_gets_no_convex_verdict(name):
    fn, allowed = NONCONVEX[name]
    rng = np.random.default_rng(7)
    for lo, hi in boxes(rng, 12):
        cert, ok_cv, ok_cc = _flags(fn, lo, hi)
        if cert is None:
            continue
        assert not ok_cv or allowed == "cv", name
        assert not ok_cc or allowed == "cc", name
        _check_relaxation(fn, lo, hi, rng)


def test_vector_outputs_and_affine_skipped():
    fn = vec(lambda x: x[0] + 2 * x[1], lambda x: jnp.exp(x[0]), lambda x: x[0] * x[1], lambda x: jnp.sqrt(x[2]))
    cert = certificate_of(fn, jnp.ones(3))
    assert cert.cv_mask.tolist() == [False, True, False, False]
    assert cert.cc_mask.tolist() == [False, False, False, True]


def test_tightens_signomial():
    """The certified relaxation is the function itself: much tighter than McCormick on a product chain."""
    fn = RULES["nsig"][0]
    lo, hi = jnp.ones(N), 10 * jnp.ones(N)
    x = jnp.array([2.0, 3.0, 4.0])
    on = make_relaxer(lambda z: jnp.atleast_1d(fn(z)), univariate=False, alphabb=False)
    off = make_relaxer(lambda z: jnp.atleast_1d(fn(z)), univariate=False, alphabb=False, convex=False)
    v_on = float(on(x, lo, hi, on.prepare(lo, hi)).cv[0])
    v_off = float(off(x, lo, hi, off.prepare(lo, hi)).cv[0])
    assert v_on == pytest.approx(float(fn(x)), rel=1e-12)
    assert v_off < v_on - 1e-3


def test_disabled_switch():
    fn = RULES["nsig"][0]
    convexity.ENABLED = False
    try:
        r = make_relaxer(lambda z: jnp.atleast_1d(fn(z)), univariate=False, alphabb=False)
        lo, hi = jnp.ones(N), 10 * jnp.ones(N)
        assert r.prepare(lo, hi) is None  # the plain McCormick relaxer, unwrapped
    finally:
        convexity.ENABLED = True


@pytest.mark.parametrize("safe", [False, True])
@pytest.mark.parametrize("name", ["nsig", "norm", "pcon", "quad_psd", "lse", "one_pos"])
def test_combined_relaxers(name, safe):
    """With the univariate hulls and alphaBB (the solver's default) and in safe mode."""
    fn, _ = RULES[name]
    rng = np.random.default_rng(3)
    for lo, hi in boxes(rng, 9):
        _check_relaxation(fn, lo, hi, rng, n_pts=100, safe=safe, univariate=True, alphabb=True)


def test_safe_size_guard(monkeypatch):
    """Safe mode keeps the base relaxation on tapes above SAFE_MAX_NODES; plain mode is unaffected."""
    fn = RULES["nsig"][0]
    lo, hi = jnp.ones(N), 10 * jnp.ones(N)
    x = jnp.array([2.0, 3.0, 4.0])
    monkeypatch.setattr(convexity, "SAFE_MAX_NODES", 0)
    kw = dict(univariate=False, alphabb=False)
    on = make_relaxer(lambda z: jnp.atleast_1d(fn(z)), safe=True, **kw)
    off = make_relaxer(lambda z: jnp.atleast_1d(fn(z)), safe=True, convex=False, **kw)
    plain = make_relaxer(lambda z: jnp.atleast_1d(fn(z)), **kw)
    v_on = float(on(x, lo, hi, on.prepare(lo, hi)).cv[0])
    v_off = float(off(x, lo, hi, off.prepare(lo, hi)).cv[0])
    assert v_on == v_off
    assert float(plain(x, lo, hi, plain.prepare(lo, hi)).cv[0]) == pytest.approx(float(fn(x)), rel=1e-12)


def test_under_jit_and_vmap():
    fn = RULES["nsig"][0]
    r = make_relaxer(lambda z: jnp.atleast_1d(fn(z)), univariate=False, alphabb=False)

    @jax.jit
    def cv(lo, hi):
        x = 0.5 * (lo + hi)
        return r(x, lo, hi, r.prepare(lo, hi)).cv[0]

    los = jnp.array([[1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]])
    his = los + 2.0
    v = jax.vmap(cv)(los, his)
    assert float(v[0]) == pytest.approx(float(fn(jnp.array([2.0, 2.0, 2.0]))), rel=1e-12)
    assert float(v[1]) < float(fn(jnp.array([0.0, 2.0, 2.0]))) or not np.isfinite(float(v[1]))


# --- MINLPLib: root-LP cuts at random points of the original box ------------------------------

NL = Path(__file__).resolve().parents[1] / "benchmarks" / "minlplib" / "nl"
INSTANCES = ["cvxnonsep_nsig30", "cvxnonsep_normcon30", "cvxnonsep_pcon40", "cvxnonsep_normcon20r",
             "smallinvDAXr3b010-011", "portfol_buyin", "batch"]  # fmt: skip


# The default run checks three instances in plain mode (a few seconds each). JAXGLOBAL_CONVEXITY_FULL=1
# checks every instance above, plain and safe (minutes each under load).
FULL = os.environ.get("JAXGLOBAL_CONVEXITY_FULL") == "1"
CASES = ([(s, n) for s in (False, True) for n in INSTANCES] if FULL
         else [(False, n) for n in ("cvxnonsep_nsig30", "cvxnonsep_normcon30", "smallinvDAXr3b010-011")])  # fmt: skip


@pytest.mark.parametrize("safe,name", CASES)
def test_minlplib_root_cuts_valid(name, safe):
    from jaxglobal.lp import _cuts, _parts_fn
    from jaxglobal.nl import read_nl

    path = NL / f"{name}.nl"
    if not path.exists():
        pytest.skip("MINLPLib .nl files not present")
    M = read_nl(path, integers="mask")
    lo, hi = jnp.asarray(M.lo, float), jnp.asarray(M.hi, float)
    # cuts on a sub-box must be valid on that sub-box: clip unbounded sides to a finite box
    lo = jnp.where(jnp.isfinite(lo), lo, jnp.where(jnp.isfinite(hi), hi - 100.0, -100.0))
    hi = jnp.where(jnp.isfinite(hi), hi, lo + 100.0)
    n = lo.size
    # univariate base without alphaBB: the convexity wrapper is what is under test (its combination
    # with alphaBB is test_combined_relaxers), and compiling the safe alphaBB prepare on these
    # instances takes many minutes (pre-existing, also with convexity off)
    prepare, parts, rs = _parts_fn(M.f, M.ineq, M.eq, True, False, safe=safe)
    data = jax.jit(prepare)(lo, hi)
    w = hi - lo
    c = 0.5 * (lo + hi)
    rng = np.random.default_rng(0)
    P = jnp.concatenate([c[None], c + jnp.concatenate([jnp.eye(n), -jnp.eye(n)]) * (w / 4),
                         lo + rng.random((8, n)) * w])  # fmt: skip
    const, coef, _ = jax.jit(jax.vmap(lambda p: _cuts(parts, data, p, lo, hi, safe)))(P)

    def true_parts(x):
        out = [jnp.atleast_1d(M.f(x))]
        if M.ineq is not None:
            out.append(jnp.atleast_1d(M.ineq(x)))
        if M.eq is not None:
            h = jnp.atleast_1d(M.eq(x))
            out += [h, -h]
        return jnp.concatenate(out)

    U = rng.random((200, n))
    Y = lo + U * w
    F = np.asarray(jax.jit(jax.vmap(true_parts))(Y))  # (200, K)
    cut = np.asarray(const)[None] + np.einsum("pkn,sn->spk", np.asarray(coef), U)  # (200, P, K)
    ok = np.isfinite(cut)
    viol = np.where(ok, cut - F[:, None, :] - 1e-9 * (1 + np.abs(F[:, None, :])), -np.inf)
    assert viol.max() <= 0, f"{name}: a root cut is above the function by {viol.max():.3g}"
    # and the certificate is in use on the convex rows of these instances
    certs = [convexity.certificate_of(lambda x, fn=fn: jnp.reshape(fn(x), (-1,)), lo) for fn in (M.f, M.ineq, M.eq)
             if fn is not None]  # fmt: skip
    if name.startswith("cvxnonsep") or name.startswith("smallinv"):
        assert any(cc is not None and cc.cv_mask.any() for cc in certs)
