"""Gaussian-process predictions with relaxations that keep the kernel structure.

``gp_mean`` is a primitive: the posterior mean ``mu(x) = k(x)^T alpha``. Relaxed term by term, the
sum of N kernels with signed weights loses their cancellation (``||alpha||_1`` can be many times
``||mu||_H``): the interval stays several times the true range on every box. For kernels
differentiable at 0 (``matern52``, ``se``) the rule intersects that relaxation with a first-order
Taylor model whose remainder is bounded in the RKHS of the kernel. ``mu`` is in it with
``||mu||_H^2 = alpha^T K_f alpha`` (``K_f`` the noise-free training covariance), so for ``c`` the box
center, ``delta = x - c`` and ``s = ||delta / l||^2``,

    |mu(x) - mu(c) - grad mu(c).delta| <= ||mu||_H ||k_x - k_c - D k_c delta||_H = ||mu||_H e(s),
    e(s)^2 = 2 sf2 (1 - phi(s) + 2 s phi'(s) - s phi'(0)),

``phi`` the unit-variance kernel as a function of ``s``. ``e`` is nondecreasing in ``s`` (its
derivative is ``phi'(s) - phi'(0) + 2 s phi''(s) >= 0``, ``phi'`` being nondecreasing and ``phi''
>= 0`` for these kernels), so its box maximum is at the farthest corner, and ``e = O(s) = O(w^2)``:
the model converges to the true range as the box shrinks, and does not involve the conditioning of
``K``. The zeroth-order bound ``|mu(x) - mu(c)| <= ||mu||_H sqrt(2 sf2 (1 - phi(s)))`` holds for
every kernel but is not used: on large boxes it makes the relaxation flat, its minimizer then says
nothing about where ``mu`` is low, and the local solves started from it found the incumbent later
(more nodes in ``benchmarks/gp_mean_ab.py``, docs/PLAN-acquisition-relaxations.md).

``gp_std`` is a primitive too: the posterior standard deviation

    sigma(x) = sqrt(max(v(x), eps)),   v(x) = sf2 - k(x)^T A k(x) = sf2 - ||W^T k(x)||^2,

with ``k_i(x) = sf2 * kernel(sum_j ((x_j - Z_ij) / l_j)**2)``, ``A`` the inverse
of the (noisy) training covariance and ``A = W W^T`` its Cholesky factor.
Relaxed as ``k^T A k``, the quadratic form is a sum of bilinear products, and
any bound through ``A`` inherits its conditioning (``~1/noise``). The rule
combines two relaxations, keeping the max of the convex and the min of the
concave sides:

* ``sqrt(max(sf2 - ||W^T k||^2, eps))``, squares of linear forms in ``k``:
  good from above;
* a Lipschitz bound from the GP itself. The posterior is a Gaussian process,
  so for any ``x`` and the box center ``c``,
  ``|sigma(x) - sigma(c)| <= sd_post(f(x) - f(c)) <= sd_prior(f(x) - f(c))
  = sqrt(2 sf2 (1 - kernel(d(x, c))))`` (the triangle inequality for standard
  deviations; conditioning never increases the variance of a linear
  functional). Over the box this gives ``sigma(c) -+ h`` with ``h`` from the
  farthest corner: first order in the box size, independent of ``A``;
* for ``matern52`` and ``se``, a first-order model (``_std_taylor``): the same
  argument with ``f(c) + grad f(c).delta`` in place of ``f(c)`` gives
  ``|sigma(x) - sqrt(q(delta))| <= e(s)``, ``e`` the prior remainder of the mean's
  model above (without ``||mu||_H``) and ``q`` the posterior variance of
  ``f(c) + grad f(c).delta``, a PSD quadratic in closed form,
  ``q = sf2 + sum_j D_j delta_j^2 - ||w + J_w delta||^2`` (``w = W^T k(c)``,
  ``J_w`` its Jacobian, ``D_j = -2 sf2 phi'(0) / l_j^2``). ``sqrt(q)`` is convex:
  its convex side is a max of tangent planes, its concave side ``sqrt`` of a
  concave overestimator of ``q``. Second order: the interval converges to the
  true range as the box shrinks.

``v >= 0`` in exact arithmetic (``A`` is positive semidefinite); ``eps`` keeps
the square root differentiable.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.extend.core import Primitive
from jax.interpreters import ad, batching, mlir

from .kernels import KERNELS
from .relax import MC, RULES, as_mc, eval_jaxpr_mc, tighten
from .univariate import SMOOTH_IF


def _kvec(x, Z, lengthscale, sf2, kernel):
    d = jnp.sum(((x - Z) / lengthscale) ** 2, axis=-1)
    return sf2 * KERNELS[kernel](d)


def _mean(x, Z, lengthscale, alpha, sf2, rkhs, *, kernel):
    return _kvec(x, Z, lengthscale, sf2, kernel) @ alpha


def _batch_of(fn):
    def batch(args, dims, **p):
        # batched evaluation goes through the formula: relaxations only ever see one point
        axes = tuple(d if isinstance(d, int) else None for d in dims)  # not_mapped -> None
        return jax.vmap(lambda *a: fn(*a, **p), in_axes=axes)(*args), 0

    return batch


gp_mean_p = Primitive("gp_mean")
gp_mean_p.def_impl(_mean)
gp_mean_p.def_abstract_eval(lambda x, Z, lengthscale, alpha, sf2, rkhs, **_: x.update(shape=x.shape[:-1]))
mlir.register_lowering(gp_mean_p, mlir.lower_fun(_mean, multiple_results=False))
# derivatives in x only (the other operands are the trained GP's constants)
ad.defjvp(gp_mean_p, lambda g, x, *rest, **p: jax.jvp(lambda u: _mean(u, *rest, **p), (x,), (g,))[1],
          None, None, None, None, None)  # fmt: skip
batching.primitive_batchers[gp_mean_p] = _batch_of(_mean)


def gp_mean(x, Z, lengthscale, alpha, sf2, kernel="matern52"):
    """Posterior mean ``k(x)^T alpha`` (``alpha = A y``) at one point ``x``.

    ``Z`` (N, n) training inputs, ``lengthscale`` (n,), ``sf2`` the signal variance, ``alpha`` the
    weights (any vector: the RKHS norm the relaxation uses is computed from it).
    """
    f = jnp.asarray
    Z, lengthscale, alpha, sf2 = f(Z, dtype=float), f(lengthscale, dtype=float), f(alpha, dtype=float), f(sf2, dtype=float)
    rkhs = _rkhs(Z, lengthscale, alpha, sf2, kernel)
    return gp_mean_p.bind(f(x, dtype=float), Z, lengthscale, alpha, sf2, rkhs, kernel=kernel)


def _rkhs(Z, lengthscale, alpha, sf2, kernel):
    """``||mu||_H = sqrt(alpha^T K_f alpha)``, ``K_f`` the noise-free training covariance, rounded up."""
    d = jnp.sum(((Z[:, None, :] - Z[None, :, :]) / lengthscale) ** 2, axis=-1)
    q = alpha @ (sf2 * KERNELS[kernel](d)) @ alpha
    # rounding of q is at most ~N eps sf2 ||alpha||_1^2 (|Kf_ij| <= sf2): add a multiple of it
    return jnp.sqrt(jnp.maximum(q, 0.0) + 4 * (alpha.size + 2) * _EPS * sf2 * jnp.sum(jnp.abs(alpha)) ** 2)


_EPS = float(jnp.finfo(jnp.float64).eps)
_SMOOTH = {"matern52", "se"}  # differentiable at 0: grad mu exists and the first-order bound holds


def _relax_formula(fn, args, ctx):
    closed = jax.make_jaxpr(fn)(*(jnp.zeros(a.lo.shape, a.lo.dtype) for a in args))
    (out,) = eval_jaxpr_mc(closed.jaxpr, closed.consts, list(args), ctx)
    return as_mc(out)


def _box(x):
    """Center ``c``, half-widths ``h`` and whether the box of ``x`` is finite."""
    fin = jnp.all(jnp.isfinite(x.lo) & jnp.isfinite(x.hi))
    c = jnp.where(jnp.isfinite(x.lo) & jnp.isfinite(x.hi), 0.5 * (x.lo + x.hi), 0.0)
    return c, jnp.maximum(c - x.lo, x.hi - c), fin


def _remainder(s, sf2, kernel):
    """``e(s)``: the prior sd of the first-order remainder at scaled squared distance ``s``, rounded up."""
    phi, d1 = KERNELS[kernel], jax.grad(KERNELS[kernel])
    dphi0 = d1(jnp.zeros_like(s))
    r2 = 2 * sf2 * (1 - phi(s) + 2 * s * d1(s) - s * dphi0)
    # the closed form cancels to O(s^2): its rounding is a few eps times its terms
    return jnp.sqrt(jnp.maximum(r2, 0.0) + 16 * _EPS * sf2 * (1 + 3 * s * jnp.abs(dphi0)))


def _mean_model(c, h, Z, lengthscale, alpha, sf2, rkhs, kernel):
    """``(m0, g, em)``: ``|mu(c + delta) - m0 - g.delta| <= em`` on the box ``|delta| <= h`` (with rounding)."""
    m0, g = jax.value_and_grad(lambda u: _mean(u, Z, lengthscale, alpha, sf2, rkhs, kernel=kernel))(c)
    e = rkhs * _remainder(jnp.sum((h / lengthscale) ** 2), sf2, kernel)  # the farthest corner
    # rounding of mu anywhere on the box (and of m0), ~N eps sum_i |alpha_i| k_i <= N eps sf2 ||alpha||_1,
    # and of g.(x - c)
    tol = 4 * (alpha.size + c.size + 2) * _EPS * (sf2 * jnp.sum(jnp.abs(alpha)) + jnp.sum(jnp.abs(g) * (jnp.abs(c) + h)))
    return m0, g, e + tol


def _mean_rule(ctx, eqn, x, Z, lengthscale, alpha, sf2, rkhs):
    kernel = eqn.params["kernel"]
    x = as_mc(x)
    base = _relax_formula(lambda u: _mean(u, Z, lengthscale, alpha, sf2, rkhs, kernel=kernel), [x], ctx)
    if kernel not in _SMOOTH or any(isinstance(v, MC) for v in (Z, lengthscale, alpha, sf2, rkhs)):
        return base  # no gradient at the training points, or not a trained GP's constants
    c, h, fin = _box(x)
    m0, g, e = _mean_model(c, h, Z, lengthscale, alpha, sf2, rkhs, kernel)
    span = jnp.sum(jnp.abs(g) * h)
    # mu(c) + g.(x - c) -+ e; g.(x - c) relaxed term by term by the sign of g (x may itself be relaxed)
    lin_cv = m0 + jnp.sum(jnp.where(g >= 0, g * (x.cv - c), g * (x.cc - c)))
    lin_cc = m0 + jnp.sum(jnp.where(g >= 0, g * (x.cc - c), g * (x.cv - c)))
    lo, hi, cv, cc = m0 - span - e, m0 + span + e, lin_cv - e, lin_cc + e
    ok = fin & jnp.isfinite(lo) & jnp.isfinite(hi) & jnp.isfinite(cv) & jnp.isfinite(cc)
    return tighten(MC(jnp.where(ok, jnp.maximum(base.lo, lo), base.lo),
                      jnp.where(ok, jnp.minimum(base.hi, hi), base.hi),
                      jnp.where(ok, jnp.maximum(base.cv, cv), base.cv),
                      jnp.where(ok, jnp.minimum(base.cc, cc), base.cc)))  # fmt: skip


RULES["gp_mean"] = _mean_rule
SMOOTH_IF["gp_mean"] = lambda p: p["kernel"] in _SMOOTH  # alphaBB applies as it did to the formula


def _var(x, Z, lengthscale, W, sf2, kernel):
    return sf2 - jnp.sum((W.T @ _kvec(x, Z, lengthscale, sf2, kernel)) ** 2)


def _std(x, Z, lengthscale, W, sf2, *, kernel, eps):
    return jnp.sqrt(jnp.maximum(_var(x, Z, lengthscale, W, sf2, kernel), eps))


gp_std_p = Primitive("gp_std")
gp_std_p.def_impl(_std)
gp_std_p.def_abstract_eval(lambda x, Z, lengthscale, W, sf2, **_: x.update(shape=x.shape[:-1]))
mlir.register_lowering(gp_std_p, mlir.lower_fun(_std, multiple_results=False))
# derivatives in x only (the other operands are the trained GP's constants)
ad.defjvp(gp_std_p, lambda g, x, Z, lengthscale, W, sf2, **p: jax.jvp(
    lambda u: _std(u, Z, lengthscale, W, sf2, **p), (x,), (g,))[1], None, None, None, None)


batching.primitive_batchers[gp_std_p] = _batch_of(_std)


def gp_std(x, Z, lengthscale, A, sf2, kernel="matern52", eps=1e-12):
    """Posterior standard deviation ``sqrt(max(sf2 - k(x)^T A k(x), eps))`` at one point ``x``.

    ``Z`` (N, n) training inputs, ``lengthscale`` (n,), ``A`` the inverse of the
    training covariance including noise (symmetric positive definite), ``sf2``
    the signal variance.
    """
    A = jnp.asarray(A)
    W = jnp.linalg.cholesky(0.5 * (A + A.T))  # A = W W^T
    f = jnp.asarray
    return gp_std_p.bind(f(x), f(Z), f(lengthscale), W, f(sf2), kernel=kernel, eps=float(eps))


def _std_model(c, h, Z, lengthscale, W, sf2, kernel, eps):
    """The first-order model of sigma on the box ``|delta| <= h``: ``|sigma(c + delta) - sqrt(q(delta))| <= e + tolV``,
    ``q = q0 + 2 b.delta + delta^T C delta``; ``errQ``, ``errG`` bound the rounding of ``q`` and its gradient."""
    N, n = Z.shape[0], c.shape[-1]
    e = _remainder(jnp.sum((h / lengthscale) ** 2), sf2, kernel)  # the farthest corner

    def wk(u):
        return W.T @ _kvec(u, Z, lengthscale, sf2, kernel)

    w, Jw = wk(c), jax.jacfwd(wk)(c)  # (N,), (N, n)
    D = -2 * sf2 * jax.grad(KERNELS[kernel])(jnp.zeros(())) / lengthscale**2  # prior covariance of grad f (diagonal)
    q0, b, C = sf2 - w @ w, -Jw.T @ w, jnp.diag(D) - Jw.T @ Jw
    # rounding: of q (and its gradient) from the cancellation sf2 + D delta^2 - ||w + Jw delta||^2, and of
    # the evaluated sigma itself (sf2 - ||W^T k||^2 with ||W^T k||^2 <= ~sf2)
    big = jnp.abs(w) + jnp.abs(Jw) @ h
    k_err = 16 * (N + n + 2) * _EPS
    errQ = k_err * (sf2 + jnp.sum(D * h**2) + big @ big)
    errG = k_err * (D * h + jnp.abs(Jw).T @ big)
    tolV = jnp.sqrt(k_err * sf2) + jnp.sqrt(eps)  # also sqrt(max(v, eps)) <= sqrt(v) + sqrt(eps)
    return q0, b, C, e, errQ, errG, tolV


def _std_taylor(x, Z, lengthscale, W, sf2, kernel, eps):
    """``(lo, hi, cv, cc)`` of sigma on the box of ``x`` from the first-order model (kernels in ``_SMOOTH``).

    ``|sigma(x) - st(delta)| <= e(s)``, ``st(delta) = sqrt(q(delta))`` the posterior sd of
    ``f(c) + grad f(c).delta``, ``q(delta) = q0 + 2 b.delta + delta^T C delta`` (``C`` PSD): convex.
    cc: ``sqrt`` of a concave overestimator of ``q`` (squares by their secant on the symmetric box,
    cross terms by the bilinear McCormick overestimators: exact at the vertices). cv: the max of
    tangent planes of ``st`` at the center, the clipped minimizer of ``q`` and the face centers.
    Both are affine or ``sqrt`` of a concave function of ``delta``, relaxed by the sign of each
    coefficient when ``x`` is itself relaxed.
    """
    c, h, _ = _box(x)
    n = c.shape[-1]
    q0, b, C, e, errQ, errG, tolV = _std_model(c, h, Z, lengthscale, W, sf2, kernel, eps)
    dcv, dcc = x.cv - c, x.cc - c  # relaxations of delta

    # concave side: q <= q0 + sum_i C_ii h_i^2 + 2 b.delta + sum_{i != j} |C_ij| (h_i h_j - |l_ij(delta)|),
    # l_ij = h_i delta_j - sgn(C_ij) h_j delta_i, and -|l| = min(l, -l), each relaxed from above
    def up(a, k):  # concave overestimator of a * delta_k (elementwise)
        return jnp.where(a >= 0, a * dcc[k], a * dcv[k])

    diag = jnp.sum(jnp.diag(C) * h**2)
    ccq = q0 + diag + jnp.sum(up(2 * b, jnp.arange(n)))
    qmax = q0 + diag + 2 * jnp.abs(b) @ h
    off = 1.0 - jnp.eye(n)
    ai = -jnp.sign(C) * h[None, :]  # coefficient of delta_i in l_ij (that of delta_j is h_i >= 0)
    rows, cols = jnp.arange(n)[:, None], jnp.arange(n)[None, :]
    m = jnp.minimum(h[:, None] * dcc[cols] + up(ai, rows), -h[:, None] * dcv[cols] + up(-ai, rows))
    hh = h[:, None] * h[None, :]
    ccq = ccq + jnp.sum(off * jnp.abs(C) * (hh + m))
    qmax = qmax + jnp.sum(off * jnp.abs(C) * hh)
    cc = jnp.sqrt(jnp.maximum(ccq, 0.0) + errQ) + e + tolV
    hi = jnp.sqrt(jnp.maximum(qmax, 0.0) + errQ) + e + tolV
    # convex side: tangent planes st(p) + g.(delta - p), g = (b + C p) / st(p), where st(p) is not tiny
    eye = jnp.eye(n)
    newton = jnp.clip(-jnp.linalg.solve(C + (1e-12 * jnp.trace(C) / n + 1e-300) * eye, b), -h, h)
    P = jnp.concatenate([jnp.zeros((1, n)), newton[None], jnp.diag(h), -jnp.diag(h)])  # (2n + 2, n)
    st = jnp.sqrt(jnp.maximum(q0 + 2 * P @ b + jnp.einsum("pi,ij,pj->p", P, C, P), 0.0))
    use = st * st >= 1e4 * errQ  # then the computed st and g are accurate to the margin below
    ss = jnp.where(use, st, 1.0)
    G = (b + P @ C) / ss[:, None]
    margin = errQ / ss + ((errG[None, :] / ss[:, None] + jnp.abs(G) * (errQ / ss**2)[:, None]) * 2 * h) @ jnp.ones(n)
    base = st - jnp.sum(G * P, axis=1) - margin - e - tolV
    tan_cv = base + jnp.sum(jnp.where(G >= 0, G * dcv, G * dcc), axis=1)
    tan_lo = base - jnp.abs(G) @ h
    cv = jnp.max(jnp.where(use, tan_cv, 0.0))  # sigma >= 0
    lo = jnp.max(jnp.where(use, tan_lo, 0.0))
    return lo, hi, cv, cc


def _rule(ctx, eqn, x, Z, lengthscale, W, sf2):
    kernel, eps = eqn.params["kernel"], eqn.params["eps"]
    x = as_mc(x)
    sig = _relax_formula(lambda u: _std(u, Z, lengthscale, W, sf2, kernel=kernel, eps=eps), [x], ctx)  # Cholesky form
    # Lipschitz bound: sigma(c) -+ sqrt(2 sf2 (1 - kernel(d_max))), d_max to the farthest corner
    fin = jnp.all(jnp.isfinite(x.lo) & jnp.isfinite(x.hi))
    c = jnp.where(jnp.isfinite(x.lo) & jnp.isfinite(x.hi), 0.5 * (x.lo + x.hi), 0.0)
    half = jnp.maximum(c - x.lo, x.hi - c)
    d_max = jnp.sum((half / lengthscale) ** 2)
    h = jnp.sqrt(jnp.maximum(2 * sf2 * (1 - KERNELS[kernel](d_max)), 0.0))
    sc = jnp.sqrt(jnp.maximum(_var(c, Z, lengthscale, W, sf2, kernel), 0.0))
    lo, hi = sc - h, sc + h + jnp.sqrt(eps)  # sqrt(max(v, eps)) <= sqrt(v) + sqrt(eps)
    cv, cc = lo, hi
    ok = fin & jnp.isfinite(h) & jnp.isfinite(sc)
    if kernel in _SMOOTH and not any(isinstance(v, MC) for v in (Z, lengthscale, W, sf2)):
        tlo, thi, tcv, tcc = _std_taylor(x, Z, lengthscale, W, sf2, kernel, eps)
        tok = fin & jnp.isfinite(tlo) & jnp.isfinite(thi) & jnp.isfinite(tcv) & jnp.isfinite(tcc)
        lo, hi = jnp.where(tok, jnp.maximum(lo, tlo), lo), jnp.where(tok, jnp.minimum(hi, thi), hi)
        cv, cc = jnp.where(tok, jnp.maximum(cv, tcv), cv), jnp.where(tok, jnp.minimum(cc, tcc), cc)
    return tighten(MC(jnp.where(ok, jnp.maximum(sig.lo, lo), sig.lo),
                      jnp.where(ok, jnp.minimum(sig.hi, hi), sig.hi),
                      jnp.where(ok, jnp.maximum(sig.cv, cv), sig.cv),
                      jnp.where(ok, jnp.minimum(sig.cc, cc), sig.cc)))  # fmt: skip


RULES["gp_std"] = _rule


# ---------------------------------------------------------------------------
# Expected improvement
# ---------------------------------------------------------------------------
#
# EI(mu, sigma) = (fmin - mu) Phi(z) + sigma phi(z), z = (fmin - mu) / sigma, is the perspective
# sigma * psi((fmin - mu) / sigma) of psi(z) = z Phi(z) + phi(z), which is convex (psi'' = phi > 0):
# so EI is jointly convex in (mu, sigma), nonincreasing in mu and nondecreasing in sigma
# [schweidtmann2021deterministic]. At sigma -> 0 it tends to max(fmin - mu, 0).

_SIG_MIN = 1e-300  # sigma floor: EI(mu, 0) is the continuous limit max(fmin - mu, 0)


def _ei(mu, sigma, fmin):
    s = jnp.maximum(sigma, _SIG_MIN)
    t = fmin - mu
    z = t / s
    return t * jax.scipy.stats.norm.cdf(z) + s * jax.scipy.stats.norm.pdf(z)


ei_p = Primitive("expected_improvement")
ei_p.def_impl(_ei)
ei_p.def_abstract_eval(lambda mu, sigma, fmin: mu.update(shape=jnp.broadcast_shapes(mu.shape, sigma.shape, fmin.shape)))
mlir.register_lowering(ei_p, mlir.lower_fun(_ei, multiple_results=False))


def _ei_z(mu, sigma, fmin):
    return (fmin - mu) / jnp.maximum(sigma, _SIG_MIN)


ad.defjvp(ei_p,
          lambda g, mu, sigma, fmin: -g * jax.scipy.stats.norm.cdf(_ei_z(mu, sigma, fmin)),
          lambda g, mu, sigma, fmin: g * jnp.where(sigma > _SIG_MIN, jax.scipy.stats.norm.pdf(_ei_z(mu, sigma, fmin)), 0.0),
          lambda g, mu, sigma, fmin: g * jax.scipy.stats.norm.cdf(_ei_z(mu, sigma, fmin)))  # fmt: skip
batching.defbroadcasting(ei_p)


def expected_improvement(mu, sigma, fmin):
    """Expected improvement below ``fmin`` of a Gaussian with mean ``mu`` and std ``sigma`` (elementwise).

    Same value as ``(fmin - mu) Phi(z) + sigma phi(z)``; relaxed as the jointly convex, monotone
    function it is, instead of through ``Phi``, ``phi`` and the division one by one.
    """
    f = jnp.asarray
    return ei_p.bind(f(mu, dtype=float), f(sigma, dtype=float), f(fmin, dtype=float))


def _ei_rule(ctx, eqn, mu, sigma, fmin):
    if isinstance(fmin, MC):  # fmin a variable: relax the formula
        closed = jax.make_jaxpr(_ei)(*(jnp.zeros(jnp.shape(v.lo if isinstance(v, MC) else v)) for v in (mu, sigma, fmin)))
        (out,) = eval_jaxpr_mc(closed.jaxpr, closed.consts, [mu, sigma, fmin], ctx)
        return as_mc(out)
    m, s = as_mc(mu), as_mc(sigma)
    mL, mU, sL, sU = m.lo, m.hi, jnp.maximum(s.lo, 0.0), jnp.maximum(s.hi, 0.0)
    f = lambda a, b: _ei(a, b, fmin)  # noqa: E731
    fLL, fLU, fUL, fUU = f(mL, sL), f(mL, sU), f(mU, sL), f(mU, sU)  # f(mu-corner, sigma-corner)
    # convex, nonincreasing in mu, nondecreasing in sigma
    cv = f(m.cc, jnp.maximum(s.cv, 0.0))
    # concave envelope over the (mu, sigma) box: the triangulation interpolant with the larger
    # center value, = the min of its two planes; every plane is nonincreasing in mu, nondecreasing
    # in sigma, so composing with (mu.cv, sigma.cc) stays concave and above EI
    dm, ds = mU - mL, sU - sL
    am, bs = jnp.where(dm > 0, 1.0 / jnp.where(dm > 0, dm, 1.0), 0.0), jnp.where(ds > 0, 1.0 / jnp.where(ds > 0, ds, 1.0), 0.0)
    pm, ps = m.cv - mL, jnp.maximum(s.cc, 0.0) - sL
    # diagonal LL-UU: triangles (LL, UL, UU) and (LL, LU, UU)
    a1 = jnp.minimum(fLL + (fUL - fLL) * am * pm + (fUU - fUL) * bs * ps,
                     fLL + (fUU - fLU) * am * pm + (fLU - fLL) * bs * ps)
    # diagonal LU-UL: triangles (LL, UL, LU) and (UU, LU, UL)
    a2 = jnp.minimum(fLL + (fUL - fLL) * am * pm + (fLU - fLL) * bs * ps,
                     fUU + (fUU - fLU) * am * (m.cv - mU) + (fUU - fUL) * bs * (jnp.maximum(s.cc, 0.0) - sU))
    cc = jnp.where(fLL + fUU >= fLU + fUL, a1, a2)
    return tighten(MC(fUL, fLU, cv, cc))


RULES["expected_improvement"] = _ei_rule


# ---------------------------------------------------------------------------
# Probability of improvement
# ---------------------------------------------------------------------------
#
# PI(mu, sigma) = Phi((t - mu) / sigma), t the target (fmin - xi). Not convex, but monotone:
# nonincreasing in mu, and in sigma nondecreasing for mu > t, nonincreasing for mu < t. So over a
# sigma interval [sL, sU] it is at most u(mu) = Phi((t - mu) / sL) for mu <= t, Phi((t - mu) / sU)
# for mu > t: a function of mu alone, nonincreasing, concave left of t and convex right of it, whose
# concave envelope composed with mu's convex relaxation bounds PI from above.


def _pi(mu, sigma, t):
    return jax.scipy.stats.norm.cdf((t - mu) / jnp.maximum(sigma, _SIG_MIN))


pi_p = Primitive("probability_of_improvement")
pi_p.def_impl(_pi)
pi_p.def_abstract_eval(lambda mu, sigma, t: mu.update(shape=jnp.broadcast_shapes(mu.shape, sigma.shape, t.shape)))
mlir.register_lowering(pi_p, mlir.lower_fun(_pi, multiple_results=False))


def _pi_dz(mu, sigma, t):  # phi(z) / s and z
    s = jnp.maximum(sigma, _SIG_MIN)
    z = (t - mu) / s
    return jax.scipy.stats.norm.pdf(z) / s, z


ad.defjvp(pi_p,
          lambda g, mu, sigma, t: -g * _pi_dz(mu, sigma, t)[0],
          lambda g, mu, sigma, t: jnp.where(sigma > _SIG_MIN, -g * _pi_dz(mu, sigma, t)[0] * _pi_dz(mu, sigma, t)[1], 0.0),
          lambda g, mu, sigma, t: g * _pi_dz(mu, sigma, t)[0])  # fmt: skip
batching.defbroadcasting(pi_p)


def probability_of_improvement(mu, sigma, target):
    """Probability that a Gaussian with mean ``mu`` and std ``sigma`` falls below ``target`` (elementwise).

    Same value as ``Phi((target - mu) / sigma)``; relaxed from its monotonicity in ``mu`` and
    ``sigma`` instead of through ``Phi`` and the division separately.
    """
    f = jnp.asarray
    return pi_p.bind(f(mu, dtype=float), f(sigma, dtype=float), f(target, dtype=float))


def _phi_cc_env(z, L, U):
    """Concave envelope of Phi on [L, U] at z (Phi convex below 0, concave above)."""
    Phi, phi = jax.scipy.stats.norm.cdf, jax.scipy.stats.norm.pdf
    L, U = jax.lax.stop_gradient(L), jax.lax.stop_gradient(U)

    def h(m):  # tangent at m (concave part) evaluated at L, minus Phi(L): increasing in m >= 0
        return Phi(m) + phi(m) * (L - m) - Phi(L)

    def bisect(_, ab):
        lo_, hi_ = ab
        mid = 0.5 * (lo_ + hi_)
        up = h(mid) >= 0
        return jnp.where(up, lo_, mid), jnp.where(up, mid, hi_)

    a0 = jnp.maximum(L, 0.0)
    _, mstar = jax.lax.fori_loop(0, 60, bisect, (a0, jnp.maximum(U, a0)))  # keep h(mstar) >= 0
    mstar = jax.lax.stop_gradient(mstar)
    tangent = jnp.where(z >= mstar, Phi(z), Phi(mstar) + phi(mstar) * (z - mstar))
    w = U - L
    chord = Phi(L) + jnp.where(w > 0, (Phi(U) - Phi(L)) / jnp.where(w > 0, w, 1.0), 0.0) * (z - L)
    # L >= 0: concave, the envelope is Phi; U <= 0, or no tangent from the concave part above
    # (L, Phi(L)): the chord; else Phi above the tangency point and its tangent below
    return jnp.where(L >= 0, Phi(z), jnp.where((U <= 0) | (h(jnp.maximum(U, 0.0)) < 0), chord, tangent))


def _phi_cv_env(z, L, U):
    """Convex envelope of Phi on [L, U] at z, from Phi(z) = 1 - Phi(-z)."""
    return 1.0 - _phi_cc_env(-z, -U, -L)


def _pi_rule(ctx, eqn, mu, sigma, t):
    if isinstance(t, MC):  # a variable target: the formula's relaxation only
        closed = jax.make_jaxpr(_pi)(*(jnp.zeros(jnp.shape(v.lo if isinstance(v, MC) else v)) for v in (mu, sigma, t)))
        (out,) = eval_jaxpr_mc(closed.jaxpr, closed.consts, [mu, sigma, t], ctx)
        return as_mc(out)
    m, s = as_mc(mu), as_mc(sigma)
    mL, mU = m.lo, m.hi
    sL, sU = jnp.maximum(s.lo, _SIG_MIN), jnp.maximum(s.hi, _SIG_MIN)
    Phi, phi = jax.scipy.stats.norm.cdf, jax.scipy.stats.norm.pdf

    def u(x):  # max of PI over sigma in [sL, sU] at mean x
        return jnp.where(x <= t, Phi((t - x) / sL), Phi((t - x) / sU))

    def lmin(x):  # min of PI over sigma
        return jnp.where(x <= t, Phi((t - x) / sU), Phi((t - x) / sL))

    hi, lo = u(mL), lmin(mU)  # exact: PI is nonincreasing in mu
    uU = u(mU)

    def du(x):  # derivative of u on its concave part (x <= t)
        return -phi((t - x) / sL) / sL

    def h(x):  # tangent of u at x, evaluated at mU, minus u(mU): decreasing on [mL, t], <= 0 at t
        return u(x) + du(x) * (mU - x) - uU

    # concave envelope of u on [mL, mU]:
    #   mU <= t: u is concave there, the envelope is u;
    #   otherwise, if a tangent from the concave part passes above (mU, u(mU)), u up to the
    #   tangency point and its tangent after it (taken at the last bisection point with h >= 0,
    #   so the tangent stays above u on both parts); else the chord from mL.
    a = jax.lax.stop_gradient(mL)
    b = jax.lax.stop_gradient(jnp.minimum(t, mU))
    tangent = (mL < t) & (h(a) >= 0)

    def bisect(_, ab):
        lo_, hi_ = ab
        mid = 0.5 * (lo_ + hi_)
        up = h(mid) >= 0
        return jnp.where(up, mid, lo_), jnp.where(up, hi_, mid)

    mstar, _ = jax.lax.fori_loop(0, 60, bisect, (a, b))
    mstar = jax.lax.stop_gradient(mstar)
    x = m.cv  # the envelope is nonincreasing: compose with mu's convex relaxation
    tan_env = jnp.where(x <= mstar, u(x), u(mstar) + du(mstar) * (x - mstar))
    w = mU - mL
    chord = u(mL) + jnp.where(w > 0, (uU - u(mL)) / jnp.where(w > 0, w, 1.0), 0.0) * (x - mL)
    env = jnp.where(mU <= t, u(x), jnp.where(tangent, tan_env, chord))
    # Phi composed with z = (t - mu) / sigma: the division's McCormick relaxation, then Phi's exact
    # envelopes (Phi is nondecreasing: concave side at z.cc, convex side at z.cv). Not the generic
    # erf rule: composed with z in a range of width ~1e3 (sigma tiny), its convex side can sit
    # ~1e-8 above Phi.
    sp = MC(jnp.maximum(s.lo, _SIG_MIN), jnp.maximum(s.hi, _SIG_MIN), jnp.maximum(s.cv, _SIG_MIN), jnp.maximum(s.cc, _SIG_MIN))
    closed = jax.make_jaxpr(lambda a, b: (t - a) / b)(jnp.zeros(jnp.shape(m.lo)), jnp.zeros(jnp.shape(sp.lo)))
    (z,) = eval_jaxpr_mc(closed.jaxpr, closed.consts, [m, sp], ctx)
    z = as_mc(z)
    zL, zU = z.lo, z.hi
    ccz = _phi_cc_env(jnp.clip(z.cc, zL, zU), zL, zU)
    cvz = _phi_cv_env(jnp.clip(z.cv, zL, zU), zL, zU)
    ok = jnp.isfinite(zL) & jnp.isfinite(zU) & jnp.isfinite(ccz) & jnp.isfinite(cvz)
    cc = jnp.minimum(jnp.minimum(env, hi), jnp.where(ok, ccz, hi))
    cv = jnp.maximum(lo, jnp.where(ok, cvz, lo))
    return tighten(MC(lo, hi, cv, cc))


RULES["probability_of_improvement"] = _pi_rule

