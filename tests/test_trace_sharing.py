"""Trace-time sharing (docs/ANALYSIS-trace.md): process-wide tapes and their analyses, numpy
dependency masks, ``fbbt.batched_init``, and the functions a kernel leaves out not being prepared."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jaxglobal.fbbt as FB
import jaxglobal.lp as LP
from jaxglobal.alphabb import _out_size
from jaxglobal.fbbt_interp import make_propagate, scalarize
from jaxglobal.univariate import _apply_masks, _dep_masks, find_blocks


def _f(x):
    return x[0] ** 2 + jnp.exp(0.3 * x[1]) * x[2] - x[0] * x[1]


def _g(x):
    return jnp.stack([x[0] * x[2] - 1.5, x[1] ** 2 + x[3] ** 2 - 4.0, x[0] + x[1] + x[2] - 3.0])


def _h(x):  # affine
    return jnp.stack([x[0] + 2.0 * x[1] - x[2] - 0.5, 0.5 * x[3] - x[1] + 0.25])


def test_tapes_and_analyses_shared():
    x = np.zeros(4)
    t1, t2 = FB.make_tape(_g, x), FB.make_tape(_g, jnp.zeros(4))
    assert t1 is t2
    assert FB.joint_tape(_f, _g, _h, x) is FB.joint_tape(_f, _g, _h, x)
    assert FB.make_tape(_g, np.zeros(5)) is not t1  # another shape
    assert scalarize(t1, 4) is scalarize(t1, 4)
    assert _dep_masks(t1, 4) is _dep_masks(t1, 4)
    tape = FB.joint_tape(_f, _g, _h, x)
    assert make_propagate(tape, _g, _h, 4, 1e-6, 3) is make_propagate(tape, _g, _h, 4, 1e-6, 3)
    assert make_propagate(tape, _g, _h, 4, 1e-6, 4) is not make_propagate(tape, _g, _h, 4, 1e-6, 3)
    assert LP._as_vec(_g) is LP._as_vec(_g)


def test_tape_closing_over_a_tracer_not_kept():
    seen = []

    def outer(c):
        tape = FB.make_tape(lambda x: x * c, jnp.zeros(3))
        seen.append(FB._has_tracer(tape))
        return c

    jax.vmap(outer)(jnp.arange(2.0))
    assert seen == [True]
    assert not any(FB._has_tracer(t) for t in FB._TAPES.values())


def _masks_jax(tape, n):
    """The former ``_dep_masks``: every rule with eager ``jnp`` operations and the primitives."""
    from jaxglobal.fbbt import Const
    from jaxglobal.univariate import MOVE, RULES

    masks = {0: jnp.eye(n)}
    with jax.ensure_compile_time_eval():
        for node in tape.nodes:
            eqn, name = node.eqn, node.eqn.primitive.name
            var = [not isinstance(a, Const) and a in masks for a in node.ins]
            if not any(var):
                continue
            out_shapes = [tape.shapes[s][0] for s in node.outs]
            ins = [masks[a] if v else (a.val if isinstance(a, Const) else None) for a, v in zip(node.ins, var)]
            try:
                if name in MOVE or name == "dot_general":
                    outs = _apply_masks(eqn, node, ins, var, tape)
                else:
                    assert name in RULES
                    acc = jnp.zeros(out_shapes[0] + (n,))
                    for m, v in zip(ins, var):
                        if v:
                            shp = jnp.shape(m)[:-1]
                            m = jnp.reshape(m, (1,) * (len(out_shapes[0]) - len(shp)) + shp + (n,))
                            acc = acc + jnp.broadcast_to(jnp.abs(m), out_shapes[0] + (n,))
                    outs = [acc] * len(node.outs)
            except Exception:  # noqa: BLE001 -- unknown structure: depends on everything (as _dep_masks)
                total = sum(jnp.sum(jnp.abs(m), axis=tuple(range(m.ndim - 1))) for m, v in zip(ins, var) if v)
                outs = [jnp.broadcast_to(total, s + (n,)) for s in out_shapes]
            for s, m in zip(node.outs, outs):
                masks[s] = jnp.abs(jnp.asarray(m, jnp.float64))
    return masks


def _structured(x):
    a = jnp.concatenate([x[:2] * 2.0, jnp.array([1.0]), x[3:]])  # slice, concatenate with a constant
    b = jnp.reshape(jnp.broadcast_to(x[1], (2, 2)), (4,))  # broadcast_in_dim, reshape
    c = jnp.stack([x[0] * x[3], jnp.sin(x[2]), x[1] + 1.0])  # stack
    d = jnp.squeeze(jnp.transpose(jnp.reshape(x, (2, 2)))[:1, :], 0)  # transpose (a primitive bind)
    return jnp.concatenate([a * b, c, d, jnp.dot(jnp.arange(4.0), x)[None]])


@pytest.mark.parametrize("fn", [_f, _g, _h, _structured])
def test_dep_masks_match_the_jax_rules(fn):
    tape = FB._make_tape(fn, np.zeros(4))  # (a fresh tape: the masks are not memoized on it yet)
    new, old = _dep_masks(tape, 4), _masks_jax(tape, 4)
    assert new.keys() == old.keys()
    for s in old:
        assert np.array_equal(np.asarray(new[s]) > 0, np.asarray(old[s]) > 0), s
    assert [(b.slot, b.dep.tolist()) for b in find_blocks(tape, 4)] == \
        [(b.slot, b.dep.tolist()) for b in find_blocks(FB._make_tape(fn, np.zeros(4)), 4)]  # fmt: skip


def test_out_size_from_the_tape():
    for fn in (_f, _g, _h, _structured, lambda x: jnp.ones(3)):
        tape = FB.make_tape(LP._as_vec(fn), np.zeros(4))
        assert _out_size(tape) == jnp.size(LP._as_vec(fn)(jnp.zeros(4)))
    assert LP._part_sizes(_f, _g, _h, 4) == (1, 3, 2)
    assert LP._n_parts(_f, _g, None, 4) == 4


def _loop(lo, k_max):
    def cond(c):
        return (c[0] < k_max) & ~c[3]

    def body(c):
        k, v, best, _ = c
        v = 0.5 * (v + lo / jnp.maximum(v, 1e-3))
        return k + 1, v, jnp.maximum(best, jnp.sum(v)), jnp.all(jnp.abs(v * v - lo) < 1e-12)

    init = FB.batched_init((0, jnp.ones(3), jnp.array(-jnp.inf), jnp.array(False)), lo)
    return jax.lax.while_loop(cond, body, init)


@pytest.mark.parametrize("on", [True, False])
def test_batched_init_values(on, monkeypatch):
    monkeypatch.setattr(FB, "BATCHED_INIT", on)
    lo = np.array([[1.0, 2.0, 3.0], [4.0, 9.0, 16.0], [0.5, 0.25, 7.0]])
    out = jax.jit(jax.vmap(_loop, in_axes=(0, None)))(lo, 30)
    for i in range(3):
        one = jax.jit(_loop)(lo[i], 30)
        for a, b in zip(out, one):
            assert np.array_equal(np.asarray(a[i]), np.asarray(b))
    # nested vmap and forward-mode through the loop
    two = jax.vmap(jax.vmap(_loop, in_axes=(0, None)), in_axes=(None, 0))(lo, jnp.array([5, 30]))
    assert np.array_equal(np.asarray(two[1][1]), np.asarray(out[1]))
    jax.jacfwd(lambda l: _loop(l, 30)[1])(lo[0])


def test_batched_init_batches_the_body_once(monkeypatch):
    passes = [0]

    @jax.custom_batching.custom_vmap
    def counted(v):
        return v

    @counted.def_vmap
    def _(axis_size, in_batched, v):  # runs once per batching of the loop body
        passes[0] += 1
        return v, in_batched[0]

    def f(lo):
        def body(c):
            k, v, flag = c
            v = counted(v * 0.5 + lo)
            return k + 1, v, flag | jnp.any(v > 3.0)

        init = FB.batched_init((0, jnp.zeros(2), jnp.array(False)), lo)
        return jax.lax.while_loop(lambda c: (c[0] < 10) & ~c[2], body, init)

    out, counts = {}, {}
    for on in (False, True):
        monkeypatch.setattr(FB, "BATCHED_INIT", on)
        passes[0] = 0
        out[on] = jax.vmap(f)(jnp.arange(6.0).reshape(3, 2))
        counts[on] = passes[0]
    assert counts[True] == 1 < counts[False]
    for x, y in zip(out[False], out[True]):
        assert np.array_equal(np.asarray(x), np.asarray(y))


def test_prepare_skips_left_out_functions():
    prepare, parts, _ = LP._parts_fn(_f, _g, _h, True, True)
    lo, hi = -np.ones(4), 2.0 * np.ones(4)
    full = prepare(lo, hi)
    part = prepare(lo, hi, (None, None, 2))
    assert part[2] is None and full[2] is not None
    x = jnp.asarray(0.3 * np.ones(4))
    v_full, _ = parts.bounds(x, lo, hi, full)
    v_part, b = parts.bounds(x, lo, hi, part, (None, None, 2))
    assert np.array_equal(np.asarray(v_full[:4]), np.asarray(v_part[:4])) and b[2] is None


def test_pool_cuts_skip_affine_functions(monkeypatch):
    lo, hi = -np.ones(4), 2.0 * np.ones(4)
    P = np.array([[0.1, 0.2, 0.3, 0.4], [1.5, -0.5, 0.0, 1.0]])
    new = jax.vmap(LP.make_pool_cuts(_f, _g, _h, univariate=True, alphabb=True), in_axes=(0, None, None))(P, lo, hi)
    monkeypatch.setattr(LP, "_nonlinear_of", lambda f, g, h: (lambda n: None))
    old = jax.vmap(LP.make_pool_cuts(_f, _g, _h, univariate=True, alphabb=True), in_axes=(0, None, None))(P, lo, hi)
    (a1, b1), (a0, b0) = new, old
    live = np.r_[np.ones(4, bool), np.zeros(4, bool)]  # f, g: relaxed; h (affine): its affine form
    assert np.array_equal(np.asarray(b1)[:, live], np.asarray(b0)[:, live])
    assert np.array_equal(np.asarray(a1)[:, live], np.asarray(a0)[:, live])
    # the affine parts' cuts are the forms themselves: the relaxed cuts up to rounding
    np.testing.assert_allclose(np.asarray(a1)[:, ~live], np.asarray(a0)[:, ~live], atol=1e-12)
    np.testing.assert_allclose(np.asarray(b1)[:, ~live], np.asarray(b0)[:, ~live], atol=1e-12)
