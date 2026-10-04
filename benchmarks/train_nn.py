"""Train the peaks tanh MLPs of jaxglobal.applications.nn (deterministic, seeded).

Usage: python benchmarks/train_nn.py [name ...]   (default: all in GRID)
Latin-hypercube samples of peaks() on [-3,3]^2, 70/15/15 split, MSE with Adam
(optax), full-batch, keeping the best-validation weights (early stopping).
Inputs are scaled by 1/3 during training; the scaling is folded into W0.
"""
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.stats import qmc

from jaxglobal.applications import nn

jax.config.update("jax_enable_x64", True)


def train(name, L, H, nsamp, seed=0, steps=15000, lr=1e-2):
    X = qmc.scale(qmc.LatinHypercube(d=2, seed=seed).random(nsamp), -nn.DOMAIN, nn.DOMAIN)
    y = nn.peaks(X[:, 0], X[:, 1])
    perm = np.random.default_rng(seed).permutation(nsamp)
    ntr, nva = int(0.7 * nsamp), int(0.15 * nsamp)
    tr, va, te = perm[:ntr], perm[ntr:ntr + nva], perm[ntr + nva:]
    Xs = jnp.asarray(X / nn.DOMAIN)
    yj = jnp.asarray(y)
    key = jax.random.PRNGKey(seed)
    dims = [2] + [H] * L + [1]
    params = []
    for a, b in zip(dims[:-1], dims[1:]):
        key, k = jax.random.split(key)
        params.append((jax.random.normal(k, (a, b)) * jnp.sqrt(1.0 / a), jnp.zeros(b)))

    def fwd(p, x):
        for W, b in p[:-1]:
            x = jnp.tanh(x @ W + b)
        return (x @ p[-1][0] + p[-1][1])[:, 0]

    mse = lambda p, i: jnp.mean((fwd(p, Xs[i]) - yj[i]) ** 2)
    opt = optax.adam(optax.cosine_decay_schedule(lr, steps))
    st = opt.init(params)

    @jax.jit
    def step(p, st):
        g = jax.grad(mse)(p, tr)
        u, st = opt.update(g, st)
        return optax.apply_updates(p, u), st

    best, bestv = params, np.inf
    for k in range(steps):
        params, st = step(params, st)
        if k % 20 == 0:
            v = float(mse(params, va))
            if v < bestv:
                best, bestv = params, v
    Ws = [np.asarray(W) for W, _ in best]
    bs = [np.asarray(b) for _, b in best]
    Ws[0] = Ws[0] / nn.DOMAIN
    np.savez(nn.DATA / f"{name}.npz", **{f"W{i}": W for i, W in enumerate(Ws)}, **{f"b{i}": b for i, b in enumerate(bs)})
    pred = nn.forward_np(Ws, bs, X)
    return [float(np.mean((pred[i] - y[i]) ** 2)) for i in (tr, va, te)]


if __name__ == "__main__":
    want = sys.argv[1:]
    for name, L, H, ns, _ in nn.GRID:
        if want and name not in want:
            continue
        t = time.time()
        m = train(name, L, H, ns)
        print(f"{name}: MSE train/val/test = {m[0]:.2e} {m[1]:.2e} {m[2]:.2e}  ({time.time()-t:.0f}s)", flush=True)
