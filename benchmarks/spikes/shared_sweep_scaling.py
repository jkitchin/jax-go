import sys, time, logging; logging.disable(logging.INFO)
import numpy as np, jax, jax.numpy as jnp; jax.config.update("jax_enable_x64", True)
sys.path.insert(0, "/home/user/jax-go")
from jaxglobal import bound_generic as BG
from jaxglobal.nl import read_nl
name = sys.argv[1]
M = read_nl(f"/home/user/jax-go/benchmarks/minlplib/nl/{name}.nl", integers="mask"); n = len(M.lo)
G = BG.make_generic_bounder(M.f, M.ineq, M.eq, n=n); d = G.data; N = G.g.N
lo = jnp.concatenate([jnp.asarray(np.maximum(M.lo, -1e3)), jnp.zeros(N - n)]); hi = jnp.concatenate([jnp.asarray(np.minimum(M.hi, 1e3)), jnp.zeros(N - n)])
x = 0.5 * (lo + hi); K = int(d.rows); S = d.cval.shape[0]
print(name, "rows", K, "slots", S, "N", N, flush=True)
sw = lambda x, lo, hi, d: BG._sweep(x, lo, hi, d, False, G.g.tier)
one = jax.jit(sw)
batched = jax.jit(jax.vmap(jax.vmap(sw, in_axes=(0, None, None, None)), in_axes=(0, 0, 0, None)))
X = jnp.tile(x, (16, 17, 1)); L = jnp.tile(lo, (16, 1)); H = jnp.tile(hi, (16, 1))
for nm, f, a in (("1 box x 1 point", one, (x, lo, hi, d)), ("16 boxes x 17 points", batched, (X, L, H, d))):
    c = f.lower(*a).compile(); jax.block_until_ready(c(*a))
    t = time.perf_counter(); [jax.block_until_ready(c(*a)) for _ in range(10)]; dt = (time.perf_counter() - t) / 10
    txt = c.as_text()
    print(f"  {nm:22s} {dt*1e3:8.2f} ms per sweep = {dt/K*1e6:8.2f} us per row; HLO copy ops: {txt.count(' copy(')}, dynamic-update-slice: {txt.count('dynamic-update-slice(')}, scatter: {txt.count(' scatter(')}", flush=True)
