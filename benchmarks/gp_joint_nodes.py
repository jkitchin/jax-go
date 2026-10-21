"""Node bounds of LCB / -EI on notebook 06 GPs: separate primitives against GPModel's joint vertex bounds."""
import sys; sys.path.insert(0, "benchmarks")
import numpy as np, jax, jax.numpy as jnp
import gp_mean_ab as ab
from jaxglobal import make_bounder
for case in ("lcb_N50", "ei_N50"):
    P = {m: ab.problem(case, m)["f"] for m in ("primitive", "joint")}
    B = {m: jax.jit(make_bounder(P[m], univariate=True, alphabb=True)) for m in P}
    fv = jax.jit(jax.vmap(P["primitive"]))
    rng = np.random.default_rng(0)
    for w in [2, 0.5, 0.1, 0.02]:
        r = []
        for _ in range(25):
            c = rng.uniform(-3 + w / 2, 3 - w / 2, 2); lo, hi = jnp.asarray(c - w / 2), jnp.asarray(c + w / 2)
            t = float(fv(jnp.asarray(rng.uniform(lo, hi, (3000, 2)))).min())
            r.append([t - float(B[m](lo, hi)[0]) for m in ("primitive", "joint")])
        r = np.array(r)
        print(f"{case} w={w}: median gap separate {np.median(r[:,0]):.2e} joint {np.median(r[:,1]):.2e}; joint tighter by >10% in {np.sum(r[:,1] < 0.9 * r[:,0])}/25")
