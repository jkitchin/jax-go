"""Compile minimize's 'bound' kernel (batch 16) of one MINLPLib model, with one option changed.

usage: bound_kernel_attribution.py NAME VARIANT
  VARIANT: default | interp (relax_interp.MIN_ROWS = 0) | key=value of _Kernels options, joined by +
  COPTS env: JSON compiler_options for the compile
prints: name,variant,trace_lower_s,hlo_lines,compile_s,run_ms_per_call (results/bound_kernel_*.csv)"""
import sys, time, os, logging
logging.disable(logging.INFO)
os.environ["JAXGLOBAL_CACHE"] = "0"
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", None)
sys.path.insert(0, "/home/user/jax-go")
from jaxglobal import bnb
from jaxglobal.nl import read_nl

name, variant = sys.argv[1], sys.argv[2]
M = read_nl(f"/home/user/jax-go/benchmarks/minlplib/nl/{name}.nl", integers="mask")
n = len(M.lo)
opts = dict(n_fw=20, n_kelley=4, fbbt=True, fbbt_rounds=5, dbbt=True, obbt="root", obbt_rounds=2,
            univariate=True, alphabb=True, multivariate=False, lifted=False, crown=False, feas_tol=1e-6,
            safe=bool(bnb.SAFE_DEFAULT), integer=tuple(int(i) for i in np.flatnonzero(M.integer)),
            quadratic=True, cut_pool=0, exact=False)
from jaxglobal import relax_interp
for kv in variant.split("+"):
    if kv == "default": continue
    if kv == "interp": relax_interp.MIN_ROWS = 0; continue
    k, v = kv.split("=")
    opts[k] = {"False": False, "True": True}.get(v, v)
K = bnb._Kernels(M.f, M.ineq, M.eq, **opts)
lo, hi = np.maximum(M.lo, -1e3), np.minimum(M.hi, 1e3)
L, H = jnp.asarray(np.tile(lo, (16, 1))), jnp.asarray(np.tile(hi, (16, 1)))
t = time.perf_counter()
low = K.bound.jitted.lower(L, H, jnp.inf)
tl = time.perf_counter() - t
hlo = low.as_text().count("\n")
print(f"{name},{variant},{tl:.2f},{hlo},", end="", flush=True)
t = time.perf_counter()
import json; c = low.compile(compiler_options=json.loads(os.environ.get("COPTS", "{}")))
print(f"{time.perf_counter() - t:.2f},", end="", flush=True)
jax.block_until_ready(c(L, H, jnp.inf))
t = time.perf_counter()
for _ in range(20): jax.block_until_ready(c(L, H, jnp.inf))
print(f"{(time.perf_counter() - t) / 20 * 1e3:.2f}ms", flush=True)
