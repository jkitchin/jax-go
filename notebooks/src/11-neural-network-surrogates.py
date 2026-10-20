# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 04. Optimizing over a neural-network surrogate
#
# A common workflow: fit a small neural network to simulation or experimental data, then find the input that minimizes the network output.
# To get a *global* answer the network has to be part of a global optimization model.
#
# We train a small tanh MLP (2 inputs, two hidden layers of 8 neurons, 1 output) in JAX, then minimize its output over the box $[-1.5, 1.5]^2$.
#
# **The training data are synthetic**: 400 uniform random points of $y = \sin(3x_1)\cos(2x_2) + 0.3(x_1^2 + x_2^2)$. The network is a stand-in for any fitted surrogate.
#
# Models compared:
# * jaxglobal: the trained network's own forward function, with and without `crown=True` (CROWN backward linear bounds as an extra lower bound);
# * Pyomo reduced space: the network written as one nested expression in the 2 input variables;
# * Pyomo full space: one variable per pre-activation and activation, one constraint per neuron;
# * OMLT (the Pyomo ecosystem's package for this) [ceccon2022omlt] in both formulations.

# %%
import numpy as np
import pyomo.environ as pyo

from nbutil import loc, model_size, save_results, solve_jax, solve_pyomo, table  # imports jaxglobal first (x64 enabled)

import jax
import jax.numpy as jnp
import optax


def target(X):
    return jnp.sin(3 * X[:, 0]) * jnp.cos(2 * X[:, 1]) + 0.3 * (X[:, 0] ** 2 + X[:, 1] ** 2)


sizes = [2, 8, 8, 1]
key = jax.random.PRNGKey(0)
params = []
for i in range(len(sizes) - 1):
    key, k = jax.random.split(key)
    params.append((jax.random.normal(k, (sizes[i], sizes[i + 1])) / np.sqrt(sizes[i]), jnp.zeros(sizes[i + 1])))


def net(params, x):
    for W, b in params[:-1]:
        x = jnp.tanh(x @ W + b)
    W, b = params[-1]
    return (x @ W + b)[..., 0]


Xtr = jax.random.uniform(jax.random.PRNGKey(1), (400, 2), minval=-1.5, maxval=1.5)
ytr = target(Xtr)
loss = lambda p: jnp.mean((net(p, Xtr) - ytr) ** 2)
opt = optax.adam(1e-2)
state = opt.init(params)


@jax.jit
def train_step(p, s):
    l, g = jax.value_and_grad(loss)(p)
    u, s = opt.update(g, s)
    return optax.apply_updates(p, u), s, l


for i in range(3000):
    params, state, l = train_step(params, state)
P = [(np.asarray(W, dtype=float), np.asarray(b, dtype=float)) for W, b in params]
LO, HI = -1.5, 1.5
print(f"training MSE {float(l):.2e}; parameters: {sum(W.size + b.size for W, b in P)}")

# %% [markdown]
# **A note on reproducibility.** The seeds above are fixed, and with a given JAX version on a given machine the training is deterministic: two runs give bit-identical weights.
# The fixed seed does not make it reproducible across JAX versions, however. The seed fixes the random numbers (the initial weights and the 400 training points), but 3000 Adam steps are sensitive to the last bit of every operation, and XLA's CPU kernels (fusion, summation order, the `tanh` implementation) change between releases and can differ between CPUs.
# The outputs stored here were produced with an earlier JAX than the one this project now requires (`jax>=0.11.2,<0.12`). Re-running the notebook with JAX 0.11.2 trains a slightly different network (training MSE 1.19e-03 instead of 1.20e-03), whose minimum is near −0.9329 rather than −0.9231, so every number below changes slightly. The comparisons and conclusions do not.
# Bit-for-bit reproducibility would require saving the trained weights to a file and loading them instead of retraining. We have not done that, so the stored outputs are tied to the JAX version that produced them.

# %% [markdown]
# ## jaxglobal model
#
# The model is the network's forward function. Nothing is rewritten; the weights are closed-over constants.

# %%
def surrogate(x):
    return net(P, x)


# %%
rows_jax = [solve_jax(surrogate, [LO, LO], [HI, HI])]
rows_jax[0]["solver"] = "jaxglobal"
rows_jax.append(solve_jax(surrogate, [LO, LO], [HI, HI], crown=True))
rows_jax[1]["solver"] = "jaxglobal crown"
print("nodes:", rows_jax[0]["nodes"], "(default),", rows_jax[1]["nodes"], "(crown)")
table(rows_jax)

# %% [markdown]
# ## Pyomo models, written by hand
#
# `pyo.tanh` is the obvious choice for the activation. It does not work with the two solvers that return certified bounds here, as the next cells show.
# We therefore also write tanh as $1 - 2/(e^{2z}+1)$, which every solver here accepts.

# %%
def tanh_exp(z):
    return 1 - 2 / (pyo.exp(2 * z) + 1)


def reduced(tanh=pyo.tanh):
    m = pyo.ConcreteModel()
    m.x = pyo.Var(range(2), bounds=(LO, HI), initialize=0.0)
    a = [m.x[0], m.x[1]]
    for W, b in P[:-1]:
        a = [tanh(sum(W[i, j] * a[i] for i in range(W.shape[0])) + b[j]) for j in range(W.shape[1])]
    W, b = P[-1]
    m.obj = pyo.Objective(expr=sum(W[i, 0] * a[i] for i in range(W.shape[0])) + b[0])
    return m


def full(tanh=pyo.tanh):
    m = pyo.ConcreteModel()
    m.x = pyo.Var(range(2), bounds=(LO, HI), initialize=0.0)
    a = [m.x[0], m.x[1]]
    m.layers = pyo.ConstraintList()
    for l, (W, b) in enumerate(P[:-1]):
        z = pyo.Var(range(W.shape[1]), initialize=0.0)
        ao = pyo.Var(range(W.shape[1]), bounds=(-1, 1), initialize=0.0)
        setattr(m, f"z{l}", z)
        setattr(m, f"a{l}", ao)
        for j in range(W.shape[1]):
            m.layers.add(z[j] == sum(W[i, j] * a[i] for i in range(W.shape[0])) + b[j])
            m.layers.add(ao[j] == tanh(z[j]))
        a = [ao[j] for j in range(W.shape[1])]
    W, b = P[-1]
    m.obj = pyo.Objective(expr=sum(W[i, 0] * a[i] for i in range(W.shape[0])) + b[0])
    return m


size_red, size_full = model_size(reduced()), model_size(full())
print(f"reduced space: {size_red[0]} variables, {size_red[1]} constraints")
print(f"full space:    {size_full[0]} variables, {size_full[1]} constraints")

# %% [markdown]
# ### Native `tanh`
#
# Each solver gets the same limit (30 s here so the notebook stays fast; the cap for all solvers is at most 60 s).

# %%
TL = 30
rows_tanh = []
for name, builder in (("reduced", reduced), ("full", full)):
    for s in ("scip", "couenne", "baron"):
        r = solve_pyomo(builder(pyo.tanh), s, tlim=TL)
        r["solver"] = f"{s} ({name}, tanh)"
        rows_tanh.append(r)
table(rows_tanh)

# %% [markdown]
# SCIP's `.nl` reader rejects `tanh`, BARON (through GAMS) returns no solution, and Couenne accepts it but does not close the gap in the time limit (its status is a limit stop, and through `.nl` it reports no bound).
# This is a solver limitation, not a modelling error.
#
# ### Exponential form of tanh

# %%
rows_exp = []
for name, builder in (("reduced", reduced), ("full", full)):
    for s in ("scip", "couenne", "baron"):
        r = solve_pyomo(builder(tanh_exp), s, tlim=TL)
        r["solver"] = f"{s} ({name}, exp)"
        rows_exp.append(r)
table(rows_exp)

# %% [markdown]
# A status of `locallyOptimal` (BARON, if it appears) means BARON stopped without proving the gap, so its bound should not be read as certified.
#
# SCIP's full-space objective (−0.923114) and BARON's (−0.923092) are slightly *below* the certified optimum −0.923084 found by every reduced-space solve.
# They do not contradict jaxglobal's proven bound (−0.923168), and they are not better points.
# A full-space solver reports the objective in its own activation variables, which satisfy the 32 neuron equations only to its feasibility tolerance (1e-6), and those small violations, carried through the layers, shift the output by a few 1e-5.
# Evaluating the network itself at the solver's `x` removes the difference: on a re-run, SCIP's full-space objective was 3e-5 below `net(x)` at its own point, with a largest equation violation of 9.8e-7.
# The reduced-space models have no such equations and report the network's own value.

# %% [markdown]
# ## OMLT
#
# [OMLT](https://github.com/cog-imperial/OMLT) builds these formulations from a network description, so the layer equations are not hand-written.
# It is installed in this environment. We load the same weights through `NetworkDefinition`.

# %%
from omlt import OmltBlock
from omlt.neuralnet import FullSpaceSmoothNNFormulation, NetworkDefinition, ReducedSpaceSmoothNNFormulation
from omlt.neuralnet.layer import DenseLayer, InputLayer


def omlt_model(formulation_cls):
    nd = NetworkDefinition(scaled_input_bounds={0: (LO, HI), 1: (LO, HI)})
    prev = InputLayer([2])
    nd.add_layer(prev)
    for k, (W, b) in enumerate(P):
        act = "tanh" if k < len(P) - 1 else "linear"
        layer = DenseLayer([W.shape[0]], [W.shape[1]], activation=act, weights=W, biases=b)
        nd.add_layer(layer)
        nd.add_edge(prev, layer)
        prev = layer
    m = pyo.ConcreteModel()
    m.nn = OmltBlock()
    m.nn.build_formulation(formulation_cls(nd))
    m.obj = pyo.Objective(expr=m.nn.outputs[0])
    return m


rows_omlt = []
for cls in (FullSpaceSmoothNNFormulation, ReducedSpaceSmoothNNFormulation):
    sz = model_size(omlt_model(cls))
    print(f"{cls.__name__}: {sz[0]} variables, {sz[1]} constraints")
    for s in ("scip", "baron", "couenne"):
        r = solve_pyomo(omlt_model(cls), s, tlim=TL)
        r["solver"] = f"{s} ({cls.__name__.replace("NNFormulation", "")})"
        rows_omlt.append(r)
table(rows_omlt)

# %% [markdown]
# OMLT emits `tanh` through Pyomo, so it inherits the same solver limits as the native-`tanh` models above.
# Its value is convenience (and support for ReLU big-M formulations, which need a MIP solver and are outside what jaxglobal does).

# %%
keep = lambda tag: [r for r in rows_exp if tag in r["solver"] and not r["solver"].startswith("couenne")]
lp, lj = loc(reduced, tanh_exp), loc(surrogate)
lp_full = loc(full, tanh_exp)
save_results("11", "tanh MLP 2-8-8-1 (reduced space)", lp, lj, size_red, rows_jax + keep("reduced"), note="exp-form tanh for Pyomo; Couenne did not converge in the limit")
save_results("11", "tanh MLP 2-8-8-1 (full space)", lp_full, lj, size_full, rows_jax + keep("full"), note="exp-form tanh for Pyomo; Couenne did not converge in the limit")
print(f"lines of model code: Pyomo reduced {lp}, Pyomo full {lp_full}, jaxglobal {lj} (the network function `net` is shared with training and counted separately: {loc(net)})")

# %% [markdown]
# ## What was hard
#
# **In Pyomo**
#
# * Choosing the formulation. Reduced space has 2 variables but a deeply nested expression; full space has 34 variables and 32 constraints, with the nonconvexity pushed into simple univariate equations that solvers handle better.
# * The activation function. A solver that does not know `tanh` (SCIP, BARON here) needs a rewritten activation, and you only find this out by running it.
# * Every weight is re-entered as a Pyomo coefficient. A change of architecture means rewriting the builder (or using OMLT).
# * Solver behaviour differs a lot on the same model (see the exp-form table): BARON proves the reduced-space model in a few seconds, SCIP is better on the full-space form and may hit the time limit on the reduced one, and
#   Couenne did not converge on either in the limit. BARON on the full-space form stopped as `locallyOptimal` with a very weak bound.
# * Strengths: where a solver does converge it gives a certified bound, and the full-space form carries into a MINLP (for example, choosing among discrete designs together with the surrogate).
#
# **In jaxglobal**
#
# * The model is the same `net` function used for training, so there is no transcription and no chance of a mismatch between the trained network and the optimization model.
# * `tanh` is a supported primitive. `crown=True` adds backward linear bounds through the layers, which tightens the bound for composed layers; on this network it reduced the node count (see above), but it is not guaranteed to help on every network.
# * Solve times are not better than the Pyomo solvers on this size of network once compile time is included, and the wall-clock figures come from a loaded machine.
# * Larger networks are the scaling concern: the number of relaxation operations grows with the number of neurons, and the dependency problem across layers weakens the bound.
