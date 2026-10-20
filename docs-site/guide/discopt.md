# jaxglobal and discopt

[discopt](https://doi.org/10.5281/zenodo.19762815) is the author's other MINLP
solver, and the more mature of the two. They solve the same class of problems
with different designs. This list compares them as of October 2026; the
discopt facts come from discopt's own README and development notes.

| | jaxglobal | discopt |
|---|---|---|
| **Model input** | a plain JAX function, traced to a jaxpr, or an `.nl` file. `jnp.where` logic, neural networks, GP surrogates and fixed-step ODE integrators (Python loops or `lax.scan`) get certified bounds. | an algebraic modeling API (`Model`, `m.continuous`, `dm.exp`, ...), or `.nl`, `.gms` and QPLIB files. Opaque Python callables (`dm.custom`, `dm.external`) are solved without a dual bound. |
| **Implementation** | pure Python and JAX: relaxations, the LP interior-point method, FBBT and OBBT are all `jit`/`vmap` kernels. | a Rust core (expression IR, B&B tree, `.nl` parser, FBBT, simplex) with a numpy relaxation layer and Python orchestration. JAX is no longer on its default solve path. |
| **Lower bounds** | McCormick and alphaBB rules applied to each jaxpr primitive; alphaBB is derived automatically from `grad(grad(prim))`. Also a lifted quadratic and polynomial LP with RLT rows, and convex outer approximation. | McCormick over 28 primitives with a table of univariate envelopes, piecewise McCormick, alphaBB, G-convexity, RLT, PSD/SOC, GMI and OA cuts. |
| **Node LPs** | a primal–dual interior-point method in JAX (Mehrotra predictor–corrector, early stop), batched over boxes, with weak-duality-certified bounds. | an in-house Rust simplex with sparse LU and warm starts; HiGHS only for pure LP/MILP. |
| **Search** | batched spatial B&B: many boxes bounded per `vmap` call, pseudocost or width branching, and symmetry pruning on integer polynomial problems. | a native Rust spatial B&B, plus **AMP** (adaptive multivariate partitioning: piecewise MILP relaxations iterated against an NLP) and a signomial global optimizer. |
| **Primal heuristics** | multi-start local NLP (pounce when installed, otherwise optax L-BFGS; scipy on request), dives, rounding repair and integer polish. | multi-start NLP, feasibility pump, diving, RINS, local branching and QUBO local search. |
| **Soundness** | float64 re-check of every point; optional `safe=True` outward rounding; MINLPLib wrong-answer checks and a fuzzer. | outward rounding, MINLPLib `.solu` checks and certificate checks. |
| **Fixed cost** | JAX compilation: seconds per new model structure. A warm cache removes most of it. | small: native code with no compile step per model. |
| **global50, 60 s** | 48 of the 50 run, 45 of those solved, 0 wrong (this repo, 58bb551, October 2026, before PR #76). | 48 of 50 "ok" (incumbent matches the known optimum), at a geomean time about 8x BARON's (discopt report, July 2026). |

**What jaxglobal lacks from discopt.** It has no AMP or piecewise MILP
relaxations, no simplex warm starts, and no GMI or PSD cuts. Its heuristics are
narrower, and every new model structure pays a compile cost. discopt is the
better choice for problems written as equations.

**What jaxglobal is for.** It takes models that are only available as JAX code:
surrogates, `where` logic and simulation loops. It bounds them by tracing,
without rewriting them as equations, and its bounding kernels batch over boxes.
B&B still runs on the host, and GPU throughput has not been measured (see
[Limitations](correctness.md#limitations)).

**Status.** Development is paused at this point. None of the discopt features
above are planned for jaxglobal yet.
