# Benchmarks and results

All times are wall clock on a laptop CPU. Cold times include tracing and
compiling the bounding kernels, which dominate small problems.

## MINLPLib

Solved instances at a 60 s limit, cold start (compilation included), each
answer checked against the library's reference bounds:

| set | jaxglobal | SCIP |
|---|---:|---:|
| `start50`: 50 continuous instances, ≤ 50 variables, weighted toward hard ones | 31 | 44 |
| `int50`: 50 mixed-integer instances, ≤ 50 variables | 29 | 45 |

jaxglobal solves no `int50` instance that SCIP does not.

The subsets are chosen with a fixed seed from MINLPLib by outcome and size band
(`benchmarks/sets/` in the repository lists the instances and how they were
selected). 0 answers were wrong. Shifted geometric mean time at 60 s, cold:
32.6 s against SCIP's 1.9 s on `start50`, and 35.9 s against 1.35 s on
`int50`. A warm rerun, with the kernels already in the disk cache, solves more
instances in the same time limit because it skips compilation.

On 328 mixed-integer MINLPLib instances with up to 100 variables (60 s, cold),
jaxglobal solved 72 of the 220 with at most 50 variables and 1 of the 108 with
51 to 100.

### Running the MINLPLib benchmark

From a clone of the repository:

```bash
python benchmarks/minlplib/fetch.py          # 322 continuous instances, ≤ 50 variables
python benchmarks/run.py --time-limit 60     # jaxglobal, SCIP, Couenne, BARON (GAMS)
python benchmarks/report.py                  # checked results, SGM time, profiles
python benchmarks/check_wrong.py             # exits nonzero on any wrong or unverified answer
```

## Built-in test problems

From [example 03](../examples/03-test-problems.ipynb).

Tolerance: gap ≤ max(1e-5, 1e-6·|f|). *Root LB* is plain interval arithmetic
on the whole box. *mstart* is the fraction of 20 random L-BFGS-B starts that
reach f\*, which shows how multimodal each problem is.

| problem | n | f\* (lit.) | found | proven LB | root LB | mstart | nodes | time |
|---|---|---|---|---|---|---|---|---|
| gramacy_lee | 1 | −0.869011 | −0.869011 | −0.869011 | −1 | 15% | 5 | 3.7s |
| six_hump_camel | 2 | −1.031628 | −1.031628 | −1.031630 | −156 | 95% | 23 | 4.2s |
| mccormick | 2 | −1.913223 | −1.913223 | −1.913225 | −13.5 | 55% | 11 | 2.0s |
| goldstein_price | 2 | 3.000000 | 3.000000 | 3.000000 | −8.79e7 | 60% | 1445 | 13.3s |
| shubert | 2 | −186.730909 | −186.730909 | −186.731063 | −225 | 0% | 303 | 5.5s |
| michalewicz_2d | 2 | −1.801303 | −1.801297 | −1.801303 | −2 | 5% | 11 | 3.6s |
| holder_table | 2 | −19.208503 | −19.208503 | −19.208519 | −33.2 | 0% | 191 | 7.7s |
| eggholder | 2 | −959.640663 | −959.640663 | −959.640663 | −1.07e3 | 0% | 47 | 3.4s |
| hartmann3 | 3 | −3.862782 | −3.862780 | −3.862786 | −8.4 | 50% | 155 | 3.7s |
| styblinski_tang_3d | 3 | −117.498497 | −117.498497 | −117.498551 | −638 | 15% | 9 | 2.1s |
| shekel10_4d | 4 | −10.536443† | −10.536443 | −10.536445 | −35.4 | 25% | 83 | 3.1s |
| hartmann6 | 6 | −3.322368 | −3.322368 | −3.322378 | −8.4 | 55% | 1083 | 4.2s |
| piecewise_2d (where/abs/max) | 2 | — | −0.823199 | −0.823206 | −3.7 | 25% | 27 | 5.0s |
| tanh_net_2d (16-unit MLP) | 2 | — | −0.553102 | −0.553110 | −2.8 | 30% | 287 | 2.7s |

† The usual −10.5364 is f(4,4,4,4); the true minimizer is slightly off that point.
Times are wall-clock on a laptop CPU for a first (cold) call, so they are
mostly JIT compilation; a second solve of the same function reuses the
compiled kernels and takes a small fraction of that (see [example 04](../examples/04-equation-based-solvers.ipynb)). For the two
problems without a published optimum, the tests check the certificate against
a 401×401 brute-force grid.

![Convergence of the bounds](../assets/figures/convergence.png)

Excluded on purpose: sum-of-squares problems with f\* = 0 (Rosenbrock,
Himmelblau, Rastrigin, Ackley, Levy) and Branin. Plain interval arithmetic
already proves their optimum at the root node, so they show nothing about the
relaxations.

## Comparison with equation-based solvers

`examples/compare.py` builds Pyomo models from **the same Python functions**,
using a `jnp`→Pyomo shim on numpy object arrays. It then runs SCIP and Couenne
(via `.nl` files) and BARON and LindoGlobal (via GAMS). `where`/`maximum` and
`tanh` had to be reformulated by hand; [`jaxglobal.export`](export.md)
now does that automatically. BARON rejects every model containing
`sin`/`cos` ("Cannot handle function 'sin'"). Results go to
`examples/compare_results.json`. The notebook
[example 04](../examples/04-equation-based-solvers.ipynb)
walks through the shim and the reformulations, re-measures jaxglobal (cold,
including compilation, and compiled), and draws `figures/benchmark.png`.

**Pyomo side by side.** The [Pyomo comparison notebooks](../pyomo/index.md)
write each model twice, in Pyomo and as a plain JAX function, and solve both:
basic and pooling problems, `where`/`clip`/table lookups, RK4 kinetics, NN
surrogates, and things Pyomo cannot express directly.

**Flagship benchmark.** `jaxglobal.applications` holds code-defined models
(ODE kinetics, regime-switching engineering models, tanh NN surrogates after
[Schweidtmann & Mitsos 2018](../references.md#ref-schweidtmann2019deterministic)), each with a hand-written Pyomo twin.
`benchmarks/flagship.py` runs jaxglobal, SCIP, Couenne and BARON on them and
judges each answer against the best known optimum.

![Comparison with equation-based solvers](../assets/figures/benchmark.png)
