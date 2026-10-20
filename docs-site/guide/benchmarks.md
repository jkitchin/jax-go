# Benchmarks and results

All times are wall clock on a laptop CPU. Cold times include tracing and
compiling the bounding kernels, which dominate small problems.

## MINLPLib

All answers are checked against the library's reference bounds. Outside
MINLPLib, before PR #76 the solver certified a wrong optimum on an
application model (two parallel lines with motor price lists: reported
optimal 130186.35 while 130000.6 is attainable), caused by invalid
subgradients at near-ties; this is now fixed.

* **MINLPLib against SCIP, October 2026** (`main` at 58bb551). An instance
  passes if jaxglobal proves it optimal within max(2 x SCIP's time, 10 s).
  "Cold" includes JAX compilation; "warm" reuses the compile caches from a
  previous run. SCIP has no compile step, so the cold numbers are the fair
  comparison. Measured at 58bb551 (2026-10-19), before PR #76 (relaxation
  subgradient fix); re-run pending.

  | set | instances | pass, cold | pass, warm | where SCIP takes ≥ 0.5 s (warm) |
  |---|---:|---:|---:|---:|
  | development: the targets the recent solver work was tuned on | 79 | 68 | 74 | 20 / 23 |
  | held out: same families, never used during development | 184 | 151 | 160 | 10 / 26 |

  - **Development set.** It is drawn from `start50` and `int50` (41 of the 79 have
    integer variables). Recent PRs fixed these instances one by one, so the
    numbers above are optimistic. The 5 still failing, and why, are listed in
    [`docs/PLAN-scip-gap-v2.md` §10](https://github.com/jkitchin/jax-go/blob/main/docs/PLAN-scip-gap-v2.md).
  - **Held-out set.** Every MINLPLib instance with a SCIP reference time
    that was not used during development: not in `start50`, `int50` or the
    `check_wrong` sample, and not named in the code, tests or docs. The set is
    easier than the development set: all continuous, with a median of 6
    variables and a median SCIP time of 0.03 s.
    - Easy instances (SCIP < 0.5 s) generalize: 150 of 158 pass, warm.
    - Harder instances do not: 10 of 26, against 20 of 23 on the development set.
    - 11 are rejected as unsupported, because bound propagation finds no finite
      bounds for some variables.
  - **Speed.** jaxglobal beats SCIP's time on only 8 of the development
    instances and 3 of the held-out ones. Most passes rely on the 10 s floor.
  - **Wrong answers:** 0 in either set.
  - **To reproduce:** run `benchmarks/run.py --solvers jaxglobal,scip --names
    <list> --time-limit 60` once with `JAXGLOBAL_CACHE=0` (cold) and again on a
    primed cache (warm), then apply the pass rule per instance against SCIP's
    time. `run.py` takes a single time limit, so the rule is applied
    afterwards. The development and held-out instance lists are not checked
    in; the held-out set is defined above.

* **global50, 60 s limit** (the 50-instance BARON comparison panel from
  discopt, `discopt_benchmarks/config/baron_global50.txt`). All solvers ran on
  the same machine. 48 of the 50 ran; casctanks and hda are missing from this
  repo's MINLPLib tables. Measured at 58bb551 (2026-10-19), before PR #76;
  re-run pending.

  | | solved in 60 s | wrong answers | not solved in 60 s |
  |---|---:|---:|---|
  | jaxglobal, cold | 45 / 48 | 0 | clay0303hfsg, cvxnonsep_psig30, tanksize |
  | jaxglobal, warm | 45 / 48 | 0 | the same three |
  | SCIP (command line, `.nl`) | 45 / 48 | 0 | cvxnonsep_nsig30, st_e36, tspn05 |
  | BARON (GAMS) | 47 / 48 | 1 (nvs05: reports 5.887 as optimal; the optimum is 5.471) | — |

  - **Development overlap.** 11 of the 48 are in the development set above, and
    every solver solves all 11. On the other 37, jaxglobal and SCIP each solve
    34.
  - **Under the 2 x SCIP / 10 s rule:** 40 of 48 pass cold and 45 warm.
    nvs09 takes 44 s cold and 2.6 s warm, so compilation dominates the cold
    misses.
  - **Speed.** SCIP and BARON solve most of these instances in under 0.1 s;
    jaxglobal takes 0.5–8 s warm. jaxglobal solves cvxnonsep_nsig30, st_e36 and
    tspn05, which SCIP leaves unsolved, but BARON solves all three faster.

* **MINLPLib, 60 s limit, cold start (compilation included)**, measured on
  2026-10-17, before the SCIP-gap work and PR #76, and not re-run since:

  | set | jaxglobal | SCIP |
  |---|---:|---:|
  | `start50`: 50 continuous instances, ≤ 50 variables, weighted toward hard ones | 31 | 44 |
  | `int50`: 50 mixed-integer instances, ≤ 50 variables | 29 | 45 |

  The sets are defined in [`benchmarks/sets/README.md`](https://github.com/jkitchin/jax-go/blob/main/benchmarks/sets/README.md);
  the 31 and 29 are the issue #14 cold runs in [`docs/ROADMAP.md`](https://github.com/jkitchin/jax-go/blob/main/docs/ROADMAP.md)
  (`results/issue14_*_cold.csv`).
  Branching comparisons ([Solver options](solver-options.md#branch_var-which-variable-to-split)) were separate runs, so their counts differ.

On 328 mixed-integer MINLPLib instances with up to 100 variables (60 s, cold,
measured on 2026-10-17, before the SCIP-gap work and PR #76; re-run pending),
jaxglobal solved 72 of the
220 with at most 50 variables and 1 of the 108 with 51 to 100.

### Running the MINLPLib benchmark

From a clone of the repository:

```bash
python benchmarks/minlplib/fetch.py          # 322 continuous instances, ≤ 50 variables
python benchmarks/run.py --time-limit 60     # jaxglobal, SCIP, Couenne, BARON (GAMS)
python benchmarks/report.py                  # checked results, SGM time, profiles
python benchmarks/check_wrong.py             # exits nonzero on any wrong or unverified answer
```

## Built-in test problems

From [notebook 03](../notebooks/03-test-problems.ipynb).

Tolerance: gap ≤ max(1e-5, 1e-6·|f|). *Root LB* is plain interval arithmetic
on the whole box. *mstart* is the fraction of 20 random L-BFGS-B starts that
reach f\*, which shows how multimodal each problem is.

| problem | n | f\* (lit.) | found | proven LB | root LB | mstart | nodes | time |
|---|---|---|---|---|---|---|---|---|
| gramacy_lee | 1 | −0.869011 | −0.869011 | −0.869011 | −1 | 15% | 5 | 10.7s |
| six_hump_camel | 2 | −1.031628 | −1.031628 | −1.031629 | −156 | 95% | 23 | 7.5s |
| mccormick | 2 | −1.913223 | −1.913223 | −1.913224 | −13.5 | 55% | 11 | 7.0s |
| goldstein_price | 2 | 3.000000 | 3.000000 | 2.999998 | −8.79e7 | 60% | 1447 | 18.1s |
| shubert | 2 | −186.730909 | −186.730909 | −186.731070 | −225 | 0% | 277 | 18.0s |
| michalewicz_2d | 2 | −1.801303 | −1.801301 | −1.801303 | −2 | 5% | 9 | 14.9s |
| holder_table | 2 | −19.208503 | −19.208503 | −19.208518 | −33.2 | 0% | 279 | 18.5s |
| eggholder | 2 | −959.640663 | −959.640663 | −959.640711 | −1.07e3 | 0% | 41 | 12.6s |
| hartmann3 | 3 | −3.862782 | −3.862780 | −3.862786 | −8.4 | 50% | 89 | 11.3s |
| styblinski_tang_3d | 3 | −117.498497 | −117.498497 | −117.498551 | −638 | 15% | 9 | 9.6s |
| shekel10_4d | 4 | −10.536443† | −10.536443 | −10.536453 | −35.4 | 25% | 67 | 11.1s |
| hartmann6 | 6 | −3.322368 | −3.322368 | −3.322377 | −8.4 | 55% | 1035 | 11.2s |
| piecewise_2d (where/abs/max) | 2 | — | −0.823199 | −0.823202 | −3.7 | 25% | 31 | 16.8s |
| tanh_net_2d (16-unit MLP) | 2 | — | −0.553102 | −0.553105 | −2.8 | 30% | 239 | 8.6s |

† The usual −10.5364 is f(4,4,4,4); the true minimizer is slightly off that point.
Times are wall-clock on a laptop CPU for a first (cold) call, so they are
mostly JIT compilation; a second solve of the same function reuses the
compiled kernels and takes a small fraction of that (see [notebook 04](../notebooks/04-equation-based-solvers.ipynb)). For the two
problems without a published optimum, the tests check the certificate against
a 401×401 brute-force grid. Re-run on 2026-10-20 on current `main` (after PRs
#76, #77 and #81) on a 4-core 2.1 GHz cloud VM; the node counts are within a few
nodes of the earlier laptop run at 2ae4c39, and the times are 4–8× longer, most of
it compilation on the slower machine.

![Convergence of the bounds](../assets/figures/convergence.png)

Excluded on purpose: sum-of-squares problems with f\* = 0 (Rosenbrock,
Himmelblau, Rastrigin, Ackley, Levy) and Branin. Plain interval arithmetic
already proves their optimum at the root node, so they show nothing about the
relaxations.

## Comparison with equation-based solvers

`notebooks/compare.py` builds Pyomo models from **the same Python functions**,
using a `jnp`→Pyomo shim on numpy object arrays. It then runs SCIP and Couenne
(via `.nl` files) and BARON and LindoGlobal (via GAMS). `where`/`maximum` and
`tanh` had to be reformulated by hand; [`jaxglobal.export`](export.md)
now does that automatically. BARON rejects every model containing
`sin`/`cos` ("Cannot handle function 'sin'"). Results go to
`notebooks/results/compare_results.json`. The notebook
[notebook 04](../notebooks/04-equation-based-solvers.ipynb)
walks through the shim and the reformulations, re-measures jaxglobal (cold,
including compilation, and compiled), and draws `figures/benchmark.png`.

**Pyomo side by side.** The [Pyomo comparison notebooks](../notebooks/index.md#comparison-with-pyomo)
write each model twice, in Pyomo and as a plain JAX function, and solve both:
basic and pooling problems, `where`/`clip`/table lookups, RK4 kinetics, NN
surrogates, and things Pyomo cannot express directly.

**Flagship benchmark.** `jaxglobal.applications` holds code-defined models
(RK4 ODE kinetics stepped with `lax.scan`, regime-switching engineering models, tanh NN surrogates after
[Schweidtmann & Mitsos 2018](../references.md#ref-schweidtmann2019deterministic)), each with a hand-written Pyomo twin.
`benchmarks/flagship.py` runs jaxglobal, SCIP, Couenne and BARON on them and
judges each answer against the best known optimum.

![Comparison with equation-based solvers](../assets/figures/benchmark.png)
