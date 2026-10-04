<p align="center"><img src="figures/logo.svg" alt="jaxglobal" width="560"></p>

# jaxglobal — global optimization of plain JAX functions (proof of concept)

Write an ordinary Python function with `jax.numpy`, give a box, and get a
**certified** global minimum: an upper bound (a feasible point) and a proven
lower bound. You write no equations and no algebraic model.

```python
import jax.numpy as jnp
from jaxglobal import minimize

def f(x):                       # Eggholder: very rugged, optimum on the boundary
    a, b = x[0], x[1]
    return -(b + 47) * jnp.sin(jnp.sqrt(jnp.abs(a / 2 + b + 47))) \
           - a * jnp.sin(jnp.sqrt(jnp.abs(a - (b + 47))))

r = minimize(f, [-512, -512], [512, 512])
# Result(fun=-959.64066, lower_bound=-959.64144, x=[512. 404.231805], converged=True, nodes=153, ...)
```

## How it works

1. **Trace.** `jax.make_jaxpr(f)` turns the Python function into a flat list of
   primitives (`sin`, `mul`, `integer_pow`, `dot_general`, `slice`, …).
   Python loops, helper functions, `jnp.where`, `jax.nn.*`, and `jit`/`custom_jvp`
   sub-calls all disappear into this list.
2. **Re-interpret the jaxpr on McCormick objects** `MC(lo, hi, cv, cc)` in
   `jaxglobal/relax.py`. Every primitive has a rule:
   * *structural / positive-linear* primitives (slice, reshape, add, reduce_sum,
     concatenate, gather, …) apply component-wise;
   * *signed linear* (neg, sub, mul or `dot_general` by a constant) swap by sign;
   * *bilinear* `mul` uses the McCormick product rule;
   * *univariate* primitives use McCormick composition with known envelopes
     (exp, log, sqrt, xⁿ, |x|, 1/x, pow, cosh, max/min) **or an automatically
     derived αBB relaxation**. For sin, cos, tanh, logistic, erf, atan, …, the
     interpreter traces `jax.grad(jax.grad(prim))`, evaluates *that* jaxpr in
     interval mode to bound f″, and builds `f(z) ± α/2 (z−lo)(z−hi)`.
     Adding a new smooth primitive needs only its range function.
   * comparisons / `select_n` (from `jnp.where`) are resolved when the interval
     decides the branch and fall back to the hull otherwise.
3. **The relaxation is itself a JAX function**, so it can be `jit`-ed,
   `vmap`-ed over hundreds of boxes, and differentiated. `jax.grad(cv)` gives
   subgradients of the convex relaxation. A few Frank–Wolfe steps on cv give
   affine cuts `cv(x) + g·(v − x)`, whose minimum over the box is a valid lower
   bound (`jaxglobal/bnb.py`).
4. **Batched spatial branch-and-bound.** Each iteration bisects the 128
   lowest-bound boxes and bounds all 256 children in one `jit(vmap(...))` call.
   L-BFGS-B (with `jax.grad`) polishes the incumbent.
5. **Bound tightening against the incumbent** (`f <= UB`), all batched:
   * *FBBT* (`jaxglobal/fbbt.py`): the jaxpr is flattened into a tape and swept
     forward (interval rules) and backward (inverse rules: `x ∈ Z − Y` for
     add, `x ∈ log Z` for exp, the two-sided root for `x²ⁿ` and `|x|`, …) to
     a fixed number of rounds before every node is bounded;
   * *duality-based range reduction*: the cut / LP-dual certificate
     `f ≥ lb + Σ r_j (x_j − lo_j)` confines each variable to where it could
     still beat the incumbent, at no extra cost;
   * *OBBT* (`lp.make_obbt`): min and max of every variable over the
     outer-approximation LP, as 2n `vmap`-ed LPs with weak-duality-certified
     results; on the root by default (`obbt="tree"` for every node).
   Children are split from the tightened box.
6. **Univariate envelopes** (`jaxglobal/univariate.py`, on by default). Every
   smooth sub-expression whose elements each depend on one variable (found
   structurally on the jaxpr: separable sums, polynomial factors,
   `sin(x)·sin(x²)^20`, ...) is sampled on a 65-point grid. The samples are
   lowered by `max(g″, 0)·h²/8`, with `g″` bounded per cell by interval
   evaluation of `jvp(jvp(g))`, and their lower convex hull is a nearly exact
   convex envelope (symmetrically for the concave one). It is intersected
   with the McCormick relaxation, so it never loosens it. On Styblinski–Tang
   the root bound goes from −612 to −118.6 (f* = −117.5).
   An extended-space LP bound (`jaxglobal/lifted.py`, `lifted=True`) is also
   available: one LP column per nonlinear jaxpr node, cut from that node's own
   McCormick rule at FBBT-tightened bounds.

![relaxations](figures/relaxations.png)

## Results (`python examples/benchmark.py`)

Tolerance: gap ≤ max(1e-5, 1e-6·|f|). *Root LB* is plain interval arithmetic
on the whole box. *mstart* is the fraction of 20 random L-BFGS-B starts that
reach f\*, which shows how multimodal each problem is.

| problem | n | f\* (lit.) | found | proven LB | root LB | mstart | nodes | time |
|---|---|---|---|---|---|---|---|---|
| gramacy_lee | 1 | −0.869011 | −0.869011 | −0.869016 | −1 | 15% | 23 | 0.4s |
| six_hump_camel | 2 | −1.031628 | −1.031628 | −1.031638 | −156 | 95% | 407 | 0.4s |
| mccormick | 2 | −1.913223 | −1.913223 | −1.913228 | −13.5 | 55% | 41 | 0.2s |
| goldstein_price | 2 | 3 | 3.000000 | 2.999990 | −8.8e7 | 60% | 8883 | 0.6s |
| shubert | 2 | −186.7309 | −186.7309 | −186.7311 | −225 | 0% | 2399 | 0.5s |
| michalewicz_2d | 2 | −1.801303 | −1.801303 | −1.801308 | −2 | 5% | 65 | 0.5s |
| holder_table | 2 | −19.20850 | −19.20850 | −19.20851 | −33.2 | 0% | 415 | 0.7s |
| eggholder | 2 | −959.6407 | −959.6407 | −959.6414 | −1070 | 0% | 153 | 0.5s |
| hartmann3 | 3 | −3.862782 | −3.862780 | −3.862790 | −8.4 | 50% | 1769 | 0.3s |
| styblinski_tang_3d | 3 | −117.4985 | −117.4985 | −117.4986 | −638 | 15% | 451 | 0.2s |
| shekel10_4d | 4 | −10.536443† | −10.536443 | −10.536452 | −35.4 | 25% | 133 | 0.5s |
| hartmann6 | 6 | −3.322368 | −3.322368 | −3.322378 | −8.4 | 55% | 6515 | 0.3s |
| piecewise_2d (where/abs/max) | 2 | — | −0.823199 | −0.823201 | −3.7 | 25% | 87 | 0.5s |
| tanh_net_2d (16-unit MLP) | 2 | — | −0.553102 | −0.553111 | −2.8 | 30% | 387 | 0.3s |

† The usual −10.5364 is f(4,4,4,4); the true minimizer is slightly off that point.
Times are wall-clock on a laptop CPU and include JIT compilation. For the two
problems without a published optimum, the tests check the certificate against
a 401×401 brute-force grid.

![convergence](figures/convergence.png)

Excluded on purpose: sum-of-squares problems with f\* = 0 (Rosenbrock,
Himmelblau, Rastrigin, Ackley, Levy) and Branin. Plain interval arithmetic
already proves their optimum at the root node, so they show nothing about the
relaxations.

## Constraints

`minimize(f, lo, hi, ineq=g, eq=h)` handles `g(x) <= 0` and `h(x) == 0`, where
`g` and `h` are plain JAX functions returning a scalar or a 1-D array.

* **Lower bound** (`jaxglobal/lp.py`): the relaxations give convex functions
  `cv_f`, `cv_g`, `cv_h`, `-cc_h`. `jax.jacfwd` turns them into affine cuts at
  the box center, 2n axis points, and 4 Kelley points. The small LP is solved by
  a fixed-iteration primal–dual interior point method in JAX, so it `vmap`s over
  boxes. The bound **does not trust the LP solver**. Weak duality gives a valid
  bound for *any* multipliers y ≥ 0, and the best value over all IPM iterates is
  kept. A badly converged LP gives a weaker bound, never a wrong one.
  Interval arithmetic prunes boxes that are provably infeasible.
* **Upper bound**: scipy SLSQP with JAX gradients and Jacobians, started from
  the LP solutions. A point counts as feasible when the violation is ≤ `feas_tol`.
* **Branching** only on variables that appear nonlinearly, detected from
  sampled Hessians.
* `status` is `"optimal"`, `"infeasible"` or `"max_iter"`.

| problem | n | f\* | found | proven LB | nodes | time |
|---|---|---|---|---|---|---|
| G06 | 2 | −6961.81388 | −6961.81388 | −6961.8200 | 61 | 2.2s |
| G08 | 2 | −0.0958250 | −0.0958250 | −0.0958257 | 99 | 3.8s |
| G11 (equality) | 2 | 0.75 | 0.7499996 | 0.7499989 | 127 | 1.3s |
| G24 (disconnected feasible set) | 2 | −5.508013 | −5.508013 | −5.508016 | 43 | 1.7s |
| G04 | 5 | −30665.539 | −30665.539 | −30665.552 | 67 | 5.8s |
| Haverly pooling | 9 | −400 | −400 | −400.0004 | 157 | 20s |
| Reactor network | 6 | −0.388811 | −0.3888114 | −0.3888124 | 1935 | 12.7s |

Equality optima can sit slightly below f\* because of `feas_tol` (1e-6).

## Comparison with equation-based solvers

`examples/compare.py` builds Pyomo models from **the same Python functions**,
using a `jnp`→Pyomo shim on numpy object arrays. It then runs SCIP and Couenne
(via `.nl` files) and BARON and LindoGlobal (via GAMS). `where`/`maximum` and
`tanh` had to be reformulated by hand. BARON rejects every model containing
`sin`/`cos` ("Cannot handle function 'sin'"). Results go to
`examples/compare_results.json`, and `examples/compare_plot.py` draws
`figures/benchmark.png`. The slides in `slides/jaxglobal.md` summarize this.

**Pyomo side by side.** The notebooks in [`notebooks/`](notebooks/README.md)
write each model twice, in Pyomo and as a plain JAX function, and solve both:
basic and pooling problems, `where`/`clip`/table lookups, RK4 kinetics, NN
surrogates, and things Pyomo cannot express directly.

**Flagship benchmark.** `jaxglobal.applications` holds code-defined models
(ODE kinetics, regime-switching engineering models, tanh NN surrogates after
[schweidtmann2019deterministic]), each with a hand-written Pyomo twin.
`benchmarks/flagship.py` runs jaxglobal, SCIP, Couenne and BARON on them and
judges each answer against the best known optimum.

## Correctness checks

`minimize(..., safe=True)` widens every relaxation outward by a few ulps so
floating-point rounding cannot cut off the global minimum (off by default, about
1.7× slower). Every result is re-evaluated in float64; a point that violates the
constraints is reported with status `"unverified"` and its `max_violation`.
`tests/test_fuzz.py` checks bounds against sampled points on random problems
(`JAXGLOBAL_FUZZ=N` for more cases), and `benchmarks/check_wrong.py` exits
nonzero on any wrong or unverified MINLPLib answer.

## Run

```bash
pip install -e '.[dev]'
pytest -q                      # soundness, cut validity, global optima (57 tests, ~30 s)
python examples/benchmark.py
python examples/figures.py

# MINLPLib benchmark (phase 0 of docs/ROADMAP.md)
python benchmarks/minlplib/fetch.py          # 322 continuous instances, ≤ 50 variables
python benchmarks/run.py --time-limit 60     # jaxglobal, SCIP, Couenne, BARON (GAMS)
python benchmarks/report.py                  # checked results, SGM time, profiles
```

Any AMPL `.nl` model (text or binary) can be loaded with
`jaxglobal.nl.read_nl(path)`, which returns `f`, `ineq`, `eq`, `lo`, `hi` for
`minimize`.

## Lessons / gotchas found so far

* **Autodiff subgradients need tie-aware rules.** At points where cv == cc (box
  vertices, which is exactly where Frank–Wolfe iterates go), `jnp.minimum`
  splits the gradient 50/50 between its arguments. In the product rule and in
  the composition `mid(·)` this blends the *concave* relaxation's gradient into
  a "subgradient" of the convex one. The result is invalid cuts and a wrong
  certificate (Eggholder reported −935 as optimal). The fix is to select
  branches with `jnp.where` on sign or ordering. `test_cuts_valid` guards this
  and fails on 9 problems if the fix is reverted.

## Limitations (it's a PoC)

* Not rigorous in floating point: no outward rounding, and αBB minimizers come
  from 60-step bisection.
* Constrained bounds use a dense LP per node: fine for ≤ 10 variables, not
  beyond. Upper bounds come from local solves with pounce (default), optax or
  scipy (`local_solver=`).
* `dot_general` between two variable arrays only for 1-D inner products.
  `lax.cond`/`while`/`scan` with variable-dependent control flow are not
  supported; Python control flow on static values is fine.
* Unsupported primitives raise `NotImplementedError` naming the primitive.
* Variables may be declared unbounded (`±inf`). Finite bounds are derived
  first (`jaxglobal/domain.py`): FBBT from the declared box, unused variables
  fixed, then FBBT against `f <= UB` once a local solve finds a feasible
  point. A variable that nothing bounds gives status `"unbounded_domain"`
  rather than a guessed box; FBBT cannot prove bounds that need joint
  reasoning over several constraints (e.g. G24 without its upper bounds).
* Bounds are first-order McCormick/αBB plus interval arithmetic. Tightened
  intermediate ranges from FBBT are not yet fed back into the relaxations
  (Wechsung-style reverse McCormick), and there is no probing.
* The first solve of a problem is mostly compilation on the small test
  problems. Compiled kernels are cached, so repeat solves of the same
  functions (on any box) are much faster; `jaxglobal.bnb.clear_cache()`
  empties the cache.

## Natural next steps

* Harder constrained sets (larger pooling problems, MINLPLib `ex8_*`).
* Feed FBBT's intermediate ranges into the McCormick rules.
* Multivariate McCormick / tighter envelopes for common patterns
  (x·exp(x), x·log(x), bilinear `dot_general`).
* Outward rounding (e.g. `nextafter`) for rigorous bounds.
* GPU: the bounding step is already a pure `jit(vmap)` kernel.

## Related work

Citation keys refer to [`docs/references.bib`](docs/references.bib). Every
entry with a DOI was checked against Crossref and its BibTeX taken from
Crossref by `docs/verify_refs.py` (116 entries). The 39 entries without a
Crossref DOI (conference papers, arXiv, software, reports) are written by
hand in `docs/refs.json`; the same script confirms each one from an
independent source: the arXiv record, the DOI record, OpenAlex, the GitHub
repository, or the entry's title on the publisher's page. The CEC 2006
report has no online record and was checked against its title page.
`tests/test_citations.py` fails on any `[key]` in the repo's `.py`, `.md` or
notebook sources that is not in the bibliography.

**McCormick relaxations of code.** The relaxation arithmetic is McCormick's
[mccormick1976computability], in the generalized and subgradient form of
[mitsos2009mccormick, scott2011generalized]. That work relaxes whole programs
by operator overloading (MC++ [chachuatmcpp, chachuat2015settheoretic]); MAiNGO
[bongartz2018maingo, bongartz2017deterministic] and EAGO.jl
[wilhelm2022eagojl] are the solvers built on it. Reduced-space branching on
the original variables follows [epperly1997reduced]. Convergence theory:
[bompadre2012convergence, najman2016convergence]. Possible tightenings:
multivariate rules [tsoukalas2014multivariate], subgradient bound tightening
[najman2019tighter], reverse propagation [wechsung2015reverse], LP from
linearizations [najman2021linearization], Taylor and McCormick–Taylor models
[makino2003taylor, sahlodin2011convex, bompadre2013convergence], DAG interval propagation [schichl2005interval],
differentiable relaxations [khan2017differentiable], and relaxations of implicit
functions [stuber2015convex], a route to relaxing `lax.while_loop`.
Embedded ML models: [schweidtmann2019deterministic,
schweidtmann2021deterministic, wilhelm2023convex]. jaxglobal's difference: no
operator overloading or code generation. The relaxation is a *re-interpretation
of the jaxpr*, so `jit`, `vmap` and `grad` apply to the bounding itself.

**αBB and interval Hessians.** αBB [androulakis1995alphabb,
adjiman1996rigorous, adjiman1998global, adjiman1998globalii] with tighter α
[hladik2015efficient, nerantzis2019tighter, kazazakis2018arbitrarily,
skjal2012generalization] and univariate underestimators
[gounaris2008tight, meyer2005convex]. The closest prior art to evaluating
`grad(grad(primitive))` on intervals is AD-based interval Hessian bounding
[monnigmann2008efficient, schulzedarup2014efficient, schulzedarup2016improved,
hladik2017effect]. We found no prior work that applies αBB *automatically per
primitive* to every nonconvex primitive of a traced program.

**GPU and parallel global optimization.** The most direct competitors:
Gottlieb, Xu & Stuber generate McCormick kernels from Julia/Symbolics for
GPU-batched lower bounding [gottlieb2024automatic, gottlieb2026automatic];
Zhang et al. add GPU interval bounding to MAiNGO [zhang2025accelerating].
Earlier GPU/parallel B&B: [barkalov2016parallel, bagoczki2017parallel,
gillner2024gpuaccelerated, gendron1994parallel, crainic2006parallel,
gmys2016gpubased]. Spatial B&B and its convergence theory: [falk1969algorithm,
horst1990global, smith1999symbolic, kannan2018convergenceorder]. Rigorous
and interval global optimization: [neumaier2004complete, kearfott1996rigorous,
hansen2004global]. Branching rules: [achterberg2005branching].

**Bound propagation in ML and jaxpr interpreters.** jaxpr-interpreting
bounders already exist, but for other jobs:
- jax_verify [stanforth2020jaxverify]: NN verification, with bilinear McCormick
  and B&B over activation/input splits.
- AutoBound [streeter2022automatically, streeter2023sharp]: Taylor enclosures,
  with a toy B&B.
- immrax [harapanahalli2024immrax]: interval reachability.

The verification literature has GPU-batched bounding in B&B:
- linear bounds: IBP [gowal2018effectiveness], CROWN [zhang2018efficient],
  DeepPoly [singh2019abstract], auto_LiRPA [xu2020automatic];
- B&B: α/β-CROWN [xu2021fast, wang2021beta], GenBaB [shi2025neural],
  [bunel2020branch];
- dual and SDP relaxations: [dvijotham2018dual, dvijotham2020efficient,
  bunel2020lagrangian];
- relaxation-gap results: [salman2019convex, tjandraatmadja2020convex].

AD at kinks can return values that are not subgradients
[kakade2018provably, lee2020correctness, bolte2020mathematical,
bolte2021conservative]. This is the failure mode behind jaxglobal's tie-aware
rules (see *Lessons* above). Nonsmooth implicit differentiation:
[bolte2024differentiating]. Principled fixes: abs-normal forms
[griewank2013stable, griewank2015solving] and lexicographic derivatives
[nesterov2005lexicographic, khan2013evaluating, khan2015vector].

**LP bounds.** Node bounds come from cuts [kelley1960cuttingplane,
frank1956algorithm, jaggi2013revisiting], fed to an interior-point LP
[mehrotra1992implementation, wright1997primaldual]. The bound is made safe by
weak duality with box bounds, following [neumaier2004safe, jansson2004rigorous,
keil2006computational]. That safety is what allows first-order GPU LP solvers
in the tree: PDLP [applegate2021practical, applegate2026pdlp,
applegate2023faster, applegate2024infeasibility], cuPDLP
[lu2025cupdlpjl, lu2025practical], and MPAX in JAX [lu2024mpax]. Batched
differentiable solvers: [amos2017optnet, agrawal2019differentiable].

**Global solvers and bound tightening** (comparison baselines and roadmap
targets):
- solvers: BARON [tawarmalani2005polyhedral, khajavirad2018hybrid], SCIP
  [vigerske2018scip, bestuzheva2025global, bolusani2024scip, hojny2025scip],
  Couenne [belotti2009branching], ANTIGONE [misener2014antigone],
  LindoGlobal [lin2009global];
- bound tightening: FBBT [belotti2010feasibilitybased], domain reduction
  [puranik2017domain, ryoo1996branchandreduce];
- checking solver claims against reference values: [neumaier2005comparison],
  which `benchmarks/report.py` follows.

**Local solvers and infrastructure.**
- local solvers: pounce derives from Ipopt [wachter2006implementation];
  L-BFGS(-B) [liu1989limited, byrd1995limited], SLSQP [kraft1988software];
- augmented Lagrangian: [hestenes1969multiplier, powell1969method];
- tools: SciPy [virtanen2020scipy], JAX [bradbury2018jax, frostig2018compiling],
  optax [deepmind2020deepmind], AD [griewank2008evaluating];
- modeling for the comparisons: Pyomo [hart2011pyomo, bynum2021pyomo], GAMS
  [brooke1988gams];
- related MINLP work by the author: discopt [kitchin2026discopt], which uses
  an algebraic modeling API rather than tracing;
- ML-embedded optimization: [ceccon2022omlt, tsay2021partition,
  kronqvist2025psplit, anderson2020strong, fischetti2018deep,
  grimstad2019relu].

**Test problems.**
- sources: [floudas1999handbook, ryoo1995global, haverly1978studies,
  michalewicz1996evolutionary, liang2006problem, surjanovic2013virtual,
  jamil2013literature];
- MINLPLib [bussieck2003minlplib].
